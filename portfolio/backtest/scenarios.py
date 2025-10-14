# portfolio/backtest/scenarios.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import polars as pl

from portfolio.backtest.engine import backtest_rebalanced
from portfolio.backtest import metrics as bt_metrics

__all__ = [
    "ShockSpec",
    "ScenarioConfig",
    "ScenarioResult",
    "run_scenarios",
]

# ─────────────────────────────────────────────────────────────────────
# Datatypes
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ShockSpec:
    """
    Scenario shock specification applied to a (T,N) daily returns matrix:
      - mean_shift: add constant daily drift (scalar or shape (N,))
      - cov_scale : scale deviations from mean (vol/dispersion scale)
      - crash     : (t_index, drop) apply a one-day gap at t_index
                    'drop' can be scalar (same for all assets) or (N,)
    """
    mean_shift: Optional[float | np.ndarray] = None
    cov_scale: float = 1.0
    crash: Optional[Tuple[int, float | np.ndarray]] = None


@dataclass(frozen=True)
class ScenarioConfig:
    """
    Configuration for a scenario run. If B>0, run block-bootstrap resamples.
    """
    name: str
    B: int = 0
    block: int = 10
    seed: int = 42
    shock: ShockSpec = ShockSpec()


@dataclass
class ScenarioResult:
    """
    Output container for each scenario/backtest pair.
    """
    name: str
    bt: Dict
    metrics: pl.DataFrame
    shock: ShockSpec


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _to_numpy_wide(df_ret_wide: pl.DataFrame, tickers: List[str]) -> np.ndarray:
    """Extract a (T,N) numpy array from a wide Polars DF."""
    R = df_ret_wide.select(tickers).to_numpy()
    return np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

def _from_numpy_wide(dates: List, tickers: List[str], R: np.ndarray) -> pl.DataFrame:
    """Build a wide Polars DF from numpy (T,N)."""
    return pl.DataFrame({"date": dates, **{t: R[:, j] for j, t in enumerate(tickers)}})

def _bootstrap_paths(R: np.ndarray, B: int, block: int, seed: int) -> List[np.ndarray]:
    """Block bootstrap paths; returns [R] when B<=0 (i.e., original chronology)."""
    if B <= 0:
        return [R.copy()]
    T, _ = R.shape
    rng = np.random.default_rng(seed)
    out: List[np.ndarray] = []
    for _ in range(B):
        idx = []
        while len(idx) < T:
            start = int(rng.integers(0, max(T - block, 1)))
            idx.extend(range(start, min(start + block, T)))
        idx = np.array(idx[:T])
        out.append(R[idx, :])
    return out

def _apply_shock(R: np.ndarray, shock: ShockSpec) -> np.ndarray:
    """Apply mean shift, cov scale, and single-day crash to returns matrix."""
    R2 = R.copy()
    T, N = R2.shape

    # mean shift
    if shock.mean_shift is not None:
        ms = np.asarray(shock.mean_shift)
        if ms.size == 1:
            R2 += float(ms)
        else:
            R2 += ms.reshape(1, -1)

    # dispersion scaling
    if abs(shock.cov_scale - 1.0) > 1e-12:
        mu = np.mean(R2, axis=0, keepdims=True)
        R2 = mu + (R2 - mu) * float(shock.cov_scale)

    # one-day crash
    if shock.crash is not None:
        t_idx, drop = shock.crash
        if 0 <= t_idx < T:
            d = np.asarray(drop)
            if d.size == 1:
                R2[t_idx, :] += float(d)
            else:
                R2[t_idx, :] += d.reshape(1, -1)

    return R2


# ─────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────

def run_scenarios(
    cfgs: List[ScenarioConfig],
    df_ret_wide: pl.DataFrame,
    allocator_factory: Callable[[], Callable[[pl.DataFrame], np.ndarray]],
    *,
    lookback: int,
    rebalance_freq: str,
    cost_bps: float,
    bench_weights: np.ndarray,
) -> List[ScenarioResult]:
    """
    For each ScenarioConfig, generate one or more return paths (bootstrap),
    apply shocks, run backtests with allocator_factory(), and collect metrics.
    """
    dates = df_ret_wide.get_column("date").to_list()
    tickers = [c for c in df_ret_wide.columns if c != "date"]
    R_base = _to_numpy_wide(df_ret_wide, tickers)

    out: List[ScenarioResult] = []
    for cfg in cfgs:
        paths = _bootstrap_paths(R_base, cfg.B, cfg.block, cfg.seed)
        for b_ix, R_path in enumerate(paths):
            R_shocked = _apply_shock(R_path, cfg.shock)
            df_path = _from_numpy_wide(dates, tickers, R_shocked)

            alloc = allocator_factory()
            bt = backtest_rebalanced(
                df_ret_wide=df_path,
                lookback=int(lookback),
                rebalance_freq=rebalance_freq,
                cost_bps=float(cost_bps),
                allocator=alloc,
                bench_weights=bench_weights,
            )
            m = bt_metrics.compute_backtest_metrics(bt)
            name = cfg.name if cfg.B <= 1 else f"{cfg.name} #{b_ix+1}"
            out.append(ScenarioResult(name=name, bt=bt, metrics=m, shock=cfg.shock))
    return out