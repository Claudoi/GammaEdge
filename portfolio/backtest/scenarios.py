# portfolio/backtest/scenarios.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl
import pandas as pd

from portfolio.backtest.engine import backtest_rebalanced
from portfolio.backtest import metrics as bt_metrics


__all__ = [
    "ShockSpec",
    "ScenarioConfig",
    "apply_shock_map_to_wide",
    "apply_shock",
    "historical_slice_returns",
    "block_bootstrap_indices",
    "run_scenarios",
]


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------
@dataclass
class ShockSpec:
    """
    Return-space shock applied to a wide return matrix (Polars).

    mean_shift: constant additive drift per period (e.g., 0.0001 = +1bp/day)
    cov_scale : scale factor for cross-sectional deviations from the mean
    crash     : optional (row_index, add_return) one-day additive gap
    """
    mean_shift: Optional[float] = None
    cov_scale: float = 1.0
    crash: Optional[Tuple[int, float]] = None


@dataclass
class ScenarioConfig:
    """Configuration for a scenario backtest."""
    name: str
    B: int = 0
    block: int = 10
    seed: int = 42
    shock: Optional[ShockSpec] = None


# ---------------------------------------------------------------------
# Shock application
# ---------------------------------------------------------------------
def apply_shock_map_to_wide(
    df_wide: pl.DataFrame,
    shock_map: Dict[str, Union[float, Tuple[int, float]]],
) -> pl.DataFrame:
    """
    Apply additive shocks to a wide return DataFrame.

    Special keys:
        '__mean__'      -> float, additive drift to all asset columns
        '__cov_scale__' -> float, scale deviations from cross-sectional mean
        '__crash__'     -> tuple (row_index, add_return) one-day additive gap

    Other keys are interpreted as column-specific additive shifts.
    """
    cols = [c for c in df_wide.columns if c != "date"]
    if not cols:
        return df_wide

    out = df_wide

    # 1) Cross-sectional deviation scaling
    cov_scale = float(shock_map.get("__cov_scale__", 1.0))  # type: ignore
    if abs(cov_scale - 1.0) > 1e-12:
        X = out.select(cols).to_numpy()
        mu = np.nanmean(X, axis=1, keepdims=True)
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = mu + cov_scale * (X - mu)
        out = out.with_columns(**{c: pl.Series(X_scaled[:, j]) for j, c in enumerate(cols)})

    # 2) Mean shift
    mean_shift = shock_map.get("__mean__", None)
    if mean_shift is not None:
        ms = float(mean_shift)  # type: ignore
        out = out.with_columns(**{c: (pl.col(c).fill_null(0.0) + ms) for c in cols})

    # 3) One-day crash
    crash = shock_map.get("__crash__", None)
    if crash is not None:
        idx, add_ret = crash  # type: ignore
        idx = int(idx)
        add_ret = float(add_ret)
        if 0 <= idx < out.height:
            out = out.with_row_count("rowid")
            out = out.with_columns([
                pl.when(pl.col("rowid") == idx)
                  .then(pl.col(c).fill_null(0.0) + add_ret)
                  .otherwise(pl.col(c))
                  .alias(c)
                for c in cols
            ]).drop("rowid")

    # 4) Column-specific additive shifts
    for k, v in shock_map.items():
        if k in {"__mean__", "__cov_scale__", "__crash__", "date"}:
            continue
        if k in out.columns:
            out = out.with_columns(pl.col(k).fill_null(0.0) + float(v))  # type: ignore

    return out


def apply_shock(df_wide: pl.DataFrame, shock: Optional[ShockSpec]) -> pl.DataFrame:
    """Convenience wrapper to apply a ShockSpec to a Polars wide return frame."""
    if shock is None:
        return df_wide
    shock_map: Dict[str, Union[float, Tuple[int, float]]] = {}
    if shock.mean_shift is not None:
        shock_map["__mean__"] = float(shock.mean_shift)
    if float(shock.cov_scale) != 1.0:
        shock_map["__cov_scale__"] = float(shock.cov_scale)
    if shock.crash is not None:
        shock_map["__crash__"] = (int(shock.crash[0]), float(shock.crash[1]))
    return apply_shock_map_to_wide(df_wide, shock_map) if shock_map else df_wide


# ---------------------------------------------------------------------
# Historical slice and bootstrap
# ---------------------------------------------------------------------
def historical_slice_returns(
    df_wide: pl.DataFrame,
    start: str,
    end: str,
    tickers: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Extract a historical time slice from a wide return matrix."""
    d0 = pl.datetime.strptime(start, fmt="%Y-%m-%d")
    d1 = pl.datetime.strptime(end, fmt="%Y-%m-%d")
    cols = ["date", *(tickers if tickers else [c for c in df_wide.columns if c != "date"])]
    out = (
        df_wide
        .select(cols)
        .filter((pl.col("date") >= d0) & (pl.col("date") <= d1))
        .sort("date")
    )
    if out.height == 0:
        raise ValueError("historical_slice_returns: empty slice.")
    return out


def block_bootstrap_indices(T: int, block: int, seed: int) -> np.ndarray:
    """Generate block bootstrap indices for synthetic paths."""
    rng = np.random.default_rng(seed)
    idx: List[int] = []
    i = 0
    while i < T:
        start = rng.integers(0, max(T - block, 1))
        take = min(block, T - i)
        idx.extend(range(start, start + take))
        i += take
    return np.array(idx, dtype=int)


# ---------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------
def run_scenarios(
    cfgs: Sequence[ScenarioConfig],
    df_ret_wide: pl.DataFrame,
    allocator_factory: Callable[[], Callable[[pl.DataFrame], np.ndarray]],
    lookback: int,
    rebalance_freq: str,
    cost_bps: float,
    bench_weights: np.ndarray,
    compute_metrics: Callable[[Dict], Union[pl.DataFrame, "pd.DataFrame"]] = bt_metrics.compute_backtest_metrics,
) -> List[Dict[str, Union[str, Dict, pl.DataFrame]]]:
    """
    Run a list of ScenarioConfig objects and return:
        [{'name': str, 'bt': dict, 'metrics': pl.DataFrame}, ...]

    - If B > 0, block-bootstrap B paths, average 1-row metrics across replications,
      and return the last backtest object for plotting.
    - If B == 0, run a single shocked (or unshocked) path.
    """
    results: List[Dict[str, Union[str, Dict, pl.DataFrame]]] = []
    T = df_ret_wide.height

    for cfg in cfgs:
        if cfg.B and cfg.B > 0:
            agg_metrics: List[pl.DataFrame] = []
            last_bt: Optional[Dict] = None

            for b in range(cfg.B):
                idx = block_bootstrap_indices(T, max(2, int(cfg.block)), int(cfg.seed) + b)
                df_b = df_ret_wide[idx]
                df_b = apply_shock(df_b, cfg.shock)

                alloc = allocator_factory()
                bt = backtest_rebalanced(
                    df_ret_wide=df_b,
                    lookback=int(lookback),
                    rebalance_freq=rebalance_freq,
                    cost_bps=float(cost_bps),
                    allocator=alloc,
                    bench_weights=bench_weights,
                )
                last_bt = bt
                try:
                    m = compute_metrics(bt)
                except Exception:
                    m = pl.DataFrame({"CAGR": [np.nan], "Sharpe": [np.nan], "MaxDD": [np.nan]})
                # Ensure Polars
                if not isinstance(m, pl.DataFrame):
                    try:
                        import pandas as pd
                        if isinstance(m, pd.DataFrame):
                            m = pl.from_pandas(m)
                    except Exception:
                        m = pl.DataFrame({"CAGR": [np.nan], "Sharpe": [np.nan], "MaxDD": [np.nan]})
                agg_metrics.append(m)

            # Average scalar metrics across replications
            cols = list({c for m in agg_metrics for c in m.columns})
            avg_row: Dict[str, float] = {}
            for c in cols:
                vals: List[float] = []
                for m in agg_metrics:
                    try:
                        if c in m.columns and m.height > 0:
                            v = m.get_column(c)[0]
                            vals.append(float(v) if np.isfinite(v) else np.nan)
                        else:
                            vals.append(np.nan)
                    except Exception:
                        vals.append(np.nan)
                if vals:
                    avg_row[c] = float(np.nanmean(vals))
            m_avg = pl.DataFrame([avg_row]) if avg_row else pl.DataFrame()

            results.append({"name": cfg.name, "bt": (last_bt or {}), "metrics": m_avg})

        else:
            # Single-path scenario
            df_b = apply_shock(df_ret_wide, cfg.shock)
            alloc = allocator_factory()
            bt = backtest_rebalanced(
                df_ret_wide=df_b,
                lookback=int(lookback),
                rebalance_freq=rebalance_freq,
                cost_bps=float(cost_bps),
                allocator=alloc,
                bench_weights=bench_weights,
            )
            try:
                m = compute_metrics(bt)
            except Exception:
                m = pl.DataFrame({"CAGR": [np.nan], "Sharpe": [np.nan], "MaxDD": [np.nan]})
            # Ensure Polars
            if not isinstance(m, pl.DataFrame):
                try:
                    import pandas as pd
                    if isinstance(m, pd.DataFrame):
                        m = pl.from_pandas(m)
                except Exception:
                    m = pl.DataFrame({"CAGR": [np.nan], "Sharpe": [np.nan], "MaxDD": [np.nan]})

            results.append({"name": cfg.name, "bt": bt, "metrics": m})

    return results