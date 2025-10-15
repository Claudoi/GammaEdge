# portfolio/backtest/scenarios.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Sequence, Union

import numpy as np
import polars as pl
import pandas as pd

from portfolio.backtest.engine import backtest_rebalanced
from portfolio.backtest import metrics as bt_metrics

try:
    from portfolio.viz.plot_utils import plot_tornado_sensitivity
    _HAS_TORNADO = True
except Exception:
    _HAS_TORNADO = False

# -----------------------------
# Datatypes
# -----------------------------
@dataclass(frozen=True)
class ShockSpec:
    """
    Scenario shock specification. All fields are optional; use only what you need.
    - mean_shift: add a constant daily drift (vector or scalar) to asset returns
    - cov_scale: multiply (idiosyncratic) vol; e.g. 1.5 makes returns noisier
    - crash: (date_index, pct_drop) applies a single-day gap in all assets (or vector)
    """
    mean_shift: Optional[float | np.ndarray] = None
    cov_scale: float = 1.0
    crash: Optional[Tuple[int, float | np.ndarray]] = None


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    B: int = 0                 # bootstrap paths (0 = deterministic/stressed original)
    block: int = 10            # block length for bootstrap
    seed: int = 42
    shock: ShockSpec = ShockSpec()


@dataclass
class ScenarioResult:
    name: str
    bt: Dict                  # full backtest dict (equity, weights, etc.)
    metrics: pl.DataFrame     # metrics DF
    shock: ShockSpec


# -----------------------------
# Helpers
# -----------------------------
def _to_numpy_wide(df_ret_wide: pl.DataFrame, tickers: List[str]) -> np.ndarray:
    R = df_ret_wide.select(tickers).to_numpy()
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    return R

def _from_numpy_wide(dates: List, tickers: List[str], R: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({"date": dates, **{t: R[:, j] for j, t in enumerate(tickers)}})

def _bootstrap_paths(R: np.ndarray, B: int, block: int, seed: int) -> List[np.ndarray]:
    """
    Block bootstrap paths over rows (time). Each path has the same length as R.
    """
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
    """
    Apply shock to returns matrix R (T, N).
    - mean_shift: add daily drift
    - cov_scale: scale deviations from mean
    - crash: inject a single-day drop
    """
    R2 = R.copy()
    T, N = R2.shape
    if shock.mean_shift is not None:
        ms = np.asarray(shock.mean_shift)
        if ms.size == 1:
            R2 += float(ms)
        else:
            ms = ms.reshape(1, -1)
            R2 += ms

    if abs(shock.cov_scale - 1.0) > 1e-12:
        mu = np.mean(R2, axis=0, keepdims=True)
        R2 = mu + (R2 - mu) * float(shock.cov_scale)

    if shock.crash is not None:
        t_idx, drop = shock.crash
        if 0 <= t_idx < T:
            d = np.asarray(drop)
            if d.size == 1:
                R2[t_idx, :] += float(d)  # drop is negative number, e.g. -0.05
            else:
                R2[t_idx, :] += d.reshape(1, -1)

    return R2


def build_index_shock(
    df_wide: pl.DataFrame,
    index_col: str,
    index_move: float = -0.05,
    lookback: int = 252,
) -> dict[str, float]:
    """
    Estimate rolling betas vs `index_col` using last `lookback` obs (OLS on demeaned returns).
    Returns a {ticker: beta * index_move} shock map (additive daily return shock).
    """
    cols = [c for c in df_wide.columns if c not in ("date",)]
    assert index_col in cols, f"{index_col=} not in df columns"
    # slice last lookback
    df_tail = df_wide.sort("date").tail(lookback).select(["date", *cols])
    X = df_tail.get_column(index_col).fill_null(0.0).to_numpy()
    X = X - np.nanmean(X)
    shock = {}
    for c in cols:
        if c == index_col:
            shock[c] = float(index_move)  # index itself moves by index_move
            continue
        y = df_tail.get_column(c).fill_null(0.0).to_numpy()
        y = y - np.nanmean(y)
        denom = float(np.sum(X * X)) + 1e-12
        beta = float(np.sum(X * y) / denom)
        shock[c] = float(beta * index_move)
    return shock

def apply_shock_map_to_wide(
    df_wide: pl.DataFrame,
    shock_map: dict[str, float],
) -> pl.DataFrame:
    """
    Add (NaN-safe) constant daily return shocks per column.
    """
    out = df_wide
    for k, v in shock_map.items():
        if k in out.columns:
            out = out.with_columns(pl.col(k).fill_null(0.0) + float(v))
    return out

def historical_slice_returns(
    df_wide: pl.DataFrame,
    start: str,
    end: str,
    tickers: list[str] | None = None,
) -> pl.DataFrame:
    """
    Return df subset between [start, end], keeping ['date', *tickers].
    """
    cols = [c for c in df_wide.columns if c != "date"] if tickers is None else list(tickers)
    df = (df_wide
          .filter((pl.col("date") >= pl.datetime.strptime(start, "%Y-%m-%d")) &
                  (pl.col("date") <= pl.datetime.strptime(end, "%Y-%m-%d")))
          .sort("date")
          .select(["date", *cols]))
    return df


# -----------------------------
# API
# -----------------------------
def run_scenarios(
    cfgs: List[ScenarioConfig],
    df_ret_wide: pl.DataFrame,                 # ['date', tickers...]
    allocator_factory: Callable[[], Callable[[pl.DataFrame], np.ndarray]],
    *,
    lookback: int,
    rebalance_freq: str,
    cost_bps: float,
    bench_weights: np.ndarray,
) -> List[ScenarioResult]:
    """
    Run one or more scenarios. For each config:
      - bootstrap paths (if B>0),
      - apply shocks,
      - backtest with provided allocator.
    Returns list of ScenarioResult (one result if B==0; B results if B>0).
    """
    # Dates / tickers
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



# --- Advanced scenario helpers (additive; safe to include) --------------------

def estimate_rolling_beta(
    df_ret_wide: pl.DataFrame,
    index_col: str,
    lookback: int = 252,
) -> tuple[np.ndarray, list[str]]:
    """
    OLS beta of each asset vs `index_col` using last `lookback` rows.
    Returns (betas, tickers) where tickers excludes ['date', index_col].
    NaN-safe: fills NaN with 0 before regression.
    """
    if index_col not in df_ret_wide.columns:
        raise ValueError(f"index_col '{index_col}' not found in returns")

    df = df_ret_wide.sort("date").tail(lookback)
    tickers = [c for c in df.columns if c not in ("date", index_col)]
    if not tickers:
        return np.zeros(0, dtype=float), []

    x = df.get_column(index_col).to_numpy()
    x = np.nan_to_num(x, nan=0.0)
    xx = float(np.dot(x, x)) + 1e-12

    betas = []
    for t in tickers:
        y = df.get_column(t).to_numpy()
        y = np.nan_to_num(y, nan=0.0)
        betas.append(float(np.dot(x, y) / xx))
    return np.array(betas, dtype=float), tickers


def build_index_shock(
    df_ret_wide: pl.DataFrame,
    index_col: str,
    index_move: float,
    lookback: int = 252,
) -> dict[str, float]:
    """
    Create a per-asset shock map via betas: shock_i ≈ beta_i * index_move.
    """
    betas, tickers = estimate_rolling_beta(df_ret_wide, index_col=index_col, lookback=lookback)
    return {tk: float(b * index_move) for tk, b in zip(tickers, betas)}


def historical_slice_returns(
    df_ret_wide: pl.DataFrame,
    start: str,
    end: str,
    tickers: list[str],
) -> pl.DataFrame:
    """
    Extract a historical episode [start, end] inclusive for a specific universe.
    Missing tickers are added as null columns (engine can treat NaN as 0 later).
    """
    # Filter + keep available tickers
    keep = ["date"] + [c for c in tickers if c in df_ret_wide.columns]
    df = (
        df_ret_wide
        .filter((pl.col("date") >= pl.lit(start)) & (pl.col("date") <= pl.lit(end)))
        .select(keep)
        .sort("date")
    )
    # Add missing as nulls to preserve column order
    miss = [t for t in tickers if t not in df.columns]
    if miss:
        df = df.with_columns(**{m: pl.lit(None, dtype=pl.Float64) for m in miss}).select(["date", *tickers])
    return df


def apply_shock_map_to_wide(
    df_ret_wide: pl.DataFrame,
    shock_map: dict[str, float],
) -> pl.DataFrame:
    """
    Apply a one-off additive shock to selected tickers (return + shock).
    Shock is applied to ALL rows (typical for instantaneous shock evaluation).
    If you want a single day only, slice first and apply here.
    """
    df = df_ret_wide
    for k, v in (shock_map or {}).items():
        if k in df.columns:
            df = df.with_columns(pl.col(k).fill_null(0.0) + float(v))
    return df






@dataclass
class ShockSpec:
    """
    Defines a return-space shock applied to a wide return matrix.

    mean_shift: constant additive drift per period (e.g., 0.0001 = +1bp/day)
    cov_scale: scaling factor for cross-sectional deviations from mean
    crash: optional (index, shock_value) one-day additive return gap
    """
    mean_shift: Optional[float] = None
    cov_scale: float = 1.0
    crash: Optional[Tuple[int, float]] = None


@dataclass
class ScenarioConfig:
    """Configuration for a single scenario backtest."""
    name: str
    B: int = 0
    block: int = 10
    seed: int = 42
    shock: Optional[ShockSpec] = None


def apply_shock_map_to_wide(
    df_wide: pl.DataFrame,
    shock_map: Dict[str, Union[float, Tuple[int, float]]],
) -> pl.DataFrame:
    """
    Apply additive shocks to a wide return DataFrame.

    Recognized special keys:
        '__mean__'     -> additive drift (float)
        '__cov_scale__'-> scale deviations from mean
        '__crash__'    -> (index, value) single-day crash
        otherwise      -> column-specific additive shifts
    """
    cols = [c for c in df_wide.columns if c != "date"]
    if not cols:
        return df_wide

    out = df_wide
    # Scale cross-sectional deviations
    cov_scale = float(shock_map.get("__cov_scale__", 1.0))  # type: ignore
    if abs(cov_scale - 1.0) > 1e-12:
        X = out.select(cols).to_numpy()
        mu = np.nanmean(X, axis=1, keepdims=True)
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = mu + cov_scale * (X - mu)
        out = out.with_columns(**{c: pl.Series(X_scaled[:, j]) for j, c in enumerate(cols)})

    # Add mean shift
    mean_shift = shock_map.get("__mean__", None)
    if mean_shift is not None:
        ms = float(mean_shift)  # type: ignore
        out = out.with_columns(**{c: (pl.col(c).fill_null(0.0) + ms) for c in cols})

    # One-day crash
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

    # Column-specific additive shocks
    for k, v in shock_map.items():
        if k in {"__mean__", "__cov_scale__", "__crash__", "date"}:
            continue
        if k in out.columns:
            out = out.with_columns(pl.col(k).fill_null(0.0) + float(v))  # type: ignore

    return out


def apply_shock(df_wide: pl.DataFrame, shock: Optional[ShockSpec]) -> pl.DataFrame:
    """Wrapper to apply a ShockSpec dataclass to a Polars DataFrame."""
    if shock is None:
        return df_wide
    shock_map = {
        "__mean__": shock.mean_shift,
        "__cov_scale__": shock.cov_scale,
        "__crash__": shock.crash,
    }
    shock_map = {k: v for k, v in shock_map.items() if v is not None}
    return apply_shock_map_to_wide(df_wide, shock_map)


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
    idx = []
    i = 0
    while i < T:
        start = rng.integers(0, max(T - block, 1))
        take = min(block, T - i)
        idx.extend(range(start, start + take))
        i += take
    return np.array(idx, dtype=int)


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
    Run a list of ScenarioConfig objects through the backtesting engine.

    Returns a list of dictionaries with keys:
        {'name': str, 'bt': dict, 'metrics': pl.DataFrame}

    - Applies block bootstrap if B > 0
    - Averages metrics across bootstrap replications
    - Applies the given ShockSpec to each path before backtesting
    """
    results: List[Dict[str, Union[str, Dict, pl.DataFrame]]] = []
    T = df_ret_wide.height

    for cfg in cfgs:
        if cfg.B and cfg.B > 0:
            agg_metrics = []
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
                agg_metrics.append(m)

            # Average scalar metrics across B replications
            cols = list({c for m in agg_metrics for c in (m.columns if hasattr(m, "columns") else [])})
            avg_row: Dict[str, float] = {}
            for c in cols:
                vals = []
                for m in agg_metrics:
                    try:
                        if hasattr(m, "get_column") and c in m.columns:
                            v = m.get_column(c).to_list()[0]
                        elif hasattr(m, "__getitem__"):
                            v = m[c].iloc[0]
                        else:
                            v = np.nan
                        vals.append(float(v) if np.isfinite(v) else np.nan)
                    except Exception:
                        vals.append(np.nan)
                if vals:
                    avg_row[c] = float(np.nanmean(vals))
            m_avg = pl.DataFrame([avg_row]) if avg_row else pl.DataFrame()
            results.append({"name": cfg.name, "bt": last_bt or {}, "metrics": m_avg})
        else:
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
            results.append({"name": cfg.name, "bt": bt, "metrics": m})

    return results