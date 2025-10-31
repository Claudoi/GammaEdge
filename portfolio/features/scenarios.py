# portfolio/backtest/scenarios.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from portfolio.backtest import metrics as bt_metrics
from portfolio.backtest.engine import backtest_rebalanced


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

    mean_shift: float | np.ndarray | None = None
    cov_scale: float = 1.0
    crash: tuple[int, float | np.ndarray] | None = None


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    B: int = 0  # bootstrap paths (0 = deterministic/stressed original)
    block: int = 10  # block length for bootstrap
    seed: int = 42
    shock: ShockSpec = ShockSpec()


@dataclass
class ScenarioResult:
    name: str
    bt: dict  # full backtest dict (equity, weights, etc.)
    metrics: pl.DataFrame  # metrics DF
    shock: ShockSpec


# -----------------------------
# Helpers
# -----------------------------
def _to_numpy_wide(df_ret_wide: pl.DataFrame, tickers: list[str]) -> np.ndarray:
    R = df_ret_wide.select(tickers).to_numpy()
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    return R.astype(float, copy=False)


def _from_numpy_wide(dates: list, tickers: list[str], R: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({"date": dates, **{t: R[:, j] for j, t in enumerate(tickers)}})


def _bootstrap_paths(R: np.ndarray, B: int, block: int, seed: int) -> list[np.ndarray]:
    """
    Block bootstrap paths over rows (time). Each path has the same length as R.
    """
    if B <= 0:
        return [R.copy()]
    T, _ = R.shape
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for _ in range(B):
        idx: list[int] = []
        while len(idx) < T:
            start = int(rng.integers(0, max(T - block, 1)))
            idx.extend(range(start, min(start + block, T)))
        take = np.array(idx[:T], dtype=int)
        out.append(R[take, :])
    return out


def _apply_shock(R: np.ndarray, shock: ShockSpec) -> np.ndarray:
    """
    Apply shock to returns matrix R (T, N).
    - mean_shift: add daily drift
    - cov_scale: scale deviations from mean
    - crash: inject a single-day drop
    """
    R2 = R.copy()
    T, _ = R2.shape

    # mean shift
    if shock.mean_shift is not None:
        ms = np.asarray(shock.mean_shift)
        if ms.size == 1:
            R2 += float(ms)
        else:
            R2 += ms.reshape(1, -1)

    # cov scale (around cross-sectional mean)
    if abs(shock.cov_scale - 1.0) > 1e-12:
        mu = np.mean(R2, axis=0, keepdims=True)
        R2 = mu + (R2 - mu) * float(shock.cov_scale)

    # single-day crash
    if shock.crash is not None:
        t_idx, drop = shock.crash
        if 0 <= t_idx < T:
            d = np.asarray(drop)
            if d.size == 1:
                R2[t_idx, :] += float(d)  # drop is usually negative, e.g., -0.05
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
    if index_col not in cols:
        raise ValueError(f"{index_col=} not in df columns")
    # slice last lookback
    df_tail = df_wide.sort("date").tail(lookback).select(["date", *cols])
    X = df_tail.get_column(index_col).fill_null(0.0).to_numpy()
    X = X - np.nanmean(X)
    shock: dict[str, float] = {}
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
            out = out.with_columns((pl.col(k).fill_null(0.0).cast(pl.Float64) + float(v)).alias(k))
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
    # Parse robustly with pandas → native Python datetimes
    start_dt = pd.to_datetime(start, utc=False).to_pydatetime()
    end_dt = pd.to_datetime(end, utc=False).to_pydatetime()

    cols = [c for c in df_wide.columns if c != "date"] if tickers is None else list(tickers)
    cols = [c for c in cols if c in df_wide.columns]

    df = (
        df_wide.filter((pl.col("date") >= pl.lit(start_dt)) & (pl.col("date") <= pl.lit(end_dt)))
        .sort("date")
        .select(["date", *cols])
    )
    return df


# -----------------------------
# API
# -----------------------------
def run_scenarios(
    cfgs: list[ScenarioConfig],
    df_ret_wide: pl.DataFrame,  # ['date', tickers...]
    allocator_factory: Callable[[], Callable[[pl.DataFrame], np.ndarray]],
    *,
    lookback: int,
    rebalance_freq: str,
    cost_bps: float,
    bench_weights: np.ndarray,
) -> list[ScenarioResult]:
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

    out: list[ScenarioResult] = []
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
            name = cfg.name if cfg.B <= 1 else f"{cfg.name} #{b_ix + 1}"
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

    betas: list[float] = []
    for t in tickers:
        y = df.get_column(t).to_numpy()
        y = np.nan_to_num(y, nan=0.0)
        betas.append(float(np.dot(x, y) / xx))
    return np.array(betas, dtype=float), tickers


def block_bootstrap_indices(T: int, block: int, seed: int) -> np.ndarray:
    """Generate block bootstrap indices for synthetic paths."""
    rng = np.random.default_rng(seed)
    idx: list[int] = []
    i = 0
    while i < T:
        start = int(rng.integers(0, max(T - block, 1)))
        take = min(block, T - i)
        idx.extend(range(start, start + take))
        i += take
    return np.array(idx, dtype=int)
