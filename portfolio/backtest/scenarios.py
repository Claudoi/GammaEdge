# portfolio/backtest/scenarios.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl

from portfolio.backtest import metrics as bt_metrics
from portfolio.backtest.engine import backtest_rebalanced

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

    mean_shift: float | None = None
    cov_scale: float = 1.0
    crash: tuple[int, float] | None = None


@dataclass
class ScenarioConfig:
    """Configuration for a scenario backtest."""

    name: str
    B: int = 0
    block: int = 10
    seed: int = 42
    shock: ShockSpec | None = None


# ---------------------------------------------------------------------
# Shock application
# ---------------------------------------------------------------------
def apply_shock_map_to_wide(
    df_wide: pl.DataFrame,
    shock_map: dict[str, float | tuple[int, float]],
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
    mean_shift = shock_map.get("__mean__")
    if mean_shift is not None:
        ms = float(mean_shift)  # type: ignore
        out = out.with_columns(**{c: (pl.col(c).fill_null(0.0) + ms) for c in cols})

    # 3) One-day crash
    crash = shock_map.get("__crash__")
    if crash is not None:
        idx, add_ret = crash  # type: ignore
        idx = int(idx)
        add_ret = float(add_ret)
        if 0 <= idx < out.height:
            out = out.with_row_count("rowid")
            out = out.with_columns(
                [
                    pl.when(pl.col("rowid") == idx)
                    .then(pl.col(c).fill_null(0.0) + add_ret)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in cols
                ]
            ).drop("rowid")

    # 4) Column-specific additive shifts
    for k, v in shock_map.items():
        if k in {"__mean__", "__cov_scale__", "__crash__", "date"}:
            continue
        if k in out.columns:
            out = out.with_columns(pl.col(k).fill_null(0.0) + float(v))  # type: ignore

    return out


def apply_shock(df_wide: pl.DataFrame, shock: ShockSpec | None) -> pl.DataFrame:
    """Convenience wrapper to apply a ShockSpec to a Polars wide return frame."""
    if shock is None:
        return df_wide
    shock_map: dict[str, float | tuple[int, float]] = {}
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
def _parse_to_py_datetime(x: str | datetime | date) -> datetime:
    """Parse input into a Python datetime (timezone-naive)."""
    if isinstance(x, datetime):
        return x.replace(tzinfo=None)
    if isinstance(x, date):
        return datetime(x.year, x.month, x.day)
    # Let pandas handle a wide range of formats, then strip tz
    dt = pd.to_datetime(x).to_pydatetime()
    return dt.replace(tzinfo=None)


def historical_slice_returns(
    df_wide: pl.DataFrame,
    start: str | datetime | date,
    end: str | datetime | date,
    tickers: list[str] | None = None,
) -> pl.DataFrame:
    """
    Extract an inclusive historical time slice from a wide returns DataFrame.

    - Accepts date strings (any common format), datetime/date objects.
    - Works with 'date' column of dtype pl.Date or pl.Datetime.
    - Returns a sorted frame with ['date', *tickers].
    """
    if "date" not in df_wide.columns:
        raise ValueError("historical_slice_returns: missing 'date' column.")

    # Parse boundaries to Python datetime
    start_ts = _parse_to_py_datetime(start)
    end_ts = _parse_to_py_datetime(end)

    # Ensure proper temporal dtype on 'date'
    date_dtype = df_wide.schema.get("date")
    if date_dtype not in (pl.Date, pl.Datetime):
        # If it's string/int, try to cast sensibly to Datetime
        df_wide = df_wide.with_columns(
            pl.col("date").str.strptime(pl.Datetime, strict=False, fmt=None)
        )

    # After possible cast, re-check dtype
    date_dtype = df_wide.schema.get("date")
    if date_dtype == pl.Date:
        # Compare using pl.Date literals
        start_lit = pl.lit(start_ts.date(), dtype=pl.Date)
        end_lit = pl.lit(end_ts.date(), dtype=pl.Date)
    else:
        # Treat everything else as Datetime (naive)
        start_lit = pl.lit(start_ts, dtype=pl.Datetime)
        end_lit = pl.lit(end_ts, dtype=pl.Datetime)

    # Choose columns
    asset_cols = tickers if tickers else [c for c in df_wide.columns if c != "date"]
    cols = ["date", *asset_cols]

    # Filter inclusive window and sort
    out = (
        df_wide.select(cols)
        .filter((pl.col("date") >= start_lit) & (pl.col("date") <= end_lit))
        .sort("date")
    )

    if out.height == 0:
        raise ValueError("historical_slice_returns: empty slice.")
    return out


def block_bootstrap_indices(T: int, block: int, seed: int) -> np.ndarray:
    """Generate block bootstrap indices for synthetic paths."""
    rng = np.random.default_rng(seed)
    idx: list[int] = []
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
    compute_metrics: Callable[
        [dict], pl.DataFrame | pd.DataFrame
    ] = bt_metrics.compute_backtest_metrics,
) -> list[dict[str, str | dict | pl.DataFrame]]:
    """
    Run a list of ScenarioConfig objects and return:
        [{'name': str, 'bt': dict, 'metrics': pl.DataFrame}, ...]

    - If B > 0, block-bootstrap B paths, average 1-row metrics across replications,
      and return the last backtest object for plotting.
    - If B == 0, run a single shocked (or unshocked) path.
    """
    results: list[dict[str, str | dict | pl.DataFrame]] = []
    T = df_ret_wide.height

    for cfg in cfgs:
        if cfg.B and cfg.B > 0:
            agg_metrics: list[pl.DataFrame] = []
            last_bt: dict | None = None

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
            avg_row: dict[str, float] = {}
            for c in cols:
                vals: list[float] = []
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
