# portfolio/backtest/metrics.py
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl

from portfolio.core.compat import dataclass_compat as dataclass

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _to_numpy_1d(x: Any, prefer_col: str | None = None) -> np.ndarray:
    """
    Convert various 1D-like inputs to a float numpy array.
    """
    import numpy as _np

    # Already ndarray
    if isinstance(x, _np.ndarray):
        arr = x.ravel().astype(float, copy=False)
        return cast(np.ndarray, _np.nan_to_num(arr, nan=_np.nan, posinf=_np.nan, neginf=_np.nan))

    # Python list/tuple
    if isinstance(x, (list, tuple)):
        arr = _np.asarray(x, dtype=float).ravel()
        return cast(np.ndarray, _np.nan_to_num(arr, nan=_np.nan, posinf=_np.nan, neginf=_np.nan))

    # Polars
    try:
        if isinstance(x, pl.Series):
            arr = x.to_numpy().astype(float, copy=False).ravel()
            return cast(
                np.ndarray, _np.nan_to_num(arr, nan=_np.nan, posinf=_np.nan, neginf=_np.nan)
            )
        if isinstance(x, pl.DataFrame):
            cols = list(x.columns)
            if not cols:
                return _np.array([], dtype=float)
            if prefer_col and prefer_col in cols:
                col: str = prefer_col
            elif "turnover" in cols:
                col = "turnover"
            elif "te" in cols:
                col = "te"
            else:
                col = cols[0]
            arr = x[col].to_numpy().astype(float, copy=False).ravel()
            return cast(
                np.ndarray, _np.nan_to_num(arr, nan=_np.nan, posinf=_np.nan, neginf=_np.nan)
            )
    except Exception:
        pass

    # Pandas
    try:
        if isinstance(x, pd.Series):
            arr = x.to_numpy(dtype=float).ravel()
            return cast(
                np.ndarray, _np.nan_to_num(arr, nan=_np.nan, posinf=_np.nan, neginf=_np.nan)
            )
        if isinstance(x, pd.DataFrame):
            cols = list(x.columns)
            if not cols:
                return _np.array([], dtype=float)
            if prefer_col and prefer_col in cols:
                col = prefer_col
            elif "turnover" in cols:
                col = "turnover"
            elif "te" in cols:
                col = "te"
            else:
                col = cols[0]
            arr = x[col].to_numpy(dtype=float).ravel()
            return cast(
                np.ndarray, _np.nan_to_num(arr, nan=_np.nan, posinf=_np.nan, neginf=_np.nan)
            )
    except Exception:
        pass

    raise TypeError(f"Unsupported type for array conversion: {type(x)}")


def _annualization_from_dates(dates: Sequence[pd.Timestamp] | pd.Index) -> float:
    """Infer simple annualization factor: D≈252, W≈52, M≈12."""
    # Construir desde list(...) evita la queja de mypy con Iterable genérico
    idx = pd.DatetimeIndex(list(dates) if not isinstance(dates, pd.Index) else dates)
    if len(idx) < 3:
        return 252.0
    dt = float(np.median(np.diff(idx.values).astype("timedelta64[D]").astype(int)))
    if dt <= 2:
        return 252.0
    if dt <= 8:
        return 52.0
    return 12.0


def _equity_to_returns(equity: np.ndarray) -> np.ndarray:
    """Compute period-to-period simple returns from an equity curve."""
    if equity.size <= 1:
        return np.array([], dtype=float)
    # Force numpy array type to avoid mypy Any inference
    r: np.ndarray = np.asarray(equity[1:] / equity[:-1] - 1.0, dtype=float)
    return r


def _max_drawdown(curve: np.ndarray) -> float:
    if curve.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(dd))


def _cagr(curve: np.ndarray, ann: float) -> float:
    n = curve.size
    if n <= 1 or curve[0] <= 0:
        return float("nan")
    years = max((n - 1) / ann, 1e-12)
    return float((curve[-1] / curve[0]) ** (1.0 / years) - 1.0)


def _sortino(ret: np.ndarray, ann: float, rf_per_period: float = 0.0) -> float:
    if ret.size == 0:
        return float("nan")
    ex = ret - rf_per_period
    downside = ex[ex < 0.0]
    denom = np.std(downside, ddof=1) if downside.size > 1 else np.nan
    mu_ann = np.nanmean(ret) * ann
    den_ann = (denom * np.sqrt(ann)) if np.isfinite(denom) and denom > 0 else np.nan
    return float(mu_ann / den_ann) if np.isfinite(den_ann) else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Métricas principales
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MetricRow:
    metric: str
    value: float


def compute_backtest_metrics(bt: Any) -> pl.DataFrame:
    """
    Compute standard backtest metrics from a dict or BacktestResult-like object.
    Returns a Polars DataFrame with columns ["metric","value"].
    """
    turnover: np.ndarray | None = None
    te_daily: np.ndarray | None = None

    if hasattr(bt, "equity") and isinstance(bt.equity, pl.DataFrame):
        eq_df: pl.DataFrame = bt.equity
        if "equity" not in eq_df.columns or "date" not in eq_df.columns:
            raise ValueError("BacktestResult.equity must include ['date','equity'].")
        dates_pd = pd.DatetimeIndex(eq_df["date"].to_pandas())
        equity = _to_numpy_1d(eq_df["equity"])
    elif isinstance(bt, dict):
        if "equity" not in bt or "dates" not in bt:
            raise ValueError("Dict backtest must contain 'equity' and 'dates'.")
        equity = _to_numpy_1d(bt["equity"])
        dates_pd = pd.DatetimeIndex(bt["dates"])
        turnover = (
            _to_numpy_1d(bt["turnover"], prefer_col="turnover")
            if "turnover" in bt and bt["turnover"] is not None
            else None
        )
        te_daily = (
            _to_numpy_1d(bt["te_daily_proxy"])
            if "te_daily_proxy" in bt and bt["te_daily_proxy"] is not None
            else None
        )
    else:
        raise TypeError("Unsupported backtest input for metrics computation.")

    # --- compute metrics ---
    ret = _equity_to_returns(equity)
    ann = _annualization_from_dates(dates_pd)

    mu = float(np.nanmean(ret) * ann) if ret.size else float("nan")
    vol = float(np.nanstd(ret, ddof=1) * np.sqrt(ann)) if ret.size > 1 else float("nan")
    sharpe = float(mu / vol) if (np.isfinite(mu) and np.isfinite(vol) and vol > 0) else float("nan")
    maxdd = _max_drawdown(equity)
    cagr = _cagr(equity, ann)
    sortino = _sortino(ret, ann)

    to_mean = (
        float(np.nanmean(turnover))
        if isinstance(turnover, np.ndarray) and turnover.size
        else float("nan")
    )
    te_ann = (
        float(np.nanstd(te_daily, ddof=1) * np.sqrt(ann))
        if isinstance(te_daily, np.ndarray) and te_daily.size > 1
        else float("nan")
    )

    rows = [
        MetricRow("CAGR", cagr),
        MetricRow("Sharpe", sharpe),
        MetricRow("Volatility_ann", vol),
        MetricRow("MaxDrawdown", maxdd),
        MetricRow("MaxDD", maxdd),
        MetricRow("Sortino", sortino),
        MetricRow("Turnover_mean", to_mean),
        MetricRow("TrackingError_ann_proxy", te_ann),
        MetricRow("AnnFactor_used", ann),
    ]
    return pl.DataFrame([r.__dict__ for r in rows], orient="row")
