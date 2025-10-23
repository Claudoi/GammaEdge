# portfolio/backtest/metrics.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

# Usa el wrapper para soportar Python 3.9 (ignora slots)
from portfolio.core.compat import dataclass_compat as dataclass


def _to_numpy_1d(x, prefer_col: str | None = None) -> np.ndarray:
    """
    Convert various 1D-like inputs to a float numpy array.
    Supports: np.ndarray, list/tuple, Polars Series/DataFrame, Pandas Series/DataFrame.
    If a DataFrame is passed, it will try:
      - prefer_col (if provided and exists),
      - a column named 'turnover' or 'te',
      - the first column.
    """
    import numpy as _np

    # Already ndarray
    if isinstance(x, _np.ndarray):
        arr = x.ravel().astype(float, copy=False)
        return _np.nan_to_num(arr, nan=_np.nan, posinf=_np.nan, neginf=_np.nan)

    # Python list/tuple
    if isinstance(x, (list, tuple)):
        arr = _np.asarray(x, dtype=float).ravel()
        return _np.nan_to_num(arr, nan=_np.nan, posinf=_np.nan, neginf=_np.nan)

    # Polars
    try:
        import polars as pl  # type: ignore

        if isinstance(x, pl.Series):
            arr = x.to_numpy().astype(float, copy=False).ravel()
            return _np.nan_to_num(arr, nan=_np.nan, posinf=_np.nan, neginf=_np.nan)
        if isinstance(x, pl.DataFrame):
            cols = list(x.columns)
            col = None
            if prefer_col and prefer_col in cols:
                col = prefer_col
            elif "turnover" in cols:
                col = "turnover"
            elif "te" in cols:
                col = "te"
            else:
                col = cols[0] if cols else None
            if col is None:
                return _np.array([], dtype=float)
            arr = x[col].to_numpy().astype(float, copy=False).ravel()
            return _np.nan_to_num(arr, nan=_np.nan, posinf=_np.nan, neginf=_np.nan)
    except Exception:
        pass

    # Pandas
    try:
        import pandas as pd  # type: ignore

        if isinstance(x, pd.Series):
            arr = x.to_numpy(dtype=float).ravel()
            return _np.nan_to_num(arr, nan=_np.nan, posinf=_np.nan, neginf=_np.nan)
        if isinstance(x, pd.DataFrame):
            cols = list(x.columns)
            col = None
            if prefer_col and prefer_col in cols:
                col = prefer_col
            elif "turnover" in cols:
                col = "turnover"
            elif "te" in cols:
                col = "te"
            else:
                col = cols[0] if cols else None
            if col is None:
                return _np.array([], dtype=float)
            arr = x[col].to_numpy(dtype=float).ravel()
            return _np.nan_to_num(arr, nan=_np.nan, posinf=_np.nan, neginf=_np.nan)
    except Exception:
        pass

    raise TypeError(f"Unsupported type for array conversion: {type(x)}")


def _annualization_from_dates(dates: Iterable[pd.Timestamp]) -> float:
    """Inferencia muy simple de factor anual: D≈252, W≈52, M≈12."""
    idx = pd.DatetimeIndex(dates)
    if len(idx) < 3:
        return 252.0
    # diferencia mediana en días
    dt = np.median(np.diff(idx.values).astype("timedelta64[D]").astype(int))
    if dt <= 2:
        return 252.0
    if dt <= 8:
        return 52.0
    return 12.0


def _equity_to_returns(equity: np.ndarray) -> np.ndarray:
    if equity.size <= 1:
        return np.array([], dtype=float)
    r = equity[1:] / equity[:-1] - 1.0
    return r.astype(float)


def _max_drawdown(curve: np.ndarray) -> float:
    if curve.size == 0:
        return np.nan
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(dd))


def _cagr(curve: np.ndarray, ann: float) -> float:
    n = curve.size
    if n <= 1 or curve[0] <= 0:
        return np.nan
    years = max((n - 1) / ann, 1e-12)
    return float((curve[-1] / curve[0]) ** (1.0 / years) - 1.0)


def _sortino(ret: np.ndarray, ann: float, rf_per_period: float = 0.0) -> float:
    if ret.size == 0:
        return np.nan
    ex = ret - rf_per_period
    downside = ex[ex < 0.0]
    denom = np.std(downside, ddof=1) if downside.size > 1 else np.nan
    mu_ann = np.nanmean(ret) * ann
    den_ann = (denom * np.sqrt(ann)) if np.isfinite(denom) and denom > 0 else np.nan
    return float(mu_ann / den_ann) if np.isfinite(den_ann) else np.nan


@dataclass(frozen=True, slots=True)
class MetricRow:
    metric: str
    value: float


def compute_backtest_metrics(bt: Any) -> pl.DataFrame:
    """
    Acepta:
      - dict estilo engine simple: {"dates": list[pd.Timestamp], "equity": np.ndarray, "turnover": np.ndarray?, "te_daily_proxy": np.ndarray?}
      - BacktestResult: con .equity (pl.DataFrame con ["date","equity","ret"])
    Devuelve: pl.DataFrame con columnas ["metric","value"] (una fila por métrica).
    """
    # --- unifica equity y fechas ---
    turnover = None
    te_daily = None

    if hasattr(bt, "equity") and isinstance(bt.equity, pl.DataFrame):
        eq_df: pl.DataFrame = bt.equity
        if "equity" not in eq_df.columns or "date" not in eq_df.columns:
            raise ValueError("BacktestResult.equity debe tener columnas ['date','equity', ...].")
        dates_pd = pd.DatetimeIndex(eq_df["date"].to_pandas())
        equity = _to_numpy_1d(eq_df["equity"])
        # turnover / te opcionales si vienen en otros campos
        if hasattr(bt, "trades") and isinstance(bt.trades, pl.DataFrame) and bt.trades.height > 0:
            # podemos estimar turnover por fecha si quieres, pero aquí lo dejamos opcional
            pass
    elif isinstance(bt, dict):
        if "equity" not in bt or "dates" not in bt:
            raise ValueError("dict de backtest debe contener 'equity' y 'dates'.")
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
        raise TypeError("Tipo de backtest no soportado para métricas.")

    # --- retornos & anualización ---
    ret = _equity_to_returns(equity)
    ann = _annualization_from_dates(dates_pd)

    mu = float(np.nanmean(ret) * ann) if ret.size else np.nan
    vol = float(np.nanstd(ret, ddof=1) * np.sqrt(ann)) if ret.size > 1 else np.nan
    sharpe = float(mu / vol) if (np.isfinite(mu) and np.isfinite(vol) and vol > 0) else np.nan
    maxdd = _max_drawdown(equity)
    cagr = _cagr(equity, ann)
    sortino = _sortino(ret, ann)

    # Turnover medio por rebalance (si viene)
    to_mean = (
        float(np.nanmean(turnover))
        if isinstance(turnover, np.ndarray) and turnover.size
        else np.nan
    )

    # TE anualizado (proxy) si viene (std diario * sqrt(ann))
    te_ann = (
        float(np.nanstd(te_daily, ddof=1) * np.sqrt(ann))
        if isinstance(te_daily, np.ndarray) and te_daily.size > 1
        else np.nan
    )

    rows = [
        MetricRow("CAGR", cagr),
        MetricRow("Sharpe", sharpe),
        MetricRow("Volatility_ann", vol),
        MetricRow("MaxDrawdown", maxdd),
        MetricRow("Sortino", sortino),
        MetricRow("Turnover_mean", to_mean),
        MetricRow("TrackingError_ann_proxy", te_ann),
        MetricRow("AnnFactor_used", ann),
    ]
    return pl.DataFrame([r.__dict__ for r in rows])
