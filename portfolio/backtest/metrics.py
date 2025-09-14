# Backtest metrics
from __future__ import annotations

from portfolio.core.compat import dataclass_compat as dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class PerfStats:
    cagr: float
    sharpe: float
    sortino: float
    maxdd: float
    vol: float
    calmar: float


def timeseries_stats_from_equity(
    equity: pl.DataFrame,
    *,
    periods_per_year: int = 252,
    rf: float = 0.0,
) -> PerfStats:
    """
    Recibe DF con columnas ["date","equity","ret"] y devuelve estadísticas estándar.
    """
    if equity.is_empty() or "ret" not in equity.columns:
        return PerfStats(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    r = equity["ret"].to_numpy()
    if r.size < 2:
        return PerfStats(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    mu = float(np.nanmean(r)) * periods_per_year
    vol = float(np.nanstd(r, ddof=1)) * (periods_per_year ** 0.5)
    neg = r[r < 0]
    dvol = float(np.nanstd(neg, ddof=1)) * (periods_per_year ** 0.5) if neg.size else np.nan

    sharpe = (mu - rf) / vol if vol > 0 else np.nan
    sortino = (mu - rf) / dvol if (dvol and dvol > 0) else np.nan

    curve = equity["equity"].to_numpy()
    peak = np.maximum.accumulate(curve)
    dd = curve / peak - 1.0
    maxdd = float(np.min(dd)) if dd.size else np.nan

    years = max(1.0, equity.height / periods_per_year)
    cagr = float((curve[-1] / curve[0]) ** (1 / years) - 1.0)
    calmar = cagr / abs(maxdd) if (maxdd and maxdd < 0) else np.nan

    return PerfStats(cagr, sharpe, sortino, maxdd, vol, calmar)