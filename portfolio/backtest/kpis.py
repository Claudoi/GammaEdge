# portfolio/backtest/kpis.py
from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


def _as_float_array(x: Any) -> np.ndarray:
    """Convert any sequence-like object to a float64 NumPy array.

    Devolvemos np.ndarray genérico para evitar choques con numpy-stubs.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64, copy=False)
    return arr  # np.ndarray (no parametric typing)


def _safe(arr: Any) -> NDArray[np.float64]:
    """Return only finite elements as a float64 array."""
    a = cast(NDArray[np.float64], _as_float_array(arr))
    mask: NDArray[np.bool_] = np.isfinite(a)
    filtered = a[mask].astype(np.float64, copy=False)
    return filtered  # <- quitamos el cast redundante


def equity_to_drawdown(equity: Any) -> NDArray[np.float64]:
    """Compute drawdown series from equity curve (as ratios, negative in drawdown).

    Preserves the input length so the output series stays aligned with the
    caller's time index (e.g., a Date index). Non-finite entries propagate
    as NaN in the running-max chain instead of being filtered out, which
    would otherwise shrink the array and misalign downstream calculations.
    """
    eq_raw = _as_float_array(equity)
    if eq_raw.size == 0:
        return np.zeros(0, dtype=np.float64)
    # Replace non-finite entries with -inf so they never become the running
    # max, but keep the array length unchanged.
    finite_mask = np.isfinite(eq_raw)
    eq_for_max = np.where(finite_mask, eq_raw, -np.inf)
    run_max = np.maximum.accumulate(eq_for_max).astype(np.float64, copy=False)
    # Positions before any finite value remain -inf; mark them NaN so the
    # division yields NaN rather than spurious values.
    run_max[~np.isfinite(run_max)] = np.nan
    run_max[run_max <= 0] = np.nan
    dd = (eq_raw / run_max) - 1.0
    return np.asarray(dd, dtype=np.float64)


def equity_to_returns(equity: Any) -> NDArray[np.float64]:
    """Compute simple returns from equity curve."""
    eq = _as_float_array(equity)
    if eq.size < 2:
        return np.zeros(0, dtype=np.float64)
    rets = np.diff(eq) / eq[:-1]
    return np.asarray(rets, dtype=np.float64)


def compute_kpis(
    equity: Any,
    rf_daily: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """
    Compute common KPIs from an equity curve on a daily grid.
    Assumes equity is a NAV-like series (positive).
    """
    # Keep the raw array to preserve the true time-grid length for any
    # annualization (years = N / periods_per_year). Filtering with _safe()
    # would shrink the array and inflate CAGR / Sharpe / Sortino whenever
    # the input contained NaN or Inf values (e.g., from data gaps).
    eq_raw = _as_float_array(equity)
    eq = _safe(eq_raw)
    # Compute returns from the filtered series to avoid divide-by-NaN/Inf
    # warnings; this only affects the per-step return distribution used for
    # Sharpe/Sortino/HitRatio, not the time-horizon annualization above.
    rets = _safe(equity_to_returns(eq))

    out: dict[str, float] = {}

    # Growth metrics
    if eq.size > 0:
        total_return = float(eq[-1] / eq[0] - 1.0) if eq[0] != 0 else np.nan
        # Use ORIGINAL series length for the time horizon, not the filtered
        # length. The multiplicative chain (eq[-1] / eq[0]) still uses the
        # filtered first/last finite values for numerical stability.
        years = max(eq_raw.size / periods_per_year, 1e-12)
        cagr = float((eq[-1] / eq[0]) ** (1.0 / years) - 1.0) if eq[0] > 0 else np.nan
        out["Total Return"] = total_return
        out["CAGR"] = cagr
    else:
        out["Total Return"] = np.nan
        out["CAGR"] = np.nan

    # Risk-adjusted metrics
    if rets.size > 0:
        excess = rets - rf_daily
        mu = float(np.nanmean(excess))
        sigma = float(np.nanstd(excess, ddof=1)) if rets.size > 1 else np.nan

        ann_mu = mu * periods_per_year
        ann_vol = sigma * np.sqrt(periods_per_year) if np.isfinite(sigma) else np.nan
        sharpe = ann_mu / ann_vol if (ann_vol is not None and ann_vol > 0) else np.nan

        neg = _safe(excess[excess < 0.0])
        downside = float(np.nanstd(neg, ddof=1)) if neg.size > 1 else np.nan
        sortino = (
            ann_mu / (downside * np.sqrt(periods_per_year))
            if (np.isfinite(downside) and downside > 0)
            else np.nan
        )

        dd = equity_to_drawdown(eq_raw)
        maxdd = float(np.nanmin(dd)) if dd.size > 0 else np.nan
        calmar = (ann_mu / abs(maxdd)) if (np.isfinite(maxdd) and maxdd < 0) else np.nan

        hit_ratio = float(np.mean(rets > 0.0))

        out.update(
            {
                "Ann. Return (excess)": ann_mu,
                "Ann. Vol": ann_vol,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "MaxDD": maxdd,
                "Calmar": calmar,
                "Hit Ratio": hit_ratio,
            }
        )
    else:
        out.update(
            {
                "Ann. Return (excess)": np.nan,
                "Ann. Vol": np.nan,
                "Sharpe": np.nan,
                "Sortino": np.nan,
                "MaxDD": np.nan,
                "Calmar": np.nan,
                "Hit Ratio": np.nan,
            }
        )

    return out
