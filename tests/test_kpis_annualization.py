# tests/test_kpis_annualization.py
"""Regression tests for the _safe() over-annualization bug in kpis.py.

The bug: `_safe(equity)` removes NaN/Inf values, which shrinks the array.
The previous implementation used the *filtered* array length to compute
`years = eq.size / periods_per_year`, which under-counted the actual time
horizon when any non-finite values were present, inflating CAGR / Sharpe /
Sortino metrics. The fix uses the ORIGINAL array length for time-based
annualization while still using the filtered array for the multiplicative
chain (to keep CAGR finite).
"""
from __future__ import annotations

import numpy as np

from portfolio.backtest.kpis import compute_kpis, equity_to_drawdown


def test_cagr_unchanged_when_no_nan():
    """Baseline: clean equity gives correct CAGR."""
    # 252 days of equity rising 10% over the year
    eq = np.linspace(1.0, 1.10, 252)
    kpis = compute_kpis(eq, periods_per_year=252)
    assert abs(kpis["CAGR"] - 0.10) < 0.01, f"CAGR {kpis['CAGR']} far from 0.10"


def test_cagr_not_inflated_by_nans():
    """CRITICAL: NaN in middle of equity must NOT inflate CAGR.

    Before the fix: _safe(equity) shrank the array, so years was too small,
    so CAGR was inflated on a 252-day series with a few NaN.
    """
    eq = np.linspace(1.0, 1.10, 252)
    eq_with_nan = eq.copy()
    # Introduce some NaN values mid-series (simulating data gaps)
    eq_with_nan[50:55] = np.nan

    kpis_clean = compute_kpis(eq, periods_per_year=252)
    kpis_nan = compute_kpis(eq_with_nan, periods_per_year=252)

    # The CAGR should be approximately equal — time horizon hasn't changed
    assert abs(kpis_nan["CAGR"] - kpis_clean["CAGR"]) < 0.005, (
        f"CAGR inflated by NaN: clean={kpis_clean['CAGR']:.4f}, " f"with_nan={kpis_nan['CAGR']:.4f}"
    )


def test_cagr_not_inflated_by_many_nans():
    """High-NaN-ratio variant that exposes the bug unambiguously.

    With ~20% of the series replaced by NaN, the previous code computed
    `years` from the filtered length, inflating CAGR by a large factor.
    """
    n = 252
    eq = np.linspace(1.0, 1.10, n)
    eq_with_nan = eq.copy()
    # Replace ~20% of values with NaN
    eq_with_nan[20:70] = np.nan

    kpis_clean = compute_kpis(eq, periods_per_year=252)
    kpis_nan = compute_kpis(eq_with_nan, periods_per_year=252)

    # CAGR should be within 0.5 percentage points; pre-fix it diverges by >2%
    assert abs(kpis_nan["CAGR"] - kpis_clean["CAGR"]) < 0.005, (
        f"CAGR inflated by NaN: clean={kpis_clean['CAGR']:.4f}, " f"with_nan={kpis_nan['CAGR']:.4f}"
    )


def test_cagr_not_inflated_by_infs():
    """Inf values must also not inflate CAGR via the same code path."""
    eq = np.linspace(1.0, 1.10, 252)
    eq_with_inf = eq.copy()
    eq_with_inf[100] = np.inf
    eq_with_inf[150] = -np.inf

    kpis_clean = compute_kpis(eq, periods_per_year=252)
    kpis_inf = compute_kpis(eq_with_inf, periods_per_year=252)

    assert abs(kpis_inf["CAGR"] - kpis_clean["CAGR"]) < 0.005, (
        f"CAGR inflated by Inf: clean={kpis_clean['CAGR']:.4f}, " f"with_inf={kpis_inf['CAGR']:.4f}"
    )


def test_drawdown_length_preserved_with_nans():
    """equity_to_drawdown should preserve the original length so the series
    aligns with the time index of the caller (e.g., a Date index)."""
    eq = np.linspace(1.0, 1.10, 100)
    eq_with_nan = eq.copy()
    eq_with_nan[10:13] = np.nan

    dd = equity_to_drawdown(eq_with_nan)
    assert dd.size == eq_with_nan.size, (
        f"drawdown length {dd.size} != equity length {eq_with_nan.size}; "
        "NaN filtering must not shrink the output series."
    )
