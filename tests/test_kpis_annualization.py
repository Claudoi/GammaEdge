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


# ---------------------------------------------------------------------------
# Sortino canonical downside deviation (Sortino & Price 1994)
# ---------------------------------------------------------------------------
#
# The previous implementation used ``np.std(neg_subset, ddof=1)``, i.e. the
# sample standard deviation taken over the *subset* of negative excess
# returns. The canonical Sortino downside deviation is the root-mean-square
# of ``min(0, r - MAR)`` over the FULL series (zero-padded for upside).
# These regression tests guard against the buggy formulation returning.


def test_sortino_uses_full_series_downside_deviation():
    """Canonical Sortino (Sortino & Price 1994) uses full-series DD, not subset std.

    Construct a series where:
    - 90% of returns are 0
    - 10% are -0.01 (small losses)

    Buggy version (std of subset): all 10 negatives have same value -> std=0 -> Sortino=nan/inf
    Canonical: DD = sqrt(mean([0, ..., 0, -0.01, ..., -0.01]^2)) = sqrt(0.10 * 0.0001) = 0.00316
    """
    returns = np.zeros(100)
    returns[::10] = -0.01  # losses at indices 0, 10, 20, ...
    equity = np.cumprod(1.0 + returns)

    kpis = compute_kpis(equity, periods_per_year=252)
    # Sortino should be finite and negative (returns are mostly 0 with consistent losses)
    assert np.isfinite(kpis["Sortino"]), (
        f"Sortino is not finite ({kpis['Sortino']!r}); the std-of-subset "
        "formulation collapses when all downside returns are identical."
    )
    assert (
        kpis["Sortino"] < 0
    ), f"Expected Sortino < 0 (more losses than gains), got {kpis['Sortino']}"
    # Canonical magnitude: with ~9 losses of -0.01 in 99 returns,
    # mean_excess ~= -9e-4, DD_per ~= sqrt(9/99) * 0.01 ~= 3.0e-3, so
    # Sortino_ann ~= (-9e-4 * 252) / (3.0e-3 * sqrt(252)) ~= -4.8.
    # The buggy std-of-subset version explodes (std of ~identical losses is
    # near machine epsilon, so |Sortino| > 1e10). A bound of |Sortino| < 100
    # cleanly discriminates the two formulations.
    assert abs(kpis["Sortino"]) < 100.0, (
        f"|Sortino| = {abs(kpis['Sortino']):.3e} is implausibly large; the "
        "std-of-subset denominator collapses to ~machine epsilon when all "
        "downside returns are (near-)identical."
    )


def test_sortino_vs_sharpe_relationship():
    """For symmetric (Gaussian) returns, Sortino / Sharpe -> sqrt(2) asymptotically.

    Theoretical: if r ~ N(mu, sigma) and MAR = mu, then:
    - sigma(r) = sigma
    - downside_dev = sigma / sqrt(2)  (half-Gaussian RMS)
    - Sortino / Sharpe = sigma / (sigma/sqrt(2)) = sqrt(2)
    """
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0001, 0.01, 10000)  # large N for asymptotic behaviour
    equity = np.cumprod(1.0 + rets)

    kpis = compute_kpis(equity, periods_per_year=252, rf_daily=0.0)
    ratio = kpis["Sortino"] / kpis["Sharpe"]
    # Expect ~sqrt(2) = 1.414, tolerate +/-0.15 for sample noise
    assert 1.25 < ratio < 1.60, (
        f"Sortino/Sharpe = {ratio:.3f}, expected ~1.414 (Gaussian theory). "
        "This indicates the downside deviation is not the canonical full-series RMS."
    )
