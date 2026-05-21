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


def test_calmar_uses_cagr_not_arithmetic_return():
    """Calmar = CAGR / |MaxDD|, not (arithmetic ann mean) / |MaxDD|.

    Construct a portfolio with strong positive skew so arithmetic and geometric
    annualizations diverge. For Gaussian returns:
      ann_mu = mean(r) * 252
      CAGR  ~= ann_mu - 0.5*sigma^2  (volatility drag)
    For sigma=0.20 annual, ann_mu = 0.10 -> CAGR ~= 0.08. ~25% difference in Calmar.
    """
    rng = np.random.default_rng(42)
    # High vol -> significant volatility drag
    daily_rets = rng.normal(0.10 / 252, 0.20 / np.sqrt(252), 252 * 5)  # 5 years
    equity = np.cumprod(1.0 + daily_rets)

    kpis = compute_kpis(equity, periods_per_year=252, rf_daily=0.0)

    cagr = kpis["CAGR"]
    maxdd = kpis["MaxDD"]
    calmar = kpis["Calmar"]

    # Canonical: Calmar = CAGR / |MaxDD|
    expected_calmar = cagr / abs(maxdd)
    assert abs(calmar - expected_calmar) < 1e-6, (
        f"Calmar = {calmar:.4f}, expected CAGR/|MaxDD| = {expected_calmar:.4f}. "
        f"CAGR={cagr:.4f}, MaxDD={maxdd:.4f}"
    )


# ---------------------------------------------------------------------------
# Sharpe rf-consistency across metrics.py and kpis.py
# ---------------------------------------------------------------------------
#
# ``portfolio.backtest.metrics.compute_backtest_metrics`` previously ignored
# the risk-free rate entirely and computed Sharpe as ``mean(r)/std(r)``,
# whereas ``portfolio.backtest.kpis.compute_kpis`` subtracts ``rf_daily``
# from per-period returns first. The two helpers therefore disagreed for any
# rf > 0 on the same portfolio. These regression tests guarantee both report
# the same Sharpe for the same equity curve and rf level.


def test_metrics_sharpe_subtracts_rf_like_kpis():
    """metrics.py and kpis.py Sharpe must agree (both subtract rf)."""
    import pandas as pd

    from portfolio.backtest.metrics import compute_backtest_metrics

    rng = np.random.default_rng(0)
    n = 252 * 3  # 3 years of daily data
    rets = rng.normal(0.0005, 0.012, n)
    equity = np.cumprod(1.0 + rets)

    rf_annual = 0.04  # 4% annual
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252) - 1.0

    # kpis.py path (already subtracts rf_daily internally)
    kpis_out = compute_kpis(equity, rf_daily=rf_daily, periods_per_year=252)
    sharpe_kpis = float(kpis_out["Sharpe"])

    # metrics.py path: build a daily ``bt`` dict that exercises the same logic
    dates = pd.bdate_range("2020-01-01", periods=equity.size)
    bt = {"dates": dates, "equity": equity}
    df_m = compute_backtest_metrics(bt, rf=rf_annual)
    sharpe_metrics = float(df_m.filter(df_m["metric"] == "Sharpe")["value"][0])

    assert np.isfinite(sharpe_kpis) and np.isfinite(
        sharpe_metrics
    ), f"Sharpe must be finite. kpis={sharpe_kpis}, metrics={sharpe_metrics}"
    assert abs(sharpe_kpis - sharpe_metrics) < 0.05, (
        f"Sharpe inconsistency: kpis={sharpe_kpis:.4f}, metrics={sharpe_metrics:.4f}. "
        "Both should subtract rf before computing the ratio."
    )


def test_metrics_sharpe_responds_to_rf():
    """Passing rf > 0 must lower the reported Sharpe vs rf=0 (positive drift)."""
    import pandas as pd

    from portfolio.backtest.metrics import compute_backtest_metrics

    rng = np.random.default_rng(1)
    n = 252 * 3
    rets = rng.normal(0.0008, 0.010, n)  # positive drift so excess > 0
    equity = np.cumprod(1.0 + rets)
    dates = pd.bdate_range("2020-01-01", periods=equity.size)
    bt = {"dates": dates, "equity": equity}

    df0 = compute_backtest_metrics(bt, rf=0.0)
    df_rf = compute_backtest_metrics(bt, rf=0.05)
    s0 = float(df0.filter(df0["metric"] == "Sharpe")["value"][0])
    s_rf = float(df_rf.filter(df_rf["metric"] == "Sharpe")["value"][0])

    assert s_rf < s0, (
        f"With positive drift, rf=5% should lower Sharpe vs rf=0. "
        f"Got rf=0 -> {s0:.4f}, rf=0.05 -> {s_rf:.4f}."
    )
