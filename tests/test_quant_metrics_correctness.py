"""Regression tests for canonical-correctness of quant_metrics functions.

Guards against re-introduction of the non-canonical Sortino downside
deviation, which previously used ``np.std(neg_subset, ddof=1)`` over the
subset of negative returns. The canonical formulation (Sortino & Price
1994) is the root-mean-square of ``min(0, r - MAR)`` over the FULL series.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from portfolio.features.quant_metrics import calculate_sortino_ratio


def test_calculate_sortino_uses_full_series_downside_dev():
    """quant_metrics.calculate_sortino_ratio must use canonical full-series DD.

    Build a 252-period series with 90% zeros and 10% identical small losses.
    Under the buggy std-of-subset formulation, all losses are identical so
    ``np.std(neg_subset, ddof=1) == 0`` and the function returns ``None``
    with the "Zero downside deviation" warning. Under the canonical
    formulation the downside deviation is non-zero, and Sortino is a finite
    negative number (mostly zero returns with consistent small losses).
    """
    n = 252
    returns = [0.0] * n
    # Inject 25 small identical losses (~10%) so the subset-std formulation
    # collapses to zero (all negatives identical), while the canonical
    # full-series RMS remains strictly positive.
    for i in range(0, n, 10):
        returns[i] = -0.01

    series = pl.Series("ret", returns)
    result = calculate_sortino_ratio(
        series,
        rf_annual=0.0,
        target_return=0.0,
        min_obs=60,
        annualize=True,
    )

    sortino = result.get("sortino_ratio")
    assert sortino is not None, (
        f"Sortino is None (warning={result.get('warning')!r}); the "
        "std-of-subset formulation collapses to zero when all downside "
        "returns are identical. Canonical RMS should yield a finite value."
    )
    assert np.isfinite(sortino), f"Sortino not finite: {sortino!r}"
    assert sortino < 0, f"Expected Sortino < 0 (consistent losses, no gains), got {sortino}"

    # Downside deviation must also be the canonical full-series RMS.
    # Losses occur at indices 0, 10, ..., 250 -> 26 losses in 252 periods.
    # DD_per_period = sqrt((26 / 252) * 0.01^2) ~= 0.003212
    # DD_annual    = DD_per_period * sqrt(252)  ~= 0.050990
    dd_ann = result.get("downside_deviation")
    assert dd_ann is not None and np.isfinite(
        dd_ann
    ), f"downside_deviation missing or not finite: {dd_ann!r}"
    n_losses = len(range(0, n, 10))
    expected_dd_ann = np.sqrt(n_losses / n) * 0.01 * np.sqrt(252)
    assert abs(dd_ann - expected_dd_ann) < 1e-4, (
        f"downside_deviation annualized = {dd_ann:.6f}, "
        f"expected ~{expected_dd_ann:.6f} (canonical full-series RMS)."
    )


def test_calculate_sortino_gaussian_vs_sharpe_ratio_relationship():
    """For symmetric Gaussian returns with MAR = 0 ~ mean, Sortino / (mean/DD)
    aligns with the half-Gaussian relationship DD = sigma / sqrt(2).

    We verify that downside_deviation (per period) is approximately
    sigma / sqrt(2) for a large Gaussian sample with mean ~ MAR.
    """
    rng = np.random.default_rng(123)
    n = 10000
    sigma = 0.01
    rets = rng.normal(0.0, sigma, n)
    series = pl.Series("ret", rets.tolist())

    result = calculate_sortino_ratio(
        series,
        rf_annual=0.0,
        target_return=0.0,
        min_obs=60,
        annualize=False,  # work in per-period space for the theoretical check
    )

    dd = result.get("downside_deviation")
    assert dd is not None and np.isfinite(dd)
    # Theoretical: DD = sigma / sqrt(2) ~= 0.00707
    expected = sigma / np.sqrt(2.0)
    # Tolerate ~5% sample noise
    assert abs(dd - expected) / expected < 0.05, (
        f"per-period downside_deviation = {dd:.6f}, expected ~{expected:.6f} "
        "(sigma/sqrt(2) under Gaussian assumption); this signals the "
        "downside deviation is not the canonical full-series RMS."
    )
