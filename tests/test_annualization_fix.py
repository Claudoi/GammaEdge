"""
Test to validate the annualization fix and prevent regression.

This test ensures that portfolio metrics are NOT double-annualized,
which was causing extreme values (5302% return, 349% volatility).
"""

from __future__ import annotations

import numpy as np
import pytest

from portfolio.core.metrics import portfolio_stats


def test_no_double_annualization():
    """
    CRITICAL REGRESSION TEST: Validate that portfolio_stats doesn't double-annualize.

    Bug History:
    - Before fix: Optimizer.py lines 454-455 re-annualized already-annual inputs
    - Result: 252² multiplier → 5302% returns instead of ~10%

    This test simulates the data flow:
    1. RiskModel outputs annualized mu/Sigma (daily × 252)
    2. Optimizer uses portfolio_stats with these inputs
    3. Result should be in annual units WITHOUT further multiplication
    """
    # Simulate realistic daily metrics for a moderate equity portfolio
    daily_mu = 0.0004  # 0.04% per day
    daily_sigma = 0.01  # 1% daily vol

    # RiskModel annualizes these (this is CORRECT behavior)
    annual_mu = daily_mu * 252  # ≈ 10% annually
    annual_sigma_sq = (daily_sigma**2) * 252  # variance annualization
    annual_sigma = np.sqrt(annual_sigma_sq)  # ≈ 15.9% annually

    # Portfolio weights (equal weight on 3 assets)
    N = 3
    w = np.full(N, 1.0 / N)

    # Simulated annualized inputs (as they come from RiskModel)
    mu_annual = np.full(N, annual_mu)
    Sigma_annual = np.eye(N) * (annual_sigma**2)

    # Call portfolio_stats (as Optimizer does)
    mu_p, sigma_p, sharpe = portfolio_stats(w, mu_annual, Sigma_annual, rf=0.0)

    # ASSERTIONS: Values should be reasonable annual metrics
    # Note: With N=3 equal-weight uncorrelated assets, portfolio vol = individual_vol / sqrt(3) ≈ 9.2%
    # If double-annualized, mu_p would be ~2,520% instead of ~10%
    assert 0.05 < mu_p < 0.20, f"Expected ~10% annual return, got {mu_p:.2%}"
    assert 0.05 < sigma_p < 0.20, f"Expected ~9% annual vol (diversified), got {sigma_p:.2%}"
    assert 0.5 < sharpe < 3.0, f"Expected reasonable Sharpe ~1.0, got {sharpe:.3f}"

    # Verify no extreme values (smoking gun for double annualization)
    assert mu_p < 3.0, f"DOUBLE ANNUALIZATION DETECTED: {mu_p:.2%} > 300%"
    assert sigma_p < 2.0, f"DOUBLE ANNUALIZATION DETECTED: {sigma_p:.2%} > 200%"


def test_annualization_multiplier_validation():
    """
    Verify that a 252x multiplier is NOT applied twice.

    This directly tests the mathematical error that caused the bug.
    """
    # Simple case: 1% daily return
    daily_ret = 0.01

    # Correct annualization
    annual_ret_correct = daily_ret * 252  # = 252% (2.52)

    # Incorrect double annualization (the bug)
    annual_ret_bug = (daily_ret * 252) * 252  # = 63,504% (635.04)

    # Set up inputs as if from RiskModel (already annualized)
    w = np.array([1.0])
    mu_already_annual = np.array([annual_ret_correct])
    Sigma_already_annual = np.array([[0.01]])  # variance

    # Get result from portfolio_stats
    mu_p, _, _ = portfolio_stats(w, mu_already_annual, Sigma_already_annual)

    # Should match the correctly annualized value, NOT the bug value
    assert (
        abs(mu_p - annual_ret_correct) < 0.01
    ), f"Expected {annual_ret_correct:.2%}, got {mu_p:.2%}"

    # Should NOT be anywhere near the bug value
    assert (
        abs(mu_p - annual_ret_bug) > 500
    ), f"CRITICAL: Result {mu_p:.2%} is close to bug value {annual_ret_bug:.2%}"


def test_sharpe_ratio_reasonable():
    """
    Sharpe ratios should be in typical range (0-3) for realistic portfolios.

    Bug symptom: Double annualization produced Sharpe > 10
    """
    # Realistic portfolio: 12% return, 18% vol
    N = 5
    w = np.full(N, 1.0 / N)
    mu_annual = np.full(N, 0.12)

    # Construct reasonable covariance (18% vol for equal-weight)
    corr = 0.3  # moderate correlation
    sigma_asset = 0.20  # 20% individual vol
    Sigma_annual = np.full((N, N), corr * sigma_asset**2)
    np.fill_diagonal(Sigma_annual, sigma_asset**2)

    rf_annual = 0.02  # 2% risk-free rate

    mu_p, sigma_p, sharpe = portfolio_stats(w, mu_annual, Sigma_annual, rf=rf_annual)

    # Sharpe should be reasonable: (0.12 - 0.02) / ~0.15 ≈ 0.67
    assert 0.0 < sharpe < 3.0, f"Unrealistic Sharpe: {sharpe:.3f}"
    assert mu_p > rf_annual, "Portfolio return should exceed risk-free rate"


if __name__ == "__main__":
    # Run tests locally
    pytest.main([__file__, "-v"])
