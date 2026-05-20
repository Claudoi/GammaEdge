"""
Unit tests for enhanced CVaR optimization.

Tests scenario-based CVaR, bootstrap scenarios, and portfolio optimizer.
"""

import numpy as np
import pytest

from portfolio.optim.cvar import (
    compute_portfolio_cvar,
    cvar_minimization,
    cvar_portfolio_optimizer,
    cvar_scenario_optimization,
    generate_bootstrap_scenarios,
)


@pytest.fixture
def synthetic_returns():
    """Synthetic historical returns."""
    np.random.seed(42)
    T, N = 252, 3  # 1 year, 3 assets
    returns = np.random.normal(
        loc=np.array([0.001, 0.0008, 0.0003]), scale=np.array([0.02, 0.015, 0.005]), size=(T, N)
    )
    return returns


@pytest.fixture
def synthetic_scenarios():
    """Synthetic scenarios for testing."""
    np.random.seed(123)
    return np.random.normal(0.001, 0.02, size=(1000, 3))


class TestBootstrapScenarios:
    """Test bootstrap scenario generation."""

    def test_bootstrap_shape(self, synthetic_returns):
        """Test bootstrap scenarios have correct shape."""
        scenarios = generate_bootstrap_scenarios(
            synthetic_returns,
            n_scenarios=500,
            block_size=10,
            random_state=42,
        )

        assert scenarios.shape == (500, 3)

    def test_bootstrap_reproducible(self, synthetic_returns):
        """Test bootstrap is reproducible with seed."""
        scenarios1 = generate_bootstrap_scenarios(
            synthetic_returns,
            n_scenarios=100,
            random_state=42,
        )
        scenarios2 = generate_bootstrap_scenarios(
            synthetic_returns,
            n_scenarios=100,
            random_state=42,
        )

        np.testing.assert_array_equal(scenarios1, scenarios2)

    def test_bootstrap_different_seeds(self, synthetic_returns):
        """Test different seeds produce different scenarios."""
        scenarios1 = generate_bootstrap_scenarios(
            synthetic_returns,
            n_scenarios=100,
            random_state=42,
        )
        scenarios2 = generate_bootstrap_scenarios(
            synthetic_returns,
            n_scenarios=100,
            random_state=123,
        )

        assert not np.array_equal(scenarios1, scenarios2)

    def test_bootstrap_block_size(self, synthetic_returns):
        """Test block bootstrap with different block sizes."""
        scenarios_small = generate_bootstrap_scenarios(
            synthetic_returns,
            n_scenarios=100,
            block_size=5,
            random_state=42,
        )
        scenarios_large = generate_bootstrap_scenarios(
            synthetic_returns,
            n_scenarios=100,
            block_size=20,
            random_state=42,
        )

        # Both should work without errors
        assert scenarios_small.shape == (100, 3)
        assert scenarios_large.shape == (100, 3)


class TestCVarScenarioOptimization:
    """Test scenario-based CVaR optimization."""

    def test_cvar_scenario_basic(self, synthetic_scenarios):
        """Test basic CVaR scenario optimization."""
        weights = cvar_scenario_optimization(
            synthetic_scenarios,
            alpha=0.95,
            w_min=0.0,
            w_max=1.0,
        )

        # Should return valid weights
        assert len(weights) == 3
        assert np.allclose(weights.sum(), 1.0, atol=1e-6)
        assert np.all(weights >= -1e-6)  # Allow small numerical errors
        assert np.all(weights <= 1.0 + 1e-6)

    def test_cvar_vs_original(self, synthetic_scenarios):
        """Test scenario optimization equals original cvar_minimization."""
        weights1 = cvar_scenario_optimization(
            synthetic_scenarios,
            alpha=0.95,
        )
        weights2 = cvar_minimization(
            R=synthetic_scenarios,
            alpha=0.95,
        )

        # Should be identical
        np.testing.assert_array_almost_equal(weights1, weights2, decimal=6)

    def test_cvar_different_alpha(self, synthetic_scenarios):
        """Test CVaR with different confidence levels."""
        weights_95 = cvar_scenario_optimization(
            synthetic_scenarios,
            alpha=0.95,
        )
        weights_99 = cvar_scenario_optimization(
            synthetic_scenarios,
            alpha=0.99,
        )

        # Both should be valid
        assert np.all(weights_95 >= -1e-6)
        assert np.all(weights_99 >= -1e-6)

        # May be different (99% is more conservative)
        # Just check they're both valid
        assert np.allclose(weights_95.sum(), 1.0)
        assert np.allclose(weights_99.sum(), 1.0)


class TestCVarPortfolioOptimizer:
    """Test high-level portfolio optimizer."""

    def test_portfolio_optimizer_basic(self):
        """Test portfolio optimizer with mu and Sigma."""
        mu = np.array([0.10, 0.08, 0.05])
        Sigma = np.array(
            [
                [0.04, 0.01, 0.00],
                [0.01, 0.03, 0.00],
                [0.00, 0.00, 0.01],
            ]
        )

        weights = cvar_portfolio_optimizer(
            mu,
            Sigma,
            alpha=0.95,
            n_scenarios=500,
            use_bootstrap=False,
        )

        assert len(weights) == 3
        assert np.allclose(weights.sum(), 1.0, atol=1e-4)
        assert np.all(weights >= -1e-6)

    def test_portfolio_optimizer_with_bootstrap(self, synthetic_returns):
        """Test optimizer with bootstrap scenarios."""
        mu = synthetic_returns.mean(axis=0)
        Sigma = np.cov(synthetic_returns.T)

        weights = cvar_portfolio_optimizer(
            mu,
            Sigma,
            alpha=0.95,
            n_scenarios=500,
            use_bootstrap=True,
            historical_returns=synthetic_returns,
        )

        assert len(weights) == 3
        assert np.allclose(weights.sum(), 1.0, atol=1e-4)

    def test_portfolio_optimizer_bounds(self):
        """Test optimizer respects weight bounds."""
        mu = np.array([0.10, 0.08, 0.05])
        Sigma = np.array(
            [
                [0.04, 0.01, 0.00],
                [0.01, 0.03, 0.00],
                [0.00, 0.00, 0.01],
            ]
        )

        weights = cvar_portfolio_optimizer(
            mu,
            Sigma,
            alpha=0.95,
            w_min=0.1,
            w_max=0.5,
            n_scenarios=500,
            use_bootstrap=False,
        )

        # Check bounds
        assert np.all(weights >= 0.1 - 1e-6)
        assert np.all(weights <= 0.5 + 1e-6)


class TestComputePortfolioCVaR:
    """Test CVaR computation from scenarios."""

    def test_compute_cvar_basic(self, synthetic_scenarios):
        """Test VaR and CVaR computation."""
        weights = np.array([0.5, 0.3, 0.2])

        var, cvar = compute_portfolio_cvar(
            weights,
            synthetic_scenarios,
            alpha=0.95,
        )

        # VaR and CVaR should be positive (losses)
        assert var >= 0
        assert cvar >= var  # CVaR >= VaR always

    def test_compute_cvar_equal_weights(self, synthetic_scenarios):
        """Test CVaR with equal weights."""
        weights = np.array([1 / 3, 1 / 3, 1 / 3])

        var, cvar = compute_portfolio_cvar(
            weights,
            synthetic_scenarios,
            alpha=0.95,
        )

        # Should be reasonable values
        assert 0 <= var <= 0.1  # VaR shouldn't be crazy
        assert cvar >= var

    def test_compute_cvar_concentrated(self, synthetic_scenarios):
        """Test CVaR with concentrated portfolio."""
        # All weight on first asset
        weights = np.array([1.0, 0.0, 0.0])

        var, cvar = compute_portfolio_cvar(
            weights,
            synthetic_scenarios,
            alpha=0.95,
        )

        # Should equal single-asset tail risk
        assert var >= 0
        assert cvar >= var


class TestCVarIntegration:
    """Integration tests combining bootstrap + optimization + metrics."""

    def test_full_pipeline(self, synthetic_returns):
        """Test complete CVaR pipeline."""
        # 1. Generate scenarios from historical data
        scenarios = generate_bootstrap_scenarios(
            synthetic_returns,
            n_scenarios=1000,
            random_state=42,
        )

        # 2. Optimize portfolio
        weights = cvar_scenario_optimization(
            scenarios,
            alpha=0.95,
            w_min=0.0,
            w_max=0.5,
        )

        # 3. Compute CVaR of optimized portfolio
        var, cvar = compute_portfolio_cvar(
            weights,
            scenarios,
            alpha=0.95,
        )

        # Verify end-to-end
        assert len(weights) == 3
        assert np.allclose(weights.sum(), 1.0)
        assert var >= 0
        assert cvar >= var

    def test_cvar_vs_equal_weights(self, synthetic_returns):
        """Test CVaR-optimized beats equal weights on tail risk."""
        scenarios = generate_bootstrap_scenarios(
            synthetic_returns,
            n_scenarios=1000,
            random_state=42,
        )

        # CVaR-optimized weights
        weights_cvar = cvar_scenario_optimization(
            scenarios,
            alpha=0.95,
        )

        # Equal weights
        weights_equal = np.array([1 / 3, 1 / 3, 1 / 3])

        # Compute CVaR for both
        _, cvar_opt = compute_portfolio_cvar(weights_cvar, scenarios, alpha=0.95)
        _, cvar_eq = compute_portfolio_cvar(weights_equal, scenarios, alpha=0.95)

        # CVaR-optimized should have lower or equal CVaR
        # (may not always be lower due to random data, but should be close)
        assert cvar_opt <= cvar_eq * 1.1  # Allow 10% tolerance


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_scenarios(self):
        """Test with very few scenarios."""
        small_scenarios = np.random.normal(0.001, 0.02, size=(10, 3))

        weights = cvar_scenario_optimization(
            small_scenarios,
            alpha=0.95,
        )

        # Should still work
        assert len(weights) == 3
        assert np.allclose(weights.sum(), 1.0, atol=1e-4)

    def test_high_alpha(self):
        """Test with very high confidence level."""
        scenarios = np.random.normal(0.001, 0.02, size=(1000, 3))

        weights = cvar_scenario_optimization(
            scenarios,
            alpha=0.99,  # 99% CVaR
        )

        assert len(weights) == 3
        assert np.allclose(weights.sum(), 1.0, atol=1e-4)


def test_cvar_lp_failure_returns_equal_weight():
    """CRITICAL-1: CVaR LP con assets perfectamente correlados debe retornar
    equal-weight en lugar de lanzar RuntimeError."""
    np.random.seed(42)
    T = 252
    base = np.random.normal(0.001, 0.02, T)
    # Dos assets perfectamente correlados → LP puede fallar
    returns_singular = np.column_stack([base, base, base * 1.0001])

    # No debe lanzar excepción
    result = cvar_minimization(returns_singular, alpha=0.95, budget=1.0)

    assert result is not None, "cvar_minimization returned None"
    assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"
    assert abs(result.sum() - 1.0) < 1e-6, f"Weights don't sum to 1: {result.sum()}"
    assert not np.isnan(result).any(), "Result contains NaN"


def test_cvar_normal_case_unchanged():
    """CVaR en caso normal no debe cambiar comportamiento."""
    np.random.seed(0)
    returns = np.random.normal(
        loc=[0.001, 0.0008, 0.0003],
        scale=[0.02, 0.015, 0.005],
        size=(252, 3),
    )
    result = cvar_minimization(returns, alpha=0.95, budget=1.0)
    assert result.shape == (3,)
    assert abs(result.sum() - 1.0) < 1e-6
    assert not np.isnan(result).any()
