"""
Unit tests for production-grade quant metrics module.

Tests verify:
- Correct mathematical formulas
- Proper data alignment (inner joins)
- Edge case handling (zero variance, short windows, etc.)
- Guard clauses and warnings
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from portfolio.features.quant_metrics import (
    TRADING_DAYS_PER_YEAR,
    calculate_beta_alpha,
    calculate_cagr,
    calculate_calmar,
    calculate_correlation_matrix,
    calculate_data_quality,
    calculate_max_drawdown,
    calculate_moments,
    calculate_returns,
    calculate_sharpe_ratio,
)


class TestReturnsCalculation:
    """Test simple returns calculation."""

    def test_returns_calculation(self):
        """Verify r_t = (P_t - P_{t-1}) / P_{t-1}"""
        prices = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "adj_close_AAPL": [100.0, 102.0, 101.0],
        })

        returns = calculate_returns(prices)

        assert "ret_AAPL" in returns.columns
        ret_values = returns["ret_AAPL"].to_list()

        # First return is None (Polars null)
        assert ret_values[0] is None
        # Second return: (102-100)/100 = 0.02
        assert abs(ret_values[1] - 0.02) < 1e-10
        # Third return: (101-102)/102 ≈ -0.0098
        assert abs(ret_values[2] - (-0.00980392)) < 1e-6


class TestBetaAlpha:
    """Test Beta and Alpha calculations."""

    def test_beta_matches_cov_var(self):
        """Beta = Cov(r_i, r_m) / Var(r_m) with synthetic data"""
        # Create synthetic data with known beta
        np.random.seed(42)
        n = 100
        r_m = np.random.normal(0.001, 0.02, n)
        beta_true = 1.5
        alpha_true = 0.0005
        r_i = alpha_true + beta_true * r_m + np.random.normal(0, 0.01, n)

        # Use pl.date_range for proper pl.Date type
        dates = pl.date_range(
            start=date(2024, 1, 1),
            end=date(2024, 4, 9),  # 100 days
            interval="1d",
            eager=True,
        )

        result = calculate_beta_alpha(
            returns=pl.Series(r_i),
            benchmark_returns=pl.Series(r_m),
            dates=dates,
            benchmark_dates=dates,
        )

        # Beta should be close to 1.5
        assert result["beta"] is not None
        assert abs(result["beta"] - beta_true) < 0.2  # Allow some noise

    def test_beta_inner_join_alignment(self):
        """Verify dates are aligned via inner join"""
        # Ticker has different dates than benchmark
        ticker_dates = pl.Series([
            date(2024, 1, 1),
            date(2024, 1, 2),
            date(2024, 1, 4),  # Missing day 3
        ])
        ticker_returns = pl.Series([0.01, 0.02, -0.01])

        bench_dates = pl.Series([
            date(2024, 1, 1),
            date(2024, 1, 2),
            date(2024, 1, 3),  # Extra day
            date(2024, 1, 4),
        ])
        bench_returns = pl.Series([0.005, 0.015, 0.01, -0.005])

        result = calculate_beta_alpha(
            returns=ticker_returns,
            benchmark_returns=bench_returns,
            dates=ticker_dates,
            benchmark_dates=bench_dates,
        )

        # Should align on 3 dates (1, 2, 4)
        assert result["n_obs"] == 3
        # Data loss: 0% (all ticker dates matched)
        assert result["data_loss_pct"] == 0.0

    def test_beta_zero_variance_returns_none(self):
        """If Var(r_m) < 1e-10 → None"""
        from datetime import date, timedelta
        start_date = date(2024, 1, 1)
        dates = pl.Series([start_date + timedelta(days=i) for i in range(32)])

        # Benchmark with zero variance
        bench_returns = pl.Series([0.0] * len(dates))
        ticker_returns = pl.Series(np.random.normal(0.001, 0.01, len(dates)))

        result = calculate_beta_alpha(
            returns=ticker_returns,
            benchmark_returns=bench_returns,
            dates=dates,
            benchmark_dates=dates,
        )

        assert result["beta"] is None
        assert result["alpha_daily"] is None


class TestSharpeRatio:
    """Test Sharpe Ratio calculations."""

    def test_sharpe_rf_zero_matches_formula(self):
        """Sharpe with rf=0: mean/std * sqrt(252)"""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0.001, 0.02, 100))

        result = calculate_sharpe_ratio(returns, rf_annual=0.0, min_obs=60)

        # Manual calculation
        r = returns.to_numpy()
        expected_sharpe = (np.mean(r) / np.std(r, ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)

        assert result["sharpe_ratio"] is not None
        assert abs(result["sharpe_ratio"] - expected_sharpe) < 1e-10

    def test_sharpe_short_window_returns_none(self):
        """n < 60 → None + warning"""
        returns = pl.Series(np.random.normal(0.001, 0.02, 50))

        result = calculate_sharpe_ratio(returns, min_obs=60)

        assert result["sharpe_ratio"] is None
        assert result["warning"] is not None
        assert "Sample size" in result["warning"]


class TestMaxDrawdown:
    """Test Maximum Drawdown calculations."""

    def test_mdd_always_non_positive(self):
        """MDD ≤ 0"""
        # Random prices
        np.random.seed(42)
        prices = pl.Series(np.cumsum(np.random.normal(0.001, 0.02, 100)) + 100)

        result = calculate_max_drawdown(prices)

        assert result["max_drawdown"] <= 0.0

    def test_mdd_matches_known_path(self):
        """[100, 120, 90] → MDD = -0.25"""
        prices = pl.Series([100.0, 120.0, 90.0])

        result = calculate_max_drawdown(prices)

        # Peak at 120, trough at 90
        # MDD = (90/120) - 1 = -0.25
        expected_mdd = -0.25
        assert abs(result["max_drawdown"] - expected_mdd) < 1e-10


class TestCAGR:
    """Test CAGR calculations."""

    def test_cagr_matches_manual(self):
        """CAGR = (P_end/P_start)**(252/n) - 1"""
        # 2 years of data (504 trading days)
        n_days = 504
        start_price = 100.0
        end_price = 150.0  # 50% total return

        prices = pl.Series(np.linspace(start_price, end_price, n_days + 1))
        from datetime import date, timedelta
        start_date = date(2022, 1, 1)
        dates = pl.Series([start_date + timedelta(days=i) for i in range(n_days + 1)])

        result = calculate_cagr(prices, dates, min_days=252)

        # Manual CAGR
        expected_cagr = (end_price / start_price) ** (252 / n_days) - 1

        assert result["cagr"] is not None
        assert abs(result["cagr"] - expected_cagr) < 1e-10


class TestCalmar:
    """Test Calmar Ratio calculations."""

    def test_calmar_nan_when_mdd_zero(self):
        """If |MDD| < 1e-6 → None"""
        cagr = 0.15
        mdd = -1e-7  # Near zero

        result = calculate_calmar(cagr, mdd)

        assert result is None

    def test_calmar_calculation(self):
        """Calmar = CAGR / |MDD|"""
        cagr = 0.185
        mdd = -0.35

        result = calculate_calmar(cagr, mdd)

        expected_calmar = 0.185 / 0.35
        assert abs(result - expected_calmar) < 1e-10


class TestMoments:
    """Test distribution moments (skewness, kurtosis)."""

    def test_kurtosis_is_excess(self):
        """Normal dist → kurtosis ≈ 0"""
        np.random.seed(42)
        # Large sample from normal distribution
        returns = pl.Series(np.random.normal(0, 1, 10000))

        result = calculate_moments(returns)

        # Excess kurtosis of normal dist should be close to 0
        assert abs(result["kurtosis"]) < 0.2  # Allow some sampling error


class TestCorrelation:
    """Test correlation matrix calculations."""

    def test_correlation_diagonal_ones(self):
        """Diagonal = 1.0"""
        returns = pl.DataFrame({
            "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(32)],
            "ret_AAPL": np.random.normal(0.001, 0.02, 32),
            "ret_MSFT": np.random.normal(0.001, 0.02, 32),
        })

        result = calculate_correlation_matrix(returns)

        corr_matrix = result["correlation_matrix"]
        assert corr_matrix is not None

        # Diagonal should be 1.0
        assert abs(corr_matrix["AAPL"][0] - 1.0) < 1e-10  # AAPL vs AAPL
        assert abs(corr_matrix["MSFT"][1] - 1.0) < 1e-10  # MSFT vs MSFT

    def test_correlation_single_ticker_returns_none(self):
        """Single ticker → None"""
        returns = pl.DataFrame({
            "date": pl.date_range(
                start=date(2024, 1, 1),
                end=date(2024, 2, 1),
                interval="1d",
                eager=True,
            ),
            "ret_AAPL": np.random.normal(0.001, 0.02, 32),
        })

        result = calculate_correlation_matrix(returns)

        assert result["correlation_matrix"] is None


class TestDataQuality:
    """Test data quality metrics."""

    def test_data_quality_coverage(self):
        """Coverage = n_obs / expected_obs"""
        # Benchmark has 10 days
        from datetime import date, timedelta
        start_date = date(2024, 1, 1)
        bench_dates = pl.Series([start_date + timedelta(days=i) for i in range(10)])

        # Ticker missing 2 days
        ticker_dates = pl.Series([
            date(2024, 1, 1),
            date(2024, 1, 2),
            date(2024, 1, 3),
            # Missing 4, 5
            date(2024, 1, 6),
            date(2024, 1, 7),
            date(2024, 1, 8),
            date(2024, 1, 9),
            date(2024, 1, 10),
        ])

        result = calculate_data_quality(ticker_dates, bench_dates, "AAPL")

        # 8 out of 10 = 80%
        assert result["coverage_pct"] == 80.0
        assert result["n_obs"] == 8
        assert result["expected_obs"] == 10
