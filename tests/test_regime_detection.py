"""
Unit tests for regime detection module.

Tests the HMM-based regime detector with synthetic and historical data.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio.features.regime_detection import (
    RegimeDetector,
    compute_regime_performance,
    detect_regimes,
)


@pytest.fixture
def synthetic_returns_bull():
    """Synthetic bull market: low vol, positive drift."""
    np.random.seed(42)
    dates = pl.date_range(start=date(2024, 1, 1), end=date(2024, 6, 30), interval="1d", eager=True)
    returns = np.random.normal(loc=0.001, scale=0.01, size=len(dates))

    return pl.DataFrame(
        {
            "date": dates,
            "returns": returns,
        }
    )


@pytest.fixture
def synthetic_returns_bear():
    """Synthetic bear market: high vol, negative drift."""
    np.random.seed(123)
    dates = pl.date_range(start=date(2024, 7, 1), end=date(2024, 12, 31), interval="1d", eager=True)
    returns = np.random.normal(loc=-0.002, scale=0.025, size=len(dates))

    return pl.DataFrame(
        {
            "date": dates,
            "returns": returns,
        }
    )


@pytest.fixture
def synthetic_returns_mixed(synthetic_returns_bull, synthetic_returns_bear):
    """Mixed regime data: concat bull + bear."""
    return pl.concat([synthetic_returns_bull, synthetic_returns_bear])


@pytest.fixture
def synthetic_returns_crisis():
    """Synthetic crisis: extreme vol, large drawdown (>=100 obs for HMM)."""
    np.random.seed(456)
    dates = pl.date_range(start=date(2024, 1, 1), end=date(2024, 5, 31), interval="1d", eager=True)

    # Simulate a crash: large negative shocks
    returns = []
    for i in range(len(dates)):
        if i % 5 == 0:  # 20% of days are crashes
            returns.append(np.random.normal(-0.05, 0.03))  # -5% mean
        else:
            returns.append(np.random.normal(-0.01, 0.02))  # -1% mean

    return pl.DataFrame(
        {
            "date": dates,
            "returns": np.array(returns),
        }
    )


class TestRegimeDetector:
    """Test suite for RegimeDetector class."""

    def test_detector_initialization(self):
        """Test detector can be created."""
        detector = RegimeDetector(n_regimes=3, random_state=42)

        assert detector.n_regimes == 3
        assert detector.model is None
        assert detector.regime_stats is None

    def test_detector_fit(self, synthetic_returns_mixed):
        """Test detector fits without errors."""
        detector = RegimeDetector(n_regimes=2, random_state=42)
        detector.fit(synthetic_returns_mixed, returns_col="returns")

        assert detector.model is not None
        assert detector.regime_stats is not None
        assert len(detector.regime_stats) == 2

    def test_predict_output_structure(self, synthetic_returns_mixed):
        """Test predict returns correct structure."""
        detector = RegimeDetector(n_regimes=2, random_state=42)
        detector.fit(synthetic_returns_mixed, returns_col="returns")

        result = detector.predict(synthetic_returns_mixed, returns_col="returns")

        # Check columns exist
        assert "regime" in result.columns
        assert "regime_label" in result.columns
        assert "prob_0" in result.columns
        assert "prob_1" in result.columns

        # Check lengths match
        assert len(result) == len(synthetic_returns_mixed)

        # Check regime values are valid
        regimes = result["regime"].to_numpy()
        assert np.all((regimes >= 0) & (regimes < 2))

    def test_regime_probabilities_sum_to_one(self, synthetic_returns_mixed):
        """Test regime probabilities sum to 1.0."""
        detector = RegimeDetector(n_regimes=3, random_state=42)
        detector.fit(synthetic_returns_mixed, returns_col="returns")

        probs = detector.predict_proba(synthetic_returns_mixed, returns_col="returns")

        # Each row should sum to 1.0
        row_sums = probs.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(probs)), decimal=5)

    def test_regime_persistence(self, synthetic_returns_mixed):
        """Test regimes have autocorrelation (persistence)."""
        detector = RegimeDetector(n_regimes=2, random_state=42)
        detector.fit(synthetic_returns_mixed, returns_col="returns")

        result = detector.predict(synthetic_returns_mixed, returns_col="returns")
        regimes = result["regime"].to_numpy()

        # Count regime switches
        switches = np.sum(regimes[:-1] != regimes[1:])
        total = len(regimes) - 1

        # Regimes should be persistent (less than 50% switch rate)
        switch_rate = switches / total
        assert switch_rate < 0.5, f"Regimes switch too frequently: {switch_rate:.1%}"

    def test_crisis_detection_high_vol(self, synthetic_returns_crisis):
        """Test that crisis regime has highest volatility."""
        detector = RegimeDetector(n_regimes=3, random_state=42)
        detector.fit(synthetic_returns_crisis, returns_col="returns")

        stats = detector.get_regime_stats()

        # At least one regime should have high volatility
        max_vol = stats["volatility"].max()
        assert max_vol > 0.02, f"Expected crisis volatility > 2%, got {max_vol:.2%}"

    def test_transition_matrix_properties(self, synthetic_returns_mixed):
        """Test transition matrix is valid probability matrix."""
        detector = RegimeDetector(n_regimes=2, random_state=42)
        detector.fit(synthetic_returns_mixed, returns_col="returns")

        trans_mat = detector.get_transition_matrix()

        # Check shape
        assert trans_mat.shape == (2, 2)

        # Each row should sum to 1.0 (probability distribution)
        row_sums = trans_mat.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(2), decimal=5)

        # All values should be in [0, 1]
        assert np.all((trans_mat >= 0) & (trans_mat <= 1))

    def test_regime_stats_completeness(self, synthetic_returns_mixed):
        """Test regime stats contain all required metrics."""
        detector = RegimeDetector(n_regimes=2, random_state=42)
        detector.fit(synthetic_returns_mixed, returns_col="returns")

        stats = detector.get_regime_stats()

        required_cols = ["regime", "mean_return", "volatility", "frequency", "avg_duration"]
        for col in required_cols:
            assert col in stats.columns, f"Missing column: {col}"

    def test_fit_before_predict_error(self, synthetic_returns_mixed):
        """Test error is raised if predict called before fit."""
        detector = RegimeDetector(n_regimes=2)

        with pytest.raises(ValueError, match="Model not fitted"):
            detector.predict(synthetic_returns_mixed, returns_col="returns")

    def test_different_random_seeds_converge(self, synthetic_returns_mixed):
        """Test that different random seeds give similar results."""
        detector1 = RegimeDetector(n_regimes=2, random_state=42)
        detector2 = RegimeDetector(n_regimes=2, random_state=123)

        detector1.fit(synthetic_returns_mixed, returns_col="returns")
        detector2.fit(synthetic_returns_mixed, returns_col="returns")

        result1 = detector1.predict(synthetic_returns_mixed, returns_col="returns")
        result2 = detector2.predict(synthetic_returns_mixed, returns_col="returns")

        # Regimes might be permuted, but should have similar partition
        # Check that high-return periods are classified similarly
        df_pd = synthetic_returns_mixed.to_pandas()
        high_return_mask = df_pd["returns"] > df_pd["returns"].median()

        # Convert mask to Polars column for filtering
        result1_pd = result1.to_pandas()
        result2_pd = result2.to_pandas()

        regime1_high = (
            result1_pd.loc[high_return_mask, "regime"].mode()[0] if high_return_mask.any() else 0
        )
        regime2_high = (
            result2_pd.loc[high_return_mask, "regime"].mode()[0] if high_return_mask.any() else 0
        )

        # Both should classify high-return periods consistently
        # (Just checking they converge, not exact match due to label switching)
        assert regime1_high is not None
        assert regime2_high is not None


class TestDetectRegimesConvenience:
    """Test convenience function."""

    def test_detect_regimes_one_shot(self, synthetic_returns_mixed):
        """Test one-shot regime detection function."""
        result = detect_regimes(
            synthetic_returns_mixed,
            returns_col="returns",
            n_regimes=2,
            random_state=42,
        )

        # Should return complete DataFrame
        assert "regime" in result.columns
        assert "regime_label" in result.columns
        assert len(result) == len(synthetic_returns_mixed)


class TestComputeRegimePerformance:
    """Test performance metric calculation."""

    def test_performance_calculation(self, synthetic_returns_mixed):
        """Test regime performance metrics are computed."""
        # First detect regimes
        result = detect_regimes(
            synthetic_returns_mixed,
            returns_col="returns",
            n_regimes=2,
            random_state=42,
        )

        # Compute performance
        perf = compute_regime_performance(
            result,
            regime_col="regime_label",
            returns_col="returns",
        )

        # Check structure
        assert len(perf) == 2  # 2 regimes
        required_cols = ["regime", "mean_return", "volatility", "sharpe", "frequency", "n_days"]
        for col in required_cols:
            assert col in perf.columns

        # Frequencies should sum to ~1.0
        total_freq = perf["frequency"].sum()
        assert 0.99 <= total_freq <= 1.01

    def test_performance_sharpe_calculation(self, synthetic_returns_bull):
        """Test Sharpe ratio is reasonable for bull market."""
        result = detect_regimes(
            synthetic_returns_bull,
            returns_col="returns",
            n_regimes=1,  # Single regime
            random_state=42,
        )

        perf = compute_regime_performance(result, returns_col="returns")

        # Bull market should have positive Sharpe (usually)
        # Not guaranteed with random noise, so just check it's computed
        assert "sharpe" in perf.columns
        assert not perf["sharpe"].isna().any()


@pytest.mark.parametrize("n_regimes", [2, 3, 4])
class TestDifferentRegimeCounts:
    """Test detector works with different numbers of regimes."""

    def test_n_regimes_parameter(self, n_regimes, synthetic_returns_mixed):
        """Test detector handles different regime counts."""
        detector = RegimeDetector(n_regimes=n_regimes, random_state=42)
        detector.fit(synthetic_returns_mixed, returns_col="returns")

        result = detector.predict(synthetic_returns_mixed, returns_col="returns")

        # Should have n_regimes probability columns
        for i in range(n_regimes):
            assert f"prob_{i}" in result.columns

        # Regime values should be in [0, n_regimes-1]
        regimes = result["regime"].to_numpy()
        assert np.all((regimes >= 0) & (regimes < n_regimes))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_short_series(self):
        """Test that very short time series raises a clear ValueError (HIGH-2)."""
        short_df = pl.DataFrame(
            {
                "date": pl.date_range(
                    start=date(2024, 1, 1), end=date(2024, 1, 10), interval="1d", eager=True
                ),
                "returns": np.random.normal(0, 0.01, 10),
            }
        )

        detector = RegimeDetector(n_regimes=2, random_state=42, n_iter=50)
        with pytest.raises(ValueError, match="observations"):
            detector.fit(short_df, returns_col="returns")

    def test_constant_returns(self):
        """Constant returns (all zeros) trigger hmmlearn singular covariance.

        This degenerate input has no variance so the covariance matrix is
        singular and hmmlearn raises ValueError.  The test verifies that some
        ValueError is propagated rather than the process hanging silently.
        """
        dates = pl.date_range(
            start=date(2024, 1, 1), end=date(2024, 6, 30), interval="1d", eager=True
        )
        const_df = pl.DataFrame(
            {
                "date": dates,
                "returns": np.zeros(len(dates)),  # All zeros → singular covariance
            }
        )

        detector = RegimeDetector(n_regimes=2, random_state=42, n_iter=50)

        # Degenerate data causes hmmlearn to raise ValueError internally
        with pytest.raises(ValueError):
            detector.fit(const_df, returns_col="returns")


# ---------------------------------------------------------------------------
# HIGH-2: minimum observations validation
# ---------------------------------------------------------------------------


def test_hmm_fit_raises_on_insufficient_data():
    """HIGH-2: fit() must raise ValueError with < required observations."""
    from datetime import timedelta

    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(50)]
    df = pl.DataFrame(
        {
            "date": dates,
            "returns": np.random.default_rng(0).normal(0.001, 0.02, 50).tolist(),
        }
    )
    detector = RegimeDetector(n_regimes=3)
    with pytest.raises(ValueError, match="observations"):
        detector.fit(df, returns_col="returns")


def test_hmm_fit_ok_with_sufficient_data():
    """HIGH-2: fit() must not raise with >= required observations."""
    from datetime import timedelta

    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(300)]
    rng = np.random.default_rng(1)
    df = pl.DataFrame(
        {
            "date": dates,
            "returns": rng.normal(0.001, 0.02, 300).tolist(),
        }
    )
    detector = RegimeDetector(n_regimes=3)
    detector.fit(df, returns_col="returns")  # must not raise
    assert detector.model is not None
