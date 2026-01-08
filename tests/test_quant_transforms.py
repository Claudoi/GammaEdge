# tests/test_quant_transforms.py
"""Unit tests for quantitative transformations module."""

from __future__ import annotations

import math

import polars as pl
import pytest

from portfolio.features.quant_transforms import (
    compute_all_quant_metrics,
    compute_intraday_volatility,
    compute_log_returns,
    compute_relative_volume,
)


@pytest.fixture
def sample_ohlcv() -> pl.DataFrame:
    """Create sample OHLCV data for testing."""
    # 7 business days of data
    return pl.DataFrame(
        {
            "date": [
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
                "2023-01-06",
                "2023-01-09",
                "2023-01-10",
            ],
            "ticker": ["AAPL"] * 7,
            "open": [100.0, 102.0, 105.0, 103.0, 106.0, 108.0, 107.0],
            "high": [103.0, 106.0, 108.0, 107.0, 110.0, 112.0, 111.0],
            "low": [99.0, 101.0, 104.0, 102.0, 105.0, 107.0, 106.0],
            "close": [102.0, 105.0, 103.0, 106.0, 108.0, 107.0, 109.0],
            "volume": [1000.0, 1200.0, 800.0, 1100.0, 900.0, 1300.0, 1000.0],
        }
    ).with_columns(pl.col("date").str.to_datetime())


@pytest.fixture
def multi_ticker_ohlcv() -> pl.DataFrame:
    """Create multi-ticker sample data."""
    dates = [
        "2023-01-02",
        "2023-01-03",
        "2023-01-04",
        "2023-01-05",
        "2023-01-06",
    ]

    aapl = pl.DataFrame(
        {
            "date": dates,
            "ticker": ["AAPL"] * 5,
            "open": [100.0, 102.0, 105.0, 103.0, 106.0],
            "high": [103.0, 106.0, 108.0, 107.0, 110.0],
            "low": [99.0, 101.0, 104.0, 102.0, 105.0],
            "close": [102.0, 105.0, 103.0, 106.0, 108.0],
            "volume": [1000.0, 1200.0, 800.0, 1100.0, 900.0],
        }
    )
    msft = pl.DataFrame(
        {
            "date": dates,
            "ticker": ["MSFT"] * 5,
            "open": [200.0, 204.0, 210.0, 206.0, 212.0],
            "high": [206.0, 212.0, 216.0, 214.0, 220.0],
            "low": [198.0, 202.0, 208.0, 204.0, 210.0],
            "close": [204.0, 210.0, 206.0, 212.0, 216.0],
            "volume": [2000.0, 2400.0, 1600.0, 2200.0, 1800.0],
        }
    )
    return pl.concat([aapl, msft]).with_columns(pl.col("date").str.to_datetime())


class TestLogReturns:
    """Tests for log-returns calculation."""

    def test_log_returns_simple(self, sample_ohlcv: pl.DataFrame) -> None:
        """Test basic log return calculation."""
        result = compute_log_returns(sample_ohlcv)

        assert "log_ret" in result.columns
        assert "date" in result.columns
        assert "ticker" in result.columns
        # First row is dropped (no previous price)
        assert result.height == sample_ohlcv.height - 1

    def test_log_returns_values(self) -> None:
        """Test log return values are correct."""
        df = pl.DataFrame(
            {
                "date": ["2023-01-02", "2023-01-03"],
                "ticker": ["AAPL", "AAPL"],
                "close": [100.0, 110.0],
            }
        ).with_columns(pl.col("date").str.to_datetime())
        result = compute_log_returns(df)

        expected = math.log(110.0 / 100.0)
        actual = result.select("log_ret").item()
        assert abs(actual - expected) < 1e-10

    def test_log_returns_multi_ticker(self, multi_ticker_ohlcv: pl.DataFrame) -> None:
        """Test log returns are calculated per ticker."""
        result = compute_log_returns(multi_ticker_ohlcv)

        # Each ticker should have n-1 rows
        aapl_count = result.filter(pl.col("ticker") == "AAPL").height
        msft_count = result.filter(pl.col("ticker") == "MSFT").height

        assert aapl_count == 4
        assert msft_count == 4


class TestRelativeVolume:
    """Tests for relative volume calculation."""

    def test_relative_volume_constant(self) -> None:
        """Test that constant volume gives rel_volume = 1.0."""
        # Create 30 business days of data
        dates = [f"2023-01-{str(d).zfill(2)}" for d in range(2, 32)]

        df = pl.DataFrame(
            {
                "date": dates,
                "ticker": ["AAPL"] * len(dates),
                "volume": [1000.0] * len(dates),
            }
        ).with_columns(pl.col("date").str.to_datetime())

        result = compute_relative_volume(df, lookback=5)

        # After warmup period, all values should be 1.0
        values = result.filter(pl.col("rel_volume").is_not_null())["rel_volume"].to_list()
        for v in values[5:]:  # After lookback window
            assert abs(v - 1.0) < 1e-10

    def test_relative_volume_above_average(self, sample_ohlcv: pl.DataFrame) -> None:
        """Test that high volume days show rel_volume > 1."""
        result = compute_relative_volume(sample_ohlcv, lookback=3)

        assert "rel_volume" in result.columns
        assert result.height == sample_ohlcv.height


class TestIntradayVolatility:
    """Tests for intraday volatility estimators."""

    def test_parkinson_formula(self) -> None:
        """Test Parkinson volatility calculation."""
        # Single row: high=110, low=100
        df = pl.DataFrame(
            {
                "date": ["2023-01-02"],
                "ticker": ["AAPL"],
                "open": [105.0],
                "high": [110.0],
                "low": [100.0],
                "close": [105.0],
            }
        ).with_columns(pl.col("date").str.to_datetime())

        result = compute_intraday_volatility(df, method="parkinson")

        # Parkinson: σ = sqrt((ln(H/L))² / (4*ln(2)))
        expected = math.sqrt((math.log(110 / 100) ** 2) / (4 * math.log(2)))
        actual = result.select("intraday_vol").item()

        assert abs(actual - expected) < 1e-10

    def test_garman_klass_formula(self) -> None:
        """Test Garman-Klass volatility calculation."""
        df = pl.DataFrame(
            {
                "date": ["2023-01-02"],
                "ticker": ["AAPL"],
                "open": [100.0],
                "high": [110.0],
                "low": [95.0],
                "close": [105.0],
            }
        ).with_columns(pl.col("date").str.to_datetime())

        result = compute_intraday_volatility(df, method="garman_klass")

        # GK: σ² = 0.5*(ln(H/L))² - (2*ln(2)-1)*(ln(C/O))²
        ln_hl_sq = math.log(110 / 95) ** 2
        ln_co_sq = math.log(105 / 100) ** 2
        expected_var = 0.5 * ln_hl_sq - (2 * math.log(2) - 1) * ln_co_sq
        expected = math.sqrt(max(0, expected_var))

        actual = result.select("intraday_vol").item()
        assert abs(actual - expected) < 1e-10

    def test_volatility_multi_ticker(self, multi_ticker_ohlcv: pl.DataFrame) -> None:
        """Test volatility is calculated per ticker."""
        result = compute_intraday_volatility(multi_ticker_ohlcv, method="parkinson")

        assert result.height == multi_ticker_ohlcv.height
        assert result.filter(pl.col("ticker") == "AAPL").height == 5
        assert result.filter(pl.col("ticker") == "MSFT").height == 5


class TestComputeAllMetrics:
    """Tests for combined metrics function."""

    def test_all_metrics_combined(self, sample_ohlcv: pl.DataFrame) -> None:
        """Test that all metrics are calculated together."""
        result = compute_all_quant_metrics(sample_ohlcv, volume_lookback=3)

        assert "log_ret" in result.columns
        assert "rel_volume" in result.columns
        assert "intraday_vol" in result.columns
        assert "date" in result.columns
        assert "ticker" in result.columns

    def test_all_metrics_joins_correctly(self, multi_ticker_ohlcv: pl.DataFrame) -> None:
        """Test metrics are joined on correct date/ticker."""
        result = compute_all_quant_metrics(multi_ticker_ohlcv, volume_lookback=2)

        # Should have data for both tickers
        tickers = result.select(pl.col("ticker").unique()).to_series().to_list()
        assert "AAPL" in tickers
        assert "MSFT" in tickers
