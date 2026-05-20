from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from portfolio.features.returns import simple_returns, to_frequency
from portfolio.io.data_loader import _fetch_yfinance_close


def test_resample_simple_returns():
    # Serie sintética: 1% diario → mensual ~ (1.01)^k - 1
    idx = pd.date_range("2020-01-01", periods=22, freq="B")
    px = pd.DataFrame({"A": [100 * (1.01) ** i for i in range(len(idx))]}, index=idx)
    r = simple_returns(px)
    rm = to_frequency(r, "M")
    assert rm.shape[0] == 1
    approx = (1.01 ** (len(idx) - 1)) - 1
    assert abs(float(rm.iloc[0, 0]) - approx) < 1e-6


def test_fetch_yfinance_missing_ticker_raises():
    """SILENT-2: If yfinance returns fewer tickers than requested, raise ValueError."""
    dates = pd.date_range("2024-01-01", periods=20, name="Date")
    rng = np.random.default_rng(0)
    # Build ticker-first MultiIndex: (ticker, field) — matches group_by="ticker" output
    arrays = [
        ["AAPL", "AAPL", "MSFT", "MSFT"],
        ["Close", "Open", "Close", "Open"],
    ]
    columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Price"])
    mock_df = pd.DataFrame(
        rng.normal(200, 10, (20, 4)),
        index=dates,
        columns=columns,
    )

    with (
        patch("portfolio.io.data_loader.yf.download", return_value=mock_df),
        pytest.raises(ValueError, match="yfinance failed to download data for:"),
    ):
        _fetch_yfinance_close(
            tickers=("AAPL", "MSFT", "INVALID_TICKER"),
            start="2024-01-01",
            end="2024-01-31",
            interval="1d",
            adjust=True,
        )


def test_fetch_yfinance_all_tickers_present_ok():
    """If yfinance returns all tickers, no exception should be raised."""
    dates = pd.date_range("2024-01-01", periods=20, name="Date")
    rng = np.random.default_rng(0)
    # Build ticker-first MultiIndex: (ticker, field) — matches group_by="ticker" output
    arrays = [
        ["AAPL", "AAPL", "MSFT", "MSFT"],
        ["Close", "Open", "Close", "Open"],
    ]
    columns = pd.MultiIndex.from_arrays(arrays, names=["Ticker", "Price"])
    mock_df = pd.DataFrame(
        rng.normal(200, 10, (20, 4)),
        index=dates,
        columns=columns,
    )

    with patch("portfolio.io.data_loader.yf.download", return_value=mock_df):
        result = _fetch_yfinance_close(
            tickers=("AAPL", "MSFT"),
            start="2024-01-01",
            end="2024-01-31",
            interval="1d",
            adjust=True,
        )
    # Result should contain both tickers
    result_cols = set(result.columns) - {"Date"}
    assert "AAPL" in result_cols
    assert "MSFT" in result_cols
