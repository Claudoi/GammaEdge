import numpy as np
import polars as pl

from portfolio.backtest.scenarios import block_bootstrap_indices, historical_slice_returns


def test_block_bootstrap_indices_basic():
    T = 30
    block = 5
    idx = block_bootstrap_indices(T, block, seed=123)
    assert isinstance(idx, np.ndarray)
    assert idx.dtype == int
    assert len(idx) == T
    assert np.all(idx >= 0) and np.all(idx < T)


def test_historical_slice_inclusive_and_datetime():
    df = pl.DataFrame(
        {
            "date": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 10),
                interval="1d",
                eager=True,
            ),
            "A": np.linspace(0, 0.009, 10),
            "B": np.linspace(0, -0.009, 10),
        }
    )
    out = historical_slice_returns(df, "2024-01-03", "2024-01-05")
    assert out.height == 3
    # Datetime y límites inclusivos
    assert out.schema["date"] in (
        pl.Datetime,
        pl.Datetime("us"),
        pl.Datetime("ms"),
        pl.Datetime("ns"),
    )
    assert out["date"][0].date().isoformat() == "2024-01-03"
    assert out["date"][-1].date().isoformat() == "2024-01-05"
