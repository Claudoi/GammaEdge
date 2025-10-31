import numpy as np
import polars as pl

from portfolio.backtest.engine import backtest_rebalanced


def test_backtest_rebalanced_smoke():
    df = pl.DataFrame(
        {
            "date": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 5),
                interval="1d",
                eager=True,
            ),
            "A": [0.01, 0.0, -0.01, 0.02, 0.0],
            "B": [0.0, 0.02, -0.01, 0.0, 0.01],
        }
    )

    def alloc(_win):
        return np.array([0.5, 0.5], dtype=float)

    bt = backtest_rebalanced(
        df_ret_wide=df,
        lookback=3,
        rebalance_freq="1d",
        cost_bps=0.0,
        allocator=alloc,
        bench_weights=np.zeros(2, dtype=float),
    )
    assert "equity" in bt and len(bt["equity"]) > 0
