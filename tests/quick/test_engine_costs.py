import numpy as np
import polars as pl

from portfolio.backtest.engine import backtest_rebalanced


def test_backtest_costs_reduce_equity():
    df = pl.DataFrame(
        {
            "date": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 10),
                interval="1d",
                eager=True,
            ),
            "A": [0.01, 0.0, -0.01, 0.02, 0.0, 0.0, -0.01, 0.01, 0.0, 0.0],
            "B": [0.0, 0.01, 0.0, -0.01, 0.02, 0.0, 0.0, -0.01, 0.01, 0.0],
        }
    )
    # Alterna pesos en cada rebalance para forzar turnover
    step = {"i": 0}

    def allocator(_win):
        if step["i"] % 2 == 0:
            w = np.array([0.8, 0.2], dtype=float)
        else:
            w = np.array([0.2, 0.8], dtype=float)
        step["i"] += 1
        return w

    kwargs = dict(
        df_ret_wide=df,
        lookback=3,
        rebalance_freq="2d",
        allocator=allocator,
        bench_weights=np.zeros(2, dtype=float),
    )

    bt_free = backtest_rebalanced(cost_bps=0.0, **kwargs)
    bt_cost = backtest_rebalanced(cost_bps=50.0, **kwargs)  # 50 bps

    assert "equity" in bt_free and "equity" in bt_cost
    assert len(bt_free["equity"]) == len(bt_cost["equity"]) > 0
    assert bt_cost["equity"][-1] < bt_free["equity"][-1]
