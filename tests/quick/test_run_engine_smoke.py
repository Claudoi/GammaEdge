# tests/quick/test_run_engine_smoke.py

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from portfolio.backtest.allocators import make_allocator
from portfolio.backtest.engine import backtest_rebalanced


def test_backtest_rebalanced_smoke_equal_weight() -> None:
    # Generate stable Python datetime-based date range
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(5)]

    df = pl.DataFrame(
        {
            "date": dates,
            "A": [0.01, 0.02, -0.01, 0.00, 0.03],
            "B": [0.00, -0.01, 0.01, 0.02, -0.02],
        }
    )

    n_cols = 2

    alloc = make_allocator(
        "Equal-Weight",
        w_min=0.0,
        w_max=1.0,
        cov_estimator="Sample",
        ewma_lambda=0.97,
        use_to_budget=False,
        max_turnover=0.0,
        band_eps=0.0,
    )

    bt = backtest_rebalanced(
        df_ret_wide=df,
        lookback=3,
        rebalance_freq="1w",
        cost_bps=0.0,
        allocator=alloc,
        bench_weights=np.full(n_cols, 1.0 / n_cols),
    )

    assert "equity" in bt
    assert len(bt["equity"]) == len(bt["dates"])
    # Smoke-level assertion: we just need at least one point
    assert len(bt["equity"]) >= 1
