import numpy as np

from portfolio.backtest import metrics as m


def test_compute_backtest_metrics_smoke():
    bt = {
        "dates": [1, 2, 3],
        "equity": np.array([1.0, 1.01, 1.00], dtype=float),
        "weights": np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float),
        "tickers": ["A", "B"],
        "rebalance_dates": [1, 2],
    }
    out = m.compute_backtest_metrics(bt)
    assert hasattr(out, "height")
    assert out.height >= 1
