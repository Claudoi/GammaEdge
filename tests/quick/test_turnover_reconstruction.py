# tests/quick/test_turnover_reconstruction.py

import numpy as np

from portfolio.backtest import metrics as bt_metrics


def test_compute_backtest_metrics_basic() -> None:
    bt = {
        "dates": [1, 2, 3, 4],
        "equity": np.array([100.0, 101.0, 99.0, 102.0], dtype=float),
        "weights": np.array(
            [
                [0.5, 0.5],
                [0.6, 0.4],
                [0.4, 0.6],
                [0.5, 0.5],
            ],
            dtype=float,
        ),
        "tickers": ["A", "B"],
        "turnover": np.array([0.0, 0.1, 0.05], dtype=float),
    }

    dfm = bt_metrics.compute_backtest_metrics(bt)

    # Support both wide format (CAGR, Sharpe, MaxDD as columns)
    # and long format (columns ['metric', 'value'])
    cols = set(dfm.columns)

    if {"metric", "value"}.issubset(cols):
        # Long format: metric/value rows
        metrics_map = {str(m): float(v) for m, v in dfm.select(["metric", "value"]).iter_rows()}
        for col in ("CAGR", "Sharpe", "MaxDD"):
            assert col in metrics_map
    else:
        # Wide format: direct columns
        for col in ("CAGR", "Sharpe", "MaxDD"):
            assert col in dfm.columns
