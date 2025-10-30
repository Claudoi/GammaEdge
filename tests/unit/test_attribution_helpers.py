# tests/unit/test_attribution_helpers.py
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from portfolio.backtest import attribution as bt_attr


def test_align_returns_and_weights_shape_ok():
    """Ensure alignment produces consistent daily shapes and valid tickers/dates."""
    dates = [datetime(2024, 1, d) for d in (1, 2, 3)]
    df = pl.DataFrame(
        {
            "date": dates,
            "A": [0.01, 0.0, -0.02],
            "B": [0.0, 0.002, 0.0],
        }
    )
    Wd = np.tile(np.array([0.6, 0.4]), (3, 1))
    aln = bt_attr.align_returns_and_weights(df, Wd)

    # Expected attributes (DailyAlignment exposes `returns` and `weights`)
    assert hasattr(aln, "returns")
    assert hasattr(aln, "weights")
    assert aln.weights.shape == (3, 2)
    assert aln.returns.shape == (3, 2)
    assert len(aln.tickers) == 2
    assert len(aln.dates) == 3


def test_expand_rebalance_weights_edge_last_block():
    """Check that expand_rebalance_weights correctly fills the last segment."""
    dates = [datetime(2024, 1, d) for d in (1, 3, 5, 7, 9)]
    rb_dates = [dates[0], dates[3]]
    W_reb = np.array([[0.5, 0.5], [0.2, 0.8]])
    Wd = bt_attr.expand_rebalance_weights(dates, rb_dates, W_reb)

    # Last block [3..end] gets the last weights
    assert np.allclose(Wd[-1], [0.2, 0.8])


def test_coerce_benchmark_weights_ew_rowsum():
    """Public API: EW benchmark must be row-normalised."""
    T, N = 4, 3
    Wb = bt_attr.coerce_benchmark_weights(None, T, N, scheme="EW")
    assert Wb.shape == (T, N)
    # Each row should sum to 1.0
    assert np.allclose(Wb.sum(axis=1), 1.0)
