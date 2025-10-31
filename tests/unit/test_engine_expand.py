# tests/unit/test_engine_expand.py
from __future__ import annotations

from datetime import datetime as dt

import numpy as np

from portfolio.backtest import attribution as bt_attr


def test_expand_rebalance_weights_basic():
    # T=4 fechas, rebalance K=2
    dates = [dt(2024, 1, d) for d in (1, 3, 5, 7)]
    rb_dates = [dates[0], dates[2]]
    W_reb = np.array([[0.6, 0.4], [0.2, 0.8]], dtype=float)

    Wd = bt_attr.expand_rebalance_weights(dates, rb_dates, W_reb)

    assert Wd.shape == (4, 2)
    # Primer bloque [0,1] con (0.6,0.4), segundo bloque [2,3] con (0.2,0.8)
    assert np.allclose(Wd[0], [0.6, 0.4])
    assert np.allclose(Wd[1], [0.6, 0.4])
    assert np.allclose(Wd[2], [0.2, 0.8])
    assert np.allclose(Wd[3], [0.2, 0.8])
    # Filas normalizadas
    assert np.allclose(Wd.sum(axis=1), 1.0)
