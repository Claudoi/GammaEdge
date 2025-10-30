# tests/test_brinson_vectorized.py
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from portfolio.backtest import attribution as bt_attr


def _toy_alignment() -> bt_attr.DailyAlignment:
    T, N = 40, 4
    rng = np.random.default_rng(0)
    R = 0.001 * rng.standard_normal((T, N)).astype(float)

    start = dt.datetime(2024, 1, 1)
    dates = [(start + dt.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(T)]

    df_ret = (
        pl.DataFrame({"date": dates, **{f"T{i}": R[:, i] for i in range(N)}})
        .with_columns(pl.col("date").str.strptime(pl.Datetime, strict=False))
        .sort("date")
    )

    K = 5
    W_reb = np.tile(np.full(N, 1.0 / N), (K, 1))
    rb_dates = [dates[i * (T // K)] for i in range(K)]
    W_daily = bt_attr.expand_rebalance_weights(
        dates=df_ret.get_column("date").to_list(), rb_dates=rb_dates, W_reb=W_reb
    )

    return bt_attr.align_returns_and_weights(df_ret, W_daily)


def test_brinson_shapes_and_finiteness():
    aln = _toy_alignment()
    T, N = len(aln.dates), len(aln.tickers)

    Wb_daily = np.tile(np.full(N, 1.0 / N), (T, 1))
    groups_idx = list(range(1, N + 1))

    res = bt_attr.brinson_fachler_vectorized(aln, Wb_daily, groups_idx, cumulative=True)

    for arr in (res.alloc, res.select, res.interact, res.total):
        assert len(arr) == T
        assert np.all(np.isfinite(arr))

    # Timeseries por grupo: el API devuelve un DataFrame con columnas
    ts_df = bt_attr.brinson_fachler_timeseries(
        aln, Wb_daily, groups_idx, cumulative=True, by_group=True
    )
    # Debe tener columnas esperadas y longitud T
    expected_cols = {"date", "alloc", "select", "interact", "total"}
    assert expected_cols.issubset(set(ts_df.columns))
    assert ts_df.height == T
    # Valores finitos en 'total'
    assert np.isfinite(ts_df.select(pl.col("total")).to_numpy()).all()
