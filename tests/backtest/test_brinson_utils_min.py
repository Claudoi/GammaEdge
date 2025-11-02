# tests/backtest/test_brinson_utils_min.py

import numpy as np
import pandas as pd
import polars as pl

from portfolio.backtest import brinson_utils as bu


def test_brinson_basic_coercers_and_align():
    # Toy data: 2 assets, 3 days
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    r_p = pd.DataFrame({"A": [0.01, -0.02, 0.03], "B": [0.00, 0.01, -0.01]}, index=idx)
    r_b = pd.DataFrame({"A": [0.00, -0.01, 0.02], "B": [0.00, 0.00, 0.00]}, index=idx)

    # Coercers should not explode
    rp = bu.to_pandas_2d_returns(r_p)
    rb = bu.to_pandas_2d_returns(r_b)

    assert isinstance(rp, pd.DataFrame)
    assert rp.shape == rb.shape
    assert all(c in rp.columns for c in ["A", "B"])

    # Column alignment
    cols = bu.align_columns(rp.columns, rb.columns)
    assert cols == ["A", "B"]


def test_to_pandas_2d_returns_numpy_and_polars():
    # NumPy -> pandas
    arr = np.array([[0.1, 0.2], [0.3, 0.4]])
    df_np = bu.to_pandas_2d_returns(arr)
    assert isinstance(df_np, pd.DataFrame)
    assert df_np.shape == (2, 2)

    # Polars -> pandas (with 'date' column)
    df_pl = pl.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "x": [0.1, 0.2],
            "y": [0.3, 0.4],
        }
    )
    df_pd = bu.to_pandas_2d_returns(df_pl)
    assert isinstance(df_pd, pd.DataFrame)
    assert df_pd.shape == (2, 2)
    assert all(c in df_pd.columns for c in ["x", "y"])


def test_coerce_brinson_timeseries_to_long_variants():
    # Global-only format
    df_global = pl.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "alloc": [0.1, 0.2],
            "select": [0.3, 0.4],
            "interact": [0.0, 0.0],
            "total": [0.4, 0.6],
        }
    )
    out = bu.coerce_brinson_timeseries_to_long(df_global)
    assert all(
        c in out.columns for c in ["date", "group_id", "alloc", "select", "interact", "total"]
    )
    assert out["group_id"].to_list() == [0, 0]

    # Long with group_id
    df_gid = df_global.with_columns(pl.lit(1).alias("group_id"))
    out_gid = bu.coerce_brinson_timeseries_to_long(df_gid)
    assert "group_id" in out_gid.columns

    # Long with group (string labels)
    df_grp = df_global.with_columns(pl.lit("A").alias("group"))
    out_grp = bu.coerce_brinson_timeseries_to_long(df_grp)
    assert set(out_grp.columns) == {"date", "group_id", "alloc", "select", "interact", "total"}

    # Wide: metric_0, metric_1 (only real (date, group_id) pairs are kept)
    df_wide = pl.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "alloc_0": [0.1, 0.2],
            "alloc_1": [0.3, 0.4],
            "select_0": [0.5, 0.6],
            "select_1": [0.7, 0.8],
            "interact_0": [0.0, 0.0],
            "interact_1": [0.1, 0.2],
            "total_0": [0.6, 0.8],
            "total_1": [0.9, 1.0],
        }
    )
    out_wide = bu.coerce_brinson_timeseries_to_long(df_wide)
    assert all(
        c in out_wide.columns for c in ["date", "group_id", "alloc", "select", "interact", "total"]
    )

    # Expect 4 rows (actual pairs), not 4 (no cartesian completion).
    assert len(out_wide) == 4

    # Ensure we have exactly the two groups present
    gids = sorted(out_wide["group_id"].unique().to_list())
    assert gids == [0, 1]

    # Normalize date in Polars to a stable string, then go to pandas
    ow_norm = (
        out_wide.sort(["date", "group_id"])
        .with_columns(pl.col("date").dt.strftime("%Y-%m-%d").alias("date_str"))
        .select(["date_str", "group_id", "alloc", "select", "interact", "total"])
        .to_pandas()
    )

    # Lookup by (date_str, group_id)
    lookup = {(row["date_str"], int(row["group_id"])): row for _, row in ow_norm.iterrows()}

    # ('2020-01-01', 0) -> metrics from *_0 columns on 2020-01-01
    assert ("2020-01-01", 0) in lookup
    row_0101_g0 = lookup[("2020-01-01", 0)]
    assert row_0101_g0["alloc"] == 0.1
    assert row_0101_g0["select"] == 0.5
    assert row_0101_g0["interact"] == 0.0
    assert row_0101_g0["total"] == 0.6

    # ('2020-01-02', 1) -> metrics from *_1 columns on 2020-01-02
    assert ("2020-01-02", 1) in lookup
    row_0102_g1 = lookup[("2020-01-02", 1)]
    assert row_0102_g1["alloc"] == 0.4
    assert row_0102_g1["select"] == 0.8
    assert row_0102_g1["interact"] == 0.2
    assert row_0102_g1["total"] == 1.0


def test_align_columns_fallback_order():
    cols_a = ["a", "b", "c"]
    cols_b = ["c", "a"]
    assert bu.align_columns(cols_a, cols_b) == ["a", "c"]

    cols_a = ["x", "y"]
    cols_b = ["y", "x"]
    assert bu.align_columns(cols_a, cols_b) == ["x", "y"]
