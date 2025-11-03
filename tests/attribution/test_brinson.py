# tests/attribution/test_brinson.py

import polars as pl

from portfolio.attribution import compute_brinson_timeseries


def test_compute_brinson_timeseries_wide_to_long():
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
    res = compute_brinson_timeseries(df_wide)
    out = res.df
    # schema and basic integrity
    assert all(
        c in out.columns for c in ["date", "group_id", "alloc", "select", "interact", "total"]
    )
    # Should keep actual pairs present in input (no cartesian completion)
    assert set(out.get_column("group_id").to_list()) == {0, 1}
