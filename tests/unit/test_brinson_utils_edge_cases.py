# tests/unit/test_brinson_utils_edge_cases.py
from __future__ import annotations

from datetime import datetime as dt

import polars as pl

from portfolio.backtest.brinson_utils import (
    coerce_brinson_timeseries_to_long as _to_long,
)
from portfolio.backtest.brinson_utils import (
    ensure_datetime as _ensure_datetime,
)


def test_ensure_datetime_already_datetime():
    df = pl.DataFrame(
        {
            "date": [dt(2024, 1, 1), dt(2024, 1, 2), dt(2024, 1, 3)],
            "x": [1.0, 2.0, 3.0],
        }
    )
    out = _ensure_datetime(df, "date")
    assert out.schema["date"] == pl.Datetime
    assert out.select(pl.col("date").is_null().sum()).item() == 0
    assert out.height == 3


def test_coerce_brinson_timeseries_to_long_multi_groups_and_types():
    # Varios grupos para cubrir la normalización y salida long
    df = pl.DataFrame(
        {
            "date": [dt(2024, 1, 1), dt(2024, 1, 2), dt(2024, 1, 1), dt(2024, 1, 2)],
            "group_id": [1, 1, 2, 2],
            "alloc": [0.01, 0.02, -0.01, 0.00],
            "select": [0.00, 0.01, 0.00, 0.02],
            "interact": [0.00, 0.00, 0.00, 0.00],
            "total": [0.01, 0.03, -0.01, 0.02],
        }
    )
    df = _ensure_datetime(df, "date")
    long = _to_long(df)

    expected_cols = {"date", "group_id", "alloc", "select", "interact", "total"}
    assert expected_cols <= set(long.columns)

    # Numéricos: acepta Float64 o Float32
    for col in ("alloc", "select", "interact", "total"):
        dtype = long.schema[col]
        assert dtype in (pl.Float64, pl.Float32)

    assert long.height == 4
    assert set(long.select("group_id").to_series().to_list()) == {1, 2}
