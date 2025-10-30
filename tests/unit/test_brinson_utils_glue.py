# tests/unit/test_brinson_utils_glue.py
from __future__ import annotations

from datetime import datetime

import polars as pl

from portfolio.backtest.brinson_utils import (
    coerce_brinson_timeseries_to_long as _to_long,
)
from portfolio.backtest.brinson_utils import (
    ensure_datetime as _ensure_datetime,
)


def test_timeseries_to_long_shapes():
    # Build a minimal Brinson timeseries with a single group_id
    df = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "group_id": [1, 1],
            "alloc": [0.10, 0.20],
            "select": [0.00, 0.10],
            "interact": [0.00, 0.00],
            "total": [0.10, 0.30],
        }
    )
    df = _ensure_datetime(df, "date")

    long = _to_long(df)

    # Basic shape/content checks
    assert {"date", "group_id", "alloc", "select", "interact", "total"} <= set(long.columns)
    assert long.height == df.height
    assert long.select("group_id").unique().height >= 1

    # Numeric columns should be floats (Float32 or Float64)
    for col in ("alloc", "select", "interact", "total"):
        assert long.schema[col] in (pl.Float32, pl.Float64)
