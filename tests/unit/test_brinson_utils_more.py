# tests/unit/test_brinson_utils_more.py
from __future__ import annotations

from datetime import datetime

import polars as pl

from portfolio.backtest.brinson_utils import (
    coerce_brinson_timeseries_to_long,
    ensure_datetime,
)


def test_ensure_datetime_casts_strings_and_datetimes():
    df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "alloc": [0.0, 0.1, 0.2],
        }
    )
    out = ensure_datetime(df, "date")
    assert out.schema["date"] == pl.Datetime

    df2 = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 4), datetime(2024, 1, 5)],
            "alloc": [0.0, 0.1],
        }
    )
    out2 = ensure_datetime(df2, "date")
    assert out2.schema["date"] == pl.Datetime


def test_coerce_brinson_timeseries_to_long_basic_shapes():
    # timeseries estilo Brinson con un group_id
    df = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "group_id": [1, 1],
            "alloc": [0.10, 0.20],
            "select": [0.00, 0.05],
            "interact": [0.00, 0.01],
            "total": [0.10, 0.26],
        }
    )
    df = ensure_datetime(df, "date")
    long = coerce_brinson_timeseries_to_long(df)

    # columnas clave presentes
    expected = {"date", "group_id", "alloc", "select", "interact", "total"}
    assert expected.issubset(set(long.columns))

    # misma altura que la fuente y tipos numéricos razonables
    assert long.height == df.height
    for c in ("alloc", "select", "interact", "total"):
        assert long.schema[c] in (pl.Float32, pl.Float64)
