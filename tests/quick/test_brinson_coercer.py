from datetime import datetime, timedelta

import polars as pl
import pytest

from portfolio.backtest.brinson_utils import (
    coerce_brinson_timeseries_to_long as coerce_long,
)
from portfolio.backtest.brinson_utils import (
    ensure_datetime,
)

# ---------- helpers ----------


def _dates(n: int = 5):
    base = datetime(2024, 1, 1)
    return [base + timedelta(days=i) for i in range(n)]


def _assert_contract(df: pl.DataFrame) -> None:
    req = {"date", "group_id", "alloc", "select", "interact", "total"}
    assert req.issubset(set(df.columns))
    assert df.schema.get("date") == pl.Datetime
    assert df.schema.get("group_id") in (pl.Int64, pl.Int32)
    for m in ("alloc", "select", "interact", "total"):
        assert df.schema.get(m) in (pl.Float64, pl.Float32)


# ---------- tests ----------


def test_long_with_group_id_ok():
    n = 5
    df = pl.DataFrame(
        {
            "date": _dates(n),  # <-- Python datetimes, no Expr
            "group_id": [0] * n,
            "alloc": [0.1] * n,
            "select": [0.2] * n,
            "interact": [0.0] * n,
            "total": [0.3] * n,
        }
    )
    out = coerce_long(df)
    out = ensure_datetime(out, "date")
    _assert_contract(out)
    assert out.height == n
    assert out.select(pl.col("group_id").unique()).height == 1


def test_long_with_group_labels_ok():
    n = 4
    df = pl.DataFrame(
        {
            "date": _dates(n),
            "group": ["Energy"] * n,  # string labels
            "alloc": [0.0] * n,
            "select": [0.0] * n,
            "interact": [0.0] * n,
            "total": [0.0] * n,
        }
    )
    out = coerce_long(df)
    out = ensure_datetime(out, "date")
    _assert_contract(out)
    gids = sorted(out.select(pl.col("group_id").unique()).to_series().to_list())
    assert gids == [0]


def test_wide_style_metric_suffix_ok():
    dates = _dates(3)
    df = pl.DataFrame(
        {
            "date": dates,
            "alloc_0": [0.01, 0.02, 0.03],
            "select_0": [0.0, 0.0, 0.0],
            "interact_0": [0.0, 0.0, 0.0],
            "total_0": [0.01, 0.02, 0.03],
            "alloc_1": [0.04, 0.05, 0.06],
            "select_1": [0.0, 0.0, 0.0],
            "interact_1": [0.0, 0.0, 0.0],
            "total_1": [0.04, 0.05, 0.06],
        }
    )
    out = coerce_long(df)
    out = ensure_datetime(out, "date")
    _assert_contract(out)
    assert out.height == 6
    gids = sorted(out.select(pl.col("group_id").unique()).to_series().to_list())
    assert gids == [0, 1]


def test_global_only_injects_group_zero():
    dates = _dates(4)
    df = pl.DataFrame(
        {
            "date": dates,
            "alloc": [0.0] * 4,
            "select": [0.0] * 4,
            "interact": [0.0] * 4,
            "total": [0.0] * 4,
        }
    )
    out = coerce_long(df)
    out = ensure_datetime(out, "date")
    _assert_contract(out)
    assert out.height == 4
    gids = out.select(pl.col("group_id").unique()).to_series().to_list()
    assert sorted(gids) == [0]


def test_idempotent_on_already_long():
    n = 3
    df = pl.DataFrame(
        {
            "date": _dates(n),
            "group_id": [1] * n,
            "alloc": [0.0] * n,
            "select": [0.1] * n,
            "interact": [0.0] * n,
            "total": [0.1] * n,
        }
    )
    out1 = coerce_long(df)
    out2 = coerce_long(out1)

    assert out1.columns == out2.columns
    assert out1.shape == out2.shape

    # Suma solo columnas numéricas (evita sumar 'date')
    num_cols = [
        c for c, t in out1.schema.items() if t in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    ]
    s1 = out1.select([pl.col(c).sum().alias(c) for c in num_cols]).to_dict(as_series=False)
    s2 = out2.select([pl.col(c).sum().alias(c) for c in num_cols]).to_dict(as_series=False)
    assert s1 == s2


def test_does_not_mutate_input_df():
    df = pl.DataFrame(
        {
            "date": _dates(2),
            "alloc_0": [0.0, 0.1],
            "select_0": [0.0, 0.0],
            "interact_0": [0.0, 0.0],
            "total_0": [0.0, 0.1],
        }
    )
    before_cols = list(df.columns)
    _ = coerce_long(df)
    assert df.columns == before_cols


def test_raises_on_unsupported_format():
    df = pl.DataFrame({"date": _dates(2), "foo": [1, 2], "bar": [3, 4]})
    with pytest.raises(ValueError):
        _ = coerce_long(df)
