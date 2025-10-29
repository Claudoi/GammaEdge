import polars as pl

from portfolio.backtest.brinson_utils import (
    coerce_brinson_timeseries_to_long as coerce_long,
)
from portfolio.backtest.brinson_utils import (
    ensure_datetime,
)


def _dates(n: int) -> pl.Series:
    # Devuelve un Series[Datetime] con n días desde 2024-01-01
    return pl.datetime_range(
        start=pl.datetime(2024, 1, 1),
        end=pl.datetime(2024, 1, 1, 0, 0) + pl.duration(days=n - 1),
        interval="1d",
        eager=True,
    )


def _numeric_sum_dict(df: pl.DataFrame) -> dict:
    # Suma solo columnas numéricas para evitar sumar Datetime
    num_df = df.select(pl.all().exclude([pl.Datetime, pl.Date, pl.Time, pl.Duration]))
    summed = num_df.select(pl.all().sum())
    # Devuelve primitivos Python: evita Series en el dict
    try:
        # Polars >= 0.20 tiene row(named=True)
        return summed.row(0, named=True)
    except Exception:
        # Fallback universal
        return summed.to_dicts()[0]


def test_long_with_group_id_ok():
    n = 5
    df = pl.DataFrame(
        {
            "date": _dates(n),
            "group_id": [0] * n,
            "alloc": [0.1] * n,
            "select": [0.2] * n,
            "interact": [0.0] * n,
            "total": [0.3] * n,
        }
    )
    out = coerce_long(df)
    out = ensure_datetime(out, "date")
    assert out.columns == ["date", "group_id", "alloc", "select", "interact", "total"]
    assert out.shape == (n, 6)


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
    # columnas esperadas
    assert out.columns == ["date", "group_id", "alloc", "select", "interact", "total"]
    # filas: 3 fechas * 2 grupos = 6
    assert out.shape == (6, 6)
    # group_id ∈ {0,1}
    gids = set(out.get_column("group_id").to_list())
    assert gids == {0, 1}


def test_global_only_injects_group0():
    df = pl.DataFrame(
        {
            "date": _dates(4),
            "alloc": [0.0] * 4,
            "select": [0.0] * 4,
            "interact": [0.0] * 4,
            "total": [0.0] * 4,
        }
    )
    out = coerce_long(df)
    assert out.columns == ["date", "group_id", "alloc", "select", "interact", "total"]
    assert set(out.get_column("group_id").unique().to_list()) == {0}


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
    # Compara sumas solo sobre numéricas
    assert _numeric_sum_dict(out1) == _numeric_sum_dict(out2)
