# tests/attribution/test_brinson.py
import polars as pl

from portfolio.attribution import brinson as attr_brinson
from portfolio.backtest import brinson_utils as bu


def _wide_example() -> pl.DataFrame:
    # Mismo esquema de antes: 2 fechas × 2 grupos en formato ancho
    return pl.DataFrame(
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


def test_compute_brinson_attribution_matches_utils_timeseries():
    df = _wide_example()

    # Motor nuevo
    result = attr_brinson.compute_brinson_attribution(df)
    ts = result.timeseries

    # Motor "bajo nivel"
    ts_ref = bu.coerce_brinson_timeseries_to_long(df)

    # Comparación de columnas y shape
    assert ts.schema == ts_ref.schema
    assert ts.height == ts_ref.height

    # Comparación de métricas (ordenando por date/group_id por si acaso)
    ts_sorted = ts.sort(["date", "group_id"])
    ts_ref_sorted = ts_ref.sort(["date", "group_id"])
    for col in ["alloc", "select", "interact", "total"]:
        assert ts_sorted[col].to_list() == ts_ref_sorted[col].to_list()


def test_compute_brinson_attribution_group_and_total_aggregates():
    df = _wide_example()
    result = attr_brinson.compute_brinson_attribution(df, how="sum")

    by_group = result.by_group.sort("group_id")
    total = result.total

    # De la construcción de _wide_example sabemos los sums:
    # alloc_0: 0.1 + 0.2 = 0.3
    # alloc_1: 0.3 + 0.4 = 0.7
    g0 = by_group.row(0, named=True)
    g1 = by_group.row(1, named=True)

    assert g0["group_id"] == 0
    assert g1["group_id"] == 1

    assert g0["alloc"] == 0.1 + 0.2
    assert g1["alloc"] == 0.3 + 0.4

    # El total debe ser igual a la suma por grupo
    total_alloc = total["alloc"][0]
    assert total_alloc == g0["alloc"] + g1["alloc"]
