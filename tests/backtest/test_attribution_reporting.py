# tests/backtest/test_attribution_reporting.py

import polars as pl

from portfolio.attribution import brinson as attr_brinson
from portfolio.backtest.attribution_reporting import (
    BrinsonReport,
    build_brinson_attribution_report,
)


def _wide_example() -> pl.DataFrame:
    # Mismo esquema que en los tests de attribution Brinson:
    # 2 fechas × 2 grupos en formato ancho.
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


def test_build_brinson_attribution_report_polars_matches_direct():
    df = _wide_example()

    direct = attr_brinson.compute_brinson_attribution(df)
    report: BrinsonReport = build_brinson_attribution_report(df, how="sum")

    # Misma estructura y tamaños
    assert set(report.keys()) == {"timeseries", "by_group", "total"}

    ts = report["timeseries"]
    by_group = report["by_group"]
    total = report["total"]

    # Timeseries
    assert ts.schema == direct.timeseries.schema
    assert ts.height == direct.timeseries.height

    # By group
    assert by_group.schema == direct.by_group.schema
    assert by_group.height == direct.by_group.height

    # Total
    assert total.schema == direct.total.schema
    assert total.height == direct.total.height


def test_build_brinson_attribution_report_accepts_pandas():
    df_pl = _wide_example()
    df_pd = df_pl.to_pandas()

    report_pd = build_brinson_attribution_report(df_pd, how="sum")
    report_pl = build_brinson_attribution_report(df_pl, how="sum")

    # Comprobamos una métrica simple (alloc total) que debe coincidir
    alloc_total_pd = report_pd["total"]["alloc"][0]
    alloc_total_pl = report_pl["total"]["alloc"][0]

    assert alloc_total_pd == alloc_total_pl
