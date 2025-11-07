import pandas as pd
import pytest

from portfolio.backtest.attribution_reporting import build_brinson_attribution_report


def _make_global_only_example() -> pd.DataFrame:
    """
    Pequeño ejemplo end-to-end en formato 'global-only':
    - índice = fechas (DatetimeIndex)
    - columnas = métricas Brinson (sin group_id)
    """
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    data = {
        "alloc": [0.1, 0.2, 0.3],
        "select": [0.0, 0.1, -0.1],
        "interact": [0.0, 0.0, 0.0],
        "total": [0.1, 0.3, 0.2],
    }
    return pd.DataFrame(data, index=idx)


def test_brinson_attribution_report_end_to_end_global_only():
    df = _make_global_only_example()

    report = build_brinson_attribution_report(df, how="sum")

    ts = report["timeseries"]
    by_group = report["by_group"]
    total = report["total"]

    # 1) timeseries debe estar en formato largo estándar
    assert set(ts.columns) == {"date", "group_id", "alloc", "select", "interact", "total"}
    assert ts.height == 3

    # 2) Solo un group_id (0) en este ejemplo global-only
    gids = sorted(ts["group_id"].unique().to_list())
    assert gids == [0]

    # 3) by_group debe tener una fila por grupo
    assert by_group.height == 1
    row = by_group.row(0, named=True)
    assert row["group_id"] == 0

    # 4) La métrica total agregada debe coincidir con la suma de la timeseries
    total_from_df = float(df["total"].sum())
    assert pytest.approx(row["total"], rel=1e-12) == total_from_df

    total_global = float(total["total"][0])
    assert pytest.approx(total_global, rel=1e-12) == total_from_df
