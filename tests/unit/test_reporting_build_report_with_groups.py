from __future__ import annotations

from datetime import datetime as dt

import numpy as np
import polars as pl

from portfolio.backtest.reporting import build_backtest_report


def test_build_report_with_groupmap_generates_tables_and_figures():
    # --- Datos mínimos coherentes ---
    dates = [dt(2024, 1, d) for d in (1, 2, 3)]

    # returns wide
    df_ret = pl.DataFrame(
        {
            "date": dates,
            "A": [0.00, 0.01, 0.00],
            "B": [0.00, -0.005, 0.001],
            "C": [0.002, 0.000, -0.001],
        }
    )
    # equity con pequeña variación
    equity = np.array([1.00, 1.008, 1.006], dtype=float)
    # pesos diarios (3x3) válidos
    Wd = np.array(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [0.4, 0.3, 0.3],
            [0.5, 0.25, 0.25],
        ],
        dtype=float,
    )
    # group map dispara rutas de tablas por grupo en reporting.py
    group_map = {"A": "G1", "B": "G1", "C": "G2"}

    report = build_backtest_report(
        df_ret_wide=df_ret,
        daily_weights=Wd,
        equity=equity,
        group_map=group_map,
        title="GammaEdge Report Smoke",
    )

    # --- Figuras esperadas ---
    assert "equity" in report.figures
    assert "drawdown" in report.figures
    assert "weights" in report.figures
    assert "top_contrib" in report.figures

    # --- Tablas esperadas (activos y grupos) ---
    # La presencia de estas tablas cubre ramas de construcción de tablas en reporting.py
    assert "contrib_asset_total" in report.tables
    assert "contrib_group_total" in report.tables

    df_asset_total = report.tables["contrib_asset_total"]
    df_group_total = report.tables["contrib_group_total"]

    # Tipos y contenido básico
    assert isinstance(df_asset_total, pl.DataFrame)
    assert isinstance(df_group_total, pl.DataFrame)
    assert df_asset_total.height > 0
    assert df_group_total.height > 0

    # Columnas típicas (no exigimos todas, sólo que estén las principales)
    assert "ticker" in df_asset_total.columns or "asset" in df_asset_total.columns
    assert "contrib_total" in df_asset_total.columns

    assert "group" in df_group_total.columns
    assert "contrib_total" in df_group_total.columns
    # que haya más de un grupo (G1/G2)
    assert df_group_total.select("group").unique().height >= 2
