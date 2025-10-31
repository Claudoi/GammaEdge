from __future__ import annotations

from datetime import datetime as dt

import plotly.graph_objects as go
import polars as pl

from portfolio.backtest.reporting import ReportFigure, build_context, render_html


def test_render_html_with_tables_and_brinson():
    # --- Tablas mínimas (asset/group) ---
    df_asset_cum = pl.DataFrame({"ticker": ["A", "B"], "contrib_total": [0.12, -0.03]})
    df_group_total = pl.DataFrame(
        {"group": ["G1", "G2"], "contrib_total": [0.10, 0.02], "weight_avg": [0.55, 0.45]}
    )

    # --- Brinson timeseries mínima ---
    df_brinson = pl.DataFrame(
        {
            "date": [dt(2024, 1, 1), dt(2024, 1, 2)],
            "alloc": [0.01, 0.02],
            "select": [0.00, 0.01],
            "interact": [0.00, 0.00],
            "total": [0.01, 0.03],
        }
    )

    # --- Métricas adicionales para cubrir la tabla de métricas ---
    extra_metrics = {"Sharpe": 1.05, "MaxDD": -0.12, "Benchmark Scheme": "Equal-Weight"}

    # --- Contexto completo (activa rutas de render de tablas + brinson) ---
    ctx = build_context(
        portfolio_name="Portfolio X",
        period=("2024-01-01", "2024-01-02"),
        df_asset_cum=df_asset_cum,
        df_group_total=df_group_total,
        df_brinson=df_brinson,
        extra_metrics=extra_metrics,
    )

    # --- Al menos una figura para cubrir el bloque de figures ---
    fig = go.Figure(data=[go.Bar(x=["A", "B"], y=[0.12, -0.03])])
    html = render_html(ctx, [ReportFigure("Top Contributors", fig)], page_title="Rep", h1="H1")

    # Validaciones básicas
    assert isinstance(html, (bytes, bytearray))
    assert b"Portfolio X" in html
    assert b"Top Contributors" in html
    # Algo del contenido tabular aparece renderizado
    assert b"G1" in html or b"G2" in html
    assert b"Brinson" in html or b"alloc" in html or b"select" in html
