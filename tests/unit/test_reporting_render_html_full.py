from __future__ import annotations

from datetime import datetime as dt

import numpy as np
import plotly.graph_objects as go
import polars as pl

from portfolio.backtest.reporting import (
    ReportFigure,
    build_backtest_report,
    build_context,
    render_html,
)


def test_render_html_with_tables_and_brinson_and_figs():
    # --- Datos mínimos coherentes ---
    dates = [dt(2024, 1, d) for d in (1, 2, 3)]
    df_ret = pl.DataFrame(
        {
            "date": dates,
            "A": [0.00, 0.01, 0.00],
            "B": [0.00, -0.005, 0.001],
        }
    )
    equity = np.array([1.00, 1.008, 1.006], dtype=float)
    Wd = np.array(
        [
            [0.5, 0.5],
            [0.6, 0.4],
            [0.55, 0.45],
        ],
        dtype=float,
    )
    group_map = {"A": "G1", "B": "G2"}

    report = build_backtest_report(
        df_ret_wide=df_ret,
        daily_weights=Wd,
        equity=equity,
        group_map=group_map,
        title="GammaEdge Report Full",
    )

    # df_brinson mínimo (cumulative)
    df_brinson = pl.DataFrame(
        {
            "date": dates,
            "alloc": [0.0, 0.01, 0.015],
            "select": [0.0, -0.002, -0.001],
            "interact": [0.0, 0.0, 0.0],
            "total": [0.0, 0.008, 0.014],
        }
    )

    # Contexto con tablas y métricas
    ctx = build_context(
        portfolio_name="Test Full",
        period=("2024-01-01", "2024-01-03"),
        df_asset_cum=report.tables.get("contrib_asset_total"),
        df_group_total=report.tables.get("contrib_group_total"),
        df_brinson=df_brinson,
        extra_metrics={"Sharpe": 1.23, "MaxDD": -0.05},
    )

    # Al menos los campos básicos deben existir
    assert ctx.get("portfolio_name") == "Test Full"
    assert ctx.get("period_start") == "2024-01-01"
    assert ctx.get("period_end") == "2024-01-03"
    assert isinstance(ctx.get("metrics"), dict)

    # Figuras: mezcla de las generadas y una dummy Plotly para cubrir la rama de serialización
    figs = [
        ReportFigure("Equity", report.figures["equity"]),
        ReportFigure("Drawdown", report.figures["drawdown"]),
        ReportFigure("Weights", report.figures["weights"]),
        ReportFigure("Top Contributors", report.figures["top_contrib"]),
        ReportFigure("Dummy", go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1])])),  # extra path
    ]

    html_bytes = render_html(ctx, figs, page_title="GammaEdge Report", h1="Backtest Report")
    assert isinstance(html_bytes, (bytes, bytearray))
    # contenido mínimo esperado
    assert b"GammaEdge Report" in html_bytes or b"Backtest Report" in html_bytes
