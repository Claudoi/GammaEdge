# tests/unit/test_reporting_render_html.py
from __future__ import annotations

import plotly.graph_objects as go

from portfolio.backtest.reporting import ReportFigure, build_context, render_html


def test_render_html_minimal_smoke():
    # Contexto mínimo
    ctx = build_context(
        portfolio_name="Smoke",
        period=("2024-01-01", "2024-01-02"),
        df_asset_cum=None,
        df_group_total=None,
        df_brinson=None,
        extra_metrics={"Sharpe": 1.0},
    )

    # Figura sencilla (cubre la ruta de figures en el render)
    fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[1.0, 1.1])])

    html = render_html(ctx, [ReportFigure("Equity", fig)], page_title="Test", h1="Report")
    assert isinstance(html, (bytes, bytearray))
    # Comprobar que mete partes del contexto/título en el HTML
    assert b"Smoke" in html
    assert b"Equity" in html
