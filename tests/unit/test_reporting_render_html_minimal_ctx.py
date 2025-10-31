from __future__ import annotations

from portfolio.backtest.reporting import build_context, render_html


def test_render_html_minimal_context_no_tables_no_figs():
    # Contexto mínimo (sin tablas ni brinson ni figuras) para cubrir ramas de fallback
    ctx = build_context(
        portfolio_name="Minimal",
        period=("2024-01-01", "2024-01-01"),
        df_asset_cum=None,
        df_group_total=None,
        df_brinson=None,
        extra_metrics={"Sharpe": 0.0},
    )
    html = render_html(ctx, figures=[], page_title="Minimal Report", h1="Minimal H1")
    assert isinstance(html, (bytes, bytearray))
    assert b"Minimal Report" in html or b"Minimal H1" in html
