from __future__ import annotations

from portfolio.backtest.reporting import build_context, render_html


def test_render_html_metrics_dense_table():
    metrics = {f"m{i}": float(i) for i in range(15)}  # fuerza tabla de métricas con varias filas
    ctx = build_context(
        portfolio_name="M",
        period=("2024-01-01", "2024-01-01"),
        df_asset_cum=None,
        df_group_total=None,
        df_brinson=None,
        extra_metrics=metrics,
    )
    html = render_html(ctx, [], page_title="M", h1="M")
    assert b"m10" in html
