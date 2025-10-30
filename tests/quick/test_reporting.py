import importlib

import plotly.graph_objects as go
import polars as pl

from portfolio.backtest.reporting import ReportFigure, build_context, render_html, render_pdf


def _fig():
    return go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 2, 1]))


def test_render_html_smoke():
    ctx = build_context(
        portfolio_name="Demo",
        period=("2020-01-01", "2020-12-31"),
        df_asset_cum=pl.DataFrame({"ticker": ["A"], "contrib_total": [0.1]}),
        df_group_total=pl.DataFrame({"group": ["G"], "contrib_total": [0.1], "avg_weight": [0.5]}),
        df_brinson=pl.DataFrame(
            {
                "date": ["2020-12-31"],
                "alloc": [0.01],
                "select": [0.02],
                "interact": [0.0],
                "total": [0.03],
            }
        ),
        extra_metrics={"Sharpe": 1.23},
    )
    figs = [
        ReportFigure("Equity", _fig(), width=600, height=400),
        ReportFigure("Drawdown", _fig(), width=600, height=400),
    ]
    html_bytes = render_html(ctx, figs, page_title="X", h1="Y")
    assert isinstance(html_bytes, bytes)
    s = html_bytes.decode("utf-8")
    assert "<html" in s and "Figures" in s


def test_render_pdf_if_available():
    rl = importlib.util.find_spec("reportlab")
    if rl is None:
        return
    ctx = build_context(
        portfolio_name="Demo",
        period=("2020-01-01", "2020-12-31"),
        df_asset_cum=None,
        df_group_total=None,
        df_brinson=None,
        extra_metrics=None,
    )
    figs = [ReportFigure("Equity", _fig(), width=600, height=400)]
    pdf = render_pdf(ctx, figs, title="Demo")
    assert isinstance(pdf, bytes)
    # PDF starts with %PDF
    assert pdf[:4] == b"%PDF"
