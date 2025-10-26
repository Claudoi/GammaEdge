import numpy as np
import polars as pl

from portfolio.backtest.reporting import build_backtest_report, render_html_report


def test_build_backtest_report(df_sample):
    equity = np.linspace(1.0, 1.1, df_sample.height)
    weights = np.full((df_sample.height, len(df_sample.columns) - 1), 0.5)
    report = build_backtest_report(df_sample, weights, equity, title="Test Report")
    assert "equity" in report.figures
    assert "contrib_asset_daily" in report.tables


def test_render_html_report(df_sample):
    bt = {
        "dates": df_sample["date"].to_list(),
        "equity": np.linspace(1.0, 1.05, df_sample.height),
        "tickers": ["A", "B"],
        "weights": np.ones((df_sample.height, 2)) * 0.5,
    }
    metrics = pl.DataFrame({"CAGR": [0.1], "Sharpe": [1.2]})
    html = render_html_report(bt, metrics)
    assert "<html>" in html
    assert "Equity" in html
