import numpy as np
import polars as pl

from portfolio.backtest.reporting import render_html_report


def test_render_html_report_contains_sections():
    dates = pl.datetime_range(
        start=pl.datetime(2024, 1, 1), end=pl.datetime(2024, 1, 5), interval="1d", eager=True
    ).to_list()
    bt = {
        "dates": dates,
        "equity": np.array([1.0, 1.01, 1.00, 1.02, 1.02], dtype=float),
        "tickers": ["A", "B"],
        "weights": np.array([[0.5, 0.5], [0.6, 0.4], [0.4, 0.6]], dtype=float),
        "rebalance_dates": dates[:3],
    }
    df_metrics = pl.DataFrame({"Metric": ["CAGR", "Sharpe"], "Value": [0.10, 1.0]})
    html = render_html_report(bt, df_metrics)
    assert isinstance(html, str) and len(html) > 0
    assert "<h2>Metrics</h2>" in html
    assert "<h2>Equity</h2>" in html
    assert "<h2>Drawdown</h2>" in html
    assert "<h2>Weights</h2>" in html
