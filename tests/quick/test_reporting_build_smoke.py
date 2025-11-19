# tests/quick/test_scenarios_bootstrap_and_slice.py

import numpy as np
import polars as pl

from portfolio.backtest.reporting import (
    BacktestReport,
    ReportFigure,
    build_backtest_report,
    build_context,
    render_html,
)


def _make_synthetic_backtest_panel(
    n_days: int = 40,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, dict[str, str]]:
    """Build a small synthetic panel compatible with the reporting helpers."""
    rng = np.random.default_rng(123)

    dates = pl.datetime_range(
        start=pl.datetime(2020, 1, 1),
        end=pl.datetime(2020, 1, 1) + pl.duration(days=n_days - 1),
        interval="1d",
        eager=True,
    )

    # Two assets with slightly different drifts and volatilities
    r_a = rng.normal(loc=0.0005, scale=0.01, size=n_days)
    r_b = rng.normal(loc=0.0002, scale=0.012, size=n_days)

    df = pl.DataFrame(
        {
            "date": dates,
            "AAA": r_a,
            "BBB": r_b,
        }
    )

    # Equal-weight daily weights (T x N)
    n_assets = 2
    daily_weights = np.full((n_days, n_assets), 1.0 / n_assets, dtype=float)

    # Build equity path from equal-weight portfolio returns
    rp = (r_a + r_b) / 2.0
    equity = np.cumprod(1.0 + rp).astype(float)

    group_map = {"AAA": "Group_A", "BBB": "Group_B"}

    return df, daily_weights, equity, group_map


def test_build_backtest_report_and_render_html_smoke():
    """Minimal smoke test for BacktestReport + build_context + render_html."""
    df_ret_wide, daily_weights, equity, group_map = _make_synthetic_backtest_panel()

    report = build_backtest_report(
        df_ret_wide=df_ret_wide,
        daily_weights=daily_weights,
        equity=equity,
        group_map=group_map,
        title="Test Backtest",
    )

    # Basic structural checks on BacktestReport
    assert isinstance(report, BacktestReport)
    assert "equity" in report.figures
    assert "drawdown" in report.figures
    assert "weights" in report.figures
    assert "top_contrib" in report.figures
    assert isinstance(report.tables, dict)
    assert report.tables  # should not be empty

    df_asset_total = report.tables.get("contrib_asset_total")
    df_group_total = report.tables.get("contrib_group_total")

    period_start = str(df_ret_wide["date"][0])
    period_end = str(df_ret_wide["date"][-1])

    ctx = build_context(
        portfolio_name="TestPortfolio",
        period=(period_start, period_end),
        df_asset_cum=df_asset_total,
        df_group_total=df_group_total,
        df_brinson=None,
        extra_metrics={"CAGR": 0.10, "Sharpe": 1.5},
    )

    # Context should at least keep the portfolio name and period fields
    assert ctx["portfolio_name"] == "TestPortfolio"
    assert "period_start" in ctx
    assert "period_end" in ctx
    assert str(ctx["period_start"]).startswith("2020-")
    assert str(ctx["period_end"]).startswith("2020-")

    # Build a small set of figures for HTML rendering
    figures = [
        ReportFigure("Equity", report.figures["equity"]),
        ReportFigure("Drawdown", report.figures["drawdown"]),
        ReportFigure("Weights", report.figures["weights"]),
    ]

    html_bytes = render_html(
        ctx,
        figures,
        page_title="GammaEdge Test Report",
        h1="Backtest Report",
    )

    assert isinstance(html_bytes, (bytes, bytearray))
    lower = html_bytes.lower()
    assert b"<html" in lower
    assert b"backtest report" in lower
    assert b"testportfolio" in lower
