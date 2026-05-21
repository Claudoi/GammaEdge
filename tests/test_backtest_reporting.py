"""Tests for portfolio/backtest/reporting — basic smoke + structure tests.

This module currently has 0% coverage. These tests provide a baseline that
catches import errors, signature drift, and basic functional regressions.

Note on circular dependency:
    `portfolio.backtest.reporting` imports from `portfolio.backtest.attribution`
    at module top-level (line ~20), but inside `export_backtest_report` (line ~919)
    there's a LOCAL import:

        from portfolio.backtest import attribution as bt_attr  # local import to avoid cycles

    This hack is preserved here intentionally — we test around it.
"""

from __future__ import annotations

import inspect
from datetime import date

import numpy as np
import polars as pl

from portfolio.backtest import reporting
from portfolio.backtest.attribution import expand_rebalance_weights
from portfolio.backtest.engine import backtest_vectorized


def _equal_weight_allocator(df: pl.DataFrame) -> np.ndarray:
    n = len([c for c in df.columns if c != "date"])
    return np.full(n, 1.0 / n)


def test_backtest_reporting_module_imports_and_exposes_public_api():
    """Smoke: module imports and exposes the expected public surface.

    Catches accidental signature drift and ensures the
    `from .attribution import (...)` top-level import keeps working
    despite the known circular-dependency hack documented in the module
    docstring.
    """
    # Core public functions used by the Streamlit UI / export CLI
    expected = [
        "build_backtest_report",
        "build_brinson_attribution_section",
        "build_context",
        "render_html",
        "render_pdf",
        "export_backtest_report",
    ]
    for fn_name in expected:
        assert hasattr(reporting, fn_name), f"missing public symbol: {fn_name}"
        fn = getattr(reporting, fn_name)
        assert callable(fn), f"{fn_name} is not callable"
        # signature must be introspectable (catches malformed defs)
        sig = inspect.signature(fn)
        assert sig is not None

    # Dataclass container used by the UI
    assert hasattr(reporting, "BacktestReport")
    assert inspect.isclass(reporting.BacktestReport)


def test_build_backtest_report_with_minimal_inputs():
    """End-to-end: run tiny backtest → expand weights → build report.

    Exercises `build_backtest_report` which is the main entry point used
    by the Streamlit backtest page. Validates that:
      - tables/figures/meta are populated
      - attribution tables have expected columns
      - meta reflects the input shape
    """
    # 1) Synthetic returns (~2 years, 3 tickers) — deterministic seed
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2021, 12, 31), interval="1d", eager=True)
    n = len(dates)
    rng = np.random.default_rng(0)
    df_ret = pl.DataFrame(
        {
            "date": dates,
            "A": rng.normal(0.001, 0.01, n),
            "B": rng.normal(0.0005, 0.008, n),
            "C": rng.normal(0.0007, 0.009, n),
        }
    )

    # 2) Run vectorized backtest to obtain sparse (rebalance-date) weights
    bt = backtest_vectorized(
        df_ret,
        lookback=120,
        rebalance_freq="1mo",
        allocator=_equal_weight_allocator,
    )
    assert "error" not in bt, f"Backtest setup failed: {bt.get('error')}"

    # 3) Align inputs for the report:
    #    - df_ret_wide must cover the same date span as the equity curve
    #    - daily_weights must be (T, N) — expand sparse rebalance weights
    bt_dates = list(bt["dates"])
    df_ret_bt = df_ret.filter(pl.col("date").is_in(bt_dates)).sort("date")

    daily_W = expand_rebalance_weights(
        dates=df_ret_bt.get_column("date").to_list(),
        rb_dates=list(bt["rebalance_dates"]),
        W_reb=np.asarray(bt["weights"]),
    )

    equity = np.asarray(bt["equity"], dtype=float)

    # 4) Build the report
    report = reporting.build_backtest_report(
        df_ret_wide=df_ret_bt,
        daily_weights=daily_W,
        equity=equity,
        title="Test Report",
    )

    # Structure checks
    assert isinstance(report, reporting.BacktestReport)
    assert "contrib_asset_daily" in report.tables
    assert "contrib_asset_total" in report.tables
    assert "top_contributors" in report.tables

    contrib_total = report.tables["contrib_asset_total"]
    assert set(contrib_total.columns) >= {"ticker", "contrib_total"}
    assert contrib_total.height == 3  # one row per ticker

    # Figures dict has the canonical chart keys
    for key in ("equity", "drawdown", "weights", "top_contrib"):
        assert key in report.figures, f"missing figure: {key}"

    # Meta dict reflects shape
    assert report.meta["n_assets"] == 3
    assert report.meta["n_days"] == len(equity)
    assert report.meta["title"] == "Test Report"


def test_build_backtest_report_with_group_map_adds_group_tables():
    """When `group_map` is supplied, group-level attribution tables appear.

    Exercises the optional grouping branch of `build_backtest_report`.
    """
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2021, 6, 30), interval="1d", eager=True)
    n = len(dates)
    rng = np.random.default_rng(7)
    df_ret = pl.DataFrame(
        {
            "date": dates,
            "A": rng.normal(0.001, 0.01, n),
            "B": rng.normal(0.0005, 0.008, n),
            "C": rng.normal(0.0007, 0.009, n),
            "D": rng.normal(0.0004, 0.012, n),
        }
    )

    bt = backtest_vectorized(
        df_ret,
        lookback=90,
        rebalance_freq="1mo",
        allocator=_equal_weight_allocator,
    )
    assert "error" not in bt

    bt_dates = list(bt["dates"])
    df_ret_bt = df_ret.filter(pl.col("date").is_in(bt_dates)).sort("date")
    daily_W = expand_rebalance_weights(
        dates=df_ret_bt.get_column("date").to_list(),
        rb_dates=list(bt["rebalance_dates"]),
        W_reb=np.asarray(bt["weights"]),
    )
    equity = np.asarray(bt["equity"], dtype=float)

    group_map = {"A": "G1", "B": "G1", "C": "G2", "D": "G2"}
    report = reporting.build_backtest_report(
        df_ret_wide=df_ret_bt,
        daily_weights=daily_W,
        equity=equity,
        group_map=group_map,
        title="Grouped",
    )

    # Group-level tables should now exist
    assert "contrib_group_daily" in report.tables
    assert "contrib_group_total" in report.tables

    group_total = report.tables["contrib_group_total"]
    assert set(group_total.columns) >= {"group", "contrib_total", "avg_weight"}
    # Both groups represented
    groups_present = set(group_total.get_column("group").to_list())
    assert groups_present == {"G1", "G2"}
