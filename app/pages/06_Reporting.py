# app/pages/06_Reporting.py
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
import streamlit as st

# ---------------------------------------------------------------------
# Repo root for local imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from portfolio.backtest.reporting import (  # noqa: E402
    BacktestReport,
    ReportFigure,
    build_backtest_report,
    build_context,
    render_html,
    render_pdf,
)

# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------
st.set_page_config(page_title="Reporting", layout="wide")
st.title("📄 Reporting")
st.caption("Generate HTML and PDF reports with equity, drawdown, weights and attribution tables.")

# ---------------------------------------------------------------------
# Inputs from previous pages
# Expectation: 04/05 stored these keys in session_state
#   - bt: dict with dates, equity, tickers, weights, optionally rebalance_dates
#   - df_ret_wide: Polars DataFrame ['date', tickers...]
#   - group_map: optional dict ticker -> group
#   - metrics_df: optional Polars/Pandas table with KPIs
# ---------------------------------------------------------------------
bt = st.session_state.get("bt")
df_ret_wide = st.session_state.get("df_ret_wide", st.session_state.get("returns_wide"))
group_map = st.session_state.get("group_map")
metrics_df = st.session_state.get("metrics_df")

if bt is None or df_ret_wide is None:
    st.warning(
        "⚠️ Run pages 02–05 first so we can build the report (risk model, backtest, attribution)."
    )
    st.stop()

# Normalize inputs
if not isinstance(df_ret_wide, pl.DataFrame):
    df_ret_wide = pl.from_pandas(df_ret_wide)

dates = bt.get("dates", [])
equity = np.asarray(bt.get("equity", []), dtype=float)
tickers = bt.get("tickers", [])
weights = np.asarray(bt.get("weights", []), dtype=float)

if len(dates) == 0 or equity.size == 0 or weights.size == 0 or len(tickers) == 0:
    st.error("Missing essentials in 'bt' (dates, equity, weights, tickers).")
    st.stop()

# ---------------------------------------------------------------------
# Build unified BacktestReport (tables + figs)
# ---------------------------------------------------------------------
report: BacktestReport = build_backtest_report(
    df_ret_wide=df_ret_wide,
    daily_weights=weights,
    equity=equity,
    group_map=group_map,
    title="GammaEdge Backtest",
)

# ---------------------------------------------------------------------
# Show figures in UI
# ---------------------------------------------------------------------
col1, col2 = st.columns(2)
col1.plotly_chart(report.figures["equity"], use_container_width=True)
col2.plotly_chart(report.figures["drawdown"], use_container_width=True)
st.plotly_chart(report.figures["weights"], use_container_width=True)
st.plotly_chart(report.figures["top_contrib"], use_container_width=True)

# ---------------------------------------------------------------------
# Show tables
# ---------------------------------------------------------------------
st.subheader("Tables")
for name, df in report.tables.items():
    st.write(f"**{name}**")
    st.dataframe(df)

# ---------------------------------------------------------------------
# Build context for export
# ---------------------------------------------------------------------
period_start = str(dates[0]) if dates else "—"
period_end = str(dates[-1]) if dates else "—"

# Optional metrics block: if you have a metrics table, convert to dict
extra_metrics: dict[str, float] | None = None
if metrics_df is not None:
    try:
        pdf = metrics_df.to_pandas() if hasattr(metrics_df, "to_pandas") else metrics_df
        # Build a compact dict like {"Sharpe": 1.12, "MaxDD": -0.23, ...}
        extra_metrics = {str(k): float(v) for k, v in zip(pdf.iloc[:, 0], pdf.iloc[:, 1])}
    except Exception:
        extra_metrics = None

# Derive helper frames for context
df_asset_total = report.tables.get("contrib_asset_total")
df_group_total = report.tables.get("contrib_group_total")
df_brinson = st.session_state.get("df_brinson")  # optional if you stored it in 05_Attribution

ctx = build_context(
    portfolio_name=st.session_state.get("portfolio_name", "Portfolio"),
    period=(period_start, period_end),
    df_asset_cum=df_asset_total,
    df_group_total=df_group_total,
    df_brinson=df_brinson,
    extra_metrics=extra_metrics,
)

# ---------------------------------------------------------------------
# Export buttons: HTML + PDF
# ---------------------------------------------------------------------
st.subheader("📤 Export")

# Build figure pack for export (static images for HTML/PDF)
figures: list[ReportFigure] = [
    ReportFigure("Equity", report.figures["equity"]),
    ReportFigure("Drawdown", report.figures["drawdown"]),
    ReportFigure("Weights", report.figures["weights"]),
    ReportFigure("Top Contributors", report.figures["top_contrib"]),
]

# HTML export (inline PNGs)
try:
    html_bytes = render_html(ctx, figures, page_title="GammaEdge Report", h1="Backtest Report")
    st.download_button(
        "Download HTML report",
        data=html_bytes,
        file_name="backtest_report.html",
        mime="text/html",
    )
except Exception as e:
    st.info(f"HTML export not available: {e}")

# PDF export (reportlab)
try:
    pdf_bytes = render_pdf(ctx, figures, title="GammaEdge Report")
    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name="backtest_report.pdf",
        mime="application/pdf",
    )
except Exception as e:
    st.info(f"PDF export not available: {e}\nInstall dependencies: plotly[kaleido], reportlab")
