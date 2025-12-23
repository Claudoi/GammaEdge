# app/pages/06_Reporting.py
from __future__ import annotations

# --- stdlib ---
import os
import sys

# --- third-party ---
import numpy as np
import polars as pl
import streamlit as st

# ---------------------------------------------------------------------
# Repo root for local imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from portfolio.backtest import attribution as bt_attr  # expand/align helpers
from portfolio.backtest.kpis import compute_kpis
from portfolio.backtest.reporting import (
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
st.caption(
    "Generate HTML and PDF reports with equity, drawdown, weights, attribution and benchmark context."
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _to_pandas(df: object):
    """Safely convert Polars or other table-like objects to pandas."""
    try:
        return df.to_pandas()
    except Exception:
        return df


# ---------------------------------------------------------------------
# Inputs from previous pages (expected in session_state)
# ---------------------------------------------------------------------
bt = st.session_state.get("bt")
df_ret_wide = st.session_state.get("df_ret_wide", st.session_state.get("returns_wide"))
group_map = st.session_state.get("group_map")
metrics_df = st.session_state.get("metrics_df")  # optional KPIs
df_brinson = st.session_state.get("df_brinson")  # optional Brinson table
bench_meta = st.session_state.get("bench_meta")  # optional metadata about benchmark

if bt is None or df_ret_wide is None:
    st.warning(
        "⚠️ Run pages 02–05 first so we can build the report (risk model, backtest, attribution)."
    )
    st.stop()

# Normalize returns table to Polars + Datetime
if not isinstance(df_ret_wide, pl.DataFrame):
    df_ret_wide = pl.from_pandas(df_ret_wide)

if df_ret_wide.schema.get("date") != pl.Datetime:
    df_ret_wide = df_ret_wide.with_columns(pl.col("date").cast(pl.Datetime, strict=False))

# ---------------------------------------------------------------------
# Extract BT artifacts
# ---------------------------------------------------------------------
dates_bt = list(bt.get("dates", []))
equity = np.asarray(bt.get("equity", []), dtype=float)
tickers = list(bt.get("tickers", []))
W_reb = np.asarray(bt.get("weights", []), dtype=float)
rb_dates_any = list(bt.get("rebalance_dates", []))

if not dates_bt or equity.size == 0 or W_reb.size == 0 or not tickers:
    st.error("Missing essentials in 'bt' (dates, equity, weights, tickers).")
    st.stop()

# ---------------------------------------------------------------------
# Align returns to backtest date grid and expand weights to daily
# ---------------------------------------------------------------------
# 1) Filter returns to backtest dates (unique & sorted)
df_ret_bt = df_ret_wide.filter(pl.col("date").is_in(dates_bt)).unique(subset=["date"]).sort("date")

# 2) Rebalance dates fallback if shapes mismatch
if len(rb_dates_any) != W_reb.shape[0]:
    K = W_reb.shape[0]
    step = max(len(dates_bt) // max(K, 1), 1)
    rb_dates_any = dates_bt[::step][:K]

# 3) Expand to daily (T×N)
daily_W = bt_attr.expand_rebalance_weights(
    dates=df_ret_bt.get_column("date").to_list(),
    rb_dates=rb_dates_any,
    W_reb=W_reb,
)

# Persist a sensible fallback for other pages that might need benchmark weights
st.session_state["Wb_daily"] = st.session_state.get("Wb_daily", daily_W)

# ---------------------------------------------------------------------
# Build unified BacktestReport (tables + figs)
# ---------------------------------------------------------------------
report: BacktestReport = build_backtest_report(
    df_ret_wide=df_ret_bt,
    daily_weights=daily_W,
    equity=equity,
    group_map=group_map,
    title="GammaEdge Backtest",
)

# ---------------------------------------------------------------------
# Display meta info (benchmark & groups)
# ---------------------------------------------------------------------
st.markdown("### 🧭 Context Summary")
colA, colB = st.columns(2)
bench_scheme = bench_meta.get("scheme") if bench_meta else "Equal-Weight"
colA.metric("Benchmark Scheme", bench_scheme)
colB.metric("Groups", f"{len(group_map) if group_map else 0} assets mapped")

# ---------------------------------------------------------------------
# Show figures
# ---------------------------------------------------------------------
st.markdown("### 📈 Charts")

col1, col2 = st.columns(2)
col1.plotly_chart(report.figures["equity"], use_container_width=True)
col2.plotly_chart(report.figures["drawdown"], use_container_width=True)

st.plotly_chart(report.figures["weights"], use_container_width=True)
st.plotly_chart(report.figures["top_contrib"], use_container_width=True)

# ---------------------------------------------------------------------
# Show tables
# ---------------------------------------------------------------------
st.subheader("📊 Tables")
for name, df in report.tables.items():
    st.write(f"**{name}**")
    st.dataframe(_to_pandas(df), width="stretch")

# ---------------------------------------------------------------------
# Build context for export
# ---------------------------------------------------------------------
period_start = str(dates_bt[0]) if dates_bt else "—"
period_end = str(dates_bt[-1]) if dates_bt else "—"

# metrics: if not provided, compute from equity
extra_metrics: dict[str, float] | None = None
if metrics_df is not None:
    try:
        pdf = metrics_df.to_pandas() if hasattr(metrics_df, "to_pandas") else metrics_df
        if pdf.shape[1] >= 2:
            extra_metrics = {
                str(k): float(v) for k, v in zip(pdf.iloc[:, 0], pdf.iloc[:, 1], strict=False)
            }
    except Exception:
        extra_metrics = None

if extra_metrics is None:
    # Robust fallback KPIs
    extra_metrics = compute_kpis(equity, rf_daily=0.0, periods_per_year=252)

# Append benchmark info to appear in HTML/PDF header
if bench_meta and "scheme" in bench_meta:
    extra_metrics["Benchmark Scheme"] = bench_meta.get("scheme", "Unknown")

df_asset_total = report.tables.get("contrib_asset_total")
df_group_total = report.tables.get("contrib_group_total")

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

figures: list[ReportFigure] = [
    ReportFigure("Equity", report.figures["equity"]),
    ReportFigure("Drawdown", report.figures["drawdown"]),
    ReportFigure("Weights", report.figures["weights"]),
    ReportFigure("Top Contributors", report.figures["top_contrib"]),
]

# HTML export
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

# PDF export
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

# ---------------------------------------------------------------------
# Optional: save to disk (reports/)
# ---------------------------------------------------------------------
save_to_disk = st.checkbox("Save also to ./reports", value=False)
if save_to_disk:
    os.makedirs("reports", exist_ok=True)
    try:
        html_bytes = render_html(ctx, figures, page_title="GammaEdge Report", h1="Backtest Report")
        with open("reports/backtest_report.html", "wb") as f:
            f.write(html_bytes)
        st.success("HTML saved to reports/backtest_report.html")
    except Exception as e:
        st.info(f"Saving HTML failed: {e}")

    try:
        pdf_bytes = render_pdf(ctx, figures, title="GammaEdge Report")
        with open("reports/backtest_report.pdf", "wb") as f:
            f.write(pdf_bytes)
        st.success("PDF saved to reports/backtest_report.pdf")
    except Exception as e:
        st.info(f"Saving PDF failed: {e}")
