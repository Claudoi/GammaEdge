# app/pages/06_Reporting.py
from __future__ import annotations

# --- stdlib ---
import logging
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

# Design System
from app.design_system import get_global_styles
from app.viz.plotly_theme import apply_gammaedge_theme
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------
st.set_page_config(page_title="Reporting", layout="wide")
st.markdown(get_global_styles(), unsafe_allow_html=True)
st.title("Reporting")
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
    except Exception as exc:
        logger.warning(
            "Reporting: _to_pandas conversion failed: %s",
            exc,
            exc_info=True,
        )
        return df


# ---------------------------------------------------------------------
# Inputs from previous pages (expected in session_state)
# ---------------------------------------------------------------------
bt = st.session_state.get("bt")
returns_wide = st.session_state.get("returns_wide")
group_map = st.session_state.get("group_map")
metrics_df = st.session_state.get("metrics_df")  # optional KPIs
df_brinson = st.session_state.get("df_brinson")  # optional Brinson table
bench_meta = st.session_state.get("bench_meta")  # optional metadata about benchmark

if returns_wide is None:
    st.warning("Carga datos en la página 01_Data antes de generar el reporte.")
    st.stop()

if bt is None:
    st.warning(
        "Run pages 02–05 first so we can build the report (risk model, backtest, attribution)."
    )
    st.stop()

# Normalize returns table to Polars + Datetime
if not isinstance(returns_wide, pl.DataFrame):
    returns_wide = pl.from_pandas(returns_wide)

if returns_wide.schema.get("date") != pl.Datetime:
    returns_wide = returns_wide.with_columns(pl.col("date").cast(pl.Datetime, strict=False))

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
df_ret_bt = returns_wide.filter(pl.col("date").is_in(dates_bt)).unique(subset=["date"]).sort("date")

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
with st.spinner("Building backtest report figures and tables..."):
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
st.markdown("### Context Summary")
colA, colB = st.columns(2)
bench_scheme = bench_meta.get("scheme") if bench_meta else "Equal-Weight"
colA.metric("Benchmark Scheme", bench_scheme)
colB.metric("Groups", f"{len(group_map) if group_map else 0} assets mapped")

# ---------------------------------------------------------------------
# Show figures
# ---------------------------------------------------------------------
st.markdown("### Charts")

col1, col2 = st.columns(2)
col1.plotly_chart(apply_gammaedge_theme(report.figures["equity"]), use_container_width=True)
col2.plotly_chart(apply_gammaedge_theme(report.figures["drawdown"]), use_container_width=True)

st.plotly_chart(apply_gammaedge_theme(report.figures["weights"]), use_container_width=True)
st.plotly_chart(apply_gammaedge_theme(report.figures["top_contrib"]), use_container_width=True)

# ---------------------------------------------------------------------
# Show tables
# ---------------------------------------------------------------------
st.subheader("Tables")
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
st.subheader("Export")

figures: list[ReportFigure] = [
    ReportFigure("Equity", apply_gammaedge_theme(report.figures["equity"])),
    ReportFigure("Drawdown", apply_gammaedge_theme(report.figures["drawdown"])),
    ReportFigure("Weights", apply_gammaedge_theme(report.figures["weights"])),
    ReportFigure("Top Contributors", apply_gammaedge_theme(report.figures["top_contrib"])),
]

# HTML export
try:
    with st.spinner("Generating HTML report..."):
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
    with st.spinner("Generating PDF report..."):
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
        with st.spinner("Saving HTML report to disk..."):
            html_bytes = render_html(
                ctx, figures, page_title="GammaEdge Report", h1="Backtest Report"
            )
            with open("reports/backtest_report.html", "wb") as f:
                f.write(html_bytes)
        st.success("HTML saved to reports/backtest_report.html")
    except Exception as e:
        st.info(f"Saving HTML failed: {e}")

    try:
        with st.spinner("Saving PDF report to disk..."):
            pdf_bytes = render_pdf(ctx, figures, title="GammaEdge Report")
            with open("reports/backtest_report.pdf", "wb") as f:
                f.write(pdf_bytes)
        st.success("PDF saved to reports/backtest_report.pdf")
    except Exception as e:
        st.info(f"Saving PDF failed: {e}")
