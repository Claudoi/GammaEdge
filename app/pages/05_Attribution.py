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
from portfolio.backtest.brinson_utils import ensure_datetime as _ensure_datetime
from portfolio.backtest.kpis import compute_kpis
from portfolio.backtest.reporting import (
    BacktestReport,
    ReportFigure,
    build_backtest_report,
    build_context,
    render_html,
    render_pdf,
)
from portfolio.viz import plot_utils as viz  # para figuras extra (Brinson) opcionales


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _coerce_dates_list(dates_any: list) -> list:
    """Fuerza una lista de fechas (str/datetime/np.datetime64) a Polars Datetime,
    y devuelve lista de Python con dtype consistente para usar en is_in()."""
    if dates_any is None or len(dates_any) == 0:
        return []
    df = pl.DataFrame({"date": dates_any})
    df = _ensure_datetime(df, "date")
    return df.get_column("date").to_list()


def _safe_metrics_to_dict(metrics_df_obj) -> dict[str, float] | None:
    """Convierte una tabla 2-col a dict Metric->Value; devuelve None si falla."""
    try:
        pdf = metrics_df_obj.to_pandas() if hasattr(metrics_df_obj, "to_pandas") else metrics_df_obj
        return {str(k): float(v) for k, v in zip(pdf.iloc[:, 0], pdf.iloc[:, 1])}
    except Exception:
        return None


# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------
st.set_page_config(page_title="Reporting", layout="wide")
st.title("📄 Reporting")
st.caption(
    "Generate HTML and PDF reports with equity, drawdown, weights, attribution and benchmark context."
)

# ---------------------------------------------------------------------
# Inputs from previous pages (expected in session_state)
# ---------------------------------------------------------------------
bt = st.session_state.get("bt")
df_ret_wide = st.session_state.get("df_ret_wide", st.session_state.get("returns_wide"))
group_map = st.session_state.get("group_map")
metrics_df = st.session_state.get("metrics_df")  # optional KPIs
df_brinson = st.session_state.get("df_brinson")  # optional Brinson table (cumulative)
bench_meta = st.session_state.get("bench_meta")  # optional metadata about benchmark
Wb_daily_session = st.session_state.get("Wb_daily")  # optional benchmark weights
groups_idx = st.session_state.get("groups_idx")  # optional groups

if bt is None or df_ret_wide is None:
    st.warning(
        "⚠️ Run pages 02–05 first so we can build the report (risk model, backtest, attribution)."
    )
    st.stop()

# Normalize returns table to Polars + Datetime
if not isinstance(df_ret_wide, pl.DataFrame):
    df_ret_wide = pl.from_pandas(df_ret_wide)
df_ret_wide = _ensure_datetime(df_ret_wide, "date")

# ---------------------------------------------------------------------
# Extract BT artifacts
# ---------------------------------------------------------------------
dates_bt_any = list(bt.get("dates", []))
equity = np.asarray(bt.get("equity", []), dtype=float)
tickers = list(bt.get("tickers", []))
W_reb = np.asarray(bt.get("weights", []), dtype=float)  # shape (K, N)
rb_dates_any = list(bt.get("rebalance_dates", []))

if not dates_bt_any or equity.size == 0 or W_reb.size == 0 or not tickers:
    st.error("Missing essentials in 'bt' (dates, equity, weights, tickers).")
    st.stop()

# Coerce BT dates to Datetime (consistent con df_ret_wide['date'])
dates_bt = _coerce_dates_list(dates_bt_any)

# ---------------------------------------------------------------------
# Align returns to backtest date grid and expand weights to daily
# ---------------------------------------------------------------------
# 1) returns solo en fechas del backtest (dtypes compatibles)
df_ret_bt = df_ret_wide.filter(pl.col("date").is_in(dates_bt)).unique(subset=["date"]).sort("date")

# 2) validar dimensiones y construir rb_dates si faltan o no cuadran con W_reb
K, N = W_reb.shape[0], W_reb.shape[1] if W_reb.ndim == 2 else (W_reb.shape[0], len(tickers))
if len(tickers) != N:
    st.error(f"Weight columns ({N}) do not match tickers ({len(tickers)}).")
    st.stop()

# Si no hay rebalance_dates válidos o el número no coincide con K, infiérelo equiespaciado
rb_dates_list = _coerce_dates_list(rb_dates_any)
if len(rb_dates_list) != K:
    step = max(len(dates_bt) // max(K, 1), 1)
    rb_dates_list = dates_bt[::step][:K]

# 3) expandir a diario (T×N) usando la malla exacta de df_ret_bt
daily_W = bt_attr.expand_rebalance_weights(
    dates=df_ret_bt.get_column("date").to_list(),
    rb_dates=rb_dates_list,
    W_reb=W_reb,
)

# persist fallback de benchmark solo si no existía
if Wb_daily_session is None:
    st.session_state["Wb_daily"] = daily_W

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
colA, colB, colC = st.columns(3)
bench_scheme = (bench_meta or {}).get("scheme", "Equal-Weight")
colA.metric("Benchmark Scheme", bench_scheme)
colB.metric("Groups mapped", f"{len(group_map) if isinstance(group_map, dict) else 0}")
colC.metric("Period", f"{str(dates_bt[0])[:10]} → {str(dates_bt[-1])[:10]}")

# ---------------------------------------------------------------------
# Show figures
# ---------------------------------------------------------------------
st.markdown("### 📈 Charts")
col1, col2 = st.columns(2)
col1.plotly_chart(report.figures["equity"], use_container_width=True)
col2.plotly_chart(report.figures["drawdown"], use_container_width=True)
st.plotly_chart(report.figures["weights"], use_container_width=True)
st.plotly_chart(report.figures["top_contrib"], use_container_width=True)

# Si tenemos df_brinson en sesión, añadimos también sus figuras para la UI
if isinstance(df_brinson, pl.DataFrame) and "date" in df_brinson.columns:
    try:
        df_brinson = _ensure_datetime(df_brinson, "date")
        # Alinear a periodo del BT por seguridad
        df_brinson = df_brinson.filter(pl.col("date").is_in(dates_bt)).sort("date")
        st.plotly_chart(
            viz.plot_brinson_cumulative(df_brinson, title="Brinson–Fachler (Cumulative)"),
            use_container_width=True,
        )
        st.plotly_chart(
            viz.plot_brinson_cumulative_components(df_brinson),
            use_container_width=True,
        )
    except Exception as e:
        st.info(f"Brinson figures skipped: {e}")

# ---------------------------------------------------------------------
# Show tables
# ---------------------------------------------------------------------
st.subheader("📊 Tables")
for name, df in report.tables.items():
    st.write(f"**{name}**")
    st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------------
# Build context for export
# ---------------------------------------------------------------------
period_start = str(dates_bt[0]) if dates_bt else "—"
period_end = str(dates_bt[-1]) if dates_bt else "—"

# metrics: si no vienen, calcula a partir de equity
extra_metrics = _safe_metrics_to_dict(metrics_df) if metrics_df is not None else None
if extra_metrics is None:
    extra_metrics = compute_kpis(equity, rf_daily=0.0, periods_per_year=252)

# añade benchmark info a extra_metrics (aparece en el HTML/PDF)
if bench_meta and "scheme" in bench_meta:
    extra_metrics["Benchmark Scheme"] = bench_meta.get("scheme")

df_asset_total = report.tables.get("contrib_asset_total")
df_group_total = report.tables.get("contrib_group_total")

# Inyecta df_brinson si existe y parece válida
df_brinson_ctx = None
if isinstance(df_brinson, pl.DataFrame) and {"date", "total"}.issubset(set(df_brinson.columns)):
    df_brinson_ctx = df_brinson

ctx = build_context(
    portfolio_name=st.session_state.get("portfolio_name", "Portfolio"),
    period=(period_start, period_end),
    df_asset_cum=df_asset_total,
    df_group_total=df_group_total,
    df_brinson=df_brinson_ctx,
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

# Si hay Brinson, añadimos una figura representativa (cumulative total)
if df_brinson_ctx is not None:
    try:
        br_cum_fig = viz.plot_brinson_cumulative(
            df_brinson_ctx, title="Brinson–Fachler (Cumulative)"
        )
        figures.append(ReportFigure("Brinson (Cumulative)", br_cum_fig))
    except Exception:
        pass

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
except Exception:
    st.info("PDF export not available: {e}\nInstall dependencies: plotly[kaleido], reportlab")

# Guardar en disco (reports/)
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
