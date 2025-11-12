# app/pages/05_Attribution.py
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

# Repo root for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from portfolio.attribution.euler import euler_risk_contributions
from portfolio.backtest import attribution as bt_attr
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
from portfolio.viz import plot_utils as viz


def _coerce_dates_list(dates_any: list) -> list:
    if dates_any is None or len(dates_any) == 0:
        return []
    df = pl.DataFrame({"date": dates_any})
    df = _ensure_datetime(df, "date")
    return df.get_column("date").to_list()


def _safe_metrics_to_dict(metrics_df_obj) -> dict[str, float] | None:
    try:
        pdf = metrics_df_obj.to_pandas() if hasattr(metrics_df_obj, "to_pandas") else metrics_df_obj
        return {str(k): float(v) for k, v in zip(pdf.iloc[:, 0], pdf.iloc[:, 1], strict=False)}
    except Exception:
        return None


st.set_page_config(page_title="Attribution & Reporting", layout="wide")
st.title("📊 Attribution & Reporting")
st.caption(
    "Performance (Brinson) and risk (Euler) attribution on top of the backtest, "
    "with exportable HTML/PDF reports."
)

# Inputs expected desde páginas previas
bt = st.session_state.get("bt")
df_ret_wide = st.session_state.get("df_ret_wide", st.session_state.get("returns_wide"))
group_map = st.session_state.get("group_map")
metrics_df = st.session_state.get("metrics_df")
df_brinson = st.session_state.get("df_brinson")
bench_meta = st.session_state.get("bench_meta")
Wb_daily_session = st.session_state.get("Wb_daily")
groups_idx = st.session_state.get("groups_idx")

if bt is None or df_ret_wide is None:
    st.warning(
        "⚠️ Run pages 02–05 first so we can build the report (risk model, backtest, attribution)."
    )
    st.stop()

# Normaliza returns a Polars + Datetime
if not isinstance(df_ret_wide, pl.DataFrame):
    df_ret_wide = pl.from_pandas(df_ret_wide)
df_ret_wide = _ensure_datetime(df_ret_wide, "date")

# Artefactos del backtest
dates_bt_any = list(bt.get("dates", []))
equity = np.asarray(bt.get("equity", []), dtype=float)
tickers = list(bt.get("tickers", []))
W_reb = np.asarray(bt.get("weights", []), dtype=float)
rb_dates_any = list(bt.get("rebalance_dates", []))

if not dates_bt_any or equity.size == 0 or W_reb.size == 0 or not tickers:
    st.error("Missing essentials in 'bt' (dates, equity, weights, tickers).")
    st.stop()

dates_bt = _coerce_dates_list(dates_bt_any)

# Alinea returns a la malla del backtest
df_ret_bt = df_ret_wide.filter(pl.col("date").is_in(dates_bt)).unique(subset=["date"]).sort("date")

# Valida dimensiones y rebalance_dates
if W_reb.ndim == 2:
    K, N = W_reb.shape
else:
    K = W_reb.shape[0]
    N = len(tickers)

if len(tickers) != N:
    st.error(f"Weight columns ({N}) do not match tickers ({len(tickers)}).")
    st.stop()

rb_dates_list = _coerce_dates_list(rb_dates_any)
if len(rb_dates_list) != K:
    step = max(len(dates_bt) // max(K, 1), 1)
    rb_dates_list = dates_bt[::step][:K]

# Expande pesos a diario
daily_W = bt_attr.expand_rebalance_weights(
    dates=df_ret_bt.get_column("date").to_list(),
    rb_dates=rb_dates_list,
    W_reb=W_reb,
)

# Guarda benchmark weights por defecto si no existe en sesión
if Wb_daily_session is None:
    st.session_state["Wb_daily"] = daily_W

# Construye BacktestReport (figuras + tablas)
report: BacktestReport = build_backtest_report(
    df_ret_wide=df_ret_bt,
    daily_weights=daily_W,
    equity=equity,
    group_map=group_map,
    title="GammaEdge Backtest",
)

df_asset_total = report.tables.get("contrib_asset_total")
df_group_total = report.tables.get("contrib_group_total")

# Normaliza df_brinson para UI y export
df_brinson_ctx: pl.DataFrame | None = None
if isinstance(df_brinson, pl.DataFrame) and "date" in df_brinson.columns:
    try:
        df_brinson_ctx = _ensure_datetime(df_brinson, "date")
        df_brinson_ctx = df_brinson_ctx.filter(pl.col("date").is_in(dates_bt)).sort("date")
    except Exception:
        df_brinson_ctx = None

# Contexto superior
st.markdown("### 🧭 Context Summary")
colA, colB, colC = st.columns(3)
bench_scheme = (bench_meta or {}).get("scheme", "Equal-Weight")
colA.metric("Benchmark Scheme", bench_scheme)
colB.metric("Groups mapped", f"{len(group_map) if isinstance(group_map, dict) else 0}")
colC.metric("Period", f"{str(dates_bt[0])[:10]} → {str(dates_bt[-1])[:10]}")

# Figuras core
st.markdown("### 📈 Core Charts")
col1, col2 = st.columns(2)
viz.show_plot(report.figures["equity"], st_obj=col1, key="equity")
viz.show_plot(report.figures["drawdown"], st_obj=col2, key="drawdown")
viz.show_plot(report.figures["weights"], key="weights")
viz.show_plot(report.figures["top_contrib"], key="top_contrib")

# Brinson
if df_brinson_ctx is not None:
    try:
        st.markdown("### 🧱 Brinson Performance Attribution")

        fig_brinson_cum = viz.plot_brinson_cumulative(
            df_brinson_ctx,
            title="Brinson–Fachler (Cumulative Total)",
        )
        viz.show_plot(fig_brinson_cum, key="brinson_cum")

        fig_brinson_components = viz.plot_brinson_cumulative_components(df_brinson_ctx)
        viz.show_plot(fig_brinson_components, key="brinson_components")

        if df_group_total is not None:
            fig_brinson_group = viz.plot_brinson_group_bar(
                df_group_total,
                metric="contrib_total",
                title="Brinson — Contribution by Group",
            )
            viz.show_plot(fig_brinson_group, key="brinson_group")

        fig_brinson_ts = viz.plot_brinson_timeseries(
            df_brinson_ctx,
            title="Brinson — Total Attribution Over Time",
            metric="total",
        )
        viz.show_plot(fig_brinson_ts, key="brinson_ts")
    except Exception as e:
        st.info(f"Brinson figures skipped: {e}")

# Euler RC último día
st.markdown("### 🧮 Euler Risk Contributions (last day)")
try:
    if df_ret_bt.width > 1 and daily_W.shape[0] > 0:
        ret_pdf = df_ret_bt.select(pl.exclude("date")).to_pandas()
        cov = ret_pdf.cov()
        w_last = daily_W[-1]
        w_series = pd.Series(w_last, index=tickers)
        rc = euler_risk_contributions(w_series, cov)
        df_euler = rc.rename("risk_contribution").reset_index().rename(columns={"index": "asset"})
        fig_euler = viz.plot_euler_contributions(df_euler)
        viz.show_plot(fig_euler, key="euler_last_day")
    else:
        st.info("Not enough data to compute Euler risk contributions.")
except Exception as e:
    st.info(f"Euler risk contributions not available: {e}")

# Tablas
st.subheader("📊 Tables")
for name, df in report.tables.items():
    st.write(f"**{name}**")
    st.dataframe(df, width="stretch")

# Contexto para export
period_start = str(dates_bt[0]) if dates_bt else "—"
period_end = str(dates_bt[-1]) if dates_bt else "—"

extra_metrics = _safe_metrics_to_dict(metrics_df) if metrics_df is not None else None
if extra_metrics is None:
    extra_metrics = compute_kpis(equity, rf_daily=0.0, periods_per_year=252)

if bench_meta and "scheme" in bench_meta:
    extra_metrics["Benchmark Scheme"] = bench_meta.get("scheme")

ctx = build_context(
    portfolio_name=st.session_state.get("portfolio_name", "Portfolio"),
    period=(period_start, period_end),
    df_asset_cum=df_asset_total,
    df_group_total=df_group_total,
    df_brinson=df_brinson_ctx,
    extra_metrics=extra_metrics,
)

# Export
st.subheader("📤 Export")

figures: list[ReportFigure] = [
    ReportFigure("Equity", report.figures["equity"]),
    ReportFigure("Drawdown", report.figures["drawdown"]),
    ReportFigure("Weights", report.figures["weights"]),
    ReportFigure("Top Contributors", report.figures["top_contrib"]),
]

if df_brinson_ctx is not None:
    try:
        br_cum_fig = viz.plot_brinson_cumulative(
            df_brinson_ctx,
            title="Brinson–Fachler (Cumulative)",
        )
        figures.append(ReportFigure("Brinson (Cumulative)", br_cum_fig))
    except Exception:
        pass

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

# Guardado opcional en ./reports
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
