# app/pages/05_Attribution.py
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st

# Repo root for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from portfolio.attribution.euler import euler_risk_contributions
from portfolio.attribution.factor_decomposition import euler_factor_contributions
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


def _coerce_dates_list(dates_any: list[Any]) -> list[Any]:
    if dates_any is None or len(dates_any) == 0:
        return []
    df = pl.DataFrame({"date": dates_any})
    df = _ensure_datetime(df, "date")
    return df.get_column("date").to_list()


def _safe_metrics_to_dict(metrics_df_obj: Any) -> dict[str, float] | None:
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

# Variables para export de figuras y tablas adicionales
fig_euler_last_day: go.Figure | None = None
fig_factor_bar: go.Figure | None = None
fig_factor_hm: go.Figure | None = None
euler_last_day_rows: list[dict[str, Any]] | None = None
factor_rc_rows: list[dict[str, Any]] | None = None

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

# Euler RC último día (asset-level)
st.markdown("### 🧮 Euler Risk Contributions (last day)")
cov_last = None
try:
    if df_ret_bt.width > 1 and daily_W.shape[0] > 0:
        ret_pdf = df_ret_bt.select(pl.exclude("date")).to_pandas()
        cov_last = ret_pdf.cov()
        w_last = daily_W[-1]
        w_series = pd.Series(w_last, index=tickers)
        rc = euler_risk_contributions(w_series, cov_last)

        df_euler = rc.rename("risk_contribution").reset_index().rename(columns={"index": "asset"})

        col_e1, col_e2 = st.columns(2)
        as_percent_euler = col_e1.checkbox(
            "Show Euler RC in %",
            value=True,
            help="Normalize risk contributions by total portfolio risk.",
            key="euler_as_percent",
        )
        top_n_euler = col_e2.slider(
            "Top assets",
            min_value=1,
            max_value=min(25, len(df_euler)),
            value=min(10, len(df_euler)),
            step=1,
            key="euler_top_n",
        )

        fig_euler_last_day = viz.plot_euler_contributions(
            df_euler,
            title="Euler risk contributions (last day)",
            as_percent=as_percent_euler,
            top_n=top_n_euler,
        )
        viz.show_plot(fig_euler_last_day, key="euler_last_day")

        # Tabla compacta para contexto (Euler last day)
        try:
            sigma_euler = float(rc.sum())
            df_export = df_euler.copy()
            if sigma_euler > 0.0:
                df_export["rc_pct"] = df_export["risk_contribution"] / sigma_euler
            else:
                df_export["rc_pct"] = np.nan

            df_export = df_export.reindex(
                df_export["risk_contribution"].abs().sort_values(ascending=False).index
            ).head(15)
            euler_last_day_rows = []
            for _, row in df_export.iterrows():
                euler_last_day_rows.append(
                    {
                        "asset": str(row["asset"]),
                        "rc": float(row["risk_contribution"]),
                        "rc_pct": (float(row["rc_pct"]) if pd.notna(row["rc_pct"]) else None),
                    }
                )
        except Exception:
            euler_last_day_rows = None
    else:
        st.info("Not enough data to compute Euler risk contributions.")
except Exception as e:
    st.info(f"Euler risk contributions not available: {e}")
    cov_last = None
    fig_euler_last_day = None
    euler_last_day_rows = None

# Factor Decomposition (Euler sobre factores PCA)
st.markdown("### 🧩 Factor Decomposition (Euler, PCA factors)")

with st.expander("Show factor risk decomposition"):
    if df_ret_bt.width <= 1 or daily_W.shape[0] == 0:
        st.info("Not enough data to compute factor decomposition.")
    else:
        try:
            # Usamos la misma matriz de retornos que para el Euler asset-level
            ret_pdf = df_ret_bt.select(pl.exclude("date")).to_pandas()
            Sigma_assets = ret_pdf.cov().astype(float)

            n_assets = Sigma_assets.shape[0]
            if n_assets < 2:
                st.info("Need at least 2 assets for PCA factor model.")
            else:
                max_factors = min(5, n_assets)
                n_factors = st.slider(
                    "Number of PCA factors",
                    min_value=1,
                    max_value=max_factors,
                    value=min(3, max_factors),
                    step=1,
                )

                # Eigen-decomposition (symmetric PSD matrix)
                eigvals, eigvecs = np.linalg.eigh(Sigma_assets.values)
                idx = np.argsort(eigvals)[::-1]  # descending
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]

                lam = np.clip(eigvals[:n_factors], a_min=0.0, a_max=None)
                factors = [f"PC{i + 1}" for i in range(n_factors)]

                B = eigvecs[:, :n_factors]  # shape (N_assets, n_factors)
                B_df = pd.DataFrame(B, index=Sigma_assets.index.tolist(), columns=factors)
                Sigma_f_df = pd.DataFrame(
                    np.diag(lam),
                    index=factors,
                    columns=factors,
                )

                # Alineamos pesos con las filas de B_df
                w_last = daily_W[-1]
                w_series = pd.Series(w_last, index=tickers, name="w").astype(float)

                common_assets = [a for a in w_series.index if a in B_df.index]
                if not common_assets:
                    st.info("No overlap between portfolio tickers and PCA factor model assets.")
                else:
                    w_aligned = w_series.loc[common_assets]
                    B_aligned = B_df.loc[common_assets]

                    fact = euler_factor_contributions(w_aligned, B_aligned, Sigma_f_df)
                    sigma_p = fact["sigma_p"]
                    factor_rc = fact["factor_rc"]
                    asset_factor_rc = fact["asset_factor_rc"]

                    st.caption(f"Portfolio sigma (PCA factor model): {sigma_p:.6f}")

                    col_f1, col_f2 = st.columns(2)
                    as_percent_factor = col_f1.checkbox(
                        "Show factor RC in %",
                        value=True,
                        help=(
                            "Normalize factor and asset × factor risk contributions "
                            "by portfolio sigma."
                        ),
                        key="factor_as_percent",
                    )
                    top_n_factors = col_f2.slider(
                        "Top factors",
                        min_value=1,
                        max_value=min(10, len(factor_rc)),
                        value=min(5, len(factor_rc)),
                        step=1,
                        key="factor_top_n",
                    )

                    fig_factor_bar = viz.plot_factor_rc_bar(
                        factor_rc,
                        title="Factor RC (Euler, PCA factors)",
                        as_percent=as_percent_factor,
                        sigma_p=sigma_p,
                        top_n=top_n_factors,
                    )
                    viz.show_plot(fig_factor_bar, key="factor_rc_bar")

                    fig_factor_hm = viz.plot_factor_rc_heatmap(
                        asset_factor_rc,
                        title="Asset × Factor RC (Euler, PCA factors)",
                        as_percent=as_percent_factor,
                        sigma_p=sigma_p,
                    )
                    viz.show_plot(fig_factor_hm, key="factor_rc_heatmap")

                    # Tabla compacta para contexto (Factor RC)
                    try:
                        # factor_rc debería ser un pd.Series indexado por factor
                        if isinstance(factor_rc, pd.DataFrame):
                            if factor_rc.shape[1] == 0:
                                s = pd.Series(dtype=float)
                            else:
                                s = factor_rc.iloc[:, 0]
                        else:
                            s = factor_rc

                        df_factor = s.rename("rc").to_frame().reset_index()
                        df_factor.columns = ["factor", "rc"]
                        if sigma_p > 0.0:
                            df_factor["rc_pct"] = df_factor["rc"] / float(sigma_p)
                        else:
                            df_factor["rc_pct"] = np.nan

                        df_factor = df_factor.reindex(
                            df_factor["rc"].abs().sort_values(ascending=False).index
                        ).head(10)

                        factor_rc_rows = []
                        for _, row in df_factor.iterrows():
                            factor_rc_rows.append(
                                {
                                    "factor": str(row["factor"]),
                                    "rc": float(row["rc"]),
                                    "rc_pct": (
                                        float(row["rc_pct"]) if pd.notna(row["rc_pct"]) else None
                                    ),
                                }
                            )
                    except Exception:
                        factor_rc_rows = None
        except Exception as e:
            st.info(f"Factor decomposition not available: {e}")
            fig_factor_bar = None
            fig_factor_hm = None
            factor_rc_rows = None

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
    euler_last_day=euler_last_day_rows,
    factor_rc=factor_rc_rows,
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

# Añadimos Euler y Factor Decomposition al reporte si están disponibles
if fig_euler_last_day is not None:
    figures.append(ReportFigure("Euler RC (last day)", fig_euler_last_day))

if fig_factor_bar is not None:
    figures.append(ReportFigure("Factor RC (PCA)", fig_factor_bar))

if fig_factor_hm is not None:
    figures.append(ReportFigure("Asset × Factor RC (PCA)", fig_factor_hm))

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
except Exception as e:
    st.info(f"PDF export not available: {e}\nInstall dependencies: plotly[kaleido], reportlab")

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
