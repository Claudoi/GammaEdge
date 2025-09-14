# app/pages/04_Backtest.py
from __future__ import annotations

# --- stdlib ---
import os
import sys
from typing import Callable

# --- third-party ---
import numpy as np
import polars as pl
import streamlit as st

# ---------------------------------------------------------------------
# Path raíz del repo para imports locales (igual que en 03_Optimizer)
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# --- backtest core ---
from portfolio.backtest.engine import backtest_rebalanced
from portfolio.backtest import metrics as bt_metrics
from portfolio.backtest import attribution as bt_attr
from portfolio.backtest import reporting as bt_report

# --- optim helpers (para allocators) ---
from portfolio.core.utils import ensure_psd, project_to_box_simplex
from portfolio.optim.hrp import hrp_weights
from portfolio.core.utils import hrp_safe
from portfolio.optim.risk_parity import risk_parity
from portfolio.optim.mean_variance import pgd_box_simplex_l2

# --- viz (usa tus funciones ya definidas) ---
from portfolio.viz.plot_utils import (
    equity_and_drawdown,
    plot_equity,
    plot_drawdown,
    plot_turnover,
    plot_weights_heatmap,
    plot_tracking_error,
    plot_top_contributors,
    plot_group_contrib,
    plot_group_contrib_area,
    plot_brinson_cumulative,
)

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Backtest", layout="wide")
st.title("📊 Backtest")

# ─────────────────────────────────────────────────────────────────────
# Handoff defensivo desde 01/02/03
# ─────────────────────────────────────────────────────────────────────
if "returns_wide" not in st.session_state:
    st.warning("No hay datos de retornos. Ve a **01_Data** y genera `returns_wide` primero.")
    st.stop()

df_ret_wide: pl.DataFrame = st.session_state["returns_wide"]
tickers = [c for c in df_ret_wide.columns if c != "date"]
N = len(tickers)
if N == 0 or df_ret_wide.height < 10:
    st.error("Dataset demasiado pequeño para backtest.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────
# Sidebar – parámetros
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parámetros")

    rebalance_freq = st.selectbox("Frecuencia de rebalanceo", ["1mo", "1w", "3mo", "6mo"], index=0)
    lookback = st.number_input("Lookback (periodos)", min_value=30, max_value=2000, value=252, step=10)
    cost_bps = st.number_input("Coste (bps por turnover)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)

    st.markdown("---")
    st.subheader("Allocator")

    alloc_kind = st.selectbox(
        "Estrategia",
        ["Equal-Weight", "Min-Var (L2 PGD)", "Risk Parity", "HRP"],
        index=0,
        help="Estrategia usada para calcular los pesos en cada rebalance (sobre la ventana de lookback).",
    )

    # Caja simple para todos los allocators (opcional)
    w_min = st.number_input("w_min", 0.0, 1.0, 0.0, 0.01)
    w_max = st.number_input("w_max", 0.0, 1.0, 0.2, 0.01)
    if N * w_min > 1.0 or N * w_max < 1.0:
        w_min = min(w_min, 1.0 / max(N, 1))
        w_max = max(w_max, 1.0 / max(N, 1))
        st.info(f"Box ajustada para ser factible: w_min≤{1.0/max(N,1):.4f}≤w_max")

# ─────────────────────────────────────────────────────────────────────
# Allocator factories (window -> weights)
# ─────────────────────────────────────────────────────────────────────
def make_allocator(kind: str) -> Callable[[pl.DataFrame], np.ndarray]:
    if kind == "Equal-Weight":
        def alloc(win: pl.DataFrame) -> np.ndarray:
            n = win.width - 1
            if n <= 0:
                return np.array([], dtype=float)
            w = np.ones(n, dtype=float) / n
            return project_to_box_simplex(w, w_min, w_max)
        return alloc

    if kind == "Min-Var (L2 PGD)":
        def alloc(win: pl.DataFrame) -> np.ndarray:
            cols = [c for c in win.columns if c != "date"]
            R = win.select(cols).to_numpy() if cols else np.zeros((0, 0), dtype=float)
            # estimadores robustos muy básicos in-window
            mu_w = np.nanmean(R, axis=0) if R.size else np.zeros(len(cols))
            Sigma_w = np.cov(R, rowvar=False) if R.size else np.eye(len(cols)) * 1e-4
            mu_w = np.nan_to_num(mu_w, nan=0.0, posinf=0.0, neginf=0.0)
            Sigma_w = np.nan_to_num(Sigma_w, nan=0.0, posinf=0.0, neginf=0.0)
            Sigma_w = ensure_psd(Sigma_w, eps=1e-10, clip=True)
            # γ moderado para aproximarse a min-var (poco peso a μ)
            w = pgd_box_simplex_l2(mu_w, Sigma_w, gamma=100.0, w_min=w_min, w_max=w_max, lam_turnover=0.0)
            return project_to_box_simplex(w, w_min, w_max)
        return alloc

    if kind == "Risk Parity":
        def alloc(win: pl.DataFrame) -> np.ndarray:
            cols = [c for c in win.columns if c != "date"]
            R = win.select(cols).to_numpy() if cols else np.zeros((0, 0), dtype=float)
            Sigma_w = np.cov(R, rowvar=False) if R.size else np.eye(len(cols)) * 1e-4
            Sigma_w = np.nan_to_num(Sigma_w, nan=0.0, posinf=0.0, neginf=0.0)
            Sigma_w = ensure_psd(Sigma_w, eps=1e-10, clip=True)
            try:
                w = risk_parity(Sigma_w, w_min=w_min, w_max=w_max)
            except Exception:
                w = np.ones(len(cols)) / max(len(cols), 1)
            return project_to_box_simplex(w, w_min, w_max)
        return alloc

    if kind == "HRP":
        def alloc(win: pl.DataFrame) -> np.ndarray:
            cols = [c for c in win.columns if c != "date"]
            R = win.select(cols).to_numpy() if cols else np.zeros((0, 0), dtype=float)
            Sigma_w = np.cov(R, rowvar=False) if R.size else np.eye(len(cols)) * 1e-4
            Sigma_w = np.nan_to_num(Sigma_w, nan=0.0, posinf=0.0, neginf=0.0)
            Sigma_w = ensure_psd(Sigma_w, eps=1e-10, clip=True)
            w = hrp_safe(hrp_func=hrp_weights, cov=Sigma_w, method="ward", optimal=True, w_min=w_min, w_max=w_max)
            return project_to_box_simplex(w, w_min, w_max)
        return alloc

    # fallback
    def alloc_fallback(win: pl.DataFrame) -> np.ndarray:
        n = win.width - 1
        w = np.ones(n, dtype=float) / max(n, 1)
        return project_to_box_simplex(w, w_min, w_max)
    return alloc_fallback


allocator = make_allocator(alloc_kind)

# ─────────────────────────────────────────────────────────────────────
# Ejecutar backtest
# ─────────────────────────────────────────────────────────────────────
bt = backtest_rebalanced(
    df_ret_wide=df_ret_wide,
    lookback=int(lookback),
    rebalance_freq=rebalance_freq,
    cost_bps=float(cost_bps),
    allocator=allocator,
    bench_weights=np.full(N, 1.0 / max(N, 1)),  # benchmark estático EW (para TE proxy)
)

st.success("✅ Backtest ejecutado.")

# ─────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────
st.subheader("📈 Métricas")
dfm = bt_metrics.compute_backtest_metrics(bt)  # asegúrate que devuelva pl.DataFrame o pd.DataFrame
st.dataframe(dfm, width="stretch")

# ─────────────────────────────────────────────────────────────────────
# Plots principales
# ─────────────────────────────────────────────────────────────────────
st.subheader("📉 Equity & Drawdown")
st.plotly_chart(equity_and_drawdown(bt["dates"], bt["equity"], title="Equity & Drawdown"), width="stretch")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_equity(bt["dates"], bt["equity"], title="Equity"), width="stretch")
with col2:
    st.plotly_chart(plot_drawdown(bt["dates"], bt["equity"], title="Drawdown"), width="stretch")

st.subheader("⚖️ Pesos & Turnover")
st.plotly_chart(
    plot_weights_heatmap(bt["dates"], bt["tickers"], bt["weights"], title="Weights (rebalance steps)"),
    width="stretch",
)
st.plotly_chart(plot_turnover(bt["dates"], bt["turnover"], title="Turnover at Rebalance"), width="stretch")

if bt.get("te_daily_proxy") is not None:
    st.plotly_chart(
        plot_tracking_error(bt["dates"], bt["te_daily_proxy"], title="Daily TE (proxy)"),
        width="stretch",
    )

# ─────────────────────────────────────────────────────────────────────
# Attribution (opcional según datos disponibles)
# ─────────────────────────────────────────────────────────────────────
st.subheader("📊 Attribution")
try:
    # Top contributors acumulados
    df_top = bt_attr.top_contributors(bt, df_ret_wide, top_n=10)
    st.plotly_chart(plot_top_contributors(df_top), width="stretch")
except Exception as e:
    st.info(f"Attribution básica no disponible: {e}")


#  Mapping de grupos/sectors (ticker -> group):
# try:
#     df_group_total, df_group_daily = bt_attr.group_contrib(bt, df_ret_wide, groups_map=...)
#     st.plotly_chart(plot_group_contrib(df_group_total), width="stretch")
#     st.plotly_chart(plot_group_contrib_area(df_group_daily), width="stretch")
# except Exception:
#     pass

# Brinson (requiere benchmark por activo):
# try:
#     df_brinson = bt_attr.brinson(bt, bench_weights_per_asset=..., groups_map=...)
#     st.plotly_chart(plot_brinson_cumulative(df_brinson), width="stretch")
# except Exception:
#     pass

# ─────────────────────────────────────────────────────────────────────
# Reporting (export)
# ─────────────────────────────────────────────────────────────────────
st.subheader("🧾 Reporting")
try:
    html_report = bt_report.render_html_report(bt, dfm)
    st.download_button("Descargar reporte HTML", html_report, file_name="backtest_report.html", mime="text/html")
except Exception:
    st.caption("Reporte HTML no disponible (implementa `render_html_report`).")