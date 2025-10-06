# app/pages/04_Backtest.py
from __future__ import annotations

# --- stdlib ---
import os
import sys
import json
from typing import Callable

# --- third-party ---
import numpy as np
import polars as pl
import streamlit as st

# ---------------------------------------------------------------------
# Add repo root to sys.path so local imports work (same pattern as in 03_Optimizer)
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# --- backtest core ---
from portfolio.backtest.engine import backtest_rebalanced
from portfolio.backtest import metrics as bt_metrics
from portfolio.backtest import attribution as bt_attr
from portfolio.backtest import reporting as bt_report

# --- optim helpers (allocators) ---
from portfolio.core.utils import ensure_psd, project_to_box_simplex
from portfolio.optim.hrp import hrp_weights
from portfolio.core.utils import hrp_safe
from portfolio.optim.risk_parity import risk_parity
from portfolio.optim.mean_variance import pgd_box_simplex_l2

# --- attribution helpers we need directly ---
from portfolio.backtest.attribution import (
    expand_rebalance_weights,
    align_returns_and_weights,
    brinson_fachler_cumulative,
)

# --- visualization utilities ---
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
# Helper to safely extract turnover data (dates, values) from various formats
# ─────────────────────────────────────────────────────────────────────
def _extract_dates_vals_turnover(obj):
    """
    Returns (dates, turnover_values) from multiple possible formats:
    - Polars/Pandas DataFrame with ['date', 'turnover']
    - dict with 'date' and 'turnover' lists
    - list of dicts [{'date':..., 'turnover':...}, ...]
    - ndarray/list of pairs [date, turnover]
    """
    # Polars DataFrame
    if hasattr(obj, "to_pandas") and hasattr(obj, "columns"):
        cols = set(obj.columns)
        if {"date", "turnover"}.issubset(cols):
            return obj["date"].to_list(), np.asarray(obj["turnover"], dtype=float)
    # Pandas DataFrame
    if hasattr(obj, "reset_index") and hasattr(obj, "columns"):
        cols = set(obj.columns)
        if {"date", "turnover"}.issubset(cols):
            return obj["date"].values, obj["turnover"].to_numpy(dtype=float)
    # dict of lists
    if isinstance(obj, dict) and "date" in obj and "turnover" in obj:
        return list(obj["date"]), np.asarray(obj["turnover"], dtype=float)
    # list of dicts
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict):
        return [r["date"] for r in obj], np.asarray([r["turnover"] for r in obj], dtype=float)
    # generic ndarray / list of pairs
    arr = np.asarray(obj, dtype=object)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 0].tolist(), np.asarray(arr[:, 1], dtype=float)
    raise ValueError("bt['turnover'] must be a DF with ['date','turnover'] or a list/ndarray of (date, turnover).")


# ─────────────────────────────────────────────────────────────────────
# Streamlit page configuration
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Backtest", layout="wide")
st.title("📊 Backtest")


# ─────────────────────────────────────────────────────────────────────
# Defensive handoff: ensure data exists in session_state
# ─────────────────────────────────────────────────────────────────────
if "returns_wide" not in st.session_state:
    st.warning("No return data found. Go to **01_Data** and generate `returns_wide` first.")
    st.stop()

df_ret_wide: pl.DataFrame = st.session_state["returns_wide"]
tickers = [c for c in df_ret_wide.columns if c != "date"]
N = len(tickers)
if N == 0 or df_ret_wide.height < 10:
    st.error("Dataset too small for backtesting.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────
# Sidebar configuration – Backtest parameters + Group mapping
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    rebalance_freq = st.selectbox("Rebalance frequency", ["1mo", "1w", "3mo", "6mo"], index=0)
    lookback = st.number_input("Lookback (periods)", min_value=30, max_value=2000, value=252, step=10)
    cost_bps = st.number_input("Cost (bps per turnover)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)

    st.markdown("---")
    st.subheader("Allocator")

    alloc_kind = st.selectbox(
        "Strategy",
        ["Equal-Weight", "Min-Var (L2 PGD)", "Risk Parity", "HRP"],
        index=0,
        help="Portfolio allocation method applied at each rebalance window.",
    )

    # Box constraints (same for all allocators)
    w_min = st.number_input("w_min", 0.0, 1.0, 0.0, 0.01)
    w_max = st.number_input("w_max", 0.0, 1.0, 0.2, 0.01)
    # Adjust box if infeasible
    if N * w_min > 1.0 or N * w_max < 1.0:
        w_min = min(w_min, 1.0 / max(N, 1))
        w_max = max(w_max, 1.0 / max(N, 1))
        st.info(f"Box adjusted for feasibility: w_min≤{1.0/max(N,1):.4f}≤w_max")

    st.markdown("---")
    st.subheader("Group mapping (optional)")
    st.caption("Paste a JSON mapping like {\"AAPL\":\"Tech\",\"JPM\":\"Financials\"}. "
               "If left blank, all tickers are assigned to 'OTHER' (plots still work).")
    mapping_text = st.text_area("groups_map JSON", value="", height=120, placeholder='{"AAPL":"Tech","MSFT":"Tech"}')
    groups_map = {t: "OTHER" for t in tickers}  # default mapping
    if mapping_text.strip():
        try:
            user_map = json.loads(mapping_text)
            if isinstance(user_map, dict):
                # start from OTHER baseline; override with user entries
                groups_map = {tk: "OTHER" for tk in tickers}
                groups_map.update({k: str(v) for k, v in user_map.items() if k in tickers})
            else:
                st.warning("Provided mapping is not a JSON object; using default 'OTHER'.")
        except Exception as e:
            st.warning(f"Invalid JSON. Using default 'OTHER'. Error: {e}")


# ─────────────────────────────────────────────────────────────────────
# Allocator factory (returns a callable window → weights)
# ─────────────────────────────────────────────────────────────────────
def make_allocator(kind: str) -> Callable[[pl.DataFrame], np.ndarray]:
    """Factory for dynamic portfolio allocation based on strategy name."""
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
            # Estimate mean/variance using in-window data
            mu_w = np.nanmean(R, axis=0) if R.size else np.zeros(len(cols))
            Sigma_w = np.cov(R, rowvar=False) if R.size else np.eye(len(cols)) * 1e-4
            mu_w = np.nan_to_num(mu_w, nan=0.0, posinf=0.0, neginf=0.0)
            Sigma_w = np.nan_to_num(Sigma_w, nan=0.0, posinf=0.0, neginf=0.0)
            Sigma_w = ensure_psd(Sigma_w, eps=1e-10, clip=True)
            # High gamma ≈ minimum variance portfolio (low sensitivity to μ)
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

    # fallback: equal-weight if unknown strategy
    def alloc_fallback(win: pl.DataFrame) -> np.ndarray:
        n = win.width - 1
        w = np.ones(n, dtype=float) / max(n, 1)
        return project_to_box_simplex(w, w_min, w_max)
    return alloc_fallback

allocator = make_allocator(alloc_kind)


# ─────────────────────────────────────────────────────────────────────
# Run backtest engine
# ─────────────────────────────────────────────────────────────────────
bt = backtest_rebalanced(
    df_ret_wide=df_ret_wide,
    lookback=int(lookback),
    rebalance_freq=rebalance_freq,
    cost_bps=float(cost_bps),
    allocator=allocator,
    bench_weights=np.full(N, 1.0 / max(N, 1)),  # static equal-weight benchmark
)

st.success("✅ Backtest executed successfully.")


# ─────────────────────────────────────────────────────────────────────
# Display metrics table
# ─────────────────────────────────────────────────────────────────────
st.subheader("📈 Metrics")
dfm = bt_metrics.compute_backtest_metrics(bt)
dfm_pd = dfm.to_pandas() if hasattr(dfm, "to_pandas") else dfm
st.dataframe(dfm_pd, width="stretch")


# ─────────────────────────────────────────────────────────────────────
# Main plots: Equity, Drawdown, Weights, Turnover, TE
# ─────────────────────────────────────────────────────────────────────
st.subheader("📉 Equity & Drawdown")
st.plotly_chart(equity_and_drawdown(bt["dates"], bt["equity"], title="Equity & Drawdown"), width="stretch")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_equity(bt["dates"], bt["equity"], title="Equity"), width="stretch")
with col2:
    st.plotly_chart(plot_drawdown(bt["dates"], bt["equity"], title="Drawdown"), width="stretch")

# Handle potential mismatch between daily grid and rebalance grid
st.subheader("⚖️ Weights & Turnover")

W = np.asarray(bt["weights"], dtype=float)                 # (n_rebalances, N)
dates_all = bt["dates"]
dates_w = bt.get("rebalance_dates", None)
if dates_w is None or len(dates_w) != W.shape[0]:
    # fallback: trim daily dates to number of weight matrices
    dates_w = dates_all[: W.shape[0]]

st.plotly_chart(
    plot_weights_heatmap(dates_w, bt["tickers"], W, title="Weights (rebalance steps)"),
    width="stretch",
)

# Turnover plot (uses rebalance dates)
try:
    to_dates, to_vals = _extract_dates_vals_turnover(bt["turnover"])
    st.plotly_chart(plot_turnover(to_dates, to_vals, title="Turnover at Rebalance"), width="stretch")
except Exception as e:
    st.info(f"Turnover plot unavailable: {e}")

# Tracking error proxy (daily or rebalance-level)
te_daily = bt.get("te_daily_proxy")
if te_daily is not None:
    if hasattr(te_daily, "columns") and "date" in te_daily.columns and "te" in te_daily.columns:
        te_dates = te_daily["date"].to_list() if hasattr(te_daily, "to_pandas") else te_daily["date"].values
        te_vals = np.asarray(te_daily["te"], dtype=float)
        st.plotly_chart(plot_tracking_error(te_dates, te_vals, title="Daily TE (proxy)"), width="stretch")
    else:
        st.plotly_chart(plot_tracking_error(bt["dates"], np.asarray(te_daily, dtype=float), title="Daily TE (proxy)"),
                        width="stretch")


# ─────────────────────────────────────────────────────────────────────
# Attribution section (asset-level + groups + Brinson)
# ─────────────────────────────────────────────────────────────────────
st.subheader("📊 Attribution")

# Top contributors (asset-level)
try:
    df_top = bt_attr.top_contributors(bt, df_ret_wide, top_n=10)
    st.plotly_chart(plot_top_contributors(df_top), width="stretch")
except Exception as e:
    st.info(f"Basic attribution not available: {e}")

# Group contributions (daily + totals)
try:
    df_group_total, df_group_daily = bt_attr.group_contrib(bt, df_ret_wide, groups_map=groups_map, other_label="OTHER")
    st.plotly_chart(plot_group_contrib(df_group_total), width="stretch")
    st.plotly_chart(plot_group_contrib_area(df_group_daily), width="stretch")
except Exception as e:
    st.info(f"Group attribution unavailable: {e}")

# Brinson–Fachler cumulative
try:
    # 1) Build daily benchmark weights (Equal-Weight static) matching bt['dates'] grid
    bench_w_daily = np.tile(np.full(N, 1.0 / max(N, 1)), (len(bt["dates"]), 1))

    # 2) Map tickers to group indices [0..G-1] from groups_map
    uniq_groups = sorted(set(groups_map.get(tk, "OTHER") for tk in tickers))
    g_index = {g: i for i, g in enumerate(uniq_groups)}
    groups_idx = [g_index[groups_map.get(tk, "OTHER")] for tk in tickers]

    # 3) Expand rebalance weights to daily and align with returns
    W_reb = np.asarray(bt["weights"], float)
    W_daily = expand_rebalance_weights(bt["dates"], dates_w, W_reb)  # dates_w are rebalance dates already aligned

    df_daily_returns = (
        df_ret_wide
        .filter(pl.col("date").is_in(bt["dates"]))
        .sort("date")
        .select(["date", *tickers])
    )
    aln = align_returns_and_weights(df_daily_returns, W_daily)

    # 4) Build Brinson cumulative dataframe and plot
    df_brinson = brinson_fachler_cumulative(aln, bench_w_daily, groups_idx)
    st.plotly_chart(plot_brinson_cumulative(df_brinson), width="stretch")
except Exception as e:
    st.info(f"Brinson–Fachler unavailable: {e}")


# ─────────────────────────────────────────────────────────────────────
# Reporting and export
# ─────────────────────────────────────────────────────────────────────
st.subheader("🧾 Reporting")
try:
    html_report = bt_report.render_html_report(bt, dfm)
    st.download_button("Download HTML report", html_report,
                       file_name="backtest_report.html", mime="text/html")
except Exception:
    st.caption("HTML report not available.")