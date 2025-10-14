# app/pages/06_Scenarios.py
from __future__ import annotations

# --- stdlib ---
import os
import sys
from typing import Callable, List

# --- third-party ---
import numpy as np
import polars as pl
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Repo root for local imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# --- local modules ---
import portfolio.backtest.scenarios as scn   # robust import (module attr access)
from portfolio.backtest import metrics as bt_metrics

# allocators & utils (same style as 04_Backtest)
from portfolio.core.utils import ensure_psd, project_to_box_simplex, hrp_safe
from portfolio.optim.hrp import hrp_weights
from portfolio.optim.risk_parity import risk_parity
from portfolio.optim.mean_variance import pgd_box_simplex_l2

# plots already present in your repo
from portfolio.viz.plot_utils import (
    equity_and_drawdown,
    plot_equity,
    plot_drawdown,
    plot_weights_heatmap,
    plot_turnover,
)

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Scenarios", layout="wide")
st.title("🧪 Scenarios")
st.caption("Stress the return matrix via mean/vol shocks and one-day crashes. Optional block bootstrap.")

# ─────────────────────────────────────────────────────────────────────
# Defensive handoff from earlier pages
# ─────────────────────────────────────────────────────────────────────
df_ret_wide = st.session_state.get("df_ret_wide", st.session_state.get("returns_wide", None))
if df_ret_wide is None:
    st.warning("Missing `returns_wide`. Run **01_Data** (and 02/03/04 if needed) first.")
    st.stop()

# Normalize to Polars + Datetime
if isinstance(df_ret_wide, pd.DataFrame):
    df_ret_wide = pl.from_pandas(df_ret_wide)
if not isinstance(df_ret_wide, pl.DataFrame):
    st.error("`returns_wide` must be a Polars/Pandas DataFrame.")
    st.stop()
if df_ret_wide.schema.get("date") != pl.Datetime:
    df_ret_wide = df_ret_wide.with_columns(pl.col("date").cast(pl.Datetime))

tickers = [c for c in df_ret_wide.columns if c != "date"]
N = len(tickers)
if N == 0 or df_ret_wide.height < 10:
    st.error("Dataset too small for scenarios.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────
# Sidebar – scenario inputs and allocator options
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Scenario Inputs")

    # Bootstrap controls
    B = st.number_input("Bootstrap paths (B)", min_value=0, max_value=500, value=0, step=1,
                        help="If B=0: original chronology. If B>0: run B block-bootstrap paths.")
    block = st.number_input("Block length", min_value=2, max_value=252, value=10, step=1)
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42, step=1)

    st.markdown("---")
    st.caption("Shocks")
    mean_shift_bps = st.number_input("Mean shift (bps per day)", -1000.0, 1000.0, 0.0, 1.0,
                                     help="Add constant daily drift to all assets.")
    cov_scale = st.slider("Vol scale (cov_scale)", 0.10, 3.00, 1.00, 0.05,
                          help="Scale deviations from mean; 1.0 = no change.")
    crash_enable = st.checkbox("Enable one-day crash", value=False)
    crash_day = st.number_input("Crash day index (0-based)", 0, max(1, df_ret_wide.height-1), 0, 1)
    crash_drop_bps = st.number_input("Crash size (bps)", -5000.0, 5000.0, -500.0, 10.0,
                                     help="e.g. −500 bps = −5% gap on the crash day.")

    st.markdown("---")
    st.header("Allocator")
    alloc_kind = st.selectbox(
        "Strategy",
        ["Equal-Weight", "Min-Var (L2 PGD)", "Risk Parity", "HRP", "Min-TE (to Bench)"],
        index=0,
    )
    w_min = st.number_input("w_min", 0.0, 1.0, 0.0, 0.01)
    w_max = st.number_input("w_max", 0.0, 1.0, 0.2, 0.01)
    if N * w_min > 1.0 or N * w_max < 1.0:
        w_min = min(w_min, 1.0 / max(N, 1))
        w_max = max(w_max, 1.0 / max(N, 1))
        st.info(f"Box adjusted for feasibility: w_min≤{1.0/max(N,1):.4f}≤w_max")

    st.caption("Covariance")
    cov_estimator = st.selectbox("Covariance estimator", ["Sample", "EWMA"], index=0)
    ewma_lambda = st.slider("EWMA λ", min_value=0.80, max_value=0.995, value=0.97, step=0.005)

    st.markdown("---")
    st.header("Backtest Params")
    rebalance_freq = st.selectbox("Rebalance frequency", ["1mo", "1w", "3mo", "6mo"], index=0)
    lookback = st.number_input("Lookback (periods)", min_value=30, max_value=2000, value=252, step=10)
    cost_bps = st.number_input("Transaction cost (bps per turnover)", 0.0, 100.0, 2.0, 0.5)

# ─────────────────────────────────────────────────────────────────────
# Allocator factory (consistent with 04)
# ─────────────────────────────────────────────────────────────────────
def _cov_ewma(R: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """EWMA covariance with PSD safeguard."""
    if R.size == 0:
        return np.eye(0)
    T, N_ = R.shape
    S = np.zeros((N_, N_), dtype=float)
    w = 0.0
    mu = np.nanmean(R, axis=0)
    for t in range(T):
        x = (R[t] - mu).reshape(1, -1)
        S = lam * S + (1 - lam) * (x.T @ x)
        w = lam * w + (1 - lam)
    S = S / max(w, 1e-12)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    return ensure_psd(S, eps=1e-10, clip=True)

def _get_cov(win: pl.DataFrame, cols: List[str]) -> np.ndarray:
    if not cols:
        return np.eye(0)
    R = win.select(cols).to_numpy()
    if R.size == 0:
        return np.eye(len(cols)) * 1e-4
    if cov_estimator == "EWMA":
        S = _cov_ewma(R, lam=float(ewma_lambda))
    else:
        S = np.cov(R, rowvar=False)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    return ensure_psd(S, eps=1e-10, clip=True)

def make_allocator(kind: str) -> Callable[[pl.DataFrame], np.ndarray]:
    """
    Returns a closure mapping a rolling window of returns to weights.
    Same strategies as 04. No turnover budget here (scenarios focus on shock paths).
    """
    if kind == "Equal-Weight":
        def base_alloc(win: pl.DataFrame) -> np.ndarray:
            n = win.width - 1
            if n <= 0:
                return np.array([], dtype=float)
            w = np.ones(n, dtype=float) / n
            return project_to_box_simplex(w, w_min, w_max)

    elif kind == "Min-Var (L2 PGD)":
        def base_alloc(win: pl.DataFrame) -> np.ndarray:
            cols = [c for c in win.columns if c != "date"]
            R = win.select(cols).to_numpy() if cols else np.zeros((0, 0), dtype=float)
            mu_w = np.nanmean(R, axis=0) if R.size else np.zeros(len(cols))
            mu_w = np.nan_to_num(mu_w, nan=0.0, posinf=0.0, neginf=0.0)
            Sigma_w = _get_cov(win, cols)
            w = pgd_box_simplex_l2(mu_w, Sigma_w, gamma=100.0, w_min=w_min, w_max=w_max, lam_turnover=0.0)
            return project_to_box_simplex(w, w_min, w_max)

    elif kind == "Risk Parity":
        def base_alloc(win: pl.DataFrame) -> np.ndarray:
            cols = [c for c in win.columns if c != "date"]
            Sigma_w = _get_cov(win, cols)
            try:
                w = risk_parity(Sigma_w, w_min=w_min, w_max=w_max)
            except Exception:
                w = np.ones(len(cols)) / max(len(cols), 1)
            return project_to_box_simplex(w, w_min, w_max)

    elif kind == "HRP":
        def base_alloc(win: pl.DataFrame) -> np.ndarray:
            cols = [c for c in win.columns if c != "date"]
            Sigma_w = _get_cov(win, cols)
            w = hrp_safe(hrp_func=hrp_weights, cov=Sigma_w, method="ward", optimal=True, w_min=w_min, w_max=w_max)
            return project_to_box_simplex(w, w_min, w_max)

    elif kind == "Min-TE (to Bench)":
        def base_alloc(win: pl.DataFrame) -> np.ndarray:
            cols = [c for c in win.columns if c != "date"]
            Sigma_w = _get_cov(win, cols)
            w_bench = np.full(len(cols), 1.0 / max(len(cols), 1))
            # min (w-wb)'Σ(w-wb) via L2 PGD with μ_eff = 2 Σ wb
            mu_eff = 2.0 * (Sigma_w @ w_bench)
            w = pgd_box_simplex_l2(mu_eff, Sigma_w, gamma=1.0, w_min=w_min, w_max=w_max, lam_turnover=0.0)
            return project_to_box_simplex(w, w_min, w_max)

    else:
        def base_alloc(win: pl.DataFrame) -> np.ndarray:
            n = win.width - 1
            w = np.ones(n, dtype=float) / max(n, 1)
            return project_to_box_simplex(w, w_min, w_max)

    return base_alloc

# ─────────────────────────────────────────────────────────────────────
# Benchmark (equal-weight by default; can be replaced from session)
# ─────────────────────────────────────────────────────────────────────
bench_session = st.session_state.get("bench_weights_daily", None)
bench_eq = np.full(N, 1.0 / max(N, 1))
# (We pass a static benchmark vector; the backtest engine handles the daily proxy if needed.)

# ─────────────────────────────────────────────────────────────────────
# Build scenario configs & run
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Run Scenarios")

# Always include a Baseline config (no shock)
cfgs: list[scn.ScenarioConfig] = [scn.ScenarioConfig(name="Baseline", B=0, block=int(block), seed=int(seed))]

# Add a Shocked config driven by the sidebar inputs
mean_shift = (mean_shift_bps / 10_000.0) if abs(mean_shift_bps) > 1e-12 else None
crash_tuple = (int(crash_day), crash_drop_bps / 10_000.0) if crash_enable else None
shock = scn.ShockSpec(mean_shift=mean_shift, cov_scale=float(cov_scale), crash=crash_tuple)
cfgs.append(scn.ScenarioConfig(name="Shocked", B=int(B), block=int(block), seed=int(seed), shock=shock))

with st.spinner("Running scenarios…"):
    results = scn.run_scenarios(
        cfgs,
        df_ret_wide=df_ret_wide,
        allocator_factory=lambda: make_allocator(alloc_kind),
        lookback=int(lookback),
        rebalance_freq=rebalance_freq,
        cost_bps=float(cost_bps),
        bench_weights=bench_eq,
    )

# ─────────────────────────────────────────────────────────────────────
# Scenario Comparison (robust metric flattening)
# ─────────────────────────────────────────────────────────────────────
st.subheader("Scenario Comparison")

def _flatten_metrics(m) -> dict:
    """
    Return a flat dict of scalar metrics from either:
      - Polars DF of shape (1, K)  → {col: value}
      - Polars DF with columns ['metric','value'] → {metric_i: value_i}
      - pandas DataFrame equivalent layouts
    Any non-scalar is ignored.
    """
    out = {}

    # Polars
    if isinstance(m, pl.DataFrame):
        try:
            if m.height == 1 and m.width >= 1:
                # wide single-row
                row = m.row(0, named=True)
                for k, v in row.items():
                    if np.isscalar(v) or isinstance(v, (np.floating, np.integer)):
                        out[str(k)] = float(v)
                return out
            cols = set(m.columns)
            if {"metric", "value"}.issubset(cols):
                met = m.get_column("metric").to_list()
                val = m.get_column("value").to_list()
                for k, v in zip(met, val):
                    if np.isscalar(v) or isinstance(v, (np.floating, np.integer)):
                        out[str(k)] = float(v)
                return out
        except Exception:
            pass

    # pandas
    if isinstance(m, pd.DataFrame):
        try:
            if len(m) == 1 and m.shape[1] >= 1:
                row = m.iloc[0].to_dict()
                for k, v in row.items():
                    if np.isscalar(v) or isinstance(v, (np.floating, np.integer)):
                        out[str(k)] = float(v)
                return out
            cols = set(m.columns)
            if {"metric", "value"}.issubset(cols):
                for _, r in m.iterrows():
                    v = r["value"]
                    if np.isscalar(v) or isinstance(v, (np.floating, np.integer)):
                        out[str(r["metric"])] = float(v)
                return out
        except Exception:
            pass

    # unknown schema → nothing
    return out

rows: list[dict] = []
for res in results:
    # Try polars first; if metrics came already in pandas, handle in flattener
    m_pl = res.metrics if isinstance(res.metrics, pl.DataFrame) else None
    m_pd = res.metrics if isinstance(res.metrics, pd.DataFrame) else None

    metrics_flat = _flatten_metrics(m_pl if m_pl is not None else m_pd)

    row = {"Scenario": res.name}
    # only extend with scalars
    for k, v in metrics_flat.items():
        # sanitize weird values just in case
        if np.isfinite(v):
            row[k] = float(v)
    rows.append(row)

# Build comparison table strictly from scalars
if rows:
    df_comp = pl.DataFrame(rows)  # list[dict] → Polars (evita ArrowInvalid)
    st.dataframe(df_comp.to_pandas(), use_container_width=True)
else:
    st.info("No metrics to display.")

# ─────────────────────────────────────────────────────────────────────
# Tabs with detailed plots per scenario (unchanged)
# ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([r.name for r in results])
for tab, res in zip(tabs, results):
    with tab:
        bt = res.bt
        st.plotly_chart(
            equity_and_drawdown(bt["dates"], bt["equity"], title=f"{res.name} · Equity & Drawdown"),
            use_container_width=True
        )
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_equity(bt["dates"], bt["equity"], title=f"{res.name} · Equity"),
                            use_container_width=True)
        with c2:
            st.plotly_chart(plot_drawdown(bt["dates"], bt["equity"], title=f"{res.name} · Drawdown"),
                            use_container_width=True)

        st.plotly_chart(
            plot_weights_heatmap(bt["dates"], bt["tickers"], bt["weights"], title=f"{res.name} · Weights"),
            use_container_width=True,
        )

        dates_w = bt.get("rebalance_dates", None)
        if dates_w is None:
            k_to = int(np.size(bt.get("turnover", [])))
            dates_w = bt["dates"][-k_to:] if k_to > 0 else []
        to_vals = np.asarray(bt.get("turnover", []), dtype=float)
        if len(dates_w) == to_vals.size and to_vals.size > 0:
            st.plotly_chart(
                plot_turnover(dates_w, to_vals, title=f"{res.name} · Turnover"),
                use_container_width=True
            )
        else:
            st.caption("Turnover series not available for this scenario.")

# ─────────────────────────────────────────────────────────────────────
# Exports (use the same df_comp we just built)
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Exports")
try:
    if rows:
        csv_bytes = pl.DataFrame(rows).write_csv()
        st.download_button(
            "Download scenario metrics (CSV)",
            csv_bytes,
            file_name="scenario_metrics.csv",
            mime="text/csv",
        )
except Exception:
    pass
