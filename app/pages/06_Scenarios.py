# app/pages/06_Scenarios.py
from __future__ import annotations

# --- stdlib ---
import os
import sys
from typing import Callable, List, Dict, Any, Optional, Sequence

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
# Backtest engine and metrics
from portfolio.backtest.engine import backtest_rebalanced
from portfolio.backtest import metrics as bt_metrics

# Scenarios utilities (centralized)
from portfolio.backtest.scenarios import (
    ShockSpec,
    ScenarioConfig,
    run_scenarios,
    apply_shock_map_to_wide,
    historical_slice_returns,
)

# Allocators & utils (consistent with 04_Backtest)
from portfolio.core.utils import ensure_psd, project_to_box_simplex, hrp_safe
from portfolio.optim.hrp import hrp_weights
from portfolio.optim.risk_parity import risk_parity
from portfolio.optim.mean_variance import pgd_box_simplex_l2

# Visualization utils (return Plotly Figures; Streamlit used only here)
from portfolio.viz.plot_utils import (
    equity_and_drawdown,
    plot_equity,
    plot_drawdown,
    plot_weights_heatmap,
    plot_turnover,
    plot_tornado_sensitivity,
    plot_equity_compare,
    plot_drawdown_compare,
    plot_metric_delta_bars,
    plot_weights_compare_heatmap,
)

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Scenarios", layout="wide")
st.title("🧪 Scenarios")
st.caption("Stress-tests on the return matrix with robust turnover reconstruction and clean comparisons vs Baseline.")

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def next_key(prefix: str = "plt") -> str:
    """Generate unique Streamlit element keys to avoid duplicate-ID errors."""
    st.session_state.setdefault("_auto_key_counter", 0)
    st.session_state["_auto_key_counter"] += 1
    return f"{prefix}-{st.session_state['_auto_key_counter']}"

def _to_numpy_2d(x) -> np.ndarray:
    """Ensure an object becomes a 2D float NumPy array (NaN-safe)."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def _safe_metrics(bt_obj: Dict[str, Any]) -> pl.DataFrame:
    """Compute metrics via your bt_metrics; fallback to equity-derived measures if needed."""
    try:
        return bt_metrics.compute_backtest_metrics(bt_obj)
    except Exception:
        eq = np.asarray(bt_obj.get("equity", []), float)
        r = (eq[1:] / eq[:-1] - 1.0) if eq.size > 1 else np.array([])
        if r.size == 0:
            return pl.DataFrame({"CAGR": [np.nan], "Sharpe": [np.nan], "MaxDD": [np.nan]})
        cagr = (np.prod(1 + r)) ** (252 / max(len(r), 1)) - 1
        sharpe = (np.mean(r) / (np.std(r) + 1e-12)) * np.sqrt(252)
        mdd = 1 - (np.cumprod(1 + r) / np.maximum.accumulate(np.cumprod(1 + r))).min()
        return pl.DataFrame({"CAGR": [float(cagr)], "Sharpe": [float(sharpe)], "MaxDD": [float(mdd)]})

def _extract_metric_scalar(dfm: pl.DataFrame | pd.DataFrame, name: str, default: float = np.nan) -> float:
    """Extract scalar metric by name from a 1-row metrics frame."""
    try:
        if isinstance(dfm, pl.DataFrame):
            if name in dfm.columns and dfm.height > 0:
                v = dfm.get_column(name)[0]
                return float(v) if np.isfinite(v) else float(default)
            lower = {c.lower(): c for c in dfm.columns}
            if name.lower() in lower:
                v = dfm.get_column(lower[name.lower()])[0]
                return float(v) if np.isfinite(v) else float(default)
        elif isinstance(dfm, pd.DataFrame) and len(dfm) > 0:
            cols = dfm.columns
            if name in cols:
                v = dfm.iloc[0][name]
                return float(v) if np.isfinite(v) else float(default)
            lower = {c.lower(): c for c in cols}
            if name.lower() in lower:
                v = dfm.iloc[0][lower[name.lower()]]
                return float(v) if np.isfinite(v) else float(default)
    except Exception:
        pass
    return float(default)

def _ensure_turnover_with_drift(bt: dict, df_wide: pl.DataFrame) -> tuple[list, np.ndarray]:
    """
    If bt['turnover'] is missing, reconstruct executed turnover:
    - Propagate previous weights via return drift up to next rebalance date (pre-trade weights).
    - Compare with target weights at next rebalance. turnover = 0.5 * ||w_new - w_pre||_1
    """
    # 1) If engine provided turnover, try to use it
    to_obj = bt.get("turnover", None)
    if to_obj is not None:
        try:
            if isinstance(to_obj, pl.DataFrame) and "turnover" in to_obj.columns:
                dates = (to_obj.get_column("date").to_list()
                         if "date" in to_obj.columns else list(range(to_obj.height)))
                vals = to_obj.get_column("turnover").to_numpy()
                return dates, np.asarray(vals, float)
            if isinstance(to_obj, pd.DataFrame) and "turnover" in to_obj.columns:
                dates = (to_obj["date"].tolist()
                         if "date" in to_obj.columns else list(range(len(to_obj))))
                return dates, np.asarray(to_obj["turnover"].values, float)
            arr = np.asarray(to_obj, float).ravel()
            if arr.size > 0:
                rb_dates = bt.get("rebalance_dates", None)
                dates = (list(rb_dates)[-arr.size:]
                         if rb_dates is not None else list(bt["dates"][-arr.size:]))
                return dates, arr
        except Exception:
            pass

    # 2) Reconstruct with drift
    rb_dates = list(bt.get("rebalance_dates", []))
    W_reb = np.asarray(bt.get("weights", []), float)   # (K, N)
    tick = list(bt.get("tickers", []))
    if W_reb.size == 0 or len(rb_dates) != W_reb.shape[0] or df_wide is None:
        return [], np.array([], float)

    have = set(df_wide.columns)
    needs = [t for t in tick if t not in have]
    if needs:
        df_wide = df_wide.with_columns(**{c: pl.lit(0.0, dtype=pl.Float64) for c in needs})
    df_wide = df_wide.select(["date", *tick]).sort("date")

    idx_map = {d: i for i, d in enumerate(df_wide.get_column("date").to_list())}

    def _norm(w: np.ndarray) -> np.ndarray:
        s = float(np.sum(w))
        return (w / s) if s > 1e-12 else w

    turns, out_dates = [], []
    w_prev = _norm(W_reb[0])
    for k in range(len(rb_dates) - 1):
        d0, d1 = rb_dates[k], rb_dates[k + 1]
        i0, i1 = idx_map.get(d0), idx_map.get(d1)

        if i0 is None or i1 is None or i1 <= i0:
            w_pre = w_prev
        else:
            R_seg = df_wide.slice(i0, i1 - i0).select(tick).to_numpy()
            G = np.prod(1.0 + np.nan_to_num(R_seg, nan=0.0), axis=0)
            w_pre = _norm(w_prev * G)

        w_new = _norm(W_reb[k + 1])
        to_k = 0.5 * float(np.sum(np.abs(w_new - w_pre)))
        turns.append(to_k)
        out_dates.append(d1)
        w_prev = w_new

    return out_dates, np.asarray(turns, float)

# ─────────────────────────────────────────────────────────────────────
# Defensive handoff from previous pages
# ─────────────────────────────────────────────────────────────────────
df_ret_wide = st.session_state.get("df_ret_wide", st.session_state.get("returns_wide", None))
if df_ret_wide is None:
    st.warning("Missing `returns_wide`. Run pages 01→04 first.")
    st.stop()

if isinstance(df_ret_wide, pd.DataFrame):
    df_ret_wide = pl.from_pandas(df_ret_wide)

if not isinstance(df_ret_wide, pl.DataFrame):
    st.error("`returns_wide` must be Polars or Pandas.")
    st.stop()

if df_ret_wide.schema.get("date") != pl.Datetime:
    df_ret_wide = df_ret_wide.with_columns(pl.col("date").cast(pl.Datetime))

tickers = [c for c in df_ret_wide.columns if c != "date"]
N = len(tickers)
if N == 0 or df_ret_wide.height < 10:
    st.error("Dataset too small for scenarios.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────
# Sidebar — common parameters
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Backtest Params")
    rebalance_freq = st.selectbox("Rebalance frequency", ["1mo", "1w", "3mo", "6mo"], index=0)
    lookback = st.number_input("Lookback (periods)", 30, 2000, 252, 10)
    cost_bps = st.number_input("Transaction cost (bps per turnover)", 0.0, 100.0, 2.0, 0.5)

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
    ewma_lambda = st.slider("EWMA λ", 0.80, 0.995, 0.97, 0.005)

# ─────────────────────────────────────────────────────────────────────
# Allocator factory (same design as page 04; no turnover budget here)
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
    """Map a rolling window to weights under box+simplex constraints."""
    if kind == "Equal-Weight":
        def base_alloc(win: pl.DataFrame) -> np.ndarray:
            n = win.width - 1
            w = np.ones(n, dtype=float) / max(n, 1)
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
            w = hrp_safe(hrp_func=hrp_weights, cov=Sigma_w, method="ward", optimal=True,
                         w_min=w_min, w_max=w_max)
            return project_to_box_simplex(w, w_min, w_max)

    elif kind == "Min-TE (to Bench)":
        def base_alloc(win: pl.DataFrame) -> np.ndarray:
            cols = [c for c in win.columns if c != "date"]
            Sigma_w = _get_cov(win, cols)
            w_bench = np.full(len(cols), 1.0 / max(len(cols), 1))
            mu_eff = 2.0 * (Sigma_w @ w_bench)  # L2-PGD proxy for tracking-error min
            w = pgd_box_simplex_l2(mu_eff, Sigma_w, gamma=1.0, w_min=w_min, w_max=w_max, lam_turnover=0.0)
            return project_to_box_simplex(w, w_min, w_max)

    else:
        def base_alloc(win: pl.DataFrame) -> np.ndarray:
            n = win.width - 1
            w = np.ones(n, dtype=float) / max(n, 1)
            return project_to_box_simplex(w, w_min, w_max)

    return base_alloc

# ─────────────────────────────────────────────────────────────────────
# Engine wrapper (homogeneous parameters for fair comparisons)
# ─────────────────────────────────────────────────────────────────────
def _run_engine(df_wide: pl.DataFrame) -> Dict[str, Any]:
    alloc = make_allocator(alloc_kind)
    bench_w = np.full(N, 1.0 / max(N, 1))  # equal-weight benchmark vector
    return backtest_rebalanced(
        df_ret_wide=df_wide.select(["date", *tickers]).sort("date"),
        lookback=int(lookback),
        rebalance_freq=rebalance_freq,
        cost_bps=float(cost_bps),
        allocator=alloc,
        bench_weights=bench_w,
    )

# ─────────────────────────────────────────────────────────────────────
# Baseline (reference) — run once
# ─────────────────────────────────────────────────────────────────────
st.subheader("Baseline (reference)")
with st.spinner("Running baseline..."):
    df_base = df_ret_wide.select(["date", *tickers]).sort("date")
    base_bt = _run_engine(df_base)

# Main equity + drawdown
st.plotly_chart(
    equity_and_drawdown(base_bt["dates"], base_bt["equity"], title="Baseline · Equity & Drawdown"),
    width="stretch",
    key=next_key("baseline-ed"),
)

# Metrics (safe)
try:
    base_m = bt_metrics.compute_backtest_metrics(base_bt)
except Exception:
    base_m = _safe_metrics(base_bt)

st.dataframe(
    base_m.to_pandas() if isinstance(base_m, pl.DataFrame) else pd.DataFrame(),
    width="stretch",
)

# Weights heatmap
st.plotly_chart(
    plot_weights_heatmap(base_bt["dates"], base_bt["tickers"], base_bt["weights"], title="Baseline · Weights"),
    width="stretch",
    key=next_key("baseline-weights"),
)

# Turnover (robust reconstruction if needed)
dates_to, vals_to = _ensure_turnover_with_drift(base_bt, df_base)
if vals_to.size > 0 and len(dates_to) == vals_to.size:
    st.plotly_chart(
        plot_turnover(dates_to, vals_to, title="Baseline · Turnover"),
        width="stretch",
        key=next_key("baseline-turnover"),
    )
else:
    st.caption("Turnover series not available for Baseline.")

# ─────────────────────────────────────────────────────────────────────
# Scenarios — settings (sidebar) and shock helpers
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.header("Scenario settings")

    B = st.number_input("Bootstrap paths (B)", 0, 500, 0, 1,
                        help="If B=0: original chronology. If B>0: B synthetic paths via block bootstrap.")
    block = st.number_input("Block length (days)", 2, 252, 10, 1)
    seed = st.number_input("Seed", 0, 100_000, 42, 1)

    st.caption("Shocks (applied to returns)")
    mean_shift_bps = st.number_input("Mean shift (bps/day)", -1000.0, 1000.0, 0.0, 1.0,
                                     help="Constant drift added to every asset per day.")
    cov_scale = st.slider("Vol scale (cov_scale)", 0.10, 3.00, 1.00, 0.05,
                          help="Scale cross-sectional deviations from mean (1.0 = no change).")
    crash_enable = st.checkbox("One-day crash", value=False)
    crash_day = st.number_input("Crash day index (0-based)", 0, max(1, df_ret_wide.height - 1), 0, 1)
    crash_drop_bps = st.number_input("Crash size (bps)", -5000.0, 5000.0, -500.0, 10.0,
                                     help="e.g., −500 bps = −5% one-day gap.")

# Build scenario list only if there is a real shock/config difference
cfgs: list[ScenarioConfig] = []
mean_shift = (mean_shift_bps / 10_000.0) if abs(mean_shift_bps) > 1e-12 else None
crash_tuple = (int(crash_day), crash_drop_bps / 10_000.0) if crash_enable else None

if mean_shift is not None or float(cov_scale) != 1.0 or crash_enable or int(B) > 0:
    cfgs.append(
        ScenarioConfig(
            name="Shocked",
            B=int(B),
            block=int(block),
            seed=int(seed),
            shock=ShockSpec(mean_shift=mean_shift, cov_scale=float(cov_scale), crash=crash_tuple),
        )
    )
else:
    st.info("No active shock parameters — Baseline already shown above. Adjust shocks to run scenarios.")

# ─────────────────────────────────────────────────────────────────────
# Scenario runs (only alternative scenarios vs baseline)
# ─────────────────────────────────────────────────────────────────────
if cfgs:
    st.markdown("---")
    st.subheader("Scenario comparison vs Baseline")

    with st.spinner("Running scenarios…"):
        results = run_scenarios(
            cfgs,
            df_ret_wide=df_base,  # keep baseline chronology for fair comparison
            allocator_factory=lambda: make_allocator(alloc_kind),
            lookback=int(lookback),
            rebalance_freq=rebalance_freq,
            cost_bps=float(cost_bps),
            bench_weights=np.full(N, 1.0 / max(N, 1)),
        )

    # Flatten metrics table (one row per scenario)
    rows: list[dict] = []
    for res in results:
        m = res["metrics"]
        if isinstance(m, pl.DataFrame) and m.height == 1:
            rows.append({"Scenario": res["name"], **m.row(0, named=True)})
        elif isinstance(m, pd.DataFrame) and len(m) == 1:
            rows.append({"Scenario": res["name"], **m.iloc[0].to_dict()})

    if rows:
        df_comp = pl.DataFrame(rows)
        st.dataframe(df_comp.to_pandas(), width="stretch")

        # ΔCAGR vs baseline (if available)
        base_cagr = _extract_metric_scalar(base_m, "CAGR", default=np.nan)
        if np.isfinite(base_cagr) and "CAGR" in df_comp.columns:
            fig_delta = plot_metric_delta_bars(df_comp.to_pandas(), baseline_value=base_cagr, metric_col="CAGR")
            st.plotly_chart(fig_delta, width="stretch", key=next_key("delta-cagr"))

    # Detailed charts by scenario
    for idx, res in enumerate(results):
        bt = res["bt"]
        sc_name = str(res["name"])
        sec_prefix = f"sc-{idx}-{sc_name.replace(' ', '').lower()}"

        st.subheader(f"Scenario: {sc_name}")

        # Equity & drawdown comparisons
        st.plotly_chart(
            plot_equity_compare(base_bt["dates"], base_bt["equity"], bt.get("equity", []),
                                name_a="Baseline", name_b=sc_name),
            width="stretch",
            key=next_key(f"{sec_prefix}-equity-compare"),
        )
        st.plotly_chart(
            plot_drawdown_compare(base_bt["dates"], base_bt["equity"], bt.get("equity", []),
                                  name_a="Baseline", name_b=sc_name),
            width="stretch",
            key=next_key(f"{sec_prefix}-dd-compare"),
        )

        # Weights delta heatmap (Scenario vs Baseline)
        try:
            fig_wcmp = plot_weights_compare_heatmap(
                dates=base_bt["dates"],
                tickers=base_bt["tickers"],
                weights_a=_to_numpy_2d(base_bt["weights"]),
                weights_b=_to_numpy_2d(bt.get("weights", np.zeros_like(base_bt["weights"]))),
                name_a="Baseline",
                name_b=sc_name,
                mode="delta",
                zmax_abs=0.05,  # clamp for readability; adjust if needed
            )
            st.plotly_chart(fig_wcmp, width="stretch", key=next_key(f"{sec_prefix}-weights-delta"))
        except Exception as e:
            st.caption(f"Weight comparison heatmap skipped: {e}")

# ─────────────────────────────────────────────────────────────────────
# Optional blocks: Beta-correlated shock, Historical slice, Tornado
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.header("Beta-correlated shock")
    use_beta = st.checkbox("Enable beta-correlated shock", value=False)
    idx_opts = [c for c in df_base.columns if c != "date"]
    index_col = st.selectbox("Index column", options=idx_opts, index=0, disabled=not use_beta)
    index_move = st.number_input("Index move (e.g., -0.05)", -0.50, 0.50, -0.05, 0.01, disabled=not use_beta)
    beta_lb = st.number_input("Beta lookback (days)", 60, 1000, 252, 10, disabled=not use_beta)

    st.markdown("---")
    st.header("Historical slice")
    use_hist = st.checkbox("Enable historical slice", value=False)
    c1, c2 = st.columns(2)
    with c1:
        hist_start = st.text_input("Start (YYYY-MM-DD)", "2020-02-15", disabled=not use_hist)
    with c2:
        hist_end = st.text_input("End (YYYY-MM-DD)", "2020-04-15", disabled=not use_hist)

    st.markdown("---")
    st.header("Tornado sensitivity")
    do_tornado = st.checkbox("Compute tornado sensitivity (±δ per asset)", value=False)
    delta = st.number_input("Shock per asset (daily return)", -0.20, 0.20, 0.02, 0.01, disabled=not do_tornado)

def _rolling_beta_last_window(df_wide: pl.DataFrame, index_col: str, lookback: int) -> dict[str, float]:
    """Compute last-window CAPM betas vs a chosen index within a lookback window."""
    df = df_wide.sort("date")
    win = df if df.height <= lookback else df.tail(lookback)
    cols = [c for c in win.columns if c != "date"]
    if index_col not in cols:
        raise ValueError(f"Index column '{index_col}' not found.")
    X = np.nan_to_num(win.select(cols).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    j = cols.index(index_col)
    idx = X[:, j]
    var_idx = float(np.var(idx, ddof=1))
    if var_idx < 1e-16:
        return {c: 0.0 for c in cols if c != index_col}
    betas = {}
    for k, c in enumerate(cols):
        if c == index_col:
            continue
        cov_ij = float(np.cov(X[:, k], idx, ddof=1)[0, 1])
        betas[c] = cov_ij / var_idx
    return betas

# Beta-correlated shock demo (independent from generic scenarios)
if use_beta:
    st.markdown("---")
    st.subheader("Beta-correlated shock")
    try:
        betas = _rolling_beta_last_window(df_base, index_col=index_col, lookback=int(beta_lb))
        shock_map = {asset: beta * float(index_move) for asset, beta in betas.items()}
        df_beta = apply_shock_map_to_wide(df_base, shock_map)
        bt_beta = _run_engine(df_beta)
        st.plotly_chart(
            equity_and_drawdown(bt_beta["dates"], bt_beta["equity"],
                                title=f"Beta shock: {index_col} move {index_move:+.1%}"),
            width="stretch",
            key=next_key("beta-ed"),
        )
        st.dataframe(_safe_metrics(bt_beta).to_pandas(), width="stretch")
    except Exception as e:
        st.info(f"Beta shock skipped: {e}")

# Historical slice replay
if use_hist:
    st.markdown("---")
    st.subheader("Historical slice (replay)")
    try:
        df_slice = historical_slice_returns(df_base, hist_start, hist_end, tickers=tickers)
        bt_slice = _run_engine(df_slice)
        st.plotly_chart(
            equity_and_drawdown(bt_slice["dates"], bt_slice["equity"],
                                title=f"Historical slice {hist_start} → {hist_end}"),
            width="stretch",
            key=next_key("hist-ed"),
        )
        st.dataframe(_safe_metrics(bt_slice).to_pandas(), width="stretch")
    except Exception as e:
        st.info(f"Historical slice skipped: {e}")

# Tornado sensitivity (robust; uses plot_utils tornado)
if do_tornado:
    st.markdown("---")
    st.subheader("One-at-a-time sensitivity (Tornado)")
    try:
        # Baseline metric (CAGR) from metrics or equity
        base_cagr = _extract_metric_scalar(base_m, "CAGR", default=np.nan)
        if not np.isfinite(base_cagr):
            eq = np.asarray(base_bt.get("equity", []), float)
            if eq.size >= 2:
                r = eq[1:] / eq[:-1] - 1.0
                base_cagr = float((np.prod(1.0 + r)) ** (252 / max(len(r), 1)) - 1.0)
            else:
                base_cagr = np.nan

        sens_rows = []
        for tk in tickers:
            # Down
            df_down = apply_shock_map_to_wide(df_base, {tk: -abs(float(delta))})
            bt_down = _run_engine(df_down)
            eqd = np.asarray(bt_down.get("equity", []), float)
            if eqd.size >= 2:
                r = eqd[1:] / eqd[:-1] - 1.0
                met_down = float((np.prod(1.0 + r)) ** (252 / max(len(r), 1)) - 1.0)
            else:
                met_down = np.nan

            # Up
            df_up = apply_shock_map_to_wide(df_base, {tk: +abs(float(delta))})
            bt_up = _run_engine(df_up)
            equ = np.asarray(bt_up.get("equity", []), float)
            if equ.size >= 2:
                r = equ[1:] / equ[:-1] - 1.0
                met_up = float((np.prod(1.0 + r)) ** (252 / max(len(r), 1)) - 1.0)
            else:
                met_up = np.nan

            sens_rows.append({"asset": tk, "metric": "CAGR", "base": base_cagr, "down": met_down, "up": met_up})

        if sens_rows:
            df_sens = pd.DataFrame(sens_rows)
            fig_tornado = plot_tornado_sensitivity(df_sens, metric_label="CAGR", down_label="Down", up_label="Up")
            st.plotly_chart(fig_tornado, width="stretch", key=next_key("tornado"))
            with st.expander("Download sensitivity (CSV)", expanded=False):
                csv_bytes = df_sens.to_csv(index=False).encode("utf-8")
                st.download_button("Download", csv_bytes, file_name="tornado_sensitivity.csv",
                                   mime="text/csv", key=next_key("tornado-download"))
        else:
            st.info("No sensitivity rows generated (empty universe).")
    except Exception as e:
        st.info(f"Tornado sensitivity skipped: {e}")

# ─────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Exports")
try:
    if isinstance(base_m, pl.DataFrame) and base_m.height > 0:
        st.download_button(
            "Download baseline metrics (CSV)",
            base_m.write_csv(),
            file_name="baseline_metrics.csv",
            mime="text/csv",
            key=next_key("dl-base-metrics"),
        )
    # Scenario metrics: rebuild if needed based on 'results' above.
except Exception:
    pass