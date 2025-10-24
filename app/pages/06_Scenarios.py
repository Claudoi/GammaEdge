# app/pages/06_Scenarios.py
from __future__ import annotations

import inspect

# --- stdlib ---
import os
import sys
from typing import Any, Callable

# --- third-party ---
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

# ---------------------------------------------------------------------
# Repo root for local imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# --- local modules ---
from portfolio.backtest import metrics as bt_metrics
from portfolio.backtest.engine import backtest_rebalanced
from portfolio.backtest.scenarios import (
    ScenarioConfig,
    ShockSpec,
    apply_shock_map_to_wide,
    historical_slice_returns,
    run_scenarios,
)
from portfolio.core.utils import ensure_psd, hrp_safe, project_to_box_simplex
from portfolio.optim.hrp import hrp_weights
from portfolio.optim.mean_variance import pgd_box_simplex_l2
from portfolio.optim.risk_parity import risk_parity
from portfolio.viz.plot_utils import (
    equity_and_drawdown,
    plot_drawdown_compare,
    plot_equity_compare,
    plot_metric_delta_bars,
    plot_tornado_sensitivity,
    plot_turnover,
    plot_weights_compare_heatmap,
    plot_weights_heatmap,
)

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Scenarios", layout="wide")
st.title("🧪 Scenarios")
st.caption(
    "Stress-tests on the return matrix with robust turnover reconstruction and clean comparisons vs Baseline."
)


# ─────────────────────────────────────────────────────────────────────
# Helpers (robust + NaN-safe)
# ─────────────────────────────────────────────────────────────────────
def next_key(prefix: str = "plt") -> str:
    """Unique Streamlit keys to avoid duplicate-ID errors."""
    st.session_state.setdefault("_auto_key_counter", 0)
    st.session_state["_auto_key_counter"] += 1
    return f"{prefix}-{st.session_state['_auto_key_counter']}"


def _to_numpy_2d(x) -> np.ndarray:
    """Force 2D float array and sanitize NaNs/infs."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_metrics(bt_obj: dict[str, Any]) -> pl.DataFrame:
    """Compute metrics via repo metrics; fallback to equity-derived if needed."""
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
        return pl.DataFrame(
            {"CAGR": [float(cagr)], "Sharpe": [float(sharpe)], "MaxDD": [float(mdd)]}
        )


def _extract_metric_scalar(
    dfm: pl.DataFrame | pd.DataFrame, name: str, default: float = np.nan
) -> float:
    """Extract scalar metric by name from a 1-row frame; case-insensitive."""
    try:
        if isinstance(dfm, pl.DataFrame) and dfm.height > 0:
            if name in dfm.columns:
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
    If engine doesn't provide 'turnover', reconstruct it:
    - Drift previous weights with realized returns up to next rebalance date.
    - Turnover_k = 0.5 * || w_target(k) - w_pre_trade(k) ||_1
    """
    # 1) Prefer engine-provided turnover if present
    to_obj = bt.get("turnover")
    if to_obj is not None:
        try:
            if isinstance(to_obj, pl.DataFrame) and "turnover" in to_obj.columns:
                dates = (
                    to_obj.get_column("date").to_list()
                    if "date" in to_obj.columns
                    else list(range(to_obj.height))
                )
                vals = to_obj.get_column("turnover").to_numpy()
                return dates, np.asarray(vals, float)
            if isinstance(to_obj, pd.DataFrame) and "turnover" in to_obj.columns:
                dates = (
                    to_obj["date"].tolist()
                    if "date" in to_obj.columns
                    else list(range(len(to_obj)))
                )
                return dates, np.asarray(to_obj["turnover"].values, float)
            arr = np.asarray(to_obj, float).ravel()
            if arr.size > 0:
                rb_dates = bt.get("rebalance_dates")
                dates = (
                    list(rb_dates)[-arr.size :]
                    if rb_dates is not None
                    else list(bt["dates"][-arr.size :])
                )
                return dates, arr
        except Exception:
            pass

    # 2) Reconstruct via drift between rebalances
    rb_dates = list(bt.get("rebalance_dates", []))
    W_reb = np.asarray(bt.get("weights", []), float)  # (K, N)
    tick = list(bt.get("tickers", []))
    if W_reb.size == 0 or len(rb_dates) != W_reb.shape[0] or df_wide is None:
        return [], np.array([], float)

    have = set(df_wide.columns)
    miss = [t for t in tick if t not in have]
    if miss:
        df_wide = df_wide.with_columns(**{c: pl.lit(0.0, dtype=pl.Float64) for c in miss})
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
# Data handoff from previous pages
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
# Sidebar — core params
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

    if alloc_kind == "Equal-Weight":
        st.info(
            "Equal-Weight is constant over time by design. Pick Risk Parity / HRP / Min-Var to see changing weights & turnover."
        )

    w_min = st.number_input("w_min", 0.0, 1.0, 0.0, 0.01)
    w_max = st.number_input("w_max", 0.0, 1.0, 0.2, 0.01)
    if N * w_min > 1.0 or N * w_max < 1.0:
        w_min = min(w_min, 1.0 / max(N, 1))
        w_max = max(w_max, 1.0 / max(N, 1))
        st.info(f"Box adjusted for feasibility: w_min≤{1.0 / max(N, 1):.4f}≤w_max")

    st.caption("Covariance")
    cov_estimator = st.selectbox("Covariance estimator", ["Sample", "EWMA"], index=0)
    ewma_lambda = st.slider("EWMA λ", 0.80, 0.995, 0.97, 0.005)


# ─────────────────────────────────────────────────────────────────────
# Allocator factory (no turnover budget here)
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


def _get_cov(win: pl.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.eye(0)
    R = win.select(cols).to_numpy()
    if R.size == 0:
        return np.eye(len(cols)) * 1e-4
    S = _cov_ewma(R, lam=float(ewma_lambda)) if cov_estimator == "EWMA" else np.cov(R, rowvar=False)
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
            w = pgd_box_simplex_l2(
                mu_w, Sigma_w, gamma=100.0, w_min=w_min, w_max=w_max, lam_turnover=0.0
            )
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
            w = hrp_safe(
                hrp_func=hrp_weights,
                cov=Sigma_w,
                method="ward",
                optimal=True,
                w_min=w_min,
                w_max=w_max,
            )
            return project_to_box_simplex(w, w_min, w_max)

    elif kind == "Min-TE (to Bench)":

        def base_alloc(win: pl.DataFrame) -> np.ndarray:
            cols = [c for c in win.columns if c != "date"]
            Sigma_w = _get_cov(win, cols)
            w_bench = np.full(len(cols), 1.0 / max(len(cols), 1))
            mu_eff = 2.0 * (Sigma_w @ w_bench)  # proxy for tracking-error min
            w = pgd_box_simplex_l2(
                mu_eff, Sigma_w, gamma=1.0, w_min=w_min, w_max=w_max, lam_turnover=0.0
            )
            return project_to_box_simplex(w, w_min, w_max)

    else:

        def base_alloc(win: pl.DataFrame) -> np.ndarray:
            n = win.width - 1
            w = np.ones(n, dtype=float) / max(n, 1)
            return project_to_box_simplex(w, w_min, w_max)

    return base_alloc


# ─────────────────────────────────────────────────────────────────────
# Engine wrapper (auto-detects param names; attaches diagnostics)
# ─────────────────────────────────────────────────────────────────────
def _run_engine(df_wide: pl.DataFrame) -> dict[str, Any]:
    """
    Call the engine with a recording allocator:
    - captures every window end-date and computed weights
    - if engine returns no/constant weights, we override with recorded ones
    - attaches diagnostics (_alloc_hook/_ret_hook, recorder stats)
    """
    # --- 1) build base allocator ---
    base_alloc = make_allocator(alloc_kind)

    class _Recorder:
        def __init__(self):
            self.dates: list = []
            self.weights: list[np.ndarray] = []
            self.calls: int = 0

        def __call__(self, win: pl.DataFrame) -> np.ndarray:
            self.calls += 1
            # window end-date = last date of the window
            try:
                d_last = win.get_column("date")[-1]
            except Exception:
                d_last = None
            w = np.asarray(base_alloc(win), float).ravel()
            # store normalized
            s = float(np.sum(w))
            w = w / s if s > 1e-12 else w
            self.dates.append(d_last)
            self.weights.append(w)
            return w

        def as_array(self) -> np.ndarray:
            if not self.weights:
                return np.zeros((0, 0), float)
            # pad ragged just in case (shouldn’t happen)
            Nmax = max(len(w) for w in self.weights)
            W = np.zeros((len(self.weights), Nmax), float)
            for i, w in enumerate(self.weights):
                W[i, : len(w)] = w
            return W

    rec = _Recorder()

    # --- 2) resolve engine signature & kwargs (same logic as before) ---
    bench_w = np.full(N, 1.0 / max(N, 1))
    df_arg = df_wide.select(["date", *tickers]).sort("date")
    sig = inspect.signature(backtest_rebalanced)
    kw: dict[str, Any] = {
        "lookback": int(lookback),
        "rebalance_freq": rebalance_freq,
        "cost_bps": float(cost_bps),
        "bench_weights": bench_w,
    }
    # returns arg
    for rn in ["df_ret_wide", "returns_wide", "df_returns", "returns", "R_wide", "data"]:
        if rn in sig.parameters:
            kw[rn] = df_arg
            ret_hook = rn
            break
    else:
        kw["df_ret_wide"] = df_arg
        ret_hook = "fallback(df_ret_wide)"

    # allocator arg
    for an in ["allocator", "allocator_factory", "weights_func", "strategy"]:
        if an in sig.parameters:
            kw[an] = (lambda: rec) if an == "allocator_factory" else rec
            alloc_hook = an
            break
    else:
        kw["allocator"] = rec
        alloc_hook = "fallback(allocator)"

    # --- 3) run engine ---
    bt = backtest_rebalanced(**kw)
    if not isinstance(bt, dict):
        bt = dict(bt)

    # --- 4) attach diagnostics & recorder dumps ---
    bt["_alloc_hook"] = alloc_hook
    bt["_ret_hook"] = ret_hook
    bt["_alloc_calls"] = rec.calls

    # engine weights
    W_eng = np.asarray(bt.get("weights", []), float)
    # recorded weights
    W_rec = rec.as_array()

    def _is_constant_panel(W: np.ndarray) -> bool:
        if W.ndim != 2 or W.size == 0 or W.shape[0] < 2:
            return True
        dif = np.max(np.abs(W[1:] - W[:-1]), axis=1)
        return float(np.max(dif)) <= 1e-12

    # If engine didn’t store weights OR panel is constant but recorder shows variation → override
    if (W_eng.ndim != 2 or W_eng.size == 0 or _is_constant_panel(W_eng)) and (
        W_rec.ndim == 2 and W_rec.size > 0 and not _is_constant_panel(W_rec)
    ):
        bt["weights"] = W_rec
        # adopt recorder dates as rebalance_dates if sensible
        if rec.dates and len(rec.dates) == W_rec.shape[0]:
            bt["rebalance_dates"] = rec.dates
        # keep original tickers if present; else assume current ticker list
        bt.setdefault("tickers", tickers)

    # Always attach recorded series for transparency
    bt["_weights_recorded"] = W_rec
    bt["_rebalance_dates_recorded"] = rec.dates

    return bt


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
    key=next_key("baseline-metrics"),
)

# Weights heatmap
st.plotly_chart(
    plot_weights_heatmap(
        base_bt["dates"], base_bt["tickers"], base_bt["weights"], title="Baseline · Weights"
    ),
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

# Rebalance diagnostics
with st.expander("Rebalance diagnostics", expanded=False):
    st.write(f"Allocator hook used by engine: **{base_bt.get('_alloc_hook', '?')}**")
    st.write(f"Returns hook used by engine: **{base_bt.get('_ret_hook', '?')}**")
    st.write(f"Allocator calls recorded: **{base_bt.get('_alloc_calls', 'n/a')}**")

    W = np.asarray(base_bt.get("weights", []), float)
    if W.ndim == 2 and W.size > 0:
        diffs = np.max(np.abs(W[1:] - W[:-1]), axis=1) if W.shape[0] > 1 else np.array([])
        uniq_rows = int(1 + np.sum(diffs > 1e-10)) if diffs.size else 1
        st.write(f"Rebalances (K): {W.shape[0]}")
        st.write(f"Unique weight rows: {uniq_rows}")
        if diffs.size:
            st.write(
                f"Max Δw per step — mean: {float(np.mean(diffs)):.8f}, max: {float(np.max(diffs)):.8f}"
            )
        if diffs.size and 0 < float(np.max(diffs)) < 5e-3:
            st.info(
                "Weights move is very small (<0.5%). If the heatmap looks flat, "
                "try a tighter color clamp in the compare heatmap (zmax_abs≈0.01)."
            )
    else:
        st.write("No weights captured from engine.")

# Data sanity checks (detect degenerate inputs)
with st.expander("Data sanity checks", expanded=False):
    try:
        cols = [c for c in df_base.columns if c != "date"]
        X = np.nan_to_num(df_base.select(cols).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        T, M = X.shape
        variances = np.var(X, axis=0, ddof=1) if T > 1 else np.zeros(M)
        zero_var = int(np.sum(variances < 1e-12))
        st.write(f"Observations: {T}, Assets: {M}")
        st.write(f"Assets with ~zero variance: {zero_var} / {M}")

        if M >= 2 and T > 2:
            C = np.corrcoef(X.T)
            np.fill_diagonal(C, np.nan)
            offdiag = C[~np.isnan(C)]
            mean_off = float(np.nanmean(offdiag)) if offdiag.size else float("nan")
            max_off = float(np.nanmax(offdiag)) if offdiag.size else float("nan")
            st.write(f"Mean off-diagonal corr: {mean_off:.3f}, max: {max_off:.3f}")

        if T >= 40:
            mid = T // 2
            S1 = np.cov(X[:mid], rowvar=False)
            S2 = np.cov(X[mid:], rowvar=False)
            diff = np.max(np.abs(S1 - S2))
            st.write(f"Max |Σ1-Σ2| across halves: {float(diff):.6f}")

        if zero_var == M:
            st.warning("All asset variances are ~0. Returns look degenerate (all zeros).")
        elif zero_var > 0:
            st.info("Some assets have ~0 variance. Optimizers may collapse to 1/N.")
    except Exception as e:
        st.write(f"Sanity checks skipped: {e}")


# ─────────────────────────────────────────────────────────────────────
# Scenarios — settings (sidebar) + run + comparisons
# ─────────────────────────────────────────────────────────────────────

# Sidebar controls for scenarios
with st.sidebar:
    st.markdown("---")
    st.header("Scenario settings")

    B = st.number_input(
        "Bootstrap paths (B)",
        0,
        500,
        0,
        1,
        help="If B=0: original chronology. If B>0: B synthetic paths via block bootstrap.",
    )
    block = st.number_input("Block length (days)", 2, 252, 10, 1)
    seed = st.number_input("Seed", 0, 100_000, 42, 1)

    st.caption("Shocks (applied to returns)")
    mean_shift_bps = st.number_input(
        "Mean shift (bps/day)",
        -1000.0,
        1000.0,
        0.0,
        1.0,
        help="Constant drift added to every asset per day.",
    )
    cov_scale = st.slider(
        "Vol scale (cov_scale)",
        0.10,
        3.00,
        1.00,
        0.05,
        help="Scale cross-sectional deviations from mean (1.0 = no change).",
    )
    crash_enable = st.checkbox("One-day crash", value=False)
    crash_day = st.number_input(
        "Crash day index (0-based)", 0, max(1, df_ret_wide.height - 1), 0, 1
    )
    crash_drop_bps = st.number_input(
        "Crash size (bps)", -5000.0, 5000.0, -500.0, 10.0, help="e.g., −500 bps = −5% one-day gap."
    )

# Build scenario list only if there is a real shock/config difference
cfgs: list[ScenarioConfig] = []
mean_shift = (mean_shift_bps / 10_000.0) if abs(mean_shift_bps) > 1e-12 else None
crash_tuple = (int(crash_day), crash_drop_bps / 10_000.0) if crash_enable else None
has_changes = (mean_shift is not None) or (float(cov_scale) != 1.0) or crash_enable or (int(B) > 0)

if has_changes:
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
    st.info(
        "No active shock parameters — Baseline already shown above. Adjust shocks to run scenarios."
    )


# Helper to flatten 1-row metrics frames
def _flatten_metrics_row(m: pl.DataFrame | pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        if isinstance(m, pl.DataFrame):
            if m.height == 1:
                for k, v in m.row(0, named=True).items():
                    try:
                        fv = float(v)
                        if np.isfinite(fv):
                            out[str(k)] = fv
                    except Exception:
                        pass
            elif {"metric", "value"}.issubset(set(m.columns)):
                for k, v in zip(m.get_column("metric").to_list(), m.get_column("value").to_list()):
                    try:
                        fv = float(v)
                        if np.isfinite(fv):
                            out[str(k)] = fv
                    except Exception:
                        pass
        elif isinstance(m, pd.DataFrame):
            if len(m) == 1:
                for k, v in m.iloc[0].to_dict().items():
                    try:
                        fv = float(v)
                        if np.isfinite(fv):
                            out[str(k)] = fv
                    except Exception:
                        pass
            elif {"metric", "value"}.issubset(set(m.columns)):
                for _, r in m.iterrows():
                    try:
                        fv = float(r["value"])
                        if np.isfinite(fv):
                            out[str(r["metric"])] = fv
                    except Exception:
                        pass
    except Exception:
        pass
    return out


# Run scenarios (only alternative vs baseline)
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

    # Metrics comparison table
    rows: list[dict] = []
    for res in results:
        m = res.get("metrics", None)
        flat = _flatten_metrics_row(m) if m is not None else {}
        rows.append({"Scenario": res.get("name", "Scenario")} | flat)

    if rows:
        df_comp = pl.DataFrame(rows)
        st.dataframe(df_comp.to_pandas(), width="stretch", key=next_key("sc-metrics-table"))

        # ΔCAGR vs baseline, if present
        base_cagr = _extract_metric_scalar(base_m, "CAGR", default=np.nan)
        if np.isfinite(base_cagr) and "CAGR" in df_comp.columns:
            fig_delta = plot_metric_delta_bars(
                df_comp.to_pandas(), baseline_value=base_cagr, metric_col="CAGR"
            )
            st.plotly_chart(fig_delta, width="stretch", key=next_key("delta-cagr"))

    # Detailed charts per scenario
    for idx, res in enumerate(results):
        bt = res.get("bt", {})
        sc_name = str(res.get("name", f"Scenario {idx + 1}"))
        sec_prefix = f"sc-{idx}-{sc_name.replace(' ', '').lower()}"

        st.subheader(f"Scenario: {sc_name}")

        # Equity & drawdown comparisons
        st.plotly_chart(
            plot_equity_compare(
                base_bt["dates"],
                base_bt["equity"],
                bt.get("equity", []),
                name_a="Baseline",
                name_b=sc_name,
            ),
            width="stretch",
            key=next_key(f"{sec_prefix}-equity-compare"),
        )
        st.plotly_chart(
            plot_drawdown_compare(
                base_bt["dates"],
                base_bt["equity"],
                bt.get("equity", []),
                name_a="Baseline",
                name_b=sc_name,
            ),
            width="stretch",
            key=next_key(f"{sec_prefix}-dd-compare"),
        )

        # Weights delta heatmap (Scenario vs Baseline)
        try:
            w_a = _to_numpy_2d(base_bt.get("weights", []))
            w_b = _to_numpy_2d(bt.get("weights", np.zeros_like(w_a)))
            fig_wcmp = plot_weights_compare_heatmap(
                dates=base_bt["dates"],
                tickers=base_bt["tickers"],
                weights_a=w_a,
                weights_b=w_b,
                name_a="Baseline",
                name_b=sc_name,
                mode="delta",
                # If moves are tiny, a tighter clamp reveals structure
                zmax_abs=(0.01 if alloc_kind in {"Risk Parity", "HRP"} else 0.05),
            )
            st.plotly_chart(fig_wcmp, width="stretch", key=next_key(f"{sec_prefix}-weights-delta"))
        except Exception as e:
            st.caption(f"Weight comparison heatmap skipped: {e}")


# ─────────────────────────────────────────────────────────────────────
# Optional blocks: Beta-correlated shock, Historical slice, Tornado
# ─────────────────────────────────────────────────────────────────────

# --- Sidebar controls for optional analyses ---
with st.sidebar:
    st.markdown("---")
    st.header("Beta-correlated shock")
    use_beta = st.checkbox("Enable beta-correlated shock", value=False)
    idx_opts = [c for c in df_base.columns if c != "date"]
    index_col = st.selectbox("Index column", options=idx_opts, index=0, disabled=not use_beta)
    index_move = st.number_input(
        "Index move (e.g., -0.05)", -0.50, 0.50, -0.05, 0.01, disabled=not use_beta
    )
    beta_lb = st.number_input("Beta lookback (days)", 60, 1000, 252, 10, disabled=not use_beta)

    st.markdown("---")
    st.header("Historical slice")
    use_hist = st.checkbox("Enable historical slice", value=False)
    c1, c2 = st.columns(2)
    with c1:
        hist_start = st.text_input("Start (YYYY-MM-DD)", "2020-02-15", disabled=not use_hist)
    with c2:
        hist_end = st.text_input("End (YYYY-MM-DD)", "2025-10-15", disabled=not use_hist)

    st.markdown("---")
    st.header("Tornado sensitivity")
    do_tornado = st.checkbox("Compute tornado sensitivity (±δ per asset)", value=False)
    delta = st.number_input(
        "Shock per asset (daily return)", -0.20, 0.20, 0.02, 0.01, disabled=not do_tornado
    )


def _rolling_beta_last_window(
    df_wide: pl.DataFrame, index_col: str, lookback: int
) -> dict[str, float]:
    """CAPM betas in the last window vs chosen index; NaN-safe."""
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


# Beta-correlated shock
if use_beta:
    st.markdown("---")
    st.subheader("Beta-correlated shock")
    try:
        betas = _rolling_beta_last_window(df_base, index_col=index_col, lookback=int(beta_lb))
        shock_map = {asset: beta * float(index_move) for asset, beta in betas.items()}
        df_beta = apply_shock_map_to_wide(df_base, shock_map)
        bt_beta = _run_engine(df_beta)
        st.plotly_chart(
            equity_and_drawdown(
                bt_beta["dates"],
                bt_beta["equity"],
                title=f"Beta shock: {index_col} move {index_move:+.1%}",
            ),
            width="stretch",
            key=next_key("beta-ed"),
        )
        st.dataframe(_safe_metrics(bt_beta).to_pandas(), width="stretch")
    except Exception as e:
        st.info(f"Beta shock skipped: {e}")

# Historical slice
if use_hist:
    st.markdown("---")
    st.subheader("Historical slice (replay)")
    try:
        df_slice = historical_slice_returns(df_base, hist_start, hist_end, tickers=tickers)
        bt_slice = _run_engine(df_slice)
        st.plotly_chart(
            equity_and_drawdown(
                bt_slice["dates"],
                bt_slice["equity"],
                title=f"Historical slice {hist_start} → {hist_end}",
            ),
            width="stretch",
            key=next_key("hist-ed"),
        )
        st.dataframe(_safe_metrics(bt_slice).to_pandas(), width="stretch")
    except Exception as e:
        st.info(f"Historical slice skipped: {e}")

# ─────────────────────────────────────────────────────────────────────
# Tornado sensitivity — engine-first; equal-weight fallback (always renders)
# ─────────────────────────────────────────────────────────────────────


def _cagr_from_portfolio_returns(rp: np.ndarray) -> float:
    """Compute CAGR from a daily return series."""
    rp = np.asarray(rp, dtype=float)
    if rp.size < 2:
        return float("nan")
    gross = float(np.prod(1.0 + rp))
    return gross ** (252 / max(rp.size, 1)) - 1.0


def _cagr_equal_weight(df_wide: pl.DataFrame) -> float:
    """Equal-weight portfolio metric; independent from engine."""
    cols = [c for c in df_wide.columns if c != "date"]
    if not cols:
        return float("nan")
    X = np.nan_to_num(df_wide.select(cols).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    rp = X.mean(axis=1)
    return _cagr_from_portfolio_returns(rp)


def _try_engine_cagr(df_wide: pl.DataFrame) -> float:
    """Attempt CAGR via engine; return NaN if unusable (to enable fallback)."""
    try:
        bt = _run_engine(df_wide)
        eq = np.asarray(bt.get("equity", []), float)
        if eq.size >= 2:
            rp = eq[1:] / eq[:-1] - 1.0
            return _cagr_from_portfolio_returns(rp)
    except Exception:
        pass
    return float("nan")


if do_tornado:
    st.markdown("---")
    st.subheader("One-at-a-time sensitivity (Tornado)")
    try:
        # Base metric: engine → fallback to equal-weight
        base_cagr = _extract_metric_scalar(base_m, "CAGR", default=np.nan)
        if not np.isfinite(base_cagr):
            base_cagr = _try_engine_cagr(df_base)
        if not np.isfinite(base_cagr):
            base_cagr = _cagr_equal_weight(df_base)

        sens_rows = []
        engine_effective = False

        for tk in tickers:
            # Down shock
            df_down = apply_shock_map_to_wide(df_base, {tk: -abs(float(delta))})
            met_down = _try_engine_cagr(df_down)
            if not np.isfinite(met_down):
                met_down = _cagr_equal_weight(df_down)
            else:
                engine_effective = True

            # Up shock
            df_up = apply_shock_map_to_wide(df_base, {tk: +abs(float(delta))})
            met_up = _try_engine_cagr(df_up)
            if not np.isfinite(met_up):
                met_up = _cagr_equal_weight(df_up)
            else:
                engine_effective = True

            sens_rows.append(
                {
                    "asset": tk,
                    "metric": "CAGR",
                    "base": base_cagr,
                    "down": met_down,
                    "up": met_up,
                }
            )

        df_sens = pd.DataFrame(sens_rows)
        fig_tornado = plot_tornado_sensitivity(
            df_sens,
            metric_label="CAGR",
            down_label="Down",
            up_label="Up",
        )
        st.plotly_chart(fig_tornado, width="stretch", key=next_key("tornado"))

        if not engine_effective:
            st.caption(
                "Tornado shown using **engine-agnostic fallback (equal-weight)** "
                "because the engine did not reflect shocks. Check the diagnostics expander above."
            )

        with st.expander("Download sensitivity (CSV)", expanded=False):
            csv_bytes = df_sens.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download",
                data=csv_bytes,
                file_name="tornado_sensitivity.csv",
                mime="text/csv",
                key=next_key("tornado-download"),
            )
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
    # Scenario metrics (if table was created in Part 2)
    if "df_comp" in locals() and isinstance(df_comp, pl.DataFrame) and df_comp.height > 0:
        st.download_button(
            "Download scenario metrics (CSV)",
            df_comp.write_csv(),
            file_name="scenario_metrics.csv",
            mime="text/csv",
            key=next_key("dl-scenario-metrics"),
        )
except Exception:
    pass
