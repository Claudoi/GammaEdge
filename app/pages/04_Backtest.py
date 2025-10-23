# app/pages/04_Backtest.py
from __future__ import annotations

import contextlib

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
# Add repository root path for local imports (same as in 03_Optimizer)
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# --- backtest core ---
from portfolio.backtest import attribution as bt_attr
from portfolio.backtest import metrics as bt_metrics
from portfolio.backtest.engine import backtest_rebalanced

# --- optim helpers (for allocators) ---
from portfolio.core.utils import ensure_psd, hrp_safe, project_to_box_simplex
from portfolio.optim.hrp import hrp_weights
from portfolio.optim.mean_variance import pgd_box_simplex_l2
from portfolio.optim.risk_parity import risk_parity

# --- visualization ---
from portfolio.viz.plot_utils import (
    equity_and_drawdown,
    plot_brinson_cumulative,
    plot_drawdown,
    plot_equity,
    plot_group_contrib,
    plot_group_contrib_area,
    plot_top_contributors,
    plot_tracking_error,
    plot_turnover,
    plot_weights_heatmap,
)


# ─────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────
def _to_pandas(df: Any):
    """Safely convert Polars or other table objects to pandas."""
    try:
        return df.to_pandas()
    except Exception:
        return df


def cov_ewma(R: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Compute EWMA covariance matrix (robust, PSD-enforced)."""
    if R.size == 0:
        return np.eye(0)
    T, N = R.shape
    S = np.zeros((N, N), dtype=float)
    w = 0.0
    mu = np.nanmean(R, axis=0)
    for t in range(T):
        x = (R[t] - mu).reshape(1, -1)
        S = lam * S + (1 - lam) * (x.T @ x)
        w = lam * w + (1 - lam)
    S = S / max(w, 1e-12)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    return ensure_psd(S, eps=1e-10, clip=True)


def enforce_turnover(
    prev_w: np.ndarray | None,
    new_w: np.ndarray,
    max_to: float = 0.10,
    band: float = 0.01,
    w_min: float = 0.0,
    w_max: float = 1.0,
) -> np.ndarray:
    """
    Enforce turnover and rebalancing rules.
    - Ignores small median weight changes (band threshold).
    - Limits portfolio turnover (L1/2) to 'max_to' budget.
    """
    if prev_w is None or prev_w.size == 0:
        return project_to_box_simplex(new_w, w_min, w_max)
    # Skip rebalance if changes are minor
    if np.median(np.abs(new_w - prev_w)) < band:
        return prev_w
    # Compute turnover
    to = 0.5 * np.sum(np.abs(new_w - prev_w))
    if to <= max_to:
        return project_to_box_simplex(new_w, w_min, w_max)
    # Rescale new weights to meet turnover budget
    lam = min(1.0, max_to / (to + 1e-12))
    w_lim = prev_w + lam * (new_w - prev_w)
    return project_to_box_simplex(w_lim, w_min, w_max)


def min_te_to_bench(
    Sigma: np.ndarray, w_bench: np.ndarray, w_min: float, w_max: float
) -> np.ndarray:
    """
    Minimize Tracking Error vs benchmark:
        min (w - w_b)' Σ (w - w_b)
    Equivalent to min w'Σw - 2 w'(Σ w_b) + const
    -> solved via L2 PGD with effective μ = 2 Σ w_b.
    """
    if Sigma.size == 0:
        return project_to_box_simplex(w_bench.copy(), w_min, w_max)
    mu_eff = 2.0 * (Sigma @ w_bench)
    w = pgd_box_simplex_l2(mu_eff, Sigma, gamma=1.0, w_min=w_min, w_max=w_max, lam_turnover=0.0)
    return project_to_box_simplex(w, w_min, w_max)


def metrics_bootstrap(
    bt: dict[str, Any], B: int = 200, block: int = 10, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Bootstrap (block-based) on daily returns to compute CI for metrics.
    Requires 'equity' or 'equity_returns' in the backtest dictionary.
    """
    if bt.get("equity_returns") is not None:
        r = np.array(bt["equity_returns"], dtype=float)
    else:
        eq = np.array(bt["equity"], dtype=float)
        r = np.diff(eq) / eq[:-1]
    T = len(r)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(B):
        idx = []
        while len(idx) < T:
            start = rng.integers(0, max(T - block, 1))
            idx.extend(range(start, min(start + block, T)))
        idx = np.array(idx[:T])
        s = r[idx]
        eqb = np.cumprod(1 + s)
        cagr = eqb[-1] ** (252 / max(len(s), 1)) - 1
        sharpe = (np.mean(s) / (np.std(s) + 1e-12)) * np.sqrt(252)
        mdd = 1 - (eqb / np.maximum.accumulate(eqb)).min()
        rows.append((cagr, sharpe, mdd))
    dfb = pl.DataFrame(rows, schema=["CAGR", "Sharpe", "MaxDD"])
    q = dfb.select(
        [
            pl.col("CAGR").quantile([0.05, 0.5, 0.95]).alias("CAGR_q"),
            pl.col("Sharpe").quantile([0.05, 0.5, 0.95]).alias("Sharpe_q"),
            pl.col("MaxDD").quantile([0.05, 0.5, 0.95]).alias("MaxDD_q"),
        ]
    )
    return dfb, q


# ─────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Backtest", layout="wide")
st.title("📊 Backtest")

# ─────────────────────────────────────────────────────────────────────
# Input validation from previous pages (01/02/03)
# ─────────────────────────────────────────────────────────────────────
if "returns_wide" not in st.session_state:
    st.warning("Missing return data. Go to **01_Data** and generate `returns_wide` first.")
    st.stop()

df_ret_wide: pl.DataFrame = st.session_state["returns_wide"]
tickers = [c for c in df_ret_wide.columns if c != "date"]
N = len(tickers)
if N == 0 or df_ret_wide.height < 10:
    st.error("Dataset too small for backtesting.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────
# Sidebar – parameters
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    rebalance_freq = st.selectbox("Rebalance frequency", ["1mo", "1w", "3mo", "6mo"], index=0)
    lookback = st.number_input(
        "Lookback (periods)", min_value=30, max_value=2000, value=252, step=10
    )
    cost_bps = st.number_input(
        "Transaction cost (bps per turnover)", min_value=0.0, max_value=100.0, value=2.0, step=0.5
    )

    st.markdown("---")
    st.subheader("Allocator")

    alloc_kind = st.selectbox(
        "Strategy",
        ["Equal-Weight", "Min-Var (L2 PGD)", "Risk Parity", "HRP", "Min-TE (to Bench)"],
        index=0,
        help="Portfolio construction method used at each rebalance window.",
    )

    # Simple box constraints
    w_min = st.number_input("w_min", 0.0, 1.0, 0.0, 0.01)
    w_max = st.number_input("w_max", 0.0, 1.0, 0.2, 0.01)
    if N * w_min > 1.0 or N * w_max < 1.0:
        w_min = min(w_min, 1.0 / max(N, 1))
        w_max = max(w_max, 1.0 / max(N, 1))
        st.info(f"Box adjusted for feasibility: w_min≤{1.0/max(N,1):.4f}≤w_max")

    st.markdown("---")
    st.subheader("Covariance Estimation")
    cov_estimator = st.selectbox("Covariance estimator", ["Sample", "EWMA"], index=0)
    ewma_lambda = st.slider("EWMA λ", min_value=0.80, max_value=0.995, value=0.97, step=0.005)

    st.markdown("---")
    st.subheader("Rebalancing Control")
    use_to_budget = st.checkbox("Limit turnover (budget)", value=True)
    max_turnover = st.slider("Max turnover per rebalance", 0.0, 0.50, 0.10, 0.01)
    band_eps = st.slider("Band threshold (median |Δw|)", 0.0, 0.05, 0.01, 0.001)

    st.markdown("---")
    st.subheader("Hyperparameter Grid Search")
    do_grid = st.checkbox("Run grid search", value=False)
    grid_lookbacks = st.text_input("Lookback values", "126,252")
    grid_costs = st.text_input("Transaction costs (bps)", "0,2,5")


# ─────────────────────────────────────────────────────────────────────
# Allocator factory (window -> weights)
# ─────────────────────────────────────────────────────────────────────
def make_allocator(kind: str) -> Callable[[pl.DataFrame], np.ndarray]:
    """
    Returns a closure that maps a rolling window of returns to weights.
    Integrates: covariance estimator, turnover control, and strategy logic.
    """
    prev = {"w": None}  # persistent state for turnover constraint

    def get_cov(win: pl.DataFrame, cols: list[str]) -> np.ndarray:
        if not cols:
            return np.eye(0)
        R = win.select(cols).to_numpy()
        if R.size == 0:
            return np.eye(len(cols)) * 1e-4
        if cov_estimator == "EWMA":
            S = cov_ewma(R, lam=float(ewma_lambda))
        else:
            S = np.cov(R, rowvar=False)
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        return ensure_psd(S, eps=1e-10, clip=True)

    # Define base allocator per strategy
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
            Sigma_w = get_cov(win, cols)
            w = pgd_box_simplex_l2(
                mu_w, Sigma_w, gamma=100.0, w_min=w_min, w_max=w_max, lam_turnover=0.0
            )
            return project_to_box_simplex(w, w_min, w_max)

    elif kind == "Risk Parity":

        def base_alloc(win: pl.DataFrame) -> np.ndarray:
            cols = [c for c in win.columns if c != "date"]
            Sigma_w = get_cov(win, cols)
            try:
                w = risk_parity(Sigma_w, w_min=w_min, w_max=w_max)
            except Exception:
                w = np.ones(len(cols)) / max(len(cols), 1)
            return project_to_box_simplex(w, w_min, w_max)

    elif kind == "HRP":

        def base_alloc(win: pl.DataFrame) -> np.ndarray:
            cols = [c for c in win.columns if c != "date"]
            Sigma_w = get_cov(win, cols)
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
            Sigma_w = get_cov(win, cols)
            w_bench = np.full(len(cols), 1.0 / max(len(cols), 1))
            w = min_te_to_bench(Sigma_w, w_bench, w_min, w_max)
            return project_to_box_simplex(w, w_min, w_max)

    else:

        def base_alloc(win: pl.DataFrame) -> np.ndarray:
            n = win.width - 1
            w = np.ones(n, dtype=float) / max(n, 1)
            return project_to_box_simplex(w, w_min, w_max)

    # Add turnover control wrapper
    def alloc(win: pl.DataFrame) -> np.ndarray:
        w_new = base_alloc(win)
        if use_to_budget:
            w_final = enforce_turnover(
                prev["w"],
                w_new,
                max_to=float(max_turnover),
                band=float(band_eps),
                w_min=w_min,
                w_max=w_max,
            )
        else:
            w_final = w_new
        prev["w"] = w_final
        return w_final

    return alloc


allocator = make_allocator(alloc_kind)


# ─────────────────────────────────────────────────────────────────────
# Cached backtest execution
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_backtest(
    df_ret_wide,
    lookback,
    rebalance_freq,
    cost_bps,
    alloc_kind,
    w_min,
    w_max,
    cov_estimator,
    ewma_lambda,
    use_to_budget,
    max_turnover,
    band_eps,
):
    alloc = make_allocator(alloc_kind)
    # compute local number of assets (exclude 'date')
    n_cols = len([c for c in df_ret_wide.columns if c != "date"])
    return backtest_rebalanced(
        df_ret_wide=df_ret_wide,
        lookback=int(lookback),
        rebalance_freq=rebalance_freq,
        cost_bps=float(cost_bps),
        allocator=alloc,
        bench_weights=np.full(n_cols, 1.0 / max(n_cols, 1)),
    )


# Run backtest or grid depending on user setting
bt = None
if not do_grid:
    with st.spinner("Running backtest..."):
        bt = cached_backtest(
            df_ret_wide,
            int(lookback),
            rebalance_freq,
            float(cost_bps),
            alloc_kind,
            w_min,
            w_max,
            cov_estimator,
            float(ewma_lambda),
            use_to_budget,
            float(max_turnover),
            float(band_eps),
        )
    st.success("✅ Backtest executed.")

    # ─────────────────────────────────────────────────────────────────
    # Handoff to 05_Attribution (persist into session_state)
    # ─────────────────────────────────────────────────────────────────
    def _export_to_05(bt_obj, df_wide_obj):
        # normalize to Polars with datetime
        if isinstance(df_wide_obj, pd.DataFrame):
            df_pl = pl.from_pandas(df_wide_obj)
        elif isinstance(df_wide_obj, pl.DataFrame):
            df_pl = df_wide_obj
        else:
            st.error("`returns_wide/df_ret_wide` must be a Polars/Pandas DataFrame.")
            return

        if df_pl.schema.get("date") != pl.Datetime:
            df_pl = df_pl.with_columns(pl.col("date").cast(pl.Datetime))

        st.session_state["bt"] = bt_obj
        st.session_state["df_ret_wide"] = df_pl
        # keep a generic copy as well (so 05 can fallback)
        st.session_state["returns_wide"] = df_pl
        with contextlib.suppress(Exception):
            st.toast("Artifacts saved for 05_Attribution.", icon="💾")

    # auto-export now that bt exists
    _export_to_05(bt, st.session_state.get("returns_wide", df_ret_wide))

    # optional manual button to re-export on demand
    with st.sidebar:
        if st.button("Export to 05_Attribution"):
            _export_to_05(bt, st.session_state.get("returns_wide", df_ret_wide))
            st.success("Exported to session_state.")

# ─────────────────────────────────────────────────────────────────────
# Grid search mode (optional)
# ─────────────────────────────────────────────────────────────────────
if do_grid:
    st.info("Running grid search…")
    Ls = [int(s) for s in grid_lookbacks.split(",") if s.strip()]
    Cs = [float(s) for s in grid_costs.split(",") if s.strip()]
    rows = []
    total = max(len(Ls) * len(Cs), 1)
    prog = st.progress(0.0)
    k = 0
    for L in Ls:
        for C in Cs:
            alloc = make_allocator(alloc_kind)
            bt_ = backtest_rebalanced(
                df_ret_wide=df_ret_wide,
                lookback=int(L),
                rebalance_freq=rebalance_freq,
                cost_bps=float(C),
                allocator=alloc,
                bench_weights=np.full(N, 1.0 / max(N, 1)),
            )
            m = bt_metrics.compute_backtest_metrics(bt_)
            mp = _to_pandas(m)
            rows.append(
                {
                    "lookback": int(L),
                    "cost_bps": float(C),
                    "CAGR": float(mp.loc[0, "CAGR"]) if "CAGR" in mp.columns else np.nan,
                    "Sharpe": float(mp.loc[0, "Sharpe"]) if "Sharpe" in mp.columns else np.nan,
                    "MaxDD": float(mp.loc[0, "MaxDD"]) if "MaxDD" in mp.columns else np.nan,
                }
            )
            k += 1
            prog.progress(k / total)
    df_grid = pl.DataFrame(rows)
    st.subheader("🔎 Grid results")
    st.dataframe(_to_pandas(df_grid.sort("Sharpe", descending=True)), width="stretch")

# ─────────────────────────────────────────────────────────────────────
# Metrics + Bootstrap CI (if not running grid)
# ─────────────────────────────────────────────────────────────────────
if not do_grid:
    st.subheader("📈 Metrics")
    dfm = bt_metrics.compute_backtest_metrics(bt)
    tab1, tab2 = st.tabs(["Overview", "Bootstrap CI"])
    with tab1:
        st.dataframe(_to_pandas(dfm), width="stretch")
    with tab2:
        run_bs = st.checkbox("Run Bootstrap (200 reps)", value=False)
        if run_bs:
            with st.spinner("Bootstrapping metrics…"):
                dfb, q = metrics_bootstrap(bt, B=200, block=10, seed=42)
            st.write("Quantiles (5%, 50%, 95%)")
            st.dataframe(_to_pandas(q), width="stretch")

# ─────────────────────────────────────────────────────────────────────
# Main plots
# ─────────────────────────────────────────────────────────────────────
if not do_grid:
    st.subheader("📉 Equity & Drawdown")
    st.plotly_chart(
        equity_and_drawdown(bt["dates"], bt["equity"], title="Equity & Drawdown"),
        width="stretch",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_equity(bt["dates"], bt["equity"], title="Equity"), width="stretch")
    with col2:
        st.plotly_chart(plot_drawdown(bt["dates"], bt["equity"], title="Drawdown"), width="stretch")

    st.subheader("⚖️ Weights & Turnover")
    st.plotly_chart(
        plot_weights_heatmap(
            bt["dates"], bt["tickers"], bt["weights"], title="Weights (rebalance steps)"
        ),
        width="stretch",
    )
    # Turnover plot (per rebalance step)
    dates_w = bt.get("rebalance_dates", None)
    if dates_w is None:
        k = int(np.size(bt.get("turnover", [])))
        dates_w = bt["dates"][-k:] if k > 0 else []

    to_vals = np.asarray(bt.get("turnover", []), dtype=float)

    if len(dates_w) == to_vals.size and to_vals.size > 0:
        fig = plot_turnover(dates_w, to_vals, title="Turnover at Rebalance")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Turnover plot unavailable (length mismatch or empty series).")

    if bt.get("te_daily_proxy") is not None:
        st.plotly_chart(
            plot_tracking_error(bt["dates"], bt["te_daily_proxy"], title="Daily TE (proxy)"),
            width="stretch",
        )

# ─────────────────────────────────────────────────────────────────────
# Attribution (only when not running grid search)
# ─────────────────────────────────────────────────────────────────────
aln = None  # Prevent undefined variable if attribution fails

if not do_grid and bt is not None:
    st.subheader("📊 Attribution")

    # ────────────────────────────────
    # Basic / Asset-level attribution
    # ────────────────────────────────
    try:
        # 1) Align the date grid
        dates_bt = list(bt["dates"])
        df_ret_bt = df_ret_wide.filter(pl.col("date").is_in(dates_bt)).sort("date")
        df_ret_bt = df_ret_bt.unique(subset=["date"]).sort("date")

        # 2) Expand rebalancing weights to daily frequency
        W_reb = np.asarray(bt["weights"], dtype=float)
        rb_dates = list(bt.get("rebalance_dates", []))
        if W_reb.size == 0 or len(rb_dates) != W_reb.shape[0]:
            K = W_reb.shape[0]
            rb_dates = dates_bt[:: max(1, len(dates_bt) // max(1, K))][:K]

        daily_W = bt_attr.expand_rebalance_weights(
            dates=df_ret_bt.get_column("date").to_list(),
            rb_dates=rb_dates,
            W_reb=W_reb,
        )

        # 3) Align and compute contributions
        aln = bt_attr.align_returns_and_weights(df_ret_bt, daily_W)
        df_contrib_asset = bt_attr.contributions_by_asset(aln)
        df_top = bt_attr.top_contributors(bt=bt, df_ret_wide=df_ret_bt, top_n=10, sign="both")

        # 4) Plots
        st.plotly_chart(plot_top_contributors(df_top), width="stretch")
        df_bottom = (
            df_contrib_asset.group_by("ticker")
            .agg(pl.col("contrib").sum().alias("contrib_total"))
            .sort("contrib_total")
            .head(10)
        )
        st.plotly_chart(
            plot_top_contributors(df_bottom, title="Bottom Contributors"), width="stretch"
        )

    except Exception as e:
        st.info(f"Basic attribution not available: {e}")

    # ────────────────────────────────
    # Group / Sector attribution
    # ────────────────────────────────
    try:
        groups_map = {tk: "OTHER" for tk in bt["tickers"]}
        df_group_daily = bt_attr.contributions_by_group(aln, groups_map)
        df_group_total = (
            df_group_daily.group_by("group")
            .agg(
                [
                    pl.col("contrib").sum().alias("contrib_total"),
                    pl.col("weight").mean().alias("avg_weight"),
                ]
            )
            .sort("contrib_total", descending=True)
        )
        st.plotly_chart(plot_group_contrib(df_group_total), width="stretch")
        st.plotly_chart(plot_group_contrib_area(df_group_daily), width="stretch")
    except Exception as e:
        st.info(f"Group attribution not available: {e}")

    # ────────────────────────────────
    # Brinson–Fachler performance attribution
    # ────────────────────────────────
    try:
        N_assets = len(bt["tickers"])
        w_bench = np.full(N_assets, 1.0 / max(N_assets, 1))
        Wb_daily = np.tile(w_bench, (len(aln.dates), 1))
        groups_idx = list(range(N_assets))
        df_brinson = bt_attr.brinson_fachler_cumulative(
            aln=aln,
            bench_weights_daily=Wb_daily,
            groups_idx=groups_idx,
        )
        st.plotly_chart(plot_brinson_cumulative(df_brinson), width="stretch")
    except Exception as e:
        st.info(f"Brinson attribution not available: {e}")
