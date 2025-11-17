# app/pages/04_Backtest.py
from __future__ import annotations

# --- stdlib ---
import contextlib
import os
import sys
from collections.abc import Callable
from typing import Any

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
    show_plot,
)


# ─────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────
def _bt_key(tag: str) -> str:
    """Generate unique keys for Backtest plots to avoid duplicate IDs."""
    st.session_state.setdefault("_bt_key_seq", 0)
    st.session_state["_bt_key_seq"] += 1
    return f"bt-{tag}-{st.session_state['_bt_key_seq']}"


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
    if np.median(np.abs(new_w - prev_w)) < band:
        return prev_w
    to = 0.5 * np.sum(np.abs(new_w - prev_w))
    if to <= max_to:
        return project_to_box_simplex(new_w, w_min, w_max)
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
    Block bootstrap on daily returns to compute CI for metrics (CAGR, Sharpe, MaxDD).
    Returns:
      dfb: bootstrap samples (B x 3)
      q:   scalar quantiles table (3 x 4: metric, q05, q50, q95)
    """
    # 1) Build daily return series
    if bt.get("equity_returns") is not None:
        r = np.asarray(bt["equity_returns"], dtype=float)
    else:
        eq = np.asarray(bt["equity"], dtype=float)
        if eq.size < 2:
            return pl.DataFrame(schema=["CAGR", "Sharpe", "MaxDD"]), pl.DataFrame(
                schema=["metric", "q05", "q50", "q95"]
            )
        r = np.diff(eq) / eq[:-1]

    T = int(r.size)
    if T == 0 or B <= 0:
        return pl.DataFrame(schema=["CAGR", "Sharpe", "MaxDD"]), pl.DataFrame(
            schema=["metric", "q05", "q50", "q95"]
        )

    # 2) Bootstrap
    rng = np.random.default_rng(seed)
    rows: list[tuple[float, float, float]] = []
    blk = max(int(block), 1)
    for _ in range(int(B)):
        idx: list[int] = []
        while len(idx) < T:
            start = int(rng.integers(0, max(T - blk, 1)))
            idx.extend(range(start, min(start + blk, T)))
        s = r[np.asarray(idx[:T])]
        eqb = np.cumprod(1.0 + s)
        cagr = float(eqb[-1] ** (252.0 / max(len(s), 1)) - 1.0)
        vol = float(np.std(s) + 1e-12)
        sharpe = float((np.mean(s) / vol) * np.sqrt(252.0))
        mdd = float(1.0 - (eqb / np.maximum.accumulate(eqb)).min())
        rows.append((cagr, sharpe, mdd))

    # 3) Samples DataFrame
    dfb = pl.DataFrame(rows, schema=["CAGR", "Sharpe", "MaxDD"])

    # 4) Scalar quantiles
    def _q(df: pl.DataFrame, col: str, p: float) -> float:
        return float(df.select(pl.col(col).quantile(p)).item())

    q = pl.DataFrame(
        {
            "metric": ["CAGR", "Sharpe", "MaxDD"],
            "q05": [_q(dfb, "CAGR", 0.05), _q(dfb, "Sharpe", 0.05), _q(dfb, "MaxDD", 0.05)],
            "q50": [_q(dfb, "CAGR", 0.50), _q(dfb, "Sharpe", 0.50), _q(dfb, "MaxDD", 0.50)],
            "q95": [_q(dfb, "CAGR", 0.95), _q(dfb, "Sharpe", 0.95), _q(dfb, "MaxDD", 0.95)],
        }
    )

    return dfb, q


def _metric_from_df(m, name: str) -> float:
    """Robust metric extraction (Polars | pandas) -> float or NaN."""
    try:
        import polars as pl  # type: ignore[import]

        if isinstance(m, pl.DataFrame):
            if name in m.columns:
                return float(m.select(pl.col(name)).item())
            low = name.lower()
            for c in m.columns:
                if str(c).lower() == low:
                    return float(m.select(pl.col(c)).item())
    except Exception:
        pass
    try:
        import pandas as pd  # type: ignore[import]

        if isinstance(m, pd.DataFrame) and len(m.index) > 0:
            if name in m.columns:
                return float(m.at[m.index[0], name])
            low = name.lower()
            for c in m.columns:
                if str(c).lower() == low:
                    return float(m.at[m.index[0], c])
    except Exception:
        pass
    return float("nan")


def _metrics_safe(bt_obj: dict) -> dict[str, float]:
    """Compute metrics with robust extraction and a fallback if needed."""
    m = bt_metrics.compute_backtest_metrics(bt_obj)
    cagr = _metric_from_df(m, "CAGR")
    sharpe = _metric_from_df(m, "Sharpe")
    maxdd = _metric_from_df(m, "MaxDD")
    if not np.isfinite(cagr) or not np.isfinite(sharpe) or not np.isfinite(maxdd):
        eq = np.asarray(bt_obj.get("equity", []), dtype=float)
        if eq.size > 1:
            r = np.diff(eq) / eq[:-1]
            if r.size > 0:
                eqb = np.cumprod(1.0 + r)
                cagr_fb = float(eqb[-1] ** (252.0 / max(len(r), 1)) - 1.0)
                vol = float(np.std(r) + 1e-12)
                sharpe_fb = float((np.mean(r) / vol) * np.sqrt(252.0))
                maxdd_fb = float(1.0 - (eqb / np.maximum.accumulate(eqb)).min())
                cagr = cagr if np.isfinite(cagr) else cagr_fb
                sharpe = sharpe if np.isfinite(sharpe) else sharpe_fb
                maxdd = maxdd if np.isfinite(maxdd) else maxdd_fb
    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": float(maxdd)}


# --- Benchmark helpers ------------------------------------------------
def build_benchmark_weights(
    mode: str,
    T: int,
    tickers: list[str],
    W_portfolio_daily: np.ndarray | None = None,
) -> np.ndarray:
    """
    Returns Wb_daily with shape (T, N).
    mode: 'equal' | 'static_first_day'
      - equal: 1/N constant.
      - static_first_day: freeze day-0 portfolio weights and keep them static.
    """
    N = len(tickers)
    if mode == "equal":
        return np.tile(np.full(N, 1.0 / max(N, 1), dtype=float), (T, 1))
    elif mode == "static_first_day":
        if W_portfolio_daily is None:
            raise ValueError("static_first_day requires W_portfolio_daily")
        w0 = np.clip(W_portfolio_daily[0], 0.0, None)
        s = float(w0.sum())
        w0 = (w0 / s) if s > 0 else np.full(N, 1.0 / max(N, 1), dtype=float)
        return np.tile(w0, (T, 1))
    else:
        raise ValueError(f"Unknown benchmark mode: {mode}")


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
        st.info(f"Box adjusted for feasibility: w_min≤{1.0 / max(N, 1):.4f}≤w_max")

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
    st.subheader("Benchmark (for Brinson)")
    bench_mode = st.selectbox("Benchmark mode", ["equal", "static_first_day"], index=0)

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

    # Turnover control wrapper
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
    n_cols = len([c for c in df_ret_wide.columns if c != "date"])
    return backtest_rebalanced(
        df_ret_wide=df_ret_wide,
        lookback=int(lookback),
        rebalance_freq=rebalance_freq,
        cost_bps=float(cost_bps),
        allocator=alloc,
        bench_weights=np.full(n_cols, 1.0 / max(n_cols, 1)),
    )


# ─────────────────────────────────────────────────────────────────────
# Cached metrics computation for grid search (per combination)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_metrics_for_grid(
    df_ret_wide: pl.DataFrame,
    lookback: int,
    cost_bps: float,
    alloc_kind: str,
    rebalance_freq: str,
    w_min: float,
    w_max: float,
    cov_estimator: str,
    ewma_lambda: float,
    use_to_budget: bool,
    max_turnover: float,
    band_eps: float,
) -> dict[str, float]:
    alloc = make_allocator(alloc_kind)
    n_cols = len([c for c in df_ret_wide.columns if c != "date"])
    bt_ = backtest_rebalanced(
        df_ret_wide=df_ret_wide,
        lookback=int(lookback),
        rebalance_freq=rebalance_freq,
        cost_bps=float(cost_bps),
        allocator=alloc,
        bench_weights=np.full(n_cols, 1.0 / max(n_cols, 1)),
    )
    return _metrics_safe(bt_)


# Run backtest or grid depending on user setting
bt: dict[str, Any] | None = None
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

    # Handoff to 05_Attribution (persist into session_state)
    def _export_to_05(bt_obj, df_wide_obj):
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
        st.session_state["returns_wide"] = df_pl
        with contextlib.suppress(Exception):
            st.toast("Artifacts saved for 05_Attribution.", icon="💾")

    _export_to_05(bt, st.session_state.get("returns_wide", df_ret_wide))

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

    rows: list[dict[str, float]] = []
    total = max(len(Ls) * len(Cs), 1)
    prog = st.progress(0.0)
    k = 0

    for L in Ls:
        for C in Cs:
            mets = _cached_metrics_for_grid(
                df_ret_wide=df_ret_wide,
                lookback=int(L),
                cost_bps=float(C),
                alloc_kind=alloc_kind,
                rebalance_freq=rebalance_freq,
                w_min=w_min,
                w_max=w_max,
                cov_estimator=cov_estimator,
                ewma_lambda=float(ewma_lambda),
                use_to_budget=use_to_budget,
                max_turnover=float(max_turnover),
                band_eps=float(band_eps),
            )
            rows.append(
                {
                    "lookback": float(L),
                    "cost_bps": float(C),
                    "CAGR": mets["CAGR"],
                    "Sharpe": mets["Sharpe"],
                    "MaxDD": mets["MaxDD"],
                }
            )
            k += 1
            prog.progress(k / total)

    df_grid = (
        pl.from_dicts(rows)
        .with_columns(pl.col(["CAGR", "Sharpe", "MaxDD"]).cast(pl.Float64, strict=False))
        .with_columns(
            [
                pl.when(pl.col("CAGR").is_finite())
                .then(pl.col("CAGR"))
                .otherwise(None)
                .alias("CAGR"),
                pl.when(pl.col("Sharpe").is_finite())
                .then(pl.col("Sharpe"))
                .otherwise(None)
                .alias("Sharpe"),
                pl.when(pl.col("MaxDD").is_finite())
                .then(pl.col("MaxDD"))
                .otherwise(None)
                .alias("MaxDD"),
            ]
        )
    )

    st.subheader("🔎 Grid results")
    st.dataframe(_to_pandas(df_grid.sort("Sharpe", descending=True)), width="stretch")


# ─────────────────────────────────────────────────────────────────────
# Metrics + Bootstrap CI (if not running grid)
# ─────────────────────────────────────────────────────────────────────
if not do_grid and bt is not None:
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
if not do_grid and bt is not None:
    st.subheader("📉 Equity & Drawdown")
    show_plot(
        equity_and_drawdown(bt["dates"], bt["equity"], title="Equity & Drawdown"),
        key=_bt_key("equity-dd"),
    )

    col1, col2 = st.columns(2)
    with col1:
        show_plot(
            plot_equity(bt["dates"], bt["equity"], title="Equity"),
            key=_bt_key("equity"),
        )
    with col2:
        show_plot(
            plot_drawdown(bt["dates"], bt["equity"], title="Drawdown"),
            key=_bt_key("dd"),
        )

    st.subheader("⚖️ Weights & Turnover")
    show_plot(
        plot_weights_heatmap(
            bt["dates"], bt["tickers"], bt["weights"], title="Weights (rebalance steps)"
        ),
        key=_bt_key("weights-heatmap"),
    )

    dates_w = bt.get("rebalance_dates", None)
    if dates_w is None:
        k = int(np.size(bt.get("turnover", [])))
        dates_w = bt["dates"][-k:] if k > 0 else []

    to_vals = np.asarray(bt.get("turnover", []), dtype=float)
    if len(dates_w) == to_vals.size and to_vals.size > 0:
        fig = plot_turnover(dates_w, to_vals, title="Turnover at Rebalance")
        show_plot(fig, key=_bt_key("turnover"))
    else:
        st.info("Turnover plot unavailable (length mismatch or empty series).")

    if bt.get("te_daily_proxy") is not None:
        show_plot(
            plot_tracking_error(bt["dates"], bt["te_daily_proxy"], title="Daily TE (proxy)"),
            key=_bt_key("te-proxy"),
        )

# ─────────────────────────────────────────────────────────────────────
# Attribution (only when not running grid search)
# ─────────────────────────────────────────────────────────────────────
aln = None

if not do_grid and bt is not None:
    st.subheader("📊 Attribution")

    # 1) Align returns to bt dates and expand rebalance weights to daily
    try:
        dates_bt = list(bt["dates"])
        df_ret_bt = (
            df_ret_wide.filter(pl.col("date").is_in(dates_bt)).unique(subset=["date"]).sort("date")
        )

        W_reb = np.asarray(bt["weights"], dtype=float)
        rb_dates = list(bt.get("rebalance_dates", []))
        if W_reb.size == 0:
            raise ValueError("bt['weights'] is empty.")

        if len(rb_dates) != W_reb.shape[0]:
            K = W_reb.shape[0]
            step = max(len(dates_bt) // max(K, 1), 1)
            rb_dates = dates_bt[::step][:K]

        daily_W = bt_attr.expand_rebalance_weights(
            dates=df_ret_bt.get_column("date").to_list(),
            rb_dates=rb_dates,
            W_reb=W_reb,
        )

        aln = bt_attr.align_returns_and_weights(df_ret_bt, daily_W)

    except Exception as e:
        st.info(f"Alignment for attribution not available: {e}")
        aln = None

    # 2) Asset-level contributors
    if aln is not None:
        try:
            df_contrib_asset = bt_attr.contributions_by_asset(aln)
            df_top = (
                df_contrib_asset.group_by("ticker")
                .agg(pl.col("contrib").sum().alias("contrib_total"))
                .sort("contrib_total", descending=True)
                .head(10)
            )
            show_plot(
                plot_top_contributors(df_top),
                key=_bt_key("top-contrib"),
            )

            df_bottom = (
                df_contrib_asset.group_by("ticker")
                .agg(pl.col("contrib").sum().alias("contrib_total"))
                .sort("contrib_total", descending=False)
                .head(10)
            )
            show_plot(
                plot_top_contributors(df_bottom, title="Bottom Contributors"),
                key=_bt_key("bottom-contrib"),
            )
        except Exception as e:
            st.info(f"Basic attribution not available: {e}")

    # 3) Group-level contributors (use user map if present; else 2 buckets)
    if aln is not None:
        try:
            user_map = st.session_state.get("group_map")  # dict[ticker -> group]
            if user_map:
                groups_map = {tk: user_map.get(tk, "OTHER") for tk in bt["tickers"]}
            else:

                def _bucket(tk: str) -> str:
                    return "A-M" if tk.upper()[:1] <= "M" else "N-Z"

                groups_map = {tk: _bucket(tk) for tk in bt["tickers"]}

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
            show_plot(
                plot_group_contrib(df_group_total),
                key=_bt_key("group-bar"),
            )
            show_plot(
                plot_group_contrib_area(df_group_daily),
                key=_bt_key("group-area"),
            )

        except Exception as e:
            st.info(f"Group attribution not available: {e}")

    # 4) Brinson–Fachler
    if aln is not None:
        try:
            T, N_assets = aln.weights.shape

            # Build benchmark by user selection (do not mirror portfolio unless asked)
            Wb_daily = build_benchmark_weights(
                bench_mode, T, aln.tickers, W_portfolio_daily=aln.weights
            )

            # Build groups_idx: prefer user mapping; else per-asset distinct groups
            user_map = st.session_state.get("group_map")
            if user_map:
                name_to_id: dict[str, int] = {}
                gid = 1
                groups_idx = np.zeros(N_assets, dtype=int)
                for j, tk in enumerate(aln.tickers):
                    gname = user_map.get(tk, "OTHER")
                    if gname not in name_to_id:
                        name_to_id[gname] = gid
                        gid += 1
                    groups_idx[j] = name_to_id[gname]
            else:
                groups_idx = np.arange(1, N_assets + 1, dtype=int)

            if len(set(groups_idx.tolist())) < 2:
                st.warning(
                    "Brinson: only 1 group detected. Add more groups to see allocation effect."
                )

            df_brinson = bt_attr.brinson_fachler_cumulative(
                aln=aln,
                bench_weights_daily=Wb_daily,
                groups_idx=groups_idx.tolist(),
            )

            # Quick debug caption to ensure it's not flat
            _alloc = df_brinson["alloc"].to_numpy()
            _select = df_brinson["select"].to_numpy()
            _total = df_brinson["total"].to_numpy()
            st.caption(
                f"Brinson debug — total[min,max]=({np.nanmin(_total):.6f}, {np.nanmax(_total):.6f}) | "
                f"alloc[min,max]=({np.nanmin(_alloc):.6f}, {np.nanmax(_alloc):.6f}) | "
                f"select[min,max]=({np.nanmin(_select):.6f}, {np.nanmax(_select):.6f})"
            )

            show_plot(
                plot_brinson_cumulative(df_brinson),
                key=_bt_key("brinson-cum"),
            )

            # Export artifacts for 05_Attribution and 06_Reporting
            st.session_state["df_brinson"] = df_brinson
            st.session_state["Wb_daily"] = Wb_daily
            st.session_state["groups_idx"] = groups_idx

        except Exception as e:
            st.info(f"Brinson attribution not available: {e}")
