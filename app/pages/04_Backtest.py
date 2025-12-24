from __future__ import annotations

# --- stdlib ---
import contextlib
import os
import sys
from typing import Any, Literal, cast

# --- third-party ---
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st

# ---------------------------------------------------------------------
# Add repository root path for local imports (same as in 03_Optimizer)
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# --- backtest core ---
from portfolio.backtest import attribution as bt_attr
from portfolio.backtest import metrics as bt_metrics
from portfolio.backtest.allocators import make_allocator
from portfolio.backtest.engine import backtest_rebalanced, backtest_vectorized

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
            empty_samples = pl.DataFrame(schema=["CAGR", "Sharpe", "MaxDD"])
            empty_q = pl.DataFrame(schema=["metric", "q05", "q50", "q95"])
            return empty_samples, empty_q
        r = np.diff(eq) / eq[:-1]

    T = int(r.size)
    if T == 0 or B <= 0:
        empty_samples = pl.DataFrame(schema=["CAGR", "Sharpe", "MaxDD"])
        empty_q = pl.DataFrame(schema=["metric", "q05", "q50", "q95"])
        return empty_samples, empty_q

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


def _metric_from_df(m: Any, name: str) -> float:
    """Robust metric extraction (Polars | pandas) -> float or NaN."""
    try:
        if isinstance(m, pl.DataFrame):
            if name in m.columns:
                v = m.select(pl.col(name)).item()
                return float(v)
            low = name.lower()
            for c in m.columns:
                if str(c).lower() == low:
                    v = m.select(pl.col(c)).item()
                    return float(v)
    except Exception:
        pass

    try:
        if isinstance(m, pd.DataFrame) and len(m.index) > 0:
            if name in m.columns:
                v = m.at[m.index[0], name]
                return float(cast(float, v))
            low = name.lower()
            for c in m.columns:
                if str(c).lower() == low:
                    v = m.at[m.index[0], c]
                    return float(cast(float, v))
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
                if not np.isfinite(cagr):
                    cagr = cagr_fb
                if not np.isfinite(sharpe):
                    sharpe = sharpe_fb
                if not np.isfinite(maxdd):
                    maxdd = maxdd_fb
    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": float(maxdd)}


def _ensure_turnover_with_drift(bt: dict, df_wide: pl.DataFrame) -> tuple[list, np.ndarray]:
    """
    Ensure we have a turnover time series:
    1) Try to use what the engine returns (DataFrame or array).
    2) If missing, reconstruct turnover between rebalance dates using drifted weights.
    """
    to_obj = bt.get("turnover")
    if to_obj is not None:
        try:
            # Case 1: Polars DataFrame with columns [date, turnover]
            if isinstance(to_obj, pl.DataFrame) and "turnover" in to_obj.columns:
                dates = (
                    to_obj.get_column("date").to_list()
                    if "date" in to_obj.columns
                    else list(range(to_obj.height))
                )
                vals = to_obj.get_column("turnover").to_numpy()
                return dates, np.asarray(vals, float)

            # Case 2: pandas DataFrame with columns [date, turnover]
            if isinstance(to_obj, pd.DataFrame) and "turnover" in to_obj.columns:
                dates = (
                    to_obj["date"].tolist()
                    if "date" in to_obj.columns
                    else list(range(len(to_obj)))
                )
                return dates, np.asarray(to_obj["turnover"].values, float)

            # Case 3: plain array like
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

    # Fallback 2: reconstruct from pre/post-drift weights
    rb_dates = list(bt.get("rebalance_dates", []))
    W_reb = np.asarray(bt.get("weights", []), float)
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

    turns: list[float] = []
    out_dates: list[Any] = []
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
    if mode == "static_first_day":
        if W_portfolio_daily is None:
            raise ValueError("static_first_day requires W_portfolio_daily")
        w0 = np.clip(W_portfolio_daily[0], 0.0, None)
        s = float(w0.sum())
        w0 = (w0 / s) if s > 0 else np.full(N, 1.0 / max(N, 1), dtype=float)
        return np.tile(w0, (T, 1))
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

df_ret_wide_raw: Any = st.session_state["returns_wide"]
if isinstance(df_ret_wide_raw, pd.DataFrame):
    df_ret_wide: pl.DataFrame = pl.from_pandas(df_ret_wide_raw)
else:
    df_ret_wide = cast(pl.DataFrame, df_ret_wide_raw)

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

    # Engine selection
    engine_type = st.radio("Engine Mode", ["Classic (Loops)", "Vectorized (Fast)"], index=1)

    # Impact Model
    st.markdown("---")
    impact_model = st.selectbox(
        "Market Impact Model",
        ["linear", "sqrt"],
        index=0,
        help="Linear: Fixed bps. Sqrt: Non-linear impact based on Volume.",
    )
    impact_c = 1.0
    if impact_model == "sqrt":
        impact_c = st.number_input("Impact Coefficient (c)", 0.1, 10.0, 1.0, 0.1)

    st.markdown("---")
    st.subheader("Allocator")

    alloc_kind = st.selectbox(
        "Strategy",
        ["Equal-Weight", "Min-Var (L2 PGD)", "Risk Parity", "HRP", "Min-TE (to Bench)"],
        index=0,
        help="Portfolio construction method used at each rebalance window.",
    )

    # Simple box constraints
    w_min: float = st.number_input("w_min", 0.0, 1.0, 0.0, 0.01)
    w_max: float = st.number_input("w_max", 0.0, 1.0, 0.2, 0.01)
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
    use_to_budget: bool = st.checkbox("Limit turnover (budget)", value=True)
    max_turnover: float = st.slider("Max turnover per rebalance", 0.0, 0.50, 0.10, 0.01)
    band_eps: float = st.slider("Band threshold (median |Δw|)", 0.0, 0.05, 0.01, 0.001)

    st.markdown("---")
    st.subheader("Benchmark (for Brinson)")
    bench_mode = st.selectbox("Benchmark mode", ["equal", "static_first_day"], index=0)

    st.markdown("---")
    st.subheader("Hyperparameter Grid Search")
    do_grid: bool = st.checkbox("Run grid search", value=False)
    grid_lookbacks = st.text_input("Lookback values", "126,252")
    grid_costs = st.text_input("Transaction costs (bps)", "0,2,5")


# ─────────────────────────────────────────────────────────────────────
# Cached backtest execution
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_backtest(
    df_ret_wide: pl.DataFrame,
    lookback: int,
    rebalance_freq: str,
    cost_bps: float,
    alloc_kind: str,
    w_min: float,
    w_max: float,
    cov_estimator: Literal["Sample", "EWMA"],
    ewma_lambda: float,
    use_to_budget: bool,
    max_turnover: float,
    band_eps: float,
    engine_mode: str,
    impact_model: str = "linear",
    impact_c: float = 1.0,
    df_volume: pl.DataFrame | None = None,
) -> dict[str, Any]:
    alloc = make_allocator(
        alloc_kind,
        w_min=w_min,
        w_max=w_max,
        cov_estimator=cov_estimator,
        ewma_lambda=ewma_lambda,
        use_to_budget=use_to_budget,
        max_turnover=max_turnover,
        band_eps=band_eps,
    )
    n_cols = len([c for c in df_ret_wide.columns if c != "date"])

    # Select engine
    engine_func = backtest_vectorized if engine_mode == "Vectorized (Fast)" else backtest_rebalanced

    return engine_func(
        df_ret_wide=df_ret_wide,
        lookback=int(lookback),
        rebalance_freq=rebalance_freq,
        cost_bps=float(cost_bps),
        allocator=alloc,
        bench_weights=np.full(n_cols, 1.0 / max(n_cols, 1)),
        impact_model=impact_model,
        impact_c=impact_c,
        df_volume=df_volume,
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
    cov_estimator: Literal["Sample", "EWMA"],
    ewma_lambda: float,
    use_to_budget: bool,
    max_turnover: float,
    band_eps: float,
) -> dict[str, float]:
    alloc = make_allocator(
        alloc_kind,
        w_min=w_min,
        w_max=w_max,
        cov_estimator=cov_estimator,
        ewma_lambda=ewma_lambda,
        use_to_budget=use_to_budget,
        max_turnover=max_turnover,
        band_eps=band_eps,
    )
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
            float(w_min),
            float(w_max),
            cast(Literal["Sample", "EWMA"], cov_estimator),
            float(ewma_lambda),
            use_to_budget,
            float(max_turnover),
            float(band_eps),
            engine_type,
            impact_model,
            float(impact_c),
            None,  # TODO: Pass df_volume from stored session state if available
        )
    st.success("✅ Backtest executed.")

    # Handoff to 05_Attribution (persist into session_state)
    def _export_to_05(bt_obj: dict[str, Any], df_wide_obj: Any) -> None:
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
                w_min=float(w_min),
                w_max=float(w_max),
                cov_estimator=cast(Literal["Sample", "EWMA"], cov_estimator),
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
    # Check for errors
    if "error" in bt:
        st.error(f"Backtest failed: {bt['error']}")
        st.stop()
    if "equity" not in bt or "dates" not in bt:
        st.warning("Backtest returned no data (possibly insufficient history).")
        st.stop()

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
        # Cost Drag Chart (New)
        if "costs" in bt and len(bt["costs"]) == len(bt["equity"]):
            costs = bt["costs"]
            # Reconstruct Gross: Net / CumProd(1 - Cost)
            # Cost is fraction of CURRENT equity lost.
            # Eq_net_t = Eq_gross_t * Decay_t
            # Decay_factor = (1-c1)*(1-c2)...

            decay = np.cumprod(1.0 - costs)
            equity_gross = bt["equity"] / decay

            # Plot
            fig_drag = go.Figure()
            fig_drag.add_trace(
                go.Scatter(
                    x=bt["dates"],
                    y=equity_gross,
                    mode="lines",
                    name="Gross Equity (No Costs)",
                    line=dict(color="gray", dash="dot", width=1),
                )
            )
            fig_drag.add_trace(
                go.Scatter(
                    x=bt["dates"],
                    y=bt["equity"],
                    mode="lines",
                    name="Net Equity",
                    fill="tonexty",  # Fill to Gross
                    line=dict(color="#636EFA"),
                    fillcolor="rgba(239, 85, 59, 0.2)",  # Red-ish for cost
                )
            )
            fig_drag.update_layout(
                title="Transaction Cost Drag (Gross vs Net)",
                yaxis_title="Equity",
                template="plotly_white",
                legend=dict(x=0, y=1),
                margin=dict(l=40, r=20, t=40, b=40),
            )
            show_plot(fig_drag, key="bt-cost-drag")
        else:
            # Fallback (e.g. vectorized engine might not support cost tracking yet)
            st.info("Cost breakdown not available for this engine mode.")

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

    dates_to, vals_to = _ensure_turnover_with_drift(bt, df_ret_wide)
    if vals_to.size > 0 and len(dates_to) == vals_to.size:
        fig = plot_turnover(dates_to, vals_to, title="Turnover at Rebalance")
        show_plot(fig, key=_bt_key("turnover"))
    else:
        st.info("Turnover plot unavailable (no turnover series could be built).")

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
            T_bt, N_assets = aln.weights.shape

            # Build benchmark by user selection (do not mirror portfolio unless asked)
            Wb_daily = build_benchmark_weights(
                bench_mode, T_bt, aln.tickers, W_portfolio_daily=aln.weights
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
