# app/pages/03_Optimizer.py
from __future__ import annotations

import io
import os
import sys

import numpy as np
import polars as pl
import streamlit as st

# ---------------------------------------------------------------------
# Repo root for local imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ── Optim solvers (low-level) ────────────────────────────────────────
from portfolio.backtest.engine import backtest_rebalanced
from portfolio.core.guards import box_feasible, validate_weights
from portfolio.core.logger import JsonRunLogger
from portfolio.core.metrics import cvar_estimate, gini, portfolio_stats
from portfolio.core.opt_helpers import solve_cvar_with_fallback, stack_Ws

# ── Core utils (guards, metrics, logger, high-level helpers) ─────────
from portfolio.core.utils import (
    clean_returns_matrix,
    cond_number,
    ensure_psd,
    hrp_safe,
    project_to_box_simplex,
)
from portfolio.optim.exposures import build_onehot_exposure
from portfolio.optim.hrp import hrp_weights
from portfolio.optim.mean_variance import (
    frontier_box_projected,
    frontier_closed_form,
    markowitz_closed_form,
    pgd_box_simplex_l1,
    pgd_box_simplex_l2,
    risk_contributions,
)
from portfolio.optim.risk_parity import risk_parity
from portfolio.optim.te import te_active_pgd, te_frontier_sweep

# ── Visualization ────────────────────────────────────────────────────
from portfolio.viz.plot_utils import (
    efficient_frontier,
    equity_and_drawdown,
    risk_contributions_bar,
    te_frontier,
    turnover_vs_gamma,
    weights_bar,
    weights_path_gammas,
)

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Optimizer", layout="wide")
st.title("🚀 Optimizer")

# ─────────────────────────────────────────────────────────────────────
# Defensive handoff from 02_RiskModel
# ─────────────────────────────────────────────────────────────────────
required_keys = ("cov_mat", "mu_vec", "asset_names", "returns_wide")
if not all(k in st.session_state for k in required_keys):
    st.warning("Risk Model artifacts not found. Go to **02_RiskModel** and export to session first.")
    st.stop()

Sigma = np.asarray(st.session_state["cov_mat"], dtype=float)
mu = np.asarray(st.session_state["mu_vec"], dtype=float)
mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)  # critical de-noising
names = list(st.session_state["asset_names"])
df_ret_wide: pl.DataFrame = st.session_state["returns_wide"]
meta_df: pl.DataFrame | None = st.session_state.get("asset_meta", None)

# PSD & conditioning
Sigma = np.nan_to_num(Sigma, nan=0.0, posinf=0.0, neginf=0.0)
Sigma = 0.5 * (Sigma + Sigma.T)
np.fill_diagonal(Sigma, np.maximum(np.diag(Sigma), 1e-12))
N = len(names)
if Sigma.shape != (N, N) or mu.shape != (N,):
    st.error("Shape mismatch between μ, Σ and asset names.")
    st.stop()
Sigma = ensure_psd(Sigma, eps=1e-10, clip=True)

# ─────────────────────────────────────────────────────────────────────
# Sidebar – options & constraints
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    # Box with gentle auto-correct
    w_min = st.number_input("w_min", 0.0, 1.0, 0.0, 0.01)
    w_max = st.number_input("w_max", 0.0, 1.0, 0.1, 0.01)
    if N > 0 and (N * w_min > 1.0 or N * w_max < 1.0):
        w_min = min(w_min, 1.0 / N)
        w_max = max(w_max, 1.0 / N)
        st.info(f"Box constraints adjusted to be feasible: w_min≤{1.0/N:.4f}≤w_max")

    rf = st.number_input("rf (annualized)", -0.5, 0.5, 0.0, 0.001, format="%.3f")

    mode = st.selectbox(
        "Optimizer",
        ["Mean-Variance (L2)", "Mean-Variance (L1)", "Risk Parity", "HRP", "CVaR", "Active (TE penalized)"],
        index=0,
    )

    st.markdown("---")
    st.caption("Benchmark (for active / turnover)")
    bench_kind = st.selectbox("Benchmark", ["Equal-Weight", "Custom"], index=0)
    if bench_kind == "Equal-Weight":
        w_bench = np.full(N, 1.0 / max(N, 1))
    else:
        w_bench_str = st.text_area("Custom weights (comma-separated)", value=",".join([f"{1/max(N,1):.6f}"] * N))
        try:
            w_bench = np.array([float(x) for x in w_bench_str.split(",")], dtype=float)
            if w_bench.shape != (N,):
                raise ValueError
        except Exception:
            st.error("Invalid custom weights; falling back to equal-weight.")
            w_bench = np.full(N, 1.0 / max(N, 1))
    w_bench = project_to_box_simplex(w_bench, w_min, w_max)

    # Active exposures (sector/country)
    st.markdown("---")
    st.caption("Active exposure constraints (sector/country)")
    use_expos = st.checkbox("Enable active exposure bounds", value=False)
    X, fac_labels = None, []
    lb, ub = None, None
    rho_expo = 0.0
    if use_expos:
        if meta_df is not None:
            X, fac_labels = build_onehot_exposure(names, meta_df, cols=("sector", "country"))
        else:
            st.info("Upload a CSV with columns: ticker, sector, country")
            csv = st.file_uploader("Asset metadata CSV", type=["csv"])
            if csv is not None:
                dfm = pl.read_csv(csv)
                X, fac_labels = build_onehot_exposure(names, dfm, cols=("sector", "country"))

        if X is not None and X.size > 0:
            rho_expo = st.number_input("ρ (penalty weight for active exposures)", 0.0, 1e6, 1000.0, 10.0)
            lb_val = st.number_input("Lower bound per factor (active)", -1.0, 1.0, -0.05, 0.01)
            ub_val = st.number_input("Upper bound per factor (active)", -1.0, 1.0, 0.05, 0.01)
            lb = np.full(X.shape[0], lb_val)
            ub = np.full(X.shape[0], ub_val)
        else:
            st.warning("Exposure matrix not available. Bounds disabled.")
            use_expos = False

# ─────────────────────────────────────────────────────────────────────
# Run logger
# ─────────────────────────────────────────────────────────────────────
logger = JsonRunLogger(run_name="optimizer")
logger.log("start", mode=mode, n_assets=N, box=dict(w_min=w_min, w_max=w_max))
if not box_feasible(N, w_min, w_max):
    logger.log("error", type="box_infeasible", N=N, w_min=w_min, w_max=w_max)
    st.error("Infeasible box: ensure N*w_min ≤ 1 ≤ N*w_max")
    st.stop()
logger.log("sigma_psd", cond=cond_number(Sigma))

# Clean returns for methods that need R
R_clean_pl: pl.DataFrame = clean_returns_matrix(df_ret_wide)
cols_available = [c for c in names if c in R_clean_pl.columns]
R_np = R_clean_pl.select(cols_available).to_numpy() if cols_available else np.zeros((0, 0), dtype=float)
logger.log("returns_cleaned", n_rows=int(R_clean_pl.height), n_cols=int(len(R_clean_pl.columns)))

# ─────────────────────────────────────────────────────────────────────
# Optimization
# ─────────────────────────────────────────────────────────────────────
w_out: np.ndarray | None = None
diag: dict = {}

if mode == "Mean-Variance (L2)":
    gamma = st.slider("γ (risk aversion)", 0.1, 200.0, 10.0, 0.1)
    lam2 = st.slider("λ (L2 turnover to bench)", 0.0, 100.0, 0.0, 0.1)
    w_out = pgd_box_simplex_l2(mu, Sigma, gamma, w_min=w_min, w_max=w_max, lam_turnover=lam2, w_ref=w_bench)
    w_out = project_to_box_simplex(w_out, w_min, w_max)
    validate_weights(w_out, w_min, w_max)
    logger.log("solution_ok", algo="MV_L2", gamma=gamma, lam2=lam2)

elif mode == "Mean-Variance (L1)":
    gamma = st.slider("γ (risk aversion)", 0.1, 200.0, 10.0, 0.1)
    lam1 = st.slider("λ (L1 turnover to bench)", 0.0, 10.0, 0.0, 0.01)
    w_out = pgd_box_simplex_l1(mu, Sigma, gamma, w_min=w_min, w_max=w_max, lam_l1=lam1, w_ref=w_bench)
    w_out = project_to_box_simplex(w_out, w_min, w_max)
    validate_weights(w_out, w_min, w_max)
    logger.log("solution_ok", algo="MV_L1", gamma=gamma, lam1=lam1)

elif mode == "Risk Parity":
    try:
        w_out = risk_parity(Sigma, w_min=w_min, w_max=w_max)
        w_out = project_to_box_simplex(w_out, w_min, w_max)
        validate_weights(w_out, w_min, w_max)
        logger.log("solution_ok", algo="RiskParity")
    except Exception as e:
        logger.log("fallback", algo="RiskParity", reason=str(e))
        w_out = np.full(N, 1.0 / max(N, 1))
        w_out = project_to_box_simplex(w_out, w_min, w_max)

elif mode == "HRP":
    w_out = hrp_safe(hrp_func=hrp_weights, cov=Sigma, method="ward", optimal=True, w_min=w_min, w_max=w_max)
    w_out = project_to_box_simplex(w_out, w_min, w_max)
    validate_weights(w_out, w_min, w_max)
    logger.log("solution_ok", algo="HRP")

elif mode == "CVaR":
    alpha = st.slider("α (CVaR)", 0.80, 0.995, st.session_state.get("cvar_alpha", 0.95), 0.005, key="cvar_alpha")
    lam_l1 = st.slider("λ L1 turnover", 0.0, 5.0, st.session_state.get("cvar_lam1", 0.0), 0.01, key="cvar_lam1")
    try:
        w_out = solve_cvar_with_fallback(
            R=R_np,
            cols_used=cols_available,
            mu=mu,
            Sigma=Sigma,
            names=names,
            w_bench=w_bench,
            w_min=w_min,
            w_max=w_max,
            alpha=alpha,
            lam_l1=lam_l1,
            mv_gamma=10.0,
        )
        validate_weights(w_out, w_min, w_max)
        logger.log("solution_ok", algo="CVaR", alpha=alpha, lam_l1=lam_l1)
    except Exception as e:
        logger.log("fallback", algo="CVaR", reason=str(e))
        w_out = pgd_box_simplex_l2(mu, Sigma, gamma=10.0, w_min=w_min, w_max=w_max, lam_turnover=0.0, w_ref=w_bench)
        w_out = project_to_box_simplex(w_out, w_min, w_max)

elif mode == "Active (TE penalized)":
    st.markdown("### Active TE optimizer (penalized)")
    gamma = st.slider("γ (tradeoff AR vs TE)", 0.001, 1000.0, 10.0, 0.001)
    lam2 = st.slider("λ L2 turnover to bench", 0.0, 50.0, 0.0, 0.1)
    iters = st.slider("Iterations", 100, 3000, 800, 50)
    rho = rho_expo if use_expos else 0.0
    X_use, lb_use, ub_use = (X, lb, ub) if use_expos else (None, None, None)
    w_out, diag = te_active_pgd(
        mu,
        Sigma,
        w_bench,
        gamma=gamma,
        w_min=w_min,
        w_max=w_max,
        lam_l2=lam2,
        w_ref=w_bench,
        X=X_use,
        lb=lb_use,
        ub=ub_use,
        rho_expo=rho,
        iters=iters,
    )
    if w_out is not None:
        w_out = project_to_box_simplex(w_out, w_min, w_max)
        validate_weights(w_out, w_min, w_max)
    logger.log("solution_ok", algo="ActiveTE", gamma=gamma, lam2=lam2, iters=iters, use_expos=bool(use_expos))

# ─────────────────────────────────────────────────────────────────────
# Results / plots
# ─────────────────────────────────────────────────────────────────────
if w_out is not None:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(weights_bar(w_out, names, sort=True, topn=min(40, N)), width="stretch")
    with c2:
        rc = risk_contributions(w_out, Sigma)
        st.plotly_chart(risk_contributions_bar(rc, names, sort=True, topn=min(30, N)), width="stretch")

    # Portfolio stats (core.metrics)
    mu_p, sigma_p, sharpe = portfolio_stats(w_out, mu, Sigma, rf=rf)
    st.caption(f"μ={mu_p:.4f} · σ={sigma_p:.4f} · Sharpe={sharpe:.3f} · Gini(weights)={gini(w_out):.3f}")

    # Export weights — defensive projection before exporting
    w_export = project_to_box_simplex(w_out, w_min, w_max)
    buf = io.StringIO()
    pl.DataFrame({"ticker": names, "weight": w_export}).write_csv(buf)
    st.download_button("Download weights.csv", buf.getvalue(), file_name="weights.csv", mime="text/csv")

    # Active diagnostics: γ-sweep and TE frontier
    if mode == "Active (TE penalized)" and diag:
        st.subheader("Diagnostics (Active)")
        colA, colB, colC = st.columns(3)
        colA.metric("TE (ann proxy)", f"{diag.get('te', np.nan):.4f}")
        colB.metric("Active return (μ'Δw)", f"{diag.get('active_ret', np.nan):.4f}")
        colC.metric("Expo penalty", f"{diag.get('expo_pen', np.nan):.4f}")

        st.markdown("### γ sweep & TE Frontier")
        gammas = np.geomspace(0.01, 1000.0, 25)
        X_use, lb_use, ub_use = (X, lb, ub) if use_expos else (None, None, None)
        Ws, TE, AR, Loss = te_frontier_sweep(
            mu,
            Sigma,
            w_bench,
            gammas,
            w_min=w_min,
            w_max=w_max,
            lam_l2=0.0,
            w_ref=w_bench,
            X=X_use,
            lb=lb_use,
            ub=ub_use,
            rho_expo=(rho_expo if use_expos else 0.0),
            iters=400,
        )

        Ws_proj = [project_to_box_simplex(wg, w_min, w_max) for wg in Ws]
        Ws_arr = stack_Ws(Ws_proj, N)
        logger.log("gamma_sweep_done", n_gammas=len(gammas))

        st.plotly_chart(weights_path_gammas(Ws_arr, gammas, names, topn=min(25, N)), width="stretch")
        st.plotly_chart(turnover_vs_gamma(Ws_arr, w_bench, gammas), width="stretch")
        st.plotly_chart(te_frontier(mu, Sigma, w_bench, Ws_arr), width="stretch")

# ─────────────────────────────────────────────────────────────────────
# Efficient Frontier (closed-form vs box-projected)
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Efficient Frontier")

try:
    # 1) Robust return range (use a denoised μ; widen if degenerate)
    mu_valid = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    r_lo = float(np.nanpercentile(mu_valid, 10))
    r_hi = float(np.nanpercentile(mu_valid, 90))
    if not (np.isfinite(r_lo) and np.isfinite(r_hi)) or r_lo >= r_hi:
        r_lo, r_hi = -0.1, 0.1  # conservative fallback if μ is flat

    # 2) Closed-form frontier (short allowed)
    risks_closed, rets_closed = frontier_closed_form(mu_valid, Sigma, r_min=r_lo, r_max=r_hi, npts=100)

    # 3) Box frontier (long-only with box)
    if not box_feasible(N, w_min, w_max):
        st.warning(
            f"Box infeasible: N*w_min={N*w_min:.3f}, N*w_max={N*w_max:.3f}. "
            "Adjust bounds so N*w_min ≤ 1 ≤ N*w_max."
        )
        risks_box = rets_box = None
    else:
        risks_box, rets_box = frontier_box_projected(
            mu_valid, Sigma, w_min=w_min, w_max=w_max, r_min=r_lo, r_max=r_hi, npts=100
        )
        if np.size(risks_box) <= 1:
            st.info("Degenerate box-frontier (single point). Relax the box or widen the μ range.")
            risks_box = rets_box = None

    # 4) GMV & Tangency (safe math)
    w_mvp, w_tan = markowitz_closed_form(mu_valid, Sigma, rf=rf)
    r_mvp = float(w_mvp @ mu_valid)
    s_mvp = float(np.sqrt(max(w_mvp @ Sigma @ w_mvp, 0.0)))
    r_tan = float(w_tan @ mu_valid)
    s_tan = float(np.sqrt(max(w_tan @ Sigma @ w_tan, 0.0)))

    st.caption(
        f"cond(Σ) = {cond_number(Sigma):.2e} · "
        f"MVP: (σ={s_mvp:.3f}, μ={r_mvp:.3f}, Gini={gini(w_mvp):.3f}) · "
        f"Tangent: (σ={s_tan:.3f}, μ={r_tan:.3f}, Gini={gini(w_tan):.3f})"
    )

    fig = efficient_frontier(
        mu=mu,
        Sigma=Sigma,
        rf=rf,
        risks_closed=risks_closed,
        rets_closed=rets_closed,
        risks_box=risks_box,
        rets_box=rets_box,
        msr_point=(s_tan, r_tan),
        minvar_point=(s_mvp, r_mvp),
        title="Efficient Frontier",
    )
    st.plotly_chart(fig, width="stretch")

except Exception as e:
    st.warning(f"Frontier plot skipped: {e}")

# ─────────────────────────────────────────────────────────────────────
# Quick Backtest (rolling rebalance)
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Backtest (quick)")

freq = st.selectbox("Rebalance frequency", ["1mo", "1w", "3mo"], index=0)
lbk = st.number_input("Lookback (periods)", min_value=30, max_value=2000, value=252, step=10)
cost = st.number_input("Cost (bps per turnover)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)

def allocator(win: pl.DataFrame) -> np.ndarray:
    """Allocator used inside the rolling backtest; long-only with PSD covariance and safe numerics."""
    cols = [c for c in win.columns if c != "date"]
    R = win.select(cols).to_numpy() if cols else np.zeros((0, 0), dtype=float)
    mu_win = np.nanmean(R, axis=0) if R.size else np.zeros(N)
    Sigma_win = np.cov(R, rowvar=False) if R.size else np.eye(N) * 1e-4

    mu_win = np.nan_to_num(mu_win, nan=0.0, posinf=0.0, neginf=0.0)
    Sigma_win = np.nan_to_num(Sigma_win, nan=0.0, posinf=0.0, neginf=0.0)
    Sigma_win = ensure_psd(Sigma_win, eps=1e-10, clip=True)

    if mode == "Risk Parity":
        try:
            w = risk_parity(Sigma_win, w_min=w_min, w_max=w_max)
        except Exception:
            w = np.full(N, 1.0 / max(N, 1))
    elif mode == "HRP":
        w = hrp_safe(hrp_func=hrp_weights, cov=Sigma_win, method="ward", optimal=True, w_min=w_min, w_max=w_max)
    elif mode == "CVaR":
        alpha = st.session_state.get("cvar_alpha", 0.95)
        lam_l1 = st.session_state.get("cvar_lam1", 0.0)
        R_win_pl = clean_returns_matrix(win)
        cols_used_win = [c for c in names if c in R_win_pl.columns]
        R_win = R_win_pl.select(cols_used_win).to_numpy() if cols_used_win else np.zeros((0, 0), dtype=float)
        w = solve_cvar_with_fallback(
            R=R_win,
            cols_used=cols_used_win,
            mu=(mu_win if mu_win.size == len(names) else mu),
            Sigma=Sigma_win,
            names=names,
            w_bench=w_bench,
            w_min=w_min,
            w_max=w_max,
            alpha=alpha,
            lam_l1=lam_l1,
            mv_gamma=10.0,
        )
    elif mode == "Active (TE penalized)":
        w, _ = te_active_pgd(
            (mu_win if mu_win.size == len(names) else mu),
            Sigma_win,
            w_bench,
            gamma=10.0,
            w_min=w_min,
            w_max=w_max,
            lam_l2=0.0,
            w_ref=w_bench,
            X=(X if use_expos else None),
            lb=(lb if use_expos else None),
            ub=(ub if use_expos else None),
            rho_expo=(rho_expo if use_expos else 0.0),
        )
    else:
        w = pgd_box_simplex_l2(
            (mu_win if mu_win.size == len(names) else mu),
            Sigma_win,
            gamma=10.0,
            w_min=w_min,
            w_max=w_max,
            lam_turnover=0.0,
            w_ref=np.full(N, 1.0 / max(N, 1)),
        )

    return project_to_box_simplex(w, w_min, w_max)

bt = backtest_rebalanced(
    df_ret_wide,
    lookback=int(lbk),
    rebalance_freq=freq,
    cost_bps=float(cost),
    allocator=allocator,
    bench_weights=w_bench,
)

# ---- Equity & drawdown (safe) ----
try:
    st.plotly_chart(
        equity_and_drawdown(bt["dates"], bt["equity"], title="Equity & Drawdown"),
        width="stretch",
    )
except Exception as e:
    st.info(f"Could not plot equity/drawdown: {e}")

# ---- Turnover mean (robust to different engine outputs) ----
def _turnover_mean(turnover_obj) -> float:
    """Return the mean turnover if possible; otherwise NaN without crashing."""
    try:
        if turnover_obj is None:
            return float("nan")
        # Polars
        if isinstance(turnover_obj, pl.DataFrame) and "turnover" in turnover_obj.columns:
            return float(turnover_obj.select(pl.col("turnover").mean()).item())
        # Pandas
        import pandas as pd  # local import to avoid hard dep
        if isinstance(turnover_obj, pd.DataFrame) and "turnover" in turnover_obj.columns:
            return float(turnover_obj["turnover"].mean())
        # Fallback: vector/array
        arr = np.asarray(turnover_obj, dtype=float).ravel()
        if arr.size > 0:
            return float(np.mean(arr))
        return float("nan")
    except Exception:
        return float("nan")

mean_to = _turnover_mean(bt.get("turnover"))
st.caption(f"Mean turnover per rebalance: {mean_to:.3f}" if np.isfinite(mean_to) else "Turnover not available.")

# ---- Quick backtest metrics (from equity) ----
try:
    eq = np.asarray(bt.get("equity", []), float)
    ret_bt = (eq[1:] / eq[:-1]) - 1.0 if eq.size > 1 else np.array([])
    mu_bt = float(np.nanmean(ret_bt)) if ret_bt.size else np.nan
    sig_bt = float(np.nanstd(ret_bt, ddof=1)) if ret_bt.size > 1 else np.nan
    sharpe_bt = mu_bt / sig_bt if (sig_bt is not None and sig_bt > 1e-12) else np.nan
    cvar_bt = cvar_estimate(ret_bt, alpha=0.95) if ret_bt.size else np.nan
    st.caption(f"Backtest: μ={mu_bt:.4f} · σ={sig_bt:.4f} · Sharpe={sharpe_bt:.3f} · CVaR(0.95)={cvar_bt:.4f}")
except Exception:
    pass

# ---- (only in CVaR mode) in-sample CVaR of resulting portfolio ----
if w_out is not None and mode == "CVaR":
    try:
        cols_used_eval = [c for c in names if c in R_clean_pl.columns]
        if cols_used_eval:
            # FIX: correct (i, n) unpack — previously swapped
            name_to_idx = {n: i for i, n in enumerate(names)}
            W_eval = np.array([w_out[name_to_idx[c]] for c in cols_used_eval], dtype=float)
            R_eval = R_clean_pl.select(cols_used_eval).to_numpy()
            port_rets = R_eval @ W_eval
            cvar_ins = cvar_estimate(port_rets, alpha=st.session_state.get("cvar_alpha", 0.95))
            st.caption(f"CVaR in-sample (α={st.session_state.get('cvar_alpha', 0.95):.3f}) = {cvar_ins:.4f}")
    except Exception:
        pass