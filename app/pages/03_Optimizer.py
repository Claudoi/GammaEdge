# app/pages/03_Optimizer.py
from __future__ import annotations

import numpy as np
import polars as pl
import streamlit as st
import io

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from portfolio.optim.mean_variance import (
    pgd_box_simplex_l2, pgd_box_simplex_l1, markowitz_closed_form, risk_contributions,
    project_to_box_simplex
)
from portfolio.optim.hrp import hrp_weights
from portfolio.optim.risk_parity import risk_parity
from portfolio.optim.cvar import cvar_minimization
from portfolio.optim.te import te_active_pgd, te_frontier_sweep
from portfolio.optim.exposures import build_onehot_exposure
from portfolio.viz.plot_utils import (
    weights_bar, efficient_frontier, risk_contributions_bar,
    weights_path_gammas, turnover_vs_gamma, te_frontier
)

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Optimizer", layout="wide")
st.title("🚀 Optimizer")

# Handoff defensivo
if "cov_mat" not in st.session_state or "mu_vec" not in st.session_state or "asset_names" not in st.session_state:
    st.warning("Risk Model artifacts not found. Go to **02_RiskModel** and export to session first.")
    st.stop()

Sigma = np.asarray(st.session_state["cov_mat"], dtype=float)
mu    = np.asarray(st.session_state["mu_vec"], dtype=float)
names = list(st.session_state["asset_names"])

N = len(names)
if Sigma.shape != (N, N) or mu.shape != (N,):
    st.error("Shape mismatch between μ, Σ and names."); st.stop()

# Opcional: metadatos de activos (sector/país) desde Data (si lo guardaste ahí)
meta_df: pl.DataFrame | None = st.session_state.get("asset_meta", None)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – opciones comunes
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    w_min = st.number_input("w_min", 0.0, 1.0, 0.0, 0.01)
    w_max = st.number_input("w_max", 0.0, 1.0, 0.1, 0.01)
    rf     = st.number_input("rf (annualized)", -0.5, 0.5, 0.0, 0.001, format="%.3f")

    mode = st.selectbox("Optimizer", [
        "Mean-Variance (L2)", "Mean-Variance (L1)", "Risk Parity", "HRP", "CVaR", "Active (TE penalized)"
    ], index=0)

    st.markdown("---")
    st.caption("Active / TE options")
    bench_kind = st.selectbox("Benchmark", ["Equal-Weight", "Custom"], index=0)
    if bench_kind == "Equal-Weight":
        w_bench = np.full(N, 1.0 / N)
    else:
        # For quick input: comma-separated weights aligned to names
        w_bench_str = st.text_area("Custom weights (comma-separated)", value=",".join([f"{1/N:.6f}"]*N))
        try:
            w_bench = np.array([float(x) for x in w_bench_str.split(",")], dtype=float)
            if w_bench.shape != (N,):
                raise ValueError
            w_bench = w_bench / max(w_bench.sum(), 1e-12)
        except Exception:
            st.error("Invalid custom weights; falling back to equal-weight.")
            w_bench = np.full(N, 1.0 / N)

    # Exposiciones activas: sector/país
    st.markdown("---")
    st.caption("Active exposure constraints (sector/country)")
    use_expos = st.checkbox("Enable active exposure bounds", value=False)
    X, fac_labels = None, []
    lb, ub = None, None
    rho_expo = 0.0
    if use_expos:
        # Construir X desde meta_df o subir CSV
        if meta_df is not None:
            X, fac_labels = build_onehot_exposure(names, meta_df, cols=("sector", "country"))
        else:
            st.info("Upload a CSV with columns: ticker, sector, country")
            csv = st.file_uploader("Asset metadata CSV", type=["csv"])
            if csv is not None:
                dfm = pl.read_csv(csv)
                X, fac_labels = build_onehot_exposure(names, dfm, cols=("sector", "country"))

        if X is not None and X.size > 0:
            rho_expo = st.number_input("ρ (penalty weight)", 0.0, 1e6, 1000.0, 10.0)
            lb_val = st.number_input("Lower bound per factor (active)", -1.0, 1.0, -0.05, 0.01)
            ub_val = st.number_input("Upper bound per factor (active)", -1.0, 1.0, 0.05, 0.01)
            lb = np.full(X.shape[0], lb_val)
            ub = np.full(X.shape[0], ub_val)
        else:
            st.warning("Exposure matrix not available. Bounds disabled.")
            use_expos = False

# ──────────────────────────────────────────────────────────────────────────────
# Optimizadores
# ──────────────────────────────────────────────────────────────────────────────
w_out = None
diag = {}

if mode == "Mean-Variance (L2)":
    gamma = st.slider("γ (risk aversion)", 0.1, 200.0, 10.0, 0.1)
    lam2  = st.slider("λ (L2 turnover to bench)", 0.0, 100.0, 0.0, 0.1)
    w_ref = w_bench.copy()
    w_out = pgd_box_simplex_l2(mu, Sigma, gamma, w_min=w_min, w_max=w_max, lam_turnover=lam2, w_ref=w_ref)

elif mode == "Mean-Variance (L1)":
    gamma = st.slider("γ (risk aversion)", 0.1, 200.0, 10.0, 0.1)
    lam1  = st.slider("λ (L1 turnover to bench)", 0.0, 10.0, 0.0, 0.01)
    w_ref = w_bench.copy()
    w_out = pgd_box_simplex_l1(mu, Sigma, gamma, w_min=w_min, w_max=w_max, lam_l1=lam1, w_ref=w_ref)

elif mode == "Risk Parity":
    w_out = risk_parity(Sigma, w_min=w_min, w_max=w_max)

elif mode == "HRP":
    w_out = hrp_weights(Sigma, method="ward", optimal=True, w_min=w_min, w_max=w_max)

elif mode == "CVaR":
    alpha = st.slider("α (CVaR)", 0.80, 0.995, 0.95, 0.005)
    # Necesitamos retornos periódicos para CVaR; los tomamos de session_state
    df_ret_wide: pl.DataFrame = st.session_state["returns_wide"]
    R = df_ret_wide.select([c for c in df_ret_wide.columns if c != "date"] ).to_numpy()
    w_ref = w_bench.copy()
    lam_l1 = st.slider("λ L1 turnover", 0.0, 5.0, 0.0, 0.01)
    w_out = cvar_minimization(R, alpha=alpha, w_min=w_min, w_max=w_max, budget=1.0, lam_l1_turnover=lam_l1, w_ref=w_ref)

elif mode == "Active (TE penalized)":
    st.markdown("### Active TE optimizer (penalized)")
    gamma = st.slider("γ (tradeoff AR vs TE)", 0.001, 1000.0, 10.0, 0.001)
    lam2  = st.slider("λ L2 turnover to bench", 0.0, 50.0, 0.0, 0.1)
    iters = st.slider("Iterations", 100, 3000, 800, 50)
    rho   = rho_expo if use_expos else 0.0
    X_use, lb_use, ub_use = (X, lb, ub) if use_expos else (None, None, None)
    w_out, diag = te_active_pgd(
        mu, Sigma, w_bench, gamma=gamma, w_min=w_min, w_max=w_max,
        lam_l2=lam2, w_ref=w_bench, X=X_use, lb=lb_use, ub=ub_use, rho_expo=rho,
        iters=iters
    )

# ──────────────────────────────────────────────────────────────────────────────
# Resultados / plots
# ──────────────────────────────────────────────────────────────────────────────
if w_out is not None:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(weights_bar(w_out, names, sort=True, topn=min(40, N)), use_container_width=True)
    with c2:
        rc = risk_contributions(w_out, Sigma)
        st.plotly_chart(risk_contributions_bar(rc, names, sort=True, topn=min(30, N)), use_container_width=True)

    # Export pesos
    buf = io.StringIO()
    pl.DataFrame({"ticker": names, "weight": w_out}).write_csv(buf)
    st.download_button("Download weights.csv", buf.getvalue(), file_name="weights.csv", mime="text/csv")

    # Si el modo es TE, mostramos diags y barrido
    if mode == "Active (TE penalized)":
        st.subheader("Diagnostics (Active)")
        if diag:
            colA, colB, colC = st.columns(3)
            colA.metric("TE (ann proxy)", f"{diag['te']:.4f}")
            colB.metric("Active return (μ'Δw)", f"{diag['active_ret']:.4f}")
            colC.metric("Expo pen", f"{diag['expo_pen']:.4f}")

        st.markdown("### γ sweep & TE Frontier")
        gammas = np.geomspace(0.01, 1000.0, 25)
        X_use, lb_use, ub_use = (X, lb, ub) if use_expos else (None, None, None)
        Ws, TE, AR, Loss = te_frontier_sweep(
            mu, Sigma, w_bench, gammas, w_min=w_min, w_max=w_max,
            lam_l2=0.0, w_ref=w_bench, X=X_use, lb=lb_use, ub=ub_use, rho_expo=(rho_expo if use_expos else 0.0),
            iters=400
        )
        st.plotly_chart(weights_path_gammas(Ws, gammas, names, topn=min(25, N)), use_container_width=True)
        st.plotly_chart(turnover_vs_gamma(Ws, w_bench, gammas), use_container_width=True)
        st.plotly_chart(te_frontier(mu, Sigma, w_bench, Ws), use_container_width=True)


# --- Quick backtest (optional) ---
st.markdown("---")
st.subheader("Backtest (quick)")

df_ret_wide: pl.DataFrame = st.session_state["returns_wide"]
freq = st.selectbox("Rebalance frequency", ["1mo", "1w", "3mo"], index=0)
lbk  = st.number_input("Lookback (periods)", min_value=30, max_value=2000, value=252, step=10)
cost = st.number_input("Cost (bps per turnover)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)

def allocator(win: pl.DataFrame) -> np.ndarray:
    # Usa el mismo modo seleccionado, recalculando con Σ y μ de la ventana
    R = win.select([c for c in win.columns if c != "date"]).to_numpy()
    mu_win = np.nanmean(R, axis=0)
    Sigma_win = np.cov(R, rowvar=False)
    if mode == "Risk Parity":
        return risk_parity(Sigma_win, w_min=w_min, w_max=w_max)
    elif mode == "HRP":
        return hrp_weights(Sigma_win, w_min=w_min, w_max=w_max)
    elif mode == "CVaR":
        return cvar_minimization(R, alpha=0.95, w_min=w_min, w_max=w_max, budget=1.0, lam_l1_turnover=0.0, w_ref=np.full(N,1.0/N))
    elif mode == "Active (TE penalized)":
        return te_active_pgd(mu_win, Sigma_win, w_bench, gamma=10.0, w_min=w_min, w_max=w_max, lam_l2=0.0, w_ref=w_bench, X=(X if use_expos else None), lb=(lb if use_expos else None), ub=(ub if use_expos else None), rho_expo=(rho_expo if use_expos else 0.0))[0]
    else:
        return pgd_box_simplex_l2(mu_win, Sigma_win, gamma=10.0, w_min=w_min, w_max=w_max, lam_turnover=0.0, w_ref=np.full(N,1.0/N))

from portfolio.backtest.engine import backtest_rebalanced
bt = backtest_rebalanced(df_ret_wide, lookback=int(lbk), rebalance_freq=freq, cost_bps=float(cost), allocator=allocator, bench_weights=w_bench)
from portfolio.viz.plot_utils import equity_and_drawdown
st.plotly_chart(equity_and_drawdown(bt["dates"], bt["equity"], title="Equity & Drawdown"), use_container_width=True)

st.write(f"Mean turnover per rebalance: {bt['turnover'].mean():.3f}")
