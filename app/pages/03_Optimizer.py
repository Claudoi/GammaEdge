# app/pages/03_Optimizer.py
from __future__ import annotations

import io
import numpy as np
import polars as pl
import streamlit as st

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ── Optimización ──────────────────────────────────────────────────────────────
from portfolio.optim.mean_variance import (
    ensure_psd, cond_number,
    markowitz_closed_form, frontier_closed_form, frontier_box_projected,
    pgd_box_simplex_l2, pgd_box_simplex_l1,
    project_to_box_simplex, risk_contributions,
)
from portfolio.optim.hrp import hrp_weights
from portfolio.optim.risk_parity import risk_parity
from portfolio.optim.cvar import cvar_minimization
from portfolio.optim.te import te_active_pgd, te_frontier_sweep
from portfolio.optim.exposures import build_onehot_exposure

# ── Visualización ─────────────────────────────────────────────────────────────
from portfolio.viz.plot_utils import (
    weights_bar, risk_contributions_bar,
    efficient_frontier,        
    weights_path_gammas, turnover_vs_gamma, te_frontier,
    equity_and_drawdown,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Optimizer", layout="wide")
st.title("🚀 Optimizer")

# ──────────────────────────────────────────────────────────────────────────────
# Handoff defensivo desde 02_RiskModel
# ──────────────────────────────────────────────────────────────────────────────
if not all(k in st.session_state for k in ("cov_mat", "mu_vec", "asset_names", "returns_wide")):
    st.warning("Risk Model artifacts not found. Go to **02_RiskModel** and export to session first.")
    st.stop()

Sigma = np.asarray(st.session_state["cov_mat"], dtype=float)
mu    = np.asarray(st.session_state["mu_vec"], dtype=float)
names = list(st.session_state["asset_names"])
df_ret_wide: pl.DataFrame = st.session_state["returns_wide"]

N = len(names)
if Sigma.shape != (N, N) or mu.shape != (N,):
    st.error("Shape mismatch between μ, Σ and names.")
    st.stop()

# Asegura PSD (por si llega tocada de la página anterior tras ajustes UI)
Sigma = ensure_psd(Sigma)

# Opcional: metadatos de activos (sector/país) desde Data (si lo guardaste allí)
meta_df: pl.DataFrame | None = st.session_state.get("asset_meta", None)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _box_feasible(n: int, w_min: float, w_max: float) -> bool:
    # Factibilidad: existe w con sum=1 y w_min ≤ w_i ≤ w_max  ⟺  n*w_min ≤ 1 ≤ n*w_max
    return (n * w_min) <= 1.0 <= (n * w_max)

def _safe_project(w: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    if not np.isfinite(w).all():
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    N = w.size
    # Si la caja es infactible, vuelve a equal-weight
    if (N * w_min - 1.0) > 1e-12 or (1.0 - N * w_max) > 1e-12:
        return np.full(N, 1.0 / N)
    # Proyecta a {sum=1, w_min ≤ w ≤ w_max} y renormaliza por seguridad
    w_proj = project_to_box_simplex(w, w_min, w_max)
    s = float(w_proj.sum())
    return (w_proj / s) if s > 1e-12 else np.full(N, 1.0 / N)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – opciones comunes
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    # Caja + factibilidad
    w_min = st.number_input("w_min", 0.0, 1.0, 0.0, 0.01)
    w_max = st.number_input("w_max", 0.0, 1.0, 0.1, 0.01)
    # Autocorrección suave de factibilidad
    if N > 0 and (N * w_min > 1.0 or N * w_max < 1.0):
        # Empuja a la frontera más cercana factible
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
        w_bench = np.full(N, 1.0 / N)
    else:
        w_bench_str = st.text_area(
            "Custom weights (comma-separated)", 
            value=",".join([f"{1/N:.6f}"]*N)
        )
        try:
            w_bench = np.array([float(x) for x in w_bench_str.split(",")], dtype=float)
            if w_bench.shape != (N,):
                raise ValueError
        except Exception:
            st.error("Invalid custom weights; falling back to equal-weight.")
            w_bench = np.full(N, 1.0 / N)

    # Normaliza y proyecta benchmark de forma segura
    w_bench = _safe_project(w_bench, w_min, w_max)
    if not _box_feasible(N, w_min, w_max):
        st.warning("Box infeasible for benchmark (N*w_min ≤ 1 ≤ N*w_max no se cumple). Usando benchmark normalizado sin proyección.")


    # Exposiciones activas: sector/país
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

# ──────────────────────────────────────────────────────────────────────────────
# Optimizadores
# ──────────────────────────────────────────────────────────────────────────────
w_out: np.ndarray | None = None
diag: dict = {}

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
    R = df_ret_wide.select([c for c in df_ret_wide.columns if c != "date"]).to_numpy()
    w_ref = w_bench.copy()
    lam_l1 = st.slider("λ L1 turnover", 0.0, 5.0, 0.0, 0.01)
    w_out = cvar_minimization(R, alpha=alpha, w_min=w_min, w_max=w_max,
                              budget=1.0, lam_l1_turnover=lam_l1, w_ref=w_ref)

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

    if w_out is not None:
        w_out = _safe_project(w_out, w_min, w_max)

# ──────────────────────────────────────────────────────────────────────────────
# Resultados / plots
# ──────────────────────────────────────────────────────────────────────────────
if w_out is not None:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(
            weights_bar(w_out, names, sort=True, topn=min(40, N)),
            use_container_width=True
        )
    with c2:
        rc = risk_contributions(w_out, Sigma)
        st.plotly_chart(
            risk_contributions_bar(rc, names, sort=True, topn=min(30, N)),
            use_container_width=True
        )

    # (5) Export pesos — usar proyección defensiva antes de exportar
    w_export = _safe_project(w_out, w_min, w_max)
    buf = io.StringIO()
    pl.DataFrame({"ticker": names, "weight": w_export}).write_csv(buf)
    st.download_button(
        "Download weights.csv",
        buf.getvalue(),
        file_name="weights.csv",
        mime="text/csv"
    )

    # Diags Active
    if mode == "Active (TE penalized)" and diag:
        st.subheader("Diagnostics (Active)")
        colA, colB, colC = st.columns(3)
        colA.metric("TE (ann proxy)", f"{diag['te']:.4f}")
        colB.metric("Active return (μ'Δw)", f"{diag['active_ret']:.4f}")
        colC.metric("Expo penalty", f"{diag['expo_pen']:.4f}")

        st.markdown("### γ sweep & TE Frontier")
        gammas = np.geomspace(0.01, 1000.0, 25)
        X_use, lb_use, ub_use = (X, lb, ub) if use_expos else (None, None, None)
        Ws, TE, AR, Loss = te_frontier_sweep(
            mu, Sigma, w_bench, gammas, w_min=w_min, w_max=w_max,
            lam_l2=0.0, w_ref=w_bench, X=X_use, lb=lb_use, ub=ub_use,
            rho_expo=(rho_expo if use_expos else 0.0), iters=400
        )

        # (7) Proyección defensiva de TODAS las carteras del sweep
        Ws = [_safe_project(w, w_min, w_max) for w in Ws]
        st.plotly_chart(weights_path_gammas(Ws, gammas, names, topn=min(25, N)), use_container_width=True)
        st.plotly_chart(turnover_vs_gamma(Ws, w_bench, gammas), use_container_width=True)
        st.plotly_chart(te_frontier(mu, Sigma, w_bench, Ws), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Efficient Frontier (closed-form vs box‑projected)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Efficient Frontier")

try:
    # 1) Rango de retornos robusto
    r_lo = float(np.nanpercentile(mu, 10))
    r_hi = float(np.nanpercentile(mu, 90))
    if not (np.isfinite(r_lo) and np.isfinite(r_hi)) or r_lo >= r_hi:
        r_lo, r_hi = float(np.nanmin(mu)), float(np.nanmax(mu))
    if not (np.isfinite(r_lo) and np.isfinite(r_hi)) or r_lo >= r_hi:
        # fallback por si μ es patológico
        r_lo, r_hi = -0.1, 0.1

    # 2) Frontera cerrada (short permitido)
    risks_closed, rets_closed = frontier_closed_form(
        mu, Sigma, r_min=r_lo, r_max=r_hi, npts=100
    )

    # 3) Comprobación de factibilidad de la caja
    N = len(mu)
    box_ok = (N * w_min <= 1.0 + 1e-12) and (1.0 <= N * w_max + 1e-12)
    if not box_ok:
        st.warning(
            f"Box infeasible: N*w_min={N*w_min:.3f}, N*w_max={N*w_max:.3f}. "
            "Ajusta límites para que N*w_min ≤ 1 ≤ N*w_max."
        )
        risks_box = rets_box = None
    else:
        # 4) Frontera con caja (aprox por proyección)
        risks_box, rets_box = frontier_box_projected(
            mu, Sigma, w_min=w_min, w_max=w_max, r_min=r_lo, r_max=r_hi, npts=100
        )
        if np.size(risks_box) <= 1:
            st.info("Box‑frontier degenerada (un único punto). Relaja caja o amplía rango de μ.")
            risks_box = rets_box = None

    # 5) GMV & Tangente (informativos)
    w_mvp, w_tan = markowitz_closed_form(mu, Sigma, rf=rf)
    r_mvp = float(w_mvp @ mu); s_mvp = float(np.sqrt(max(w_mvp @ Sigma @ w_mvp, 0.0)))
    r_tan = float(w_tan @ mu); s_tan = float(np.sqrt(max(w_tan @ Sigma @ w_tan, 0.0)))

    st.caption(
        f"cond(Σ) = {cond_number(Sigma):.2e} · "
        f"MVP: (σ={s_mvp:.3f}, μ={r_mvp:.3f}) · "
        f"Tangent: (σ={s_tan:.3f}, μ={r_tan:.3f})"
    )

    # 6) Plot (usa la versión nueva de plot_utils.efficient_frontier)
    fig = efficient_frontier(
        mu=mu, Sigma=Sigma, rf=rf,
        risks_closed=risks_closed, rets_closed=rets_closed,
        risks_box=risks_box,     rets_box=rets_box,
        msr_point=(s_tan, r_tan), minvar_point=(s_mvp, r_mvp),
        title="Efficient Frontier"
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"Frontier plot skipped: {e}")



# ──────────────────────────────────────────────────────────────────────────────
# Quick Backtest (rolling rebalance)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Backtest (quick)")

freq = st.selectbox("Rebalance frequency", ["1mo", "1w", "3mo"], index=0)
lbk  = st.number_input("Lookback (periods)", min_value=30, max_value=2000, value=252, step=10)
cost = st.number_input("Cost (bps per turnover)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)

def allocator(win: pl.DataFrame) -> np.ndarray:
    # Usa el mismo modo seleccionado, recalculando con Σ y μ de la ventana
    R = win.select([c for c in win.columns if c != "date"]).to_numpy()
    mu_win = np.nanmean(R, axis=0)
    Sigma_win = np.cov(R, rowvar=False)

    # higiene numérica
    mu_win = np.where(np.isfinite(mu_win), mu_win, 0.0)
    Sigma_win = np.where(np.isfinite(Sigma_win), Sigma_win, 0.0)

    if mode == "Risk Parity":
        w = risk_parity(Sigma_win, w_min=w_min, w_max=w_max)
    elif mode == "HRP":
        w = hrp_weights(Sigma_win, w_min=w_min, w_max=w_max)
    elif mode == "CVaR":
        w = cvar_minimization(
            R, alpha=0.95, w_min=w_min, w_max=w_max, budget=1.0,
            lam_l1_turnover=0.0, w_ref=np.full(N, 1.0/N)
        )
    elif mode == "Active (TE penalized)":
        w, _diag = te_active_pgd(
            mu_win, Sigma_win, w_bench, gamma=10.0,
            w_min=w_min, w_max=w_max, lam_l2=0.0, w_ref=w_bench,
            X=(X if use_expos else None),
            lb=(lb if use_expos else None),
            ub=(ub if use_expos else None),
            rho_expo=(rho_expo if use_expos else 0.0)
        )
    else:
        w = pgd_box_simplex_l2(
            mu_win, Sigma_win, gamma=10.0,
            w_min=w_min, w_max=w_max,
            lam_turnover=0.0, w_ref=np.full(N, 1.0/N)
        )

    # Proyección final segura para el backtest (normaliza y proyecta si es factible)
    w = _safe_project(w, w_min, w_max)
    return w



from portfolio.backtest.engine import backtest_rebalanced
bt = backtest_rebalanced(df_ret_wide, lookback=int(lbk), rebalance_freq=freq,
                         cost_bps=float(cost), allocator=allocator, bench_weights=w_bench)

st.plotly_chart(equity_and_drawdown(bt["dates"], bt["equity"], title="Equity & Drawdown"), use_container_width=True)
st.write(f"Mean turnover per rebalance: {bt['turnover'].mean():.3f}")
