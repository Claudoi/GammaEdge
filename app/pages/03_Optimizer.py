# app/pages/03_Optimizer.py
from __future__ import annotations

import io
import json
import hashlib
from typing import Optional, Tuple

import numpy as np
import polars as pl
import streamlit as st

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Viz
from portfolio.viz.plot_utils import (
    efficient_frontier,
    weights_bar,
    equity_and_drawdown,
    loss_distribution,
    risk_contributions_bar,
)

# Return kind for quick backtest
from portfolio.features.returns import infer_return_kind


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Optimizer", layout="wide")
st.title("🎯 Portfolio Optimizer")

# ──────────────────────────────────────────────────────────────────────────────
# Defensive handoff validation
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_ndarray(x, name: str) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x, dtype=float)
    except Exception:
        st.error(f"'{name}' is not array-like.")
        st.stop()

def _ensure_psd(Sigma: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if Sigma.size == 0:
        return Sigma
    A = 0.5 * (Sigma + Sigma.T)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    S = (V @ np.diag(w) @ V.T)
    return 0.5 * (S + S.T)

def _cond_number(S: np.ndarray) -> float:
    if S.size == 0:
        return float("nan")
    Ssym = 0.5 * (S + S.T)
    w = np.linalg.eigvalsh(Ssym)
    if w.size == 0:
        return float("nan")
    lam_min = float(np.min(w))
    lam_max = float(np.max(w))
    return float(lam_max / max(lam_min, 1e-16))

# Handoff from 02_RiskModel
missing_keys = [k for k in ["mu_vec", "cov_mat", "asset_names"] if k not in st.session_state]
if missing_keys:
    st.warning("Risk model not found in session. Please go to **02_RiskModel** and compute μ/Σ first.")
    st.stop()

mu_ann = _ensure_ndarray(st.session_state["mu_vec"], "mu_vec")
Sigma_ann = _ensure_ndarray(st.session_state["cov_mat"], "cov_mat")
names = list(st.session_state["asset_names"])

if Sigma_ann.ndim != 2 or Sigma_ann.shape[0] != Sigma_ann.shape[1]:
    st.error("Σ must be a square matrix."); st.stop()
if mu_ann.ndim != 1 or mu_ann.shape[0] != Sigma_ann.shape[0]:
    st.error("μ length must match Σ dimension."); st.stop()
N = len(names)
if N != mu_ann.shape[0] or N != Sigma_ann.shape[0]:
    st.error("Asset labels and μ/Σ shapes are inconsistent."); st.stop()

# Rescue ridge if ill-conditioned
kappa0 = _cond_number(Sigma_ann)
if not np.isfinite(kappa0) or kappa0 > 1e10:
    st.warning("Σ is ill-conditioned. Applying small ridge ε=1e-8.")
    Sigma_ann = Sigma_ann + np.eye(N) * 1e-8

# Risk meta if present
risk_meta = st.session_state.get("risk_meta", {})
risk_cfg  = st.session_state.get("risk_config", {})
per_year = int(risk_meta.get("params", {}).get("per_year", 252))

# Returns (optional) for quick backtest
df_ret_wide: Optional[pl.DataFrame] = st.session_state.get("returns_wide", None)


# ──────────────────────────────────────────────────────────────────────────────
# Solvers (closed-form + PGD prox with box+simplex)
# ──────────────────────────────────────────────────────────────────────────────
def _markowitz_closed_form(mu: np.ndarray, Sigma: np.ndarray, rf: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (w_minvar, w_tan) in annual scale, short-allowed, fully invested."""
    invS = np.linalg.pinv(Sigma)
    one = np.ones_like(mu)

    A = float(one @ invS @ one)
    B = float(one @ invS @ mu)

    w_mvp = (invS @ one) / A

    ex = mu - rf * one
    k = float(one @ invS @ ex)
    w_tan = (invS @ ex) / k if abs(k) >= 1e-18 else np.copy(w_mvp)
    return w_mvp, w_tan

def _weights_stats(mu: np.ndarray, Sigma: np.ndarray, w: np.ndarray, rf: float) -> Tuple[float, float, float]:
    mu_p = float(w @ mu)
    var_p = float(w @ Sigma @ w)
    sig_p = np.sqrt(max(var_p, 0.0))
    sharpe = (mu_p - rf) / sig_p if sig_p > 0 else np.nan
    return mu_p, sig_p, sharpe

def _frontier_closed_form(mu: np.ndarray, Sigma: np.ndarray, r_min: float, r_max: float, npts: int):
    """Short-allowed frontier via equality return targeting."""
    invS = np.linalg.pinv(Sigma)
    one = np.ones_like(mu)
    A = float(one @ invS @ one)
    B = float(one @ invS @ mu)
    C = float(mu @ invS @ mu)
    D = A * C - B * B
    Rgrid = np.linspace(r_min, r_max, npts)
    risks = np.empty_like(Rgrid)
    for i, R in enumerate(Rgrid):
        w = ((C - B * R) / D) * (invS @ one) + ((A * R - B) / D) * (invS @ mu)
        risks[i] = np.sqrt(max(float(w @ Sigma @ w), 0.0))
    return risks, Rgrid

# --- Projections --------------------------------------------------------------
def _project_to_box_simplex(v: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    """
    Project onto { w_min ≤ w_i ≤ w_max, sum w_i = 1 } by bisection.
    Assumes feasibility: N*w_min ≤ 1 ≤ N*w_max and w_min ≤ w_max.
    """
    v = np.asarray(v, dtype=float)
    lo = float(w_min); hi = float(w_max)
    L = float(np.min(v - hi)) - 1.0
    U = float(np.max(v - lo)) + 1.0
    for _ in range(80):
        theta = 0.5 * (L + U)
        w = np.clip(v - theta, lo, hi)
        s = w.sum()
        if s > 1.0: L = theta
        else:       U = theta
    theta = 0.5 * (L + U)
    return np.clip(v - theta, lo, hi)

def _soft_threshold(u: np.ndarray, t: float) -> np.ndarray:
    """Componentwise soft threshold."""
    return np.sign(u) * np.maximum(np.abs(u) - t, 0.0)

def _pgd_box_simplex_l2(
    mu: np.ndarray,
    Sigma: np.ndarray,
    gamma: float,
    *,
    w_min: float,
    w_max: float,
    lam_turnover: float = 0.0,
    w_ref: Optional[np.ndarray] = None,
    iters: int = 400,
    step: Optional[float] = None,
) -> np.ndarray:
    """
    Min  1/2 w'Σw − γ μ'w + (λ/2)||w − w_ref||^2
    s.t. ∑w=1,  w_min ≤ w_i ≤ w_max.
    """
    n = mu.shape[0]
    if w_ref is None: w_ref = np.full(n, 1.0 / n)
    lam = float(max(lam_turnover, 0.0))

    if step is None:
        lam_max = float(np.max(np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))))
        L = max(lam_max, 1e-8) + lam
        step = 1.0 / L

    w = _project_to_box_simplex(np.full(n, 1.0 / n), w_min, w_max)
    for _ in range(iters):
        grad = (Sigma @ w) - gamma * mu + lam * (w - w_ref)
        w = w - step * grad
        w = _project_to_box_simplex(w, w_min, w_max)
    return w

def _pgd_box_simplex_l1(
    mu: np.ndarray,
    Sigma: np.ndarray,
    gamma: float,
    *,
    w_min: float,
    w_max: float,
    lam_l1: float,
    w_ref: Optional[np.ndarray] = None,
    iters: int = 600,
    step: Optional[float] = None,
) -> np.ndarray:
    """
    Min  1/2 w'Σw − γ μ'w + λ||w − w_ref||_1
    s.t. ∑w=1,  w_min ≤ w_i ≤ w_max.

    Prox step on (w − w_ref):  u = soft( (w − η∇f) − w_ref, ηλ ) + w_ref
    then project to box+simplex.
    """
    n = mu.shape[0]
    if w_ref is None: w_ref = np.full(n, 1.0 / n)
    lam = float(max(lam_l1, 0.0))

    if step is None:
        lam_max = float(np.max(np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))))
        L = max(lam_max, 1e-8)         # only smooth part
        step = 1.0 / L

    w = _project_to_box_simplex(np.full(n, 1.0 / n), w_min, w_max)
    for _ in range(iters):
        grad = (Sigma @ w) - gamma * mu
        z = w - step * grad
        u = _soft_threshold(z - w_ref, step * lam)
        w = w_ref + u
        w = _project_to_box_simplex(w, w_min, w_max)
    return w

def _sparsify_topk_and_project(w: np.ndarray, k: int, w_min: float, w_max: float) -> np.ndarray:
    """
    Soft cardinality: zero-out all but top-k by weight, then reproject to box+simplex.
    If w_min>0, many names will get w_min after projection (soft, not exact k).
    """
    if k is None or k <= 0 or k >= w.size:
        return _project_to_box_simplex(w, w_min, w_max)
    idx = np.argsort(w)[::-1]
    keep = idx[:k]
    w2 = np.zeros_like(w)
    w2[keep] = w[keep]
    return _project_to_box_simplex(w2, w_min, w_max)


# ──────────────────────────────────────────────────────────────────────────────
# UI – parameters
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Parameters")

c1, c2, c3 = st.columns(3)
with c1:
    solver = st.selectbox(
        "Solver",
        ["Unconstrained (closed-form)",
         "Long-only (PGD, L2 turnover)",
         "Long-only (Prox PGD, L1 turnover)"],
        index=2
    )
with c2:
    rf_annual = st.number_input("Risk-free (annual)", min_value=-1.0, max_value=1.0, value=0.0, step=0.001, format="%.3f")
with c3:
    n_frontier = st.slider("Frontier points", min_value=20, max_value=200, value=80, step=10)

# Bounds
c4, c5 = st.columns(2)
with c4:
    w_min_user = st.slider("Min weight per asset (w_min)", min_value=0.0, max_value=0.10, value=0.0, step=0.005)
with c5:
    default_cap = min(0.15, 1.0)
    w_max_user = st.slider("Max weight per asset (w_max)", min_value=0.0, max_value=1.0, value=float(default_cap), step=0.01)

# Turnover refs
c6, c7 = st.columns(2)
with c6:
    wref_src = st.selectbox("Reference weights (w_ref)", ["Equal-weight", "Last optimized (if any)"], index=0)
with c7:
    card_k = st.slider("Top-K (sparsify after solve)", min_value=0, max_value=max(0, min(N, 50)), value=0, step=1,
                       help="0 = off. Soft cardinality: keep top-K weights, reproject to box+simplex.")

# Penalties
c8, c9 = st.columns(2)
with c8:
    lam_l2 = st.slider("λ (L2 turnover)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
with c9:
    lam_l1 = st.slider("λ (L1 turnover)", min_value=0.0, max_value=0.50, value=0.05 if "L1" in solver else 0.0, step=0.01)

# Target return marker
target_enabled = st.checkbox("Set target return (annualized)", value=False)
target_ret = None
if target_enabled:
    target_ret = st.number_input("Target μ (annualized)", value=float(np.median(mu_ann)), step=0.001, format="%.4f")

# Build w_ref
if wref_src.startswith("Last") and isinstance(st.session_state.get("opt_weights"), np.ndarray):
    w_ref = st.session_state["opt_weights"]
    if w_ref.shape != (N,):
        w_ref = np.full(N, 1.0 / N)
else:
    w_ref = np.full(N, 1.0 / N)

# Feasibility adjustments
w_min_eff = float(w_min_user)
w_max_eff = float(max(w_max_user, w_min_eff))
changed = False
if N * w_min_eff > 1.0:
    w_min_eff = 1.0 / N; changed = True
if N * w_max_eff < 1.0:
    w_max_eff = max(w_max_eff, 1.0 / N); changed = True
if changed:
    st.warning(f"Bounds adjusted: w_min={w_min_eff:.3f}, w_max={w_max_eff:.3f} (need N*w_min ≤ 1 ≤ N*w_max).")

st.caption(
    f"Universe: **{N}** · Σ κ≈**{_cond_number(Sigma_ann):.2e}** · per_year=**{per_year}** · "
    f"w_min={w_min_eff:.3f}, w_max={w_max_eff:.3f}, λ₂={lam_l2:.2f}, λ₁={lam_l1:.2f}, Top-K={card_k}"
)

# Persist
st.session_state.setdefault("opt_payload", None)
st.session_state.setdefault("opt_ready", False)


# ──────────────────────────────────────────────────────────────────────────────
# Optimize
# ──────────────────────────────────────────────────────────────────────────────
def _run_optimizer():
    rf = float(rf_annual)

    if solver.startswith("Unconstrained"):
        w_mvp, w_tan = _markowitz_closed_form(mu_ann, Sigma_ann, rf)
        mu_mvp, s_mvp, sh_mvp = _weights_stats(mu_ann, Sigma_ann, w_mvp, rf)
        mu_tan, s_tan, sh_tan = _weights_stats(mu_ann, Sigma_ann, w_tan, rf)

        # Frontier (short-allowed)
        r_min, r_max = float(min(mu_ann)), float(max(mu_ann))
        if target_enabled and target_ret is not None:
            r_max = max(r_max, float(target_ret))
            r_min = min(r_min, float(target_ret))
        risks, rets = _frontier_closed_form(mu_ann, Sigma_ann, r_min, r_max, int(n_frontier))

        # Exact target
        w_target = None
        if target_enabled and target_ret is not None:
            invS = np.linalg.pinv(Sigma_ann)
            one = np.ones_like(mu_ann)
            A = float(one @ invS @ one)
            B = float(one @ invS @ mu_ann)
            C = float(mu_ann @ invS @ mu_ann)
            D = A * C - B * B
            R = float(target_ret)
            w_target = ((C - B * R) / D) * (invS @ one) + ((A * R - B) / D) * (invS @ mu_ann)

        payload = {
            "solver": "closed_form",
            "w_mvp": w_mvp, "w_tan": w_tan, "w_target": w_target,
            "frontier_risks": risks, "frontier_rets": rets,
            "stats": {"mvp": (mu_mvp, s_mvp, sh_mvp), "tan": (mu_tan, s_tan, sh_tan)},
            "w_min": None, "w_max": None, "lam_turn": 0.0, "lam_l1": 0.0,
        }
        return payload

    # Long-only — gamma sweep
    gammas = np.geomspace(1e-4, 1e2, int(n_frontier))
    Ws = []
    for g in gammas:
        if "L1" in solver:
            w = _pgd_box_simplex_l1(
                mu_ann, Sigma_ann, gamma=float(g),
                w_min=w_min_eff, w_max=w_max_eff,
                lam_l1=float(lam_l1), w_ref=w_ref,
                iters=600
            )
        else:
            w = _pgd_box_simplex_l2(
                mu_ann, Sigma_ann, gamma=float(g),
                w_min=w_min_eff, w_max=w_max_eff,
                lam_turnover=float(lam_l2), w_ref=w_ref,
                iters=500
            )
        # Sparsify (soft) if requested
        if card_k and card_k > 0:
            w = _sparsify_topk_and_project(w, int(card_k), w_min_eff, w_max_eff)
        Ws.append(w)
    Ws = np.vstack(Ws)

    # stats
    rf = float(rf_annual)
    rets = Ws @ mu_ann
    risks = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Ws, Sigma_ann, Ws), 0.0))
    sharpes = (rets - rf) / np.where(risks > 0, risks, np.nan)

    idx_mvp = int(np.nanargmin(risks))
    idx_msr = int(np.nanargmax(sharpes))
    w_mvp = Ws[idx_mvp]
    w_tan = Ws[idx_msr]

    w_target = None
    if target_enabled and target_ret is not None:
        R = float(target_ret)
        idx = int(np.nanargmin(np.abs(rets - R)))
        w_target = Ws[idx]

    payload = {
        "solver": "pgd_box_l1" if "L1" in solver else "pgd_box_l2",
        "gammas": gammas,
        "w_mvp": w_mvp, "w_tan": w_tan, "w_target": w_target,
        "frontier_risks": risks, "frontier_rets": rets,
        "Ws": Ws,
        "w_min": w_min_eff, "w_max": w_max_eff,
        "lam_turn": float(lam_l2), "lam_l1": float(lam_l1),
        "w_ref": w_ref.tolist(),
        "card_k": int(card_k or 0),
    }
    return payload


if st.button("Optimize", type="primary"):
    try:
        payload = _run_optimizer()
        st.session_state["opt_payload"] = payload
        st.session_state["opt_ready"] = True
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Render
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.get("opt_ready"):
    p = st.session_state["opt_payload"]
    risks, rets = p["frontier_risks"], p["frontier_rets"]

    w_mvp = p["w_mvp"]; w_tan = p["w_tan"]; w_target = p.get("w_target", None)

    rf = float(rf_annual)
    mu_mvp, s_mvp, sh_mvp = _weights_stats(mu_ann, Sigma_ann, w_mvp, rf)
    mu_tan, s_tan, sh_tan = _weights_stats(mu_ann, Sigma_ann, w_tan, rf)

    msr_point = (s_tan, mu_tan)
    minvar_point = (s_mvp, mu_mvp)
    custom = {}
    if w_target is not None:
        mu_tar, s_tar, _ = _weights_stats(mu_ann, Sigma_ann, w_target, rf)
        custom["Target"] = (s_tar, mu_tar)

    st.subheader("Efficient Frontier")
    st.plotly_chart(
        efficient_frontier(
            risks, rets, msr_point=msr_point, minvar_point=minvar_point, custom_points=custom
        ),
        use_container_width=True
    )

    choice = st.selectbox(
        "Portfolio to inspect",
        ["Max Sharpe", "Min Variance"] + (["Target"] if w_target is not None else []),
        index=0
    )
    w_sel = w_tan if choice == "Max Sharpe" else (w_mvp if choice == "Min Variance" else w_target)

    st.subheader(f"Weights — {choice}")
    st.plotly_chart(weights_bar(w_sel, names, sort=True, topn=min(40, len(names))), use_container_width=True)

    st.subheader("Risk Contributions")
    rc = w_sel * (Sigma_ann @ w_sel)
    st.plotly_chart(risk_contributions_bar(rc, names, sort=True, topn=min(40, len(names))), use_container_width=True)

    # Turnover diagnostics
    l1_turn = float(np.sum(np.abs(w_sel - w_ref)))
    l2_turn = float(np.sqrt(np.sum((w_sel - w_ref) ** 2)))
    cA, cB, cC = st.columns(3)
    cA.metric("Cardinality (nz)", int(np.sum(w_sel > 1e-10)))
    cB.metric("Turnover L1 vs ref", f"{l1_turn:.3f}")
    cC.metric("Turnover L2 vs ref", f"{l2_turn:.3f}")

    # Backtest (periodic)
    if isinstance(df_ret_wide, pl.DataFrame) and "date" in df_ret_wide.columns:
        st.subheader("Backtest (periodic)")
        try:
            Rm = df_ret_wide.select(names).to_numpy()
            r_p = Rm @ w_sel
            kind = infer_return_kind(df_ret_wide)
            dates = df_ret_wide["date"].to_list()

            if kind == "log": eq = np.exp(np.cumsum(r_p))
            else:            eq = np.cumprod(1.0 + r_p)

            st.plotly_chart(equity_and_drawdown(dates, eq, title=f"Equity & Drawdown — {choice}"), use_container_width=True)
            st.plotly_chart(loss_distribution(-r_p, title="Loss distribution (periodic)"), use_container_width=True)
        except Exception as e:
            st.warning(f"Backtest not available: {e}")
    else:
        st.info("Backtest requires `returns_wide` in session_state (from 01_Data).")

    # ──────────────────────────────────────────────────────────────────────────
    # Exports & Handoff
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("📤 Export")
    col1, col2, col3 = st.columns(3)

    with col1:
        buf_w = io.StringIO()
        pl.DataFrame({"ticker": names, "weight": w_sel}).write_csv(buf_w)
        st.download_button("Download Weights (CSV)", buf_w.getvalue(),
                           file_name=f"weights_{choice.replace(' ','_').lower()}.csv", mime="text/csv")

    with col2:
        buf_f = io.StringIO()
        pl.DataFrame({"risk": risks, "return": rets}).write_csv(buf_f)
        st.download_button("Download Frontier (CSV)", buf_f.getvalue(), file_name="frontier.csv", mime="text/csv")

    with col3:
        meta = {
            "solver": p["solver"],
            "choice": choice,
            "risk_free_annual": rf,
            "bounds": {"w_min": p.get("w_min", None), "w_max": p.get("w_max", None)},
            "turnover": {"lambda_L2": p.get("lam_turn", 0.0), "lambda_L1": p.get("lam_l1", 0.0), "w_ref": p.get("w_ref", None)},
            "cardinality": p.get("card_k", 0),
            "stats": {
                "mu": float(w_sel @ mu_ann),
                "sigma": float(np.sqrt(max(w_sel @ Sigma_ann @ w_sel, 0.0))),
                "turnover_L1": l1_turn,
                "turnover_L2": l2_turn,
            },
            "risk_meta": risk_meta,
            "risk_config": risk_cfg,
            "tickers": names,
            "weights": w_sel.tolist(),
        }
        blob = json.dumps(meta, ensure_ascii=False, indent=2)
        st.download_button("Download portfolio.json", blob.encode("utf-8"), file_name="portfolio.json", mime="application/json")

    # Handoff for next modules
    st.session_state["opt_weights"] = w_sel
    st.session_state["opt_choice"]  = choice
    st.session_state["opt_wmin"]    = p.get("w_min", None)
    st.session_state["opt_wmax"]    = p.get("w_max", None)
    st.session_state["opt_lam_l1"]  = p.get("lam_l1", 0.0)
    st.session_state["opt_lam_l2"]  = p.get("lam_turn", 0.0)

    st.success(
        f"Handoff ready: opt_weights ({choice}), bounds (w_min={p.get('w_min', None)}, w_max={p.get('w_max', None)}), "
        f"λ₁={p.get('lam_l1', 0.0)}, λ₂={p.get('lam_turn', 0.0)}, Top-K={p.get('card_k', 0)}."
    )

