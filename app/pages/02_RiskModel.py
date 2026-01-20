# app/pages/02_RiskModel.py
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
import streamlit as st

# Local project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from portfolio.core.compat import UTC
from portfolio.features.risk_models import (
    black_litterman_mu,
    capm_mu,
    compute_mu_sigma,
    correlation_from_cov,
    pca_factor_cov,
)
from portfolio.io.cache import save_json
from portfolio.optim.robust import clean_covariance_rmt
from portfolio.viz.plot_utils import (
    HeatmapOrder,
    corr_dendrogram,
    corr_heatmap,
    corr_heatmap_gl,
    covariance_spectrum,
    network_corr_graph,
    risk_contributions_bar,
    scree_plot,
    show_plot,
)
from portfolio.viz.rmt_plots import plot_eigenvalue_spectrum

# Design System
from app.design_system import COLORS, get_global_styles

# ──────────────────────────────────────────────────────────────────────────────
# Config & guards
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Risk Model", layout="wide")

# Apply global styles
st.markdown(get_global_styles(), unsafe_allow_html=True)

# Page title with Apple-style
st.markdown(f"""
<div style="margin-bottom: 32px;">
    <h1 style="
        font-size: 2.5rem;
        font-weight: 600;
        color: {COLORS['text_primary']};
        margin-bottom: 8px;
    ">
        📐 Risk Model
    </h1>
    <p style="
        font-size: 1rem;
        color: {COLORS['text_secondary']};
        line-height: 1.5;
    ">
        Estimate expected returns (μ) and covariance matrix (Σ) with configurable shrinkage and PCA
    </p>
</div>
""", unsafe_allow_html=True)

if "returns_wide" not in st.session_state:
    st.warning("Load data first in the Data page.")
    st.stop()

df_ret_wide: pl.DataFrame = st.session_state["returns_wide"]


def _validate_returns_wide(df: pl.DataFrame) -> pl.DataFrame:
    """Validate and clean wide returns frame for risk modeling."""
    if not isinstance(df, pl.DataFrame):
        st.error("returns_wide in session_state is not a Polars DataFrame.")
        st.stop()
    if "date" not in df.columns:
        st.error("returns_wide must include a 'date' column.")
        st.stop()

    # Normalize date type and order; enforce unique dates
    df = df.with_columns(pl.col("date").cast(pl.Datetime, strict=False)).sort("date")
    if df["date"].n_unique() < df.height:
        st.warning("Duplicate dates detected — keeping last per timestamp.")
        df = df.unique(subset=["date"], keep="last")

    value_cols = [c for c in df.columns if c != "date"]

    # Force numeric dtype for all asset columns
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in value_cols])

    # Replace non-finite values with null
    df = df.with_columns(
        [
            pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
            for c in value_cols
        ]
    )

    # Drop columns that are entirely null
    null_counts_row = df.select([pl.col(c).is_null().sum().alias(c) for c in value_cols]).row(0)
    n_rows = df.height
    drop_cols = [
        c
        for c, nnull in zip(value_cols, null_counts_row)  # noqa: B905
        if nnull is not None and int(nnull) == n_rows
    ]
    if drop_cols:
        st.warning(f"Dropping empty return columns: {', '.join(drop_cols)}")
        df = df.drop(drop_cols)
        value_cols = [c for c in value_cols if c not in drop_cols]

    # Drop near-constant columns (σ≈0) that break covariance estimation
    if value_cols:
        stds_row = df.select([pl.col(c).std(ddof=1).alias(c) for c in value_cols]).row(0)
        const_cols = []
        for c, s in zip(value_cols, stds_row, strict=False):
            if s is None or not np.isfinite(s) or float(s) <= 1e-14:
                const_cols.append(c)
        if const_cols:
            st.warning(f"Dropping near-constant columns (σ≈0): {', '.join(const_cols)}")
            df = df.drop(const_cols)

    # Ensure enough data for risk modeling
    if df.height < 2 or len([c for c in df.columns if c != "date"]) == 0:
        st.error("Not enough valid data after validation for risk modeling.")
        st.stop()

    return df


df_ret_wide = _validate_returns_wide(df_ret_wide)
tickers = [c for c in df_ret_wide.columns if c != "date"]
if not tickers:
    st.error("No return columns found.")
    st.stop()

# Persistent state
st.session_state.setdefault("risk_payload", None)
st.session_state.setdefault("risk_ready", False)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _json_default(o: Any) -> Any:
    """JSON serializer for numpy and datetime objects."""
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            pass

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _infer_per_year(dates: pl.Series) -> int:
    """Infer periods per year from median calendar spacing."""
    s = dates.sort()
    if s.len() < 2:
        return 252
    dt_days = (s.diff().dt.total_days()).drop_nulls()
    med = float(dt_days.median())
    if med <= 3.0:
        return 252  # daily
    if med <= 9.0:
        return 52  # weekly
    return 12  # monthly (approx)


def _apply_fill_policy(
    df_wide: pl.DataFrame,
    policy: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return filled wide returns and an imputation report."""
    if policy == "drop":
        original_h = df_wide.height
        df_filled = df_wide.drop_nulls()
        dropped = original_h - df_filled.height
        report = pl.DataFrame(
            {"policy": ["drop"], "rows_dropped": [int(dropped)], "imputed_pct": [0.0]}
        )
        return df_filled, report

    # Mean-impute per asset
    value_cols = [c for c in df_wide.columns if c != "date"]
    means = df_wide.select([pl.col(c).mean().alias(c) for c in value_cols])
    na_counts = df_wide.select([pl.col(c).is_null().sum().alias(c) for c in value_cols])

    df_filled = df_wide.clone()
    for c in value_cols:
        m = means.select(c).item()
        df_filled = df_filled.with_columns(
            pl.when(pl.col(c).is_null()).then(pl.lit(m)).otherwise(pl.col(c)).alias(c)
        )

    total_cells = df_wide.height * len(value_cols)
    total_na = int(sum(int(x or 0) for x in na_counts.row(0)))
    imputed_pct = (100.0 * total_na / total_cells) if total_cells else 0.0

    report = pl.DataFrame({"policy": ["mean"], "rows_dropped": [0], "imputed_pct": [imputed_pct]})
    return df_filled, report


def _annualize(mu: np.ndarray, Sigma: np.ndarray, per_year: int) -> tuple[np.ndarray, np.ndarray]:
    """Annualize mean and covariance."""
    mu_a = mu * float(per_year)
    Sigma_a = Sigma * float(per_year)
    return mu_a, Sigma_a


def _apply_ridge(Sigma: np.ndarray, eps: float) -> np.ndarray:
    """Diagonal ridge regularization."""
    if eps <= 0:
        return Sigma
    n = Sigma.shape[0]
    return Sigma + np.eye(n) * float(eps)


def _cond_number(S: np.ndarray) -> float:
    """Condition number κ(Σ) based on eigenvalues."""
    if S.size == 0:
        return float("nan")
    S_sym = 0.5 * (S + S.T)
    vals = np.linalg.eigvalsh(S_sym)
    if vals.size == 0:
        return float("nan")
    lam_min = float(np.min(vals))
    lam_max = float(np.max(vals))
    return float(lam_max / max(lam_min, 1e-16))


def _fingerprint(names: list[str], params: dict[str, Any]) -> str:
    """Short stable fingerprint based on tickers and params."""
    blob = json.dumps({"tickers": names, "params": params}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _ewma_default(per_year: int) -> float:
    """Reasonable EWMA lambda default given sampling frequency."""
    if per_year >= 250:
        return 0.94
    if per_year >= 50:
        return 0.80
    return 0.60


# ──────────────────────────────────────────────────────────────────────────────
# Black–Litterman helpers (simple views builder)
# ──────────────────────────────────────────────────────────────────────────────
def _build_PQ_absolute(names: list[str], asset: str, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Absolute view: r_i = q."""
    N = len(names)
    P = np.zeros((1, N), dtype=float)
    try:
        idx = names.index(asset)
    except ValueError as e:
        raise ValueError(f"Asset '{asset}' not found in universe.") from e
    P[0, idx] = 1.0
    Q = np.array([q], dtype=float)
    return P, Q


def _build_PQ_relative(
    names: list[str],
    long: str,
    short: str,
    q: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Relative view: r_i − r_j = q."""
    N = len(names)
    P = np.zeros((1, N), dtype=float)
    try:
        i = names.index(long)
        j = names.index(short)
    except ValueError as e:
        raise ValueError(f"Relative view asset not in universe: {e}") from e
    P[0, i] = 1.0
    P[0, j] = -1.0
    Q = np.array([q], dtype=float)
    return P, Q


def _omega_from_confidence(
    P: np.ndarray,
    Sigma: np.ndarray,
    tau: float,
    confidence: float | list[float],
) -> np.ndarray:
    """
    Diagonal Ω from P, Σ and tau.
    Higher confidence → lower variance (Ω scaled by 1/confidence).
    """
    K = P.shape[0]
    base = np.diag(np.diag(P @ (tau * Sigma) @ P.T))  # (K, K)

    if np.isscalar(confidence):
        c = float(np.clip(float(confidence), 1e-3, 1.0))
        return base * (1.0 / c)

    conf = np.asarray(confidence, dtype=float).reshape(-1)
    if conf.shape[0] != K:
        raise ValueError("Length of confidence list must equal number of views.")
    conf = np.clip(conf, 1e-3, 1.0)
    return base * (1.0 / conf)


# ──────────────────────────────────────────────────────────────────────────────
# Core risk pipeline
# ──────────────────────────────────────────────────────────────────────────────
def _risk_pipeline(
    df_ret_wide: pl.DataFrame,
    *,
    mu_method: str,
    mu_span: int,
    mu_shrink_target: str | None,
    mu_rf_annual: float | None,
    cov_method: str,
    ewma_lambda: float,
    n_factors: int,
    fill_policy: str,
    per_year: int,
    enforce_psd: bool,
    ridge_eps: float,
    stress_test: bool,
    heatmap_method: str,
    heatmap_optimal: bool,
) -> dict[str, Any]:
    """Compute μ, Σ and diagnostics for the selected configuration."""
    # 1) Handle missing data
    df_clean, fill_report = _apply_fill_policy(df_ret_wide, fill_policy)
    names = [c for c in df_clean.columns if c != "date"]
    X = df_clean.select(pl.exclude("date")).to_numpy()

    # 2) Mean vector target for shrinkage
    mu_shrink_to_vec: np.ndarray | None = None
    if mu_method == "shrunk" and mu_shrink_target is not None:
        if mu_shrink_target == "zero":
            mu_shrink_to_vec = np.zeros(len(names))
        elif mu_shrink_target == "equal":
            base = np.nanmean(X, axis=0)
            target_val = float(np.nanmean(base))
            mu_shrink_to_vec = np.full(len(names), target_val, dtype=float)
        elif mu_shrink_target == "rf":
            rf_ann = float(mu_rf_annual or 0.0)
            rf_per = rf_ann / max(per_year, 1)
            mu_shrink_to_vec = np.full(len(names), rf_per, dtype=float)

    # 3) Mean / covariance estimation (non-annualized)
    if mu_method == "capm":
        mu = capm_mu(df_clean, market="SPY", annualize=False)
        Sigma = np.cov(X, rowvar=False)
    elif mu_method == "black-litterman":
        mu = black_litterman_mu(df_clean, Sigma=None, annualize=False)
        Sigma = np.cov(X, rowvar=False)
    elif cov_method == "pca":
        mu, Sigma, names_out, _ = pca_factor_cov(
            df_clean,
            mu_method=mu_method,
            mu_span=int(mu_span),
            n_factors=int(n_factors),
            annualize=False,
        )
        names = list(names_out)
    else:
        mu, Sigma, names_out = compute_mu_sigma(
            df_clean,
            mu_method=mu_method,
            mu_span=int(mu_span),
            mu_shrink_to=mu_shrink_to_vec,
            cov_method=cov_method,
            ewma_lambda=float(ewma_lambda),
            fill="none",  # already filled
            annualize=False,
            psd=enforce_psd,
        )
        names = list(names_out)

    # 4) Annualization
    mu, Sigma = _annualize(mu, Sigma, per_year)

    # 5) Pre-ridge conditioning
    cond_pre = _cond_number(Sigma)

    # 6) Ridge regularization
    Sigma = _apply_ridge(Sigma, ridge_eps)

    # 7) Optional stress test
    if stress_test:
        vol = np.sqrt(np.diag(Sigma))
        Corr = correlation_from_cov(Sigma)
        Sigma = np.outer(vol * 1.2, vol * 1.2) * (Corr * 0.5)

    # 8) Post diagnostics
    S_sym = 0.5 * (Sigma + Sigma.T)
    eigvals = np.linalg.eigvalsh(S_sym)
    if eigvals.size > 0:
        lam_min = float(np.min(eigvals))
        lam_max = float(np.max(eigvals))
        cond_post = float(lam_max / max(lam_min, 1e-16))
        eff_rank = float((eigvals.sum() ** 2) / np.sum(eigvals**2))
    else:
        lam_min = float("nan")
        lam_max = float("nan")
        cond_post = float("nan")
        eff_rank = float("nan")

    params = {
        "mu_method": mu_method,
        "mu_span": int(mu_span),
        "mu_shrink_target": mu_shrink_target,
        "mu_rf_annual": float(mu_rf_annual or 0.0),
        "cov_method": cov_method,
        "ewma_lambda": float(ewma_lambda),
        "n_factors": int(n_factors),
        "fill_policy": fill_policy,
        "per_year": int(per_year),
        "enforce_psd": bool(enforce_psd),
        "ridge_eps": float(ridge_eps),
        "stress_test": bool(stress_test),
        "heatmap_method": heatmap_method,
        "heatmap_optimal": bool(heatmap_optimal),
    }
    meta = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "params": params,
        "diagnostics": {
            "lambda_min": lam_min,
            "lambda_max": lam_max,
            "cond_kappa_pre": cond_pre,
            "cond_kappa_post": cond_post,
            "effective_rank": eff_rank,
        },
        "tickers": names,
        "fingerprint": _fingerprint(names, params),
    }

    return {
        "mu": mu,
        "Sigma": Sigma,
        "names": names,
        "eigvals": eigvals,
        "cond_pre": cond_pre,
        "cond_post": cond_post,
        "eff_rank": eff_rank,
        "fill_report": fill_report,
        "meta": meta,
    }


# ──────────────────────────────────────────────────────────────────────────────
# UI – inputs
# ──────────────────────────────────────────────────────────────────────────────
default_per_year = _infer_per_year(df_ret_wide["date"])
ewma_default = _ewma_default(default_per_year)
st.caption(
    f"Annualization default inferred: **{default_per_year}** periods/year · "
    f"EWMA λ default: **{ewma_default:.2f}**"
)

c1, c2, c3 = st.columns(3)
with c1:
    mu_method = st.selectbox(
        "Expected returns (μ)",
        ["historical", "ema", "shrunk", "capm", "black-litterman"],
        index=0,
    )
    mu_span = st.number_input(
        "EMA span (if μ=ema)",
        min_value=5,
        max_value=360,
        value=60,
        step=5,
    )

with c2:
    cov_method = st.selectbox(
        "Covariance (Σ)",
        ["sample", "oas", "lw", "ewma", "pca"],
        index=0,
    )
    ewma_lambda = st.slider(
        "EWMA λ",
        min_value=0.80,
        max_value=0.995,
        value=float(ewma_default),
        step=0.005,
    )
    n_factors = st.slider(
        "PCA factors (if Σ=pca)",
        min_value=1,
        max_value=len(tickers),
        value=min(5, len(tickers)),
    )

with c3:
    fill_policy = st.selectbox("NaN policy", ["drop", "mean"], index=0)
    enforce_psd = st.checkbox("Enforce PSD (clip eigenvalues)", value=True)

# Extra controls for μ-shrunk
mu_shrink_target: str | None = None
mu_rf_annual: float = 0.0
if mu_method == "shrunk":
    st.markdown("**μ shrinkage target**")
    _tcol1, _tcol2 = st.columns(2)
    with _tcol1:
        mu_shrink_target = st.selectbox("Target", ["zero", "equal", "rf"], index=0)
    with _tcol2:
        mu_rf_annual = st.number_input(
            "Risk-free (annual, in return units)",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.001,
            format="%.3f",
        )

c4, c5, c6 = st.columns(3)
with c4:
    per_year = st.selectbox(
        "Annualization: periods/year",
        [252, 260, 52, 12],
        index=[252, 260, 52, 12].index(
            default_per_year if default_per_year in (252, 260, 52, 12) else 252
        ),
    )
with c5:
    ridge_eps = st.number_input(
        "Ridge ε on Σ (0 = off)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.0001,
        format="%.4f",
    )
with c6:
    stress_test = st.checkbox("Enable Stress Testing (+20% vol, -50% corr)", value=False)

h1, h2, h3 = st.columns(3)
with h1:
    heatmap_method = st.selectbox(
        "Heatmap linkage",
        ["single", "complete", "average", "ward"],
        index=2,
    )
with h2:
    heatmap_optimal = st.checkbox("Optimal leaf ordering", value=True)
with h3:
    show_dendro = st.checkbox("Show dendrogram", value=False)


# ──────────────────────────────────────────────────────────────────────────────
# Action – compute risk model and store in session_state
# ──────────────────────────────────────────────────────────────────────────────
if st.button("Estimate μ and Σ", type="primary"):
    payload = _risk_pipeline(
        df_ret_wide,
        mu_method=mu_method,
        mu_span=int(mu_span),
        mu_shrink_target=mu_shrink_target,
        mu_rf_annual=float(mu_rf_annual or 0.0),
        cov_method=cov_method,
        ewma_lambda=float(ewma_lambda),
        n_factors=int(n_factors),
        fill_policy=fill_policy,
        per_year=int(per_year),
        enforce_psd=bool(enforce_psd),
        ridge_eps=float(ridge_eps),
        stress_test=bool(stress_test),
        heatmap_method=heatmap_method,
        heatmap_optimal=bool(heatmap_optimal),
    )
    st.session_state["risk_payload"] = payload
    st.session_state["risk_ready"] = True


# ──────────────────────────────────────────────────────────────────────────────
# Render & handoff (only when risk model is ready)
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.get("risk_ready"):
    p = st.session_state["risk_payload"]
    mu: np.ndarray = p["mu"]
    Sigma: np.ndarray = p["Sigma"]
    names: list[str] = p["names"]
    eigvals: np.ndarray = p["eigvals"]
    cond_pre: float = p["cond_pre"]
    cond_post: float = p["cond_post"]
    fill_report: pl.DataFrame = p["fill_report"]
    meta: dict[str, Any] = p["meta"]

    # NaN policy report
    st.subheader("NaN Policy Report")
    st.dataframe(fill_report.to_pandas(), width="stretch")

    # Diagnostics
    st.subheader("Diagnostics")
    lam_min = float(eigvals.min()) if eigvals.size else np.nan
    lam_max = float(eigvals.max()) if eigvals.size else np.nan
    cA, cB, cC, cD, cE = st.columns(5)
    cA.metric("Assets (N)", len(names))
    cB.metric("Obs (T)", int(df_ret_wide.height))
    cC.metric("λ_min", f"{lam_min:.2e}")
    cD.metric("κ (pre-ridge)", f"{cond_pre:.2e}")
    cE.metric("κ (post-ridge)", f"{cond_post:.2e}")

    # μ table
    st.subheader("Expected returns (μ, annualized)")
    mu_df = pl.DataFrame({"ticker": names, "mu": mu}).sort("mu", descending=True)
    st.dataframe(mu_df.to_pandas().round(6), width="stretch")

    # Correlation heatmap
    st.subheader("Correlation heatmap (clustered)")
    order_cfg = HeatmapOrder(
        clustered=True,
        method=meta["params"]["heatmap_method"],
        optimal=bool(meta["params"]["heatmap_optimal"]),
    )

    if len(names) > 200:
        show_plot(
            corr_heatmap_gl(Sigma, labels=names, is_cov=True, order=order_cfg),
            key="risk-heatmap-gl",
        )
    else:
        show_plot(
            corr_heatmap(Sigma, labels=names, is_cov=True, order=order_cfg),
            key="risk-heatmap",
        )

    # Dendrogram
    if show_dendro:
        st.subheader("Correlation dendrogram")
        show_plot(
            corr_dendrogram(
                Sigma,
                labels=names,
                is_cov=True,
                method=meta["params"]["heatmap_method"],
            ),
            key="risk-dendrogram",
        )

    # Covariance spectrum
    # Covariance spectrum with RMT overlay
    st.subheader("Random Matrix Theory: Signal vs Noise")

    # RMT Calc
    T_obs = int(df_ret_wide.height)
    N_assets = len(names)

    # 1. Spectrum Plot
    st.markdown("#### Eigenvalue Spectrum (Marcenko-Pastur)")
    fig_rmt = plot_eigenvalue_spectrum(Sigma, T=T_obs, N=N_assets)
    show_plot(fig_rmt, key="risk-rmt-spectrum")

    # 2. Denoising
    st.markdown("#### RMT Denoising")
    apply_rmt = st.toggle("Show RMT-Cleaned Correlation Heatmap", value=False)

    if apply_rmt:
        Sigma_clean = clean_covariance_rmt(Sigma, T=T_obs, N=N_assets)
        show_plot(
            corr_heatmap(
                Sigma_clean,
                labels=names,
                is_cov=True,
                order=order_cfg,
                title="RMT Cleaned Correlation",
            ),
            key="risk-heatmap-rmt",
        )
    else:
        st.info("Toggle above to clean the matrix using Marcenko-Pastur filtering.")

    # Standard Spectrum (legacy)
    with st.expander("Legacy Spectrum View"):
        show_plot(covariance_spectrum(Sigma), key="risk-cov-spectrum")

    # Scree plot
    st.subheader("Scree Plot (explained variance)")
    show_plot(scree_plot(eigvals), key="risk-scree")

    # Correlation network
    st.subheader("Correlation Network Graph")
    show_plot(network_corr_graph(Sigma, names), key="risk-netgraph")

    # Risk contributions (equal-weight benchmark)
    st.subheader("Risk Contributions (Equal-Weight Benchmark)")
    if len(names) > 0:
        w_eq = np.full(len(names), 1.0 / len(names))
        rc = w_eq * (Sigma @ w_eq)
        show_plot(
            risk_contributions_bar(rc, names, sort=True, topn=min(30, len(names))),
            key="risk-rc-ew",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Black–Litterman (views) – interactive posterior μ
    # ──────────────────────────────────────────────────────────────────────
    with st.expander("🧠 Black–Litterman (Views)", expanded=False):
        st.caption("Build absolute or relative views and adjust Ω from confidence.")
        col_bl1, col_bl2, col_bl3 = st.columns(3)
        with col_bl1:
            bl_tau = st.number_input(
                "τ (tau)",
                min_value=0.0001,
                max_value=1.0,
                value=0.05,
                step=0.01,
                format="%.4f",
            )
        with col_bl2:
            bl_conf_mode = st.selectbox("Confidence mode", ["scalar", "per-view"], index=0)
        with col_bl3:
            bl_conf_scalar = st.slider(
                "Confidence (scalar)",
                min_value=0.05,
                max_value=1.0,
                value=0.5,
                step=0.05,
            )

        if "bl_views" not in st.session_state:
            st.session_state["bl_views"] = []  # list[dict[str, Any]]

        st.markdown("**Add view**")
        kind = st.radio("Type", ["absolute", "relative"], horizontal=True)

        if kind == "absolute":
            ca1, ca2 = st.columns([2, 1])
            with ca1:
                a_asset = st.selectbox("Asset", names, index=0, key="bl_abs_asset")
            with ca2:
                a_q = st.number_input(
                    "q (target return, per-period)",
                    value=0.001,
                    step=0.001,
                    format="%.4f",
                    key="bl_abs_q",
                )
            a_conf = st.slider(
                "Confidence (this view)",
                0.05,
                1.0,
                0.5,
                0.05,
                key="bl_abs_conf",
            )
            if st.button("➕ Add absolute view"):
                st.session_state["bl_views"].append(
                    {
                        "kind": "absolute",
                        "asset": a_asset,
                        "q": float(a_q),
                        "conf": float(a_conf),
                    }
                )
        else:
            cr1, cr2, cr3 = st.columns([2, 2, 1])
            with cr1:
                long_a = st.selectbox("Long asset", names, index=0, key="bl_rel_long")
            with cr2:
                short_a = st.selectbox("Short asset", names, index=1, key="bl_rel_short")
            with cr3:
                r_q = st.number_input(
                    "q (r_i - r_j)",
                    value=0.0,
                    step=0.001,
                    format="%.4f",
                    key="bl_rel_q",
                )
            r_conf = st.slider(
                "Confidence (this view)",
                0.05,
                1.0,
                0.5,
                0.05,
                key="bl_rel_conf",
            )
            if st.button("➕ Add relative view"):
                if long_a == short_a:
                    st.warning("Long and short assets cannot be the same.")
                else:
                    st.session_state["bl_views"].append(
                        {
                            "kind": "relative",
                            "long": long_a,
                            "short": short_a,
                            "q": float(r_q),
                            "conf": float(r_conf),
                        }
                    )

        # Current views
        if st.session_state["bl_views"]:
            st.write("**Current views**")
            st.dataframe(pl.DataFrame(st.session_state["bl_views"]).to_pandas(), width="stretch")
            if st.button("🗑️ Clear views"):
                st.session_state["bl_views"] = []

        # Compute posterior μ
        if st.session_state["bl_views"]:
            P_list, Q_list, conf_list = [], [], []
            for v in st.session_state["bl_views"]:
                if v["kind"] == "absolute":
                    P, Q = _build_PQ_absolute(names, v["asset"], v["q"])
                else:
                    P, Q = _build_PQ_relative(names, v["long"], v["short"], v["q"])
                P_list.append(P)
                Q_list.append(Q)
                conf_list.append(v["conf"])

            P = np.vstack(P_list)  # (K, N)
            Q = np.concatenate(Q_list)  # (K,)

            per_y = int(meta["params"]["per_year"])
            mu_prior = np.asarray(mu, dtype=float)
            mu_prior_per = mu_prior / max(per_y, 1)
            Sigma_per = Sigma / max(per_y, 1)

            Omega = (
                _omega_from_confidence(P, Sigma_per, bl_tau, bl_conf_scalar)
                if bl_conf_mode == "scalar"
                else _omega_from_confidence(P, Sigma_per, bl_tau, conf_list)
            )

            inv_term = np.linalg.inv(P @ (bl_tau * Sigma_per) @ P.T + Omega)
            middle = (bl_tau * Sigma_per) @ P.T @ inv_term
            mu_post_per = mu_prior_per + middle @ (Q - P @ mu_prior_per)
            mu_post = mu_post_per * float(per_y)

            st.subheader("μ prior vs μ posterior (annualized)")
            mu_tbl = pl.DataFrame(
                {
                    "ticker": names,
                    "mu_prior": mu_prior,
                    "mu_post": mu_post,
                    "delta": mu_post - mu_prior,
                }
            ).sort("delta", descending=True)
            st.dataframe(mu_tbl.to_pandas().round(6), width="stretch")

            use_bl_for_optimizer = st.toggle("Use BL posterior μ for Optimizer", value=False)
            if use_bl_for_optimizer:
                st.session_state["mu_vec"] = mu_post
                meta_bl = {
                    **meta,
                    "bl": {
                        "tau": float(bl_tau),
                        "K": int(P.shape[0]),
                        "conf_mode": bl_conf_mode,
                    },
                }
                st.session_state["risk_meta"] = meta_bl
                st.success("Using Black–Litterman posterior μ for Optimizer handoff.")
            else:
                # If not using BL, keep original μ in session
                st.session_state["mu_vec"] = mu

        else:
            # No views → use original μ
            st.session_state["mu_vec"] = mu

    # ──────────────────────────────────────────────────────────────────────
    # Handoff to downstream pages + export artifacts
    # ──────────────────────────────────────────────────────────────────────
    mu_vec = np.asarray(st.session_state.get("mu_vec", mu), dtype=float).ravel()
    Sigma_clean = np.asarray(Sigma, dtype=float)
    Sigma_clean = np.nan_to_num(Sigma_clean, nan=0.0, posinf=0.0, neginf=0.0)
    Sigma_clean = 0.5 * (Sigma_clean + Sigma_clean.T)

    # Persist to session_state for Optimizer / Backtest
    st.session_state["mu_vec"] = mu_vec
    st.session_state["cov_mat"] = Sigma_clean
    st.session_state["asset_names"] = names

    # Risk meta / config
    risk_meta = st.session_state.get("risk_meta", meta)
    if not isinstance(risk_meta, dict) or "params" not in risk_meta:
        risk_meta = meta

    params = risk_meta.get("params", {})
    risk_cfg = {
        "tickers": ",".join(names),
        "per_year": int(params.get("per_year", 252)),
        "mu_method": str(params.get("mu_method", "mean")),
        "cov_method": str(params.get("cov_method", "sample_cov")),
        "ewma_lambda": float(params.get("ewma_lambda", 0.94)),
        "ridge_eps": float(params.get("ridge_eps", 0.0)),
        "psd": bool(params.get("enforce_psd", True)),
        "fingerprint": str(risk_meta.get("fingerprint", meta.get("fingerprint", "ad-hoc"))),
    }

    st.session_state["risk_meta"] = risk_meta
    st.session_state["risk_config"] = risk_cfg
    st.session_state["risk_timestamp"] = datetime.now(UTC).isoformat()

    # Persist risk_model metadata to cache
    try:
        save_json("risk_model", risk_cfg, risk_meta)
        st.caption(
            f"Saved risk model metadata to cache (fingerprint: **{risk_cfg['fingerprint']}**)."
        )
    except Exception as e:
        st.warning(f"Could not persist risk_model.json to cache: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Export artifacts
    # ──────────────────────────────────────────────────────────────────────
    st.subheader("📤 Export artifacts")
    colx, coly, colz, colw, colm = st.columns(5)

    # μ (expected returns) CSV export
    with colx:
        buf_mu = io.StringIO()
        pl.DataFrame({"ticker": names, "mu": mu_vec}).write_csv(buf_mu)
        st.download_button(
            "Download μ (CSV)",
            buf_mu.getvalue(),
            file_name="mu.csv",
            mime="text/csv",
        )

    # Σ (covariance matrix) .npy export
    with coly:
        buf_cov = io.BytesIO()
        np.save(buf_cov, Sigma_clean)
        st.download_button(
            "Download Σ (.npy)",
            buf_cov.getvalue(),
            file_name="covariance.npy",
        )

    # ρ (correlation matrix) .npy export
    with colz:
        Corr = correlation_from_cov(Sigma_clean)
        buf_corr = io.BytesIO()
        np.save(buf_corr, Corr)
        st.download_button(
            "Download ρ (.npy)",
            buf_corr.getvalue(),
            file_name="correlation.npy",
        )

    # Σ (covariance matrix) CSV wide export with shape checks
    with colw:
        n_cols = len(names)
        if Sigma_clean.shape != (n_cols, n_cols):
            st.warning(
                f"Σ shape {Sigma_clean.shape} != ({n_cols},{n_cols}). Exporting trimmed square."
            )
            m = min(n_cols, Sigma_clean.shape[0], Sigma_clean.shape[1])
            Sigma_csv = Sigma_clean[:m, :m]
            cols_csv = names[:m]
        else:
            Sigma_csv = Sigma_clean
            cols_csv = names

        buf_cov_csv = io.StringIO()
        pl.DataFrame(Sigma_csv, schema=cols_csv).write_csv(buf_cov_csv)
        st.download_button(
            "Download Σ (CSV wide)",
            buf_cov_csv.getvalue(),
            file_name="covariance.csv",
            mime="text/csv",
        )

    # JSON metadata export
    with colm:
        meta_blob = json.dumps(
            {**risk_meta, "exported_at_utc": datetime.now(UTC).isoformat()},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        st.download_button(
            f"Download risk_model_{risk_cfg['fingerprint']}.json",
            meta_blob.encode("utf-8"),
            file_name=f"risk_model_{risk_cfg['fingerprint']}.json",
            mime="application/json",
        )

    # Conditioning warnings
    if np.isfinite(cond_post) and cond_post > 1e6:
        st.warning("High post-conditioning number κ; consider increasing Ridge ε or using OAS/PCA.")
    if eigvals.size and float(np.min(eigvals)) < 0:
        st.warning("Σ has negative eigenvalues; PSD clipping or higher Ridge ε recommended.")
