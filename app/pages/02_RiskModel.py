# app/pages/02_RiskModel.py
from __future__ import annotations

import io
import numpy as np
import polars as pl
import streamlit as st

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from portfolio.features.risk_models import (
    compute_mu_sigma,
    correlation_from_cov,
)
from portfolio.viz.plot_utils import (
    corr_heatmap,
    corr_dendrogram,
    covariance_spectrum,
)

st.set_page_config(page_title="Risk Model", layout="wide")
st.title("📐 Risk Model")

# ---------------------------------------------------------------------
# Guards & inputs
# ---------------------------------------------------------------------
if "returns_wide" not in st.session_state:
    st.warning("Load data first in the Data page.")
    st.stop()

df_ret_wide: pl.DataFrame = st.session_state["returns_wide"]
ret_kind = st.session_state.get("ret_kind", "log")

tickers = [c for c in df_ret_wide.columns if c != "date"]
if not tickers:
    st.error("No return columns found.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    mu_method = st.selectbox("Expected returns (μ)", ["ema", "historical", "shrunk"], index=0)
    mu_span = st.number_input("EMA span (if μ=ema)", min_value=5, max_value=360, value=60, step=5)
with c2:
    cov_method = st.selectbox("Covariance (Σ)", ["oas", "lw", "sample", "ewma"], index=0)
    ewma_lambda = st.slider("EWMA λ (if Σ=ewma)", min_value=0.80, max_value=0.995, value=0.94, step=0.005)
with c3:
    fill_policy = st.selectbox("NaN policy", ["drop", "mean"], index=0, help="Tratamiento de NaNs antes de μ/Σ")
    psd_enforce = st.checkbox("Enforce PSD (clip eigenvalues)", value=True)

c4, c5 = st.columns([1,1])
with c4:
    annualize = st.checkbox("Annualize μ and Σ", value=True)
with c5:
    show_dendro = st.checkbox("Show dendrogram", value=False)

# ---------------------------------------------------------------------
# Compute μ & Σ
# ---------------------------------------------------------------------
if st.button("Estimate μ and Σ", type="primary"):
    with st.spinner("Estimating expected returns and covariance…"):
        mu, Sigma, names = compute_mu_sigma(
            df_ret_wide,
            mu_method=mu_method,
            mu_span=int(mu_span),
            cov_method=cov_method,
            ewma_lambda=float(ewma_lambda),
            fill=fill_policy,
            annualize=annualize,
            psd=psd_enforce,
        )

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------
    st.subheader("Diagnostics")
    eigvals = np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))
    cond_num = float(np.max(eigvals) / max(np.min(eigvals), 1e-16)) if eigvals.size else np.nan
    min_eig = float(np.min(eigvals)) if eigvals.size else np.nan
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Assets (N)", len(names))
    cB.metric("Obs (T)", int(df_ret_wide.height))
    cC.metric("Min eigenvalue", f"{min_eig:.3e}")
    cD.metric("Condition number κ", f"{cond_num:.2e}")

    # -----------------------------------------------------------------
    # μ table
    # -----------------------------------------------------------------
    st.subheader("Expected returns (μ)")
    mu_df = pl.DataFrame({"ticker": names, "mu": mu}).sort("mu", descending=True)
    st.dataframe(mu_df.to_pandas().round(6), use_container_width=True)

    # -----------------------------------------------------------------
    # Σ/ρ plots
    # -----------------------------------------------------------------
    st.subheader("Correlation heatmap (clustered)")
    fig_hm = corr_heatmap(Sigma, labels=names, is_cov=True)
    st.plotly_chart(fig_hm, use_container_width=True)

    if show_dendro:
        st.subheader("Correlation dendrogram")
        fig_den = corr_dendrogram(Sigma, labels=names, is_cov=True)
        st.plotly_chart(fig_den, use_container_width=True)

    st.subheader("Covariance spectrum")
    fig_spec = covariance_spectrum(Sigma, title="Covariance Spectrum (eigenvalues)")
    st.plotly_chart(fig_spec, use_container_width=True)

    # -----------------------------------------------------------------
    # Hand-off & export
    # -----------------------------------------------------------------
    st.session_state["mu_vec"] = mu
    st.session_state["cov_mat"] = Sigma
    st.session_state["asset_names"] = names

    st.success("μ and Σ stored in session_state for the Optimizer.")

    st.subheader("📤 Export artifacts")
    colx, coly, colz = st.columns(3)

    # μ CSV
    with colx:
        buf_mu = io.StringIO()
        (pl.DataFrame({"ticker": names, "mu": mu})).write_csv(buf_mu)
        st.download_button(
            "Download μ (CSV)",
            data=buf_mu.getvalue().encode("utf-8"),
            file_name="mu.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Σ NPY
    with coly:
        buf_cov = io.BytesIO()
        np.save(buf_cov, Sigma)
        st.download_button(
            "Download Σ (NumPy .npy)",
            data=buf_cov.getvalue(),
            file_name="covariance.npy",
            mime="application/octet-stream",
            use_container_width=True,
        )

    # ρ NPY
    with colz:
        Corr = correlation_from_cov(Sigma)
        buf_corr = io.BytesIO()
        np.save(buf_corr, Corr)
        st.download_button(
            "Download ρ (NumPy .npy)",
            data=buf_corr.getvalue(),
            file_name="correlation.npy",
            mime="application/octet-stream",
            use_container_width=True,
        )
