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
    black_litterman_mu,
    capm_mu,
    pca_factor_cov,
)
from portfolio.viz.plot_utils import (
    corr_heatmap,
    corr_dendrogram,
    covariance_spectrum,
    scree_plot,
    network_corr_graph,
)

st.set_page_config(page_title="Risk Model", layout="wide")
st.title("📐 Risk Model")

# ───────────────────────────────────────────────
# Guards
# ───────────────────────────────────────────────
if "returns_wide" not in st.session_state:
    st.warning("Load data first in the Data page.")
    st.stop()

df_ret_wide: pl.DataFrame = st.session_state["returns_wide"]
ret_kind = st.session_state.get("ret_kind", "log")
tickers = [c for c in df_ret_wide.columns if c != "date"]

if not tickers:
    st.error("No return columns found.")
    st.stop()

# ───────────────────────────────────────────────
# Sidebar Inputs
# ───────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    mu_method = st.selectbox("Expected returns (μ)", ["historical", "ema", "shrunk", "capm", "black-litterman"], index=0)
    mu_span = st.number_input("EMA span (if μ=ema)", min_value=5, max_value=360, value=60, step=5)
with c2:
    cov_method = st.selectbox("Covariance (Σ)", ["sample", "oas", "lw", "ewma", "pca"], index=0)
    ewma_lambda = st.slider("EWMA λ", min_value=0.80, max_value=0.995, value=0.94, step=0.005)
    n_factors = st.slider("PCA factors (if Σ=pca)", min_value=1, max_value=len(tickers), value=min(5, len(tickers)))
with c3:
    fill_policy = st.selectbox("NaN policy", ["drop", "mean"], index=0)
    psd_enforce = st.checkbox("Enforce PSD (clip eigenvalues)", value=True)

c4, c5, c6 = st.columns(3)
with c4:
    annualize = st.checkbox("Annualize μ and Σ", value=True)
with c5:
    show_dendro = st.checkbox("Show dendrogram", value=False)
with c6:
    stress_test = st.checkbox("Enable Stress Testing", value=False)

# ───────────────────────────────────────────────
# Estimation
# ───────────────────────────────────────────────
if st.button("Estimate μ and Σ", type="primary"):
    with st.spinner("Estimating expected returns and covariance…"):
        if mu_method == "capm":
            mu = capm_mu(df_ret_wide, market="SPY")  # usar SPY o benchmark elegido
            Sigma = df_ret_wide.select(pl.exclude("date")).to_numpy().cov()
            names = tickers
        elif mu_method == "black-litterman":
            mu = black_litterman_mu(tickers)  # inputs priors + views
            Sigma = df_ret_wide.select(pl.exclude("date")).to_numpy().cov()
            names = tickers
        elif cov_method == "pca":
            mu, Sigma, names, factors = pca_factor_cov(
                df_ret_wide,
                mu_method=mu_method,
                mu_span=int(mu_span),
                n_factors=n_factors,
                annualize=annualize,
            )
        else:
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

    # ───────────────────────────────────────────
    # Stress test (opcional)
    # ───────────────────────────────────────────
    if stress_test:
        st.warning("Stress test enabled: +20% vol, -50% correlations")
        vol = np.sqrt(np.diag(Sigma))
        Corr = correlation_from_cov(Sigma)
        Sigma_stress = np.outer(vol * 1.2, vol * 1.2) * (Corr * 0.5)
        Sigma = Sigma_stress

    # ───────────────────────────────────────────
    # Diagnostics
    # ───────────────────────────────────────────
    st.subheader("Diagnostics")
    eigvals = np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))
    cond_num = float(np.max(eigvals) / max(np.min(eigvals), 1e-16)) if eigvals.size else np.nan
    eff_rank = (eigvals.sum()**2) / np.sum(eigvals**2)
    var_explained = np.cumsum(eigvals[::-1]) / eigvals.sum()

    cA, cB, cC, cD = st.columns(4)
    cA.metric("Assets (N)", len(names))
    cB.metric("Obs (T)", int(df_ret_wide.height))
    cC.metric("Eff. Rank", f"{eff_rank:.1f}")
    cD.metric("Cond. κ", f"{cond_num:.2e}")

    # ───────────────────────────────────────────
    # μ Table
    # ───────────────────────────────────────────
    st.subheader("Expected returns (μ)")
    mu_df = pl.DataFrame({"ticker": names, "mu": mu}).sort("mu", descending=True)
    st.dataframe(mu_df.to_pandas().round(6), use_container_width=True)

    # ───────────────────────────────────────────
    # Σ/ρ Visualizations
    # ───────────────────────────────────────────
    st.subheader("Correlation heatmap (clustered)")
    st.plotly_chart(corr_heatmap(Sigma, labels=names, is_cov=True), use_container_width=True)

    if show_dendro:
        st.subheader("Correlation dendrogram")
        st.plotly_chart(corr_dendrogram(Sigma, labels=names, is_cov=True), use_container_width=True)

    st.subheader("Covariance spectrum")
    st.plotly_chart(covariance_spectrum(Sigma), use_container_width=True)

    st.subheader("Scree Plot (explained variance)")
    st.plotly_chart(scree_plot(eigvals), use_container_width=True)

    st.subheader("Correlation Network Graph")
    st.plotly_chart(network_corr_graph(Sigma, names), use_container_width=True)

    # ───────────────────────────────────────────
    # Session & Export
    # ───────────────────────────────────────────
    st.session_state["mu_vec"] = mu
    st.session_state["cov_mat"] = Sigma
    st.session_state["asset_names"] = names

    st.subheader("📤 Export artifacts")
    colx, coly, colz, colw = st.columns(4)

    with colx:
        buf_mu = io.StringIO()
        pl.DataFrame({"ticker": names, "mu": mu}).write_csv(buf_mu)
        st.download_button("Download μ (CSV)", buf_mu.getvalue(), file_name="mu.csv", mime="text/csv")

    with coly:
        buf_cov = io.BytesIO()
        np.save(buf_cov, Sigma)
        st.download_button("Download Σ (.npy)", buf_cov.getvalue(), file_name="covariance.npy")

    with colz:
        Corr = correlation_from_cov(Sigma)
        buf_corr = io.BytesIO()
        np.save(buf_corr, Corr)
        st.download_button("Download ρ (.npy)", buf_corr.getvalue(), file_name="correlation.npy")

    with colw:
        buf_cov_csv = io.StringIO()
        pl.DataFrame(Sigma, schema=names).write_csv(buf_cov_csv)
        st.download_button("Download Σ (CSV wide)", buf_cov_csv.getvalue(), file_name="covariance.csv", mime="text/csv")

    st.success("μ and Σ stored in session_state.")
