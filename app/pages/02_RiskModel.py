import streamlit as st
import pandas as pd

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from portfolio.features.risk_models import expected_returns, covariance
from portfolio.viz.plot_utils import corr_heatmap

st.title("Risk Model")

if "returns" not in st.session_state:
    st.warning("Load data first in the Data page.")
    st.stop()

r = st.session_state["returns"]

col1, col2 = st.columns(2)
with col1:
    mu_method = st.selectbox("Expected returns", ["ema", "historical", "shrunk"], index=0)
    span = st.number_input("EMA span (if ema)", min_value=5, max_value=360, value=60, step=5)
with col2:
    cov_method = st.selectbox("Covariance", ["oas", "lw", "sample", "ewma"], index=0)
    lam = st.slider("EWMA λ (if ewma)", min_value=0.80, max_value=0.995, value=0.94, step=0.005)

if st.button("Estimate μ and Σ"):
    mu = expected_returns(r, method=mu_method, span=span)
    Sigma = covariance(r, method=cov_method, ewma_lambda=float(lam))

    st.subheader("Expected returns (annualized)")
    st.dataframe(mu.sort_values(ascending=False).to_frame("mu").round(6), use_container_width=True)

    st.subheader("Covariance (annualized) — shape")
    st.write(Sigma.shape)

    st.subheader("Correlation heatmap (clustered)")
    st.plotly_chart(corr_heatmap(Sigma), use_container_width=True)

    st.session_state["mu"] = mu
    st.session_state["Sigma"] = Sigma
    st.success("μ and Σ stored in session_state for the Optimizer tab.")
