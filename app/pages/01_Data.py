import streamlit as st
import pandas as pd

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from portfolio.io.data_loader import get_prices
from portfolio.features.returns import simple_returns, log_returns, to_frequency, winsorize, missing_report

st.title("Data")

col1, col2, col3 = st.columns([3,2,2])
with col1:
    tickers = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,META")
with col2:
    start = st.date_input("Start", pd.to_datetime("2018-01-01"))
with col3:
    end = st.date_input("End", pd.to_datetime("today"))

with st.expander("Options"):
    freq = st.selectbox("Return frequency", ["D","W","M"], index=0)
    use_log = st.checkbox("Use log-returns", value=False)
    do_wins = st.checkbox("Winsorize outliers (1%)", value=True)
    force_refresh = st.checkbox("Force refresh (ignore cache)", value=False)

if st.button("Load & Preview"):
    tks = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    prices = get_prices(tks, start, end, adjust=True, force_refresh=force_refresh)

    st.subheader("Prices (tail)")
    st.caption(f"Shape: {prices.shape}")
    st.dataframe(prices.tail().round(2), use_container_width=True)

    st.subheader("Missing report")
    mr = missing_report(prices)
    st.dataframe(mr.sort_values("missing_pct", ascending=False), use_container_width=True)

    st.subheader("Returns")
    r = log_returns(prices) if use_log else simple_returns(prices)
    r = to_frequency(r, freq)
    if do_wins:
        r = winsorize(r, q=0.01)
    st.caption(f"Returns shape: {r.shape}")
    st.dataframe(r.tail().round(6), use_container_width=True)

    st.session_state["prices"] = prices
    st.session_state["returns"] = r
    st.success("Data loaded and stored in session_state.")
