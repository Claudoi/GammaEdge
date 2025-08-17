# app/pages/01_Data.py
from __future__ import annotations

import io
import time
import streamlit as st
import polars as pl

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from portfolio.io.data_loader import get_prices_long
from portfolio.io.cache import save_pl, load_pl, cache_path, invalidate, age_seconds
from portfolio.features.returns import (
    compute_returns_from_prices_long,
    winsorize_long,
    long_to_wide,
    returns_to_frequency_wide,
    summary_stats,
    missing_report_wide,
)
from datetime import date

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Data", layout="wide")
st.title("📦 Data Module")

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar / Inputs
# ──────────────────────────────────────────────────────────────────────────────
with st.container():
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        tickers_input = st.text_input(
            "Tickers (comma-separated)",
            "AAPL,MSFT,GOOGL,AMZN,META",
        )
    with col2:
        start = st.date_input("Start", value=date(2018, 1, 1))
    with col3:
        end = st.date_input("End", value=date.today())

with st.expander("Options", expanded=False):
    colA, colB, colC, colD = st.columns(4)
    with colA:
        freq_prices = st.selectbox(
            "Price resample",
            options=[("Daily", "1d"), ("Weekly", "1w"), ("Monthly", "1mo")],
            index=1,
            format_func=lambda x: x[0],
            help="Re-muestreo de precios antes de calcular retornos (último precio de la ventana).",
        )[1]
    with colB:
        ret_kind = st.selectbox(
            "Return type",
            options=["log", "simple"],
            index=0,
            help="Retornos logarítmicos o simples.",
        )
    with colC:
        winsor_p = st.slider(
            "Winsor p (per tail)",
            min_value=0.0,
            max_value=0.10,
            value=0.01,
            step=0.005,
            help="Recorta el 1% por defecto de cada cola por ticker.",
        )
    with colD:
        freq_returns = st.selectbox(
            "Output return frequency",
            options=[("Daily", "1d"), ("Weekly", "1w"), ("Monthly", "1mo")],
            index=1,
            format_func=lambda x: x[0],
            help="Frecuencia final deseada de los retornos para el resto del pipeline.",
        )[1]

    colE, colF, colG = st.columns(3)
    with colE:
        force_refresh = st.checkbox("Force refresh (ignore cache)", value=False)
    with colF:
        invalidate_old = st.checkbox("Invalidate cache > 24h", value=True)
    with colG:
        show_stats = st.checkbox("Show summary stats", value=True)

# Normaliza tickers
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Clave de caché consistente (precios largos)
price_cfg = {
    "tickers": ",".join(sorted(tickers)),
    "start": str(start),
    "end": str(end),
    "interval": "1d",
    "adjust": True,
    "schema": "prices_v1",
}

# ──────────────────────────────────────────────────────────────────────────────
# Acciones
# ──────────────────────────────────────────────────────────────────────────────
if st.button("Load & Preview", type="primary"):
    if not tickers:
        st.error("Please provide at least one ticker.")
        st.stop()

    t0 = time.perf_counter()

    # Invalida caché antigua si procede
    if invalidate_old:
        age = age_seconds("prices_long", price_cfg)
        if age is not None and age > 24 * 3600:
            invalidate("prices_long", price_cfg)

    # 1) Fetch precios (long, Polars) con caché
    try:
        df_prices = None if force_refresh else load_pl("prices_long", price_cfg)
        if df_prices is None:
            with st.spinner("Fetching prices from Yahoo Finance..."):
                df_prices = get_prices_long(
                    tickers=tickers,
                    start=str(start),   # <— date → 'YYYY-MM-DD'
                    end=str(end),
                    interval="1d",
                    adjust=True,
                    force_refresh=force_refresh,
                    use_cache=True,
                )
                # Persistimos para futuras sesiones
                save_pl("prices_long", price_cfg, df_prices)
    except Exception as e:
        st.exception(e)
        st.stop()

    # 2) Re-muestreo de precios y cálculo de retornos (lazy → collect)
    with st.spinner("Computing returns… (resample + returns + winsor)"):
        # Retornos a la frecuencia de precios elegida (1d/1w/1mo)
        df_ret = compute_returns_from_prices_long(
            df_prices, freq=freq_prices, kind=ret_kind, drop_first=True
        ).collect()

        # Winsorización por ticker
        df_ret_w = winsorize_long(df_ret, ret_col="ret", q=float(winsor_p))

        # A ancho (para métricas y módulos siguientes)
        df_ret_wide = long_to_wide(df_ret_w, value_col="ret_w")

        # Frecuencia final deseada (por si difiere de resample de precios)
        if freq_returns != freq_prices:
            df_ret_wide = returns_to_frequency_wide(df_ret_wide, freq=freq_returns, kind=ret_kind)

    # 3) Previews
    st.subheader("Prices (tail)")
    with st.container(border=True):
        st.caption(f"Rows: {df_prices.height:,}")
        st.dataframe(df_prices.tail(10).to_pandas(), use_container_width=True)

    st.subheader("Returns (tail)")
    with st.container(border=True):
        st.caption(f"Returns shape: {df_ret_wide.shape[0]} x {df_ret_wide.shape[1]-1}")
        st.dataframe(df_ret_wide.tail(10).to_pandas().round(6), use_container_width=True)

    # 4) Data Health Panel
    st.subheader("🩺 Data Health")
    c1, c2, c3, c4 = st.columns(4)
    n_rows = df_prices.height
    n_tickers = df_prices.select(pl.col("ticker").n_unique()).item()
    n_dates = df_prices.select(pl.col("date").n_unique()).item()
    missing_prices = df_prices.filter(pl.col("price").is_null()).height
    c1.metric("Tickers", n_tickers)
    c2.metric("Dates", n_dates)
    c3.metric("Rows", n_rows)
    c4.metric("Missing Prices", missing_prices)

    # Missing report sobre retornos (wide)
    mr = missing_report_wide(df_ret_wide).sort("missing_pct", descending=True)
    st.write("Missing report (returns, wide)")
    st.dataframe(mr.to_pandas(), use_container_width=True)

    # Summary stats (por activo) si se pide
    if show_stats:
        st.subheader("Summary stats (per asset, periodic)")
        stats = summary_stats(df_ret_wide, risk_free=0.0)
        st.dataframe(stats.sort("sharpe", nulls_last=True, descending=True).to_pandas(), use_container_width=True)

    # 5) Export artefacts
    st.subheader("📤 Export")
    colP, colR = st.columns(2)
    with colP:
        buf_p = io.BytesIO()
        df_prices.write_parquet(buf_p)
        st.download_button(
            "Download Prices (parquet)",
            data=buf_p.getvalue(),
            file_name="prices_long.parquet",
            mime="application/octet-stream",
            use_container_width=True,
        )
    with colR:
        buf_r = io.BytesIO()
        df_ret_wide.write_parquet(buf_r)
        st.download_button(
            "Download Returns (parquet)",
            data=buf_r.getvalue(),
            file_name="returns_wide.parquet",
            mime="application/octet-stream",
            use_container_width=True,
        )

    # 6) Session state handoff para siguientes páginas
    st.session_state["prices_long"] = df_prices
    st.session_state["returns_wide"] = df_ret_wide
    st.session_state["ret_kind"] = ret_kind
    st.session_state["freq_returns"] = freq_returns

    st.success(f"Data loaded in {time.perf_counter() - t0:.2f}s and stored in session_state.")
