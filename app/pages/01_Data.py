# app/pages/01_Data.py
from __future__ import annotations

import io
import json
import time
from datetime import date, datetime, timezone
import hashlib

import polars as pl
import streamlit as st
import plotly.express as px

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from portfolio.io.data_loader import get_prices_long
from portfolio.io.cache import (
    save_pl, load_pl, cache_path, invalidate, age_seconds, save_json
)
from portfolio.features.returns import (
    compute_returns_from_prices_long,
    winsorize_long,
    long_to_wide,
    returns_to_frequency_wide,
    summary_stats,
    missing_report_wide,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers (internos de esta página)
# ──────────────────────────────────────────────────────────────────────────────
def _fmt_age(sec: float | None) -> str:
    if sec is None:
        return "n/a"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d: return f"{d}d {h}h"
    if h: return f"{h}h {m}m"
    if m: return f"{m}m"
    return f"{s}s"

def gaps_report(df_long: pl.DataFrame, threshold_days: int = 3) -> pl.DataFrame:
    """Reporte de gaps por ticker en días (umbral configurable)."""
    df = df_long.sort(["ticker", "date"]).with_columns(
        (pl.col("date") - pl.col("date").shift(1)).dt.total_days().alias("gap_days")
    )
    out = (
        df.group_by("ticker")
          .agg([
              pl.col("gap_days").max().fill_null(0).alias("max_gap_days"),
              (pl.col("gap_days") > threshold_days).cast(pl.Int64).sum().alias("n_gaps_gt_thr"),
              pl.when(pl.col("gap_days") > threshold_days).then(pl.col("date")).otherwise(None).min().alias("first_gap"),
              pl.when(pl.col("gap_days") > threshold_days).then(pl.col("date")).otherwise(None).max().alias("last_gap"),
          ])
          .sort(["max_gap_days", "n_gaps_gt_thr"], descending=[True, True])
    )
    return out

def top_abs_moves(df_ret_long: pl.DataFrame, k: int = 5) -> pl.DataFrame:
    """Top-k movimientos absolutos por ticker (pre-winsor)."""
    df = df_ret_long.with_columns(pl.col("ret").abs().alias("abs_ret"))
    df = df.with_columns(pl.col("abs_ret").rank(method="dense", descending=True).over("ticker").alias("rank"))
    out = (
        df.filter(pl.col("rank") <= k)
          .select(["ticker", "date", "ret", "abs_ret", "rank"])
          .sort(["ticker", "rank", "date"])
    )
    return out

def _json_default(o):
    # datetime.date/datetime → ISO
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            pass
    # NumPy → Python
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    # Polars datatypes caen en str por defecto si llegan aquí
    return str(o)

def _fingerprint(obj: dict) -> str:
    blob = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,   # serializador seguro
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]

def _run_data_pipeline(
    tickers,
    start,
    end,
    freq_prices,
    ret_kind,
    winsor_p,
    freq_returns,
    force_refresh,
    invalidate_old,
    price_cfg,
    gap_thr: int,
    topk_out: int,
):
    t0 = time.perf_counter()

    # Invalida caché antigua si procede
    if invalidate_old:
        age = age_seconds("prices_long", price_cfg)
        if age is not None and age > 24 * 3600:
            invalidate("prices_long", price_cfg)

    # 1) Fetch precios (con caché)
    df_prices = None if force_refresh else load_pl("prices_long", price_cfg)
    if df_prices is None:
        df_prices = get_prices_long(
            tickers=tickers, start=str(start), end=str(end),
            interval="1d", adjust=True,
            force_refresh=force_refresh, use_cache=True,
        )
        save_pl("prices_long", price_cfg, df_prices)

    # Normaliza dtypes/orden
    df_prices = df_prices.with_columns([
        pl.col("date").cast(pl.Datetime),
        pl.col("ticker").cast(pl.Utf8),
        pl.col("price").cast(pl.Float64),
    ]).sort(["ticker","date"])

    # 2) Cálculo de retornos (raw) + winsor + wide + frecuencia final
    df_ret_raw_long = compute_returns_from_prices_long(
        df_prices, freq=freq_prices, kind=ret_kind, drop_first=True
    ).collect()
    df_ret_w   = winsorize_long(df_ret_raw_long, ret_col="ret", q=float(winsor_p))
    df_ret_wide = long_to_wide(df_ret_w, value_col="ret_w")
    if freq_returns != freq_prices:
        df_ret_wide = returns_to_frequency_wide(df_ret_wide, freq=freq_returns, kind=ret_kind)

    # 3) Salud/diagnóstico
    mr   = missing_report_wide(df_ret_wide)
    gaps = gaps_report(df_prices, threshold_days=int(gap_thr))
    out_top = top_abs_moves(df_ret_raw_long, k=int(topk_out))
    stats = summary_stats(df_ret_wide, risk_free=0.0)

    # 4) Metadata reproducible
    eff = (
        df_prices.group_by("ticker")
                 .agg([pl.col("date").min().alias("start_eff"),
                       pl.col("date").max().alias("end_eff"),
                       pl.len().alias("n_rows")])
                 .sort("ticker")
    )
    eff_json = (
        eff.with_columns([
            pl.col("start_eff").dt.to_string().alias("start_eff"),
            pl.col("end_eff").dt.to_string().alias("end_eff"),
        ]).to_dicts()
    )
    meta = {
        "provider": "Yahoo Finance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "tickers": tickers, "start": str(start), "end": str(end),
            "interval": "1d", "adjust": True,
            "ret_kind": ret_kind, "freq_prices": freq_prices,
            "winsor_p": float(winsor_p), "freq_returns": freq_returns,
        },
        "stats": {
            "n_rows_prices": int(df_prices.height),
            "n_unique_dates": int(df_prices.select(pl.col("date").n_unique()).item()),
            "n_tickers": int(df_prices.select(pl.col("ticker").n_unique()).item()),
            "missing_prices": int(df_prices.filter(pl.col("price").is_null()).height),
        },
        "effective_ranges": eff_json,
        "cache": {
            "file": str(cache_path("prices_long", price_cfg)),
            "age_seconds": float(age_seconds("prices_long", price_cfg) or 0.0),
        },
    }
    meta["fingerprint"] = _fingerprint(meta)
    save_json("data_meta", price_cfg, meta)

    return {
        "df_prices": df_prices,
        "df_ret_raw_long": df_ret_raw_long,
        "df_ret_wide": df_ret_wide,
        "mr": mr, "gaps": gaps, "out_top": out_top,
        "stats": stats, "eff": eff, "meta": meta,
        "t_elapsed": time.perf_counter() - t0,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Data", layout="wide")
st.title("📦 Data Module")

# Inicializa session_state
if "data_payload" not in st.session_state:
    st.session_state["data_payload"] = None
if "data_ready" not in st.session_state:
    st.session_state["data_ready"] = False

# ──────────────────────────────────────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────────────────────────────────────
with st.container():
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        tickers_input = st.text_input(
            "Tickers (comma-separated)",
            "AAPL,MSFT,GOOGL,AMZN,META",
            help="Símbolos separados por coma. Ej: AAPL,MSFT,AMZN",
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

    colE, colF, colG, colH = st.columns(4)
    with colE:
        force_refresh = st.checkbox("Force refresh (ignore cache)", value=False)
    with colF:
        invalidate_old = st.checkbox("Invalidate cache > 24h", value=True)
    with colG:
        gap_thr = st.number_input("Gap threshold (days)", min_value=1, max_value=30, value=3, step=1)
    with colH:
        topk_out = st.number_input("Top-K outliers", min_value=3, max_value=20, value=5, step=1)

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
# Acción (cálculo) – guarda en session_state
# ──────────────────────────────────────────────────────────────────────────────
if st.button("Load & Preview", type="primary"):
    if not tickers:
        st.error("Please provide at least one ticker.")
        st.stop()

    payload = _run_data_pipeline(
        tickers=tickers,
        start=start, end=end,
        freq_prices=freq_prices, ret_kind=ret_kind,
        winsor_p=winsor_p, freq_returns=freq_returns,
        force_refresh=force_refresh, invalidate_old=invalidate_old,
        price_cfg=price_cfg, gap_thr=int(gap_thr), topk_out=int(topk_out),
    )
    st.session_state["data_payload"] = payload
    st.session_state["data_ready"] = True

# ──────────────────────────────────────────────────────────────────────────────
# Render (si hay datos en session_state)
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.get("data_ready"):
    p = st.session_state["data_payload"]

    df_prices       = p["df_prices"]
    df_ret_raw_long = p["df_ret_raw_long"]
    df_ret_wide     = p["df_ret_wide"]
    mr              = p["mr"]
    gaps            = p["gaps"]
    out_top         = p["out_top"]
    stats           = p["stats"]
    eff             = p["eff"]
    meta            = p["meta"]

    # Previews
    st.subheader("Prices (tail)")
    with st.container(border=True):
        st.caption(f"Rows: {df_prices.height:,}")
        st.dataframe(df_prices.tail(10).to_pandas(), use_container_width=True)

    st.subheader("Returns (tail)")
    with st.container(border=True):
        st.caption(f"Returns shape: {df_ret_wide.shape[0]} x {df_ret_wide.shape[1]-1}")
        st.dataframe(df_ret_wide.tail(10).to_pandas().round(6), use_container_width=True)

    # Data Health
    st.subheader("🩺 Data Health")
    c1, c2, c3, c4, c5 = st.columns(5)
    n_rows = df_prices.height
    n_tickers = df_prices.select(pl.col("ticker").n_unique()).item()
    n_dates = df_prices.select(pl.col("date").n_unique()).item()
    missing_prices = df_prices.filter(pl.col("price").is_null()).height
    data_age = age_seconds("prices_long", price_cfg)
    c1.metric("Tickers", n_tickers)
    c2.metric("Dates", n_dates)
    c3.metric("Rows", n_rows)
    c4.metric("Missing Prices", missing_prices)
    c5.metric("Data age", _fmt_age(data_age))

    # Universe snapshot
    uni = (
        df_prices.group_by("ticker").agg(pl.len().alias("n_obs"))
                 .sort("n_obs", descending=True)
    )
    st.write("Universe snapshot (observations per ticker)")
    fig_uni = px.bar(uni.to_pandas(), x="n_obs", y="ticker", orientation="h")
    st.plotly_chart(fig_uni, use_container_width=True)

    # Missing report (returns, wide)
    st.write("Missing report (returns, wide)")
    st.dataframe(mr.sort("missing_pct", descending=True).to_pandas(), use_container_width=True)

    # Gaps & Calendar
    st.subheader("🧩 Gaps & Calendar")
    st.dataframe(gaps.to_pandas(), use_container_width=True)

    # Outliers (pre-winsor)
    st.subheader("⚠️ Outliers (pre-winsor)")
    col_prev, col_k = st.columns([3,1])
    with col_k:
        st.caption(f"Top-{int(topk_out)} por ticker")
    st.dataframe(out_top.to_pandas().round(6), use_container_width=True)
    with col_prev:
        if st.checkbox("Preview non-winsorized returns (wide)", value=False, key="prev_nowinsor"):
            prev_wide = long_to_wide(df_ret_raw_long, value_col="ret")
            st.dataframe(prev_wide.tail(10).to_pandas().round(6), use_container_width=True)

    # Summary stats (per asset)
    if st.checkbox("Show summary stats", value=True, key="show_stats"):
        st.subheader("Summary stats (per asset, periodic)")
        st.dataframe(
            stats.sort("sharpe", nulls_last=True, descending=True).to_pandas(),
            use_container_width=True,
        )

    # Metadata
    st.subheader("🔖 Metadata")
    st.dataframe(eff.to_pandas(), use_container_width=True)

    # Export
    st.subheader("📤 Export")
    colP, colR, colJ = st.columns(3)
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
    with colJ:
        st.download_button(
            "Download data_config.json",
            data=json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default).encode("utf-8"),
            file_name="data_config.json",
            mime="application/json",
            use_container_width=True,
        )

    st.success(f"Data loaded in {p['t_elapsed']:.2f}s and stored in session_state.")
    # Handoff explícito para otros módulos (RiskModel, Optimization, etc.)
    st.session_state["returns_wide"] = df_ret_wide
