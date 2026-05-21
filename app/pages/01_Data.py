# app/pages/01_Data.py
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import struct
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Design System
from app.design_system import COLORS, get_global_styles, metric_grid, section_header
from app.viz.plotly_theme import apply_gammaedge_theme
from portfolio.core.compat import UTC
from portfolio.features.quant_charts import (
    generate_correlation_heatmap_preview,
    generate_drawdown_chart,
    generate_equity_curve_normalized,
    generate_monthly_returns_bar,
    generate_returns_distribution,
    generate_rolling_sharpe,
    generate_rolling_volatility,
)
from portfolio.features.quant_preview import generate_metrics_summary
from portfolio.features.returns import (
    compute_returns_from_prices_long,
    long_to_wide,
    missing_report_wide,
    returns_to_frequency_wide,
    summary_stats,
    winsorize_long,
)
from portfolio.io.cache import age_seconds, cache_path, invalidate, load_pl, save_json, save_pl
from portfolio.io.data_loader import get_prices_long
from portfolio.io.excel_export import export_quant_metrics_to_excel
from portfolio.viz.plot_utils import show_plot

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Local helpers
# ──────────────────────────────────────────────────────────────────────────────
def _fmt_age(sec: float | None) -> str:
    if sec is None:
        return "n/a"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m"
    return f"{s}s"


def gaps_report(df_long: pl.DataFrame, threshold_days: int = 3) -> pl.DataFrame:
    """Gap report per ticker in days (configurable threshold)."""
    df = df_long.sort(["ticker", "date"]).with_columns(
        (pl.col("date") - pl.col("date").shift(1)).dt.total_days().alias("gap_days")
    )
    out = (
        df.group_by("ticker")
        .agg(
            [
                pl.col("gap_days").max().fill_null(0).alias("max_gap_days"),
                (pl.col("gap_days") > threshold_days).cast(pl.Int64).sum().alias("n_gaps_gt_thr"),
                pl.when(pl.col("gap_days") > threshold_days)
                .then(pl.col("date"))
                .otherwise(None)
                .min()
                .alias("first_gap"),
                pl.when(pl.col("gap_days") > threshold_days)
                .then(pl.col("date"))
                .otherwise(None)
                .max()
                .alias("last_gap"),
            ]
        )
        .sort(["max_gap_days", "n_gaps_gt_thr"], descending=[True, True])
    )
    return out


def top_abs_moves(df_ret_long: pl.DataFrame, k: int = 5) -> pl.DataFrame:
    """Top-k absolute moves per ticker (pre-winsor)."""
    df = df_ret_long.with_columns(pl.col("ret").abs().alias("abs_ret"))
    df = df.with_columns(
        pl.col("abs_ret").rank(method="dense", descending=True).over("ticker").alias("rank")
    )
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

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()

    return str(o)


def _fingerprint(obj: dict) -> str:
    blob = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
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
    """Full data loading + cleaning pipeline, returning a rich payload."""
    t0 = time.perf_counter()

    empty_df = pl.DataFrame()
    coverage_full = empty_df

    # Invalidate old cache if requested
    if invalidate_old:
        age = age_seconds("prices_long", price_cfg)
        if age is not None and age > 24 * 3600:
            invalidate("prices_long", price_cfg)

    # 1) Fetch prices (cache-aware)
    df_prices: pl.DataFrame | None = None
    if not force_refresh:
        cached = load_pl("prices_long", price_cfg)
        if cached is not None:
            df_prices = cached

    if df_prices is None:
        df_prices = get_prices_long(
            tickers=tickers,
            start=str(start),
            end=str(end),
            interval="1d",
            adjust=True,
            force_refresh=force_refresh,
            use_cache=True,
        )
        save_pl("prices_long", price_cfg, df_prices)

    # Normalise dtypes/order
    df_prices = df_prices.with_columns(
        [
            pl.col("date").cast(pl.Datetime),
            pl.col("ticker").cast(pl.Utf8),
            pl.col("price").cast(pl.Float64),
        ]
    ).sort(["ticker", "date"])

    # Price coverage per ticker
    price_coverage = (
        df_prices.group_by("ticker")
        .agg(
            [
                pl.len().alias("n_rows"),
                pl.col("price").is_null().sum().alias("n_na"),
                pl.col("date").min().alias("start_eff"),
                pl.col("date").max().alias("end_eff"),
            ]
        )
        .with_columns((pl.col("n_rows") - pl.col("n_na")).alias("n_valid"))
        .sort("ticker")
    )

    alive = price_coverage.filter(pl.col("n_valid") >= 2)
    dropped_prices_df = price_coverage.filter(pl.col("n_valid") < 2)

    if dropped_prices_df.height > 0:
        alive_set = set(alive["ticker"].to_list())
        df_prices = df_prices.filter(pl.col("ticker").is_in(list(alive_set)))

    dropped_tickers_prices = (
        dropped_prices_df["ticker"].to_list() if dropped_prices_df.height else []
    )

    # If no ticker survives, build minimal payload and exit
    if df_prices.select(pl.col("ticker").n_unique()).item() == 0:
        meta_partial = {
            "provider": "Yahoo Finance",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "params": {
                "tickers": tickers,
                "start": str(start),
                "end": str(end),
                "interval": "1d",
                "adjust": True,
                "ret_kind": ret_kind,
                "freq_prices": freq_prices,
                "winsor_p": float(winsor_p),
                "freq_returns": freq_returns,
            },
            "data_quality": {
                "requested_period": {"start": str(start), "end": str(end)},
                "dropped_tickers": tickers,
            },
            "cache": {
                "file": str(cache_path("prices_long", price_cfg)),
                "age_seconds": float(age_seconds("prices_long", price_cfg) or 0.0),
            },
        }
        return {
            "df_prices": df_prices,
            "df_ret_raw_long": empty_df,
            "df_ret_wide": empty_df,
            "mr": empty_df,
            "gaps": empty_df,
            "out_top": empty_df,
            "stats": empty_df,
            "eff": empty_df,
            "meta": meta_partial,
            "coverage": empty_df,
            "dropped_tickers": tickers,
            "t_elapsed": time.perf_counter() - t0,
        }

    # 2) Returns, winsor, wide, final frequency
    df_ret_raw_long = compute_returns_from_prices_long(
        df_prices, freq=freq_prices, kind=ret_kind, drop_first=True
    ).collect()
    df_ret_w = winsorize_long(df_ret_raw_long, ret_col="ret", q=float(winsor_p))
    df_ret_wide = long_to_wide(df_ret_w, value_col="ret_w")
    if freq_returns != freq_prices:
        df_ret_wide = returns_to_frequency_wide(df_ret_wide, freq=freq_returns, kind=ret_kind)

    # Coverage per ticker in returns
    value_cols = [c for c in df_ret_wide.columns if c != "date"]
    total_dates = int(df_ret_wide.height)

    cov_exprs = []
    for c in value_cols:
        cov_exprs.extend(
            [
                pl.col(c).is_not_null().sum().alias(f"{c}__n_obs"),
                pl.col(c).is_null().sum().alias(f"{c}__n_na"),
            ]
        )
    tmp = df_ret_wide.select(cov_exprs)

    rows = []
    for c in value_cols:
        n_obs = int(tmp.select(f"{c}__n_obs").item() or 0)
        n_na = int(tmp.select(f"{c}__n_na").item() or 0)
        cov_pct = (100.0 * n_obs / total_dates) if total_dates else 0.0
        rows.append((c, n_obs, n_na, total_dates, cov_pct))

    ret_coverage = pl.DataFrame(
        rows,
        schema=["ticker", "n_obs", "n_na", "n_dates", "coverage_pct"],
        orient="row",
    ).with_columns(
        pl.col("ticker").cast(pl.Utf8),
        pl.col("n_obs").cast(pl.Int64),
        pl.col("n_na").cast(pl.Int64),
        pl.col("n_dates").cast(pl.Int64),
        pl.col("coverage_pct").cast(pl.Float64),
    )

    if total_dates > 0:
        first_row = df_ret_wide.head(1)
        last_row = df_ret_wide.tail(1)
        flags = []
        for c in value_cols:
            first_missing = bool(first_row.select(pl.col(c).is_null()).item())
            last_missing = bool(last_row.select(pl.col(c).is_null()).item())
            flags.append((c, first_missing, last_missing))

        flags_df = pl.DataFrame(
            flags,
            schema=["ticker", "start_missing", "end_missing"],
            orient="row",
        ).with_columns(pl.col("ticker").cast(pl.Utf8))

        ret_coverage = ret_coverage.join(flags_df, on="ticker", how="left")

    dropped_tickers_returns = (
        ret_coverage.filter(pl.col("n_obs") < 2)["ticker"].to_list()
        if total_dates > 0
        else value_cols
    )

    if dropped_tickers_returns:
        keep = ["date"] + [c for c in value_cols if c not in dropped_tickers_returns]
        df_ret_wide = df_ret_wide.select(keep)

    # Full coverage table for UI/meta
    coverage_full = ret_coverage.join(
        price_coverage.select(["ticker", "start_eff", "end_eff"]),
        on="ticker",
        how="left",
    )

    # Final list of dropped tickers
    dropped_tickers = sorted(set(dropped_tickers_prices) | set(dropped_tickers_returns))

    # 3) Health / diagnostics
    mr = missing_report_wide(df_ret_wide)
    gaps = gaps_report(df_prices, threshold_days=int(gap_thr))
    out_top = top_abs_moves(df_ret_raw_long, k=int(topk_out))
    stats = summary_stats(df_ret_wide, risk_free=0.0)

    # 4) Effective ranges
    eff = (
        df_prices.group_by("ticker")
        .agg(
            [
                pl.col("date").min().alias("start_eff"),
                pl.col("date").max().alias("end_eff"),
                pl.len().alias("n_rows"),
            ]
        )
        .sort("ticker")
    )
    eff_json = eff.with_columns(
        [
            pl.col("start_eff").dt.to_string().alias("start_eff"),
            pl.col("end_eff").dt.to_string().alias("end_eff"),
        ]
    ).to_dicts()

    # 5) Metadata
    meta = {
        "provider": "Yahoo Finance",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "params": {
            "tickers": tickers,
            "start": str(start),
            "end": str(end),
            "interval": "1d",
            "adjust": True,
            "ret_kind": ret_kind,
            "freq_prices": freq_prices,
            "winsor_p": float(winsor_p),
            "freq_returns": freq_returns,
        },
        "stats": {
            "n_rows_prices": int(df_prices.height),
            "n_unique_dates": int(df_prices.select(pl.col("date").n_unique()).item()),
            "n_tickers": int(df_prices.select(pl.col("ticker").n_unique()).item()),
            "missing_prices": int(df_prices.filter(pl.col("price").is_null()).height),
        },
        "effective_ranges": eff_json,
        "data_quality": {
            "requested_period": {"start": str(start), "end": str(end)},
            "dropped_tickers": dropped_tickers,
        },
        "coverage_table": coverage_full.with_columns(
            [
                pl.col("start_eff").dt.to_string(),
                pl.col("end_eff").dt.to_string(),
            ]
        ).to_dicts(),
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
        "mr": mr,
        "gaps": gaps,
        "out_top": out_top,
        "stats": stats,
        "eff": eff,
        "meta": meta,
        "coverage": coverage_full,
        "dropped_tickers": dropped_tickers,
        "t_elapsed": time.perf_counter() - t0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Audit / Frozen Data Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load_frozen_dataset_index(base_path: str = "datasets") -> list[dict]:
    """Carga el índice global de datasets."""
    index_path = Path(base_path) / "INDEX.json"
    if not index_path.exists():
        return []
    try:
        with open(index_path) as f:
            return json.load(f)
    except Exception:
        return []


def _read_ssh_string(blob, offset):
    """Helper to read SSH string from blob."""
    length = struct.unpack(">I", blob[offset : offset + 4])[0]
    offset += 4
    val = blob[offset : offset + length]
    offset += length
    return val, offset


def _verify_openssh_signature_pure(
    sig_path: Path, data_path: Path, allowed_signers_path: Path
) -> bool:
    """Pure Python verification of OpenSSH signatures to avoid subprocess hangs."""
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise ImportError(
            "Audit Mode requires the 'cryptography' package. "
            "Install it with: pip install cryptography"
        ) from exc

    try:
        # 1. Load Public Key
        with open(allowed_signers_path) as f:
            line = f.read().strip()
        parts = line.split()
        if len(parts) < 3:
            raise ValueError("Invalid allowed_signers format")

        pub_key_bytes = base64.b64decode(parts[2])
        key_type, offset = _read_ssh_string(pub_key_bytes, 0)
        raw_key, offset = _read_ssh_string(pub_key_bytes, offset)
        public_key = Ed25519PublicKey.from_public_bytes(raw_key)

        # 2. Parse Signature
        with open(sig_path) as f:
            content = f.read()

        body = (
            content.replace("-----BEGIN SSH SIGNATURE-----", "")
            .replace("-----END SSH SIGNATURE-----", "")
            .strip()
        )
        sig_blob = base64.b64decode(body)

        offset = 0
        magic = sig_blob[offset : offset + 6]
        offset += 6
        if magic != b"SSHSIG":
            raise ValueError("Invalid signature magic")

        version = struct.unpack(">I", sig_blob[offset : offset + 4])[0]  # noqa: F841
        offset += 4
        pk_blob, offset = _read_ssh_string(sig_blob, offset)
        namespace, offset = _read_ssh_string(sig_blob, offset)
        reserved, offset = _read_ssh_string(sig_blob, offset)
        hash_algo, offset = _read_ssh_string(sig_blob, offset)

        nested_sig, offset = _read_ssh_string(sig_blob, offset)
        ns_algo, ns_off = _read_ssh_string(nested_sig, 0)
        raw_signature, ns_off = _read_ssh_string(nested_sig, ns_off)

        # 3. Hash Data & Verify
        with open(data_path, "rb") as f:
            raw_data = f.read()

        digest = hashes.Hash(hashes.SHA512())
        digest.update(raw_data)
        data_hash = digest.finalize()

        def ssh_string(data):
            return struct.pack(">I", len(data)) + data

        signed_payload = (
            b"SSHSIG"
            + ssh_string(namespace)
            + ssh_string(reserved)
            + ssh_string(hash_algo)
            + ssh_string(data_hash)
        )

        public_key.verify(raw_signature, signed_payload)
        return True

    except Exception as e:
        logger.exception("Verification Failed: %s", e)
        raise e


def _verify_and_load_dataset(dataset_path: Path) -> dict:
    """Verifica firma Ed25519 y hash, carga parquet, y adapta a formato app."""
    res = {"verified": False, "error": None, "payload": None}

    # 1. Verify Signature (Pure Python)
    try:
        _verify_openssh_signature_pure(
            dataset_path / "RELEASE.sig",
            dataset_path / "RELEASE.json",
            Path("keys/allowed_signers"),
        )
    except Exception as e:
        res["error"] = f"Signature Verification Failed: {e}"
        return res

    # 2. Verify Content Hash check
    try:
        with open(dataset_path / "RELEASE.json") as f:
            manifest = json.load(f)

        expected_hash = manifest["hashes"]["content_full"]
        parquet_path = dataset_path / "data.parquet"

        sha256 = hashlib.sha256()
        with open(parquet_path, "rb") as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                sha256.update(data)
        actual_hash = sha256.hexdigest()

        if actual_hash != expected_hash:
            res["error"] = "Content Hash Mismatch! integrity compromised."
            return res

    except Exception as e:
        res["error"] = f"Hash Check Error: {e}"
        return res

    # 3. Load logic
    try:
        df_wide = pl.read_parquet(parquet_path)

        # Determine tickers from columns (close_QQQ -> QQQ)
        tickers = []
        for c in df_wide.columns:
            if c.startswith("close_"):
                tickers.append(c.replace("close_", ""))

        # Melt to Long (df_prices)
        close_cols = [f"close_{t}" for t in tickers]
        df_long = (
            df_wide.select(["date"] + close_cols)
            .unpivot(index="date", on=close_cols, variable_name="ticker_raw", value_name="price")
            .with_columns(pl.col("ticker_raw").str.replace("close_", "").alias("ticker"))
            .select(["date", "ticker", "price"])
            .sort(["ticker", "date"])
        )

        # Build payload
        meta = {
            "provider": manifest["provider"],
            "generated_at_utc": manifest["created_at"],
            "params": {
                "tickers": tickers,
                "mode": "frozen_audit",
                "dataset_id": manifest["dataset_id"],
            },
            "audit": {
                "verified": True,
                "signature": "Ed25519 (OpenSSH)",
                "hash": actual_hash,
                "schema_version": manifest.get("schema_version", "unknown"),
                "env": manifest.get("environment", {}),
            },
        }

        stats = summary_stats(df_wide, risk_free=0.0)

        res["verified"] = True
        res["payload"] = {
            "df_prices": df_long,
            "df_ret_raw_long": pl.DataFrame(),
            "df_ret_wide": df_wide,
            "mr": missing_report_wide(df_wide),
            "gaps": gaps_report(df_long, threshold_days=3),
            "out_top": pl.DataFrame(),
            "stats": stats,
            "eff": pl.DataFrame(),
            "meta": meta,
            "coverage": pl.DataFrame(),
            "dropped_tickers": [],
            "t_elapsed": 0.0,
            "manifest": manifest,
        }
        return res

    except Exception as e:
        res["error"] = f"Loading Error: {e}"
        return res


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Data", layout="wide")

# Apply global styles
st.markdown(get_global_styles(), unsafe_allow_html=True)

# Page title with Apple-style
st.markdown(
    f"""
<div style="margin-bottom: 32px;">
<h1 style="font-size: 2.5rem; font-weight: 600; color: {COLORS['text_primary']}; margin-bottom: 8px;">
Data Module
</h1>
<p style="font-size: 1rem; color: {COLORS['text_secondary']}; line-height: 1.5;">
Load historical data, clean series, and compute quantitative metrics
</p>
</div>
""",
    unsafe_allow_html=True,
)

# Initialise session_state
if "data_payload" not in st.session_state:
    st.session_state["data_payload"] = None
if "data_ready" not in st.session_state:
    st.session_state["data_ready"] = False

# ──────────────────────────────────────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────────────────────────────────────
# Defaults for controls (avoid NameError if mode skips them)
topk_out = 5
gap_thr = 3

mode = st.radio(
    "Mode",
    ["Live Mode (Research)", "Audit Mode (Frozen)"],
    horizontal=True,
    help="Live Mode fetches fresh data. Audit Mode loads verified frozen datasets.",
)

price_cfg = {}  # Scope safety

if mode == "Audit Mode (Frozen)":
    st.info("**Audit Mode**: Loading signed datasets verified with Ed25519 keys.")

    datasets_index = _load_frozen_dataset_index()
    if not datasets_index:
        st.warning("No datasets found in `datasets/INDEX.json`.")
    else:
        # Select box
        options = [f"{d['dataset_id']} | {d['created_at'][:10]}" for d in datasets_index]
        selected_option = st.selectbox("Select Frozen Dataset", options)

        # Get selected metadata
        selected_idx = options.index(selected_option)
        selected_meta = datasets_index[selected_idx]

        col_verify, col_status = st.columns([1, 4])

        if col_verify.button("Verify & Load", type="primary"):
            with st.spinner("Cryptographic Verification in progress (ssh-keygen)..."):
                # Construct path
                ds_path = Path("datasets") / selected_meta["path"]

                result = _verify_and_load_dataset(ds_path)

                if result["verified"]:
                    st.session_state["data_payload"] = result["payload"]
                    st.session_state["data_ready"] = True
                    st.success("**Signature Validated**. Dataset Integrity Confirmed.")
                    st.balloons()
                else:
                    st.error(result["error"])
                    st.session_state["data_ready"] = False

elif mode == "Live Mode (Research)":
    with st.container():
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            tickers_input = st.text_input(
                "Tickers (comma-separated)",
                "AAPL,MSFT,GOOGL,AMZN,META",
                help="Símbolos separados por coma. Ej: AAPL,MSFT,AMZN",
            )
        with col2:
            start = st.date_input(
                "Start", value=date(2010, 1, 1), min_value=date(1900, 1, 1), max_value=date.today()
            )
        with col3:
            end = st.date_input(
                "End", value=date.today(), min_value=date(1900, 1, 1), max_value=date.today()
            )

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
            gap_thr = st.number_input(
                "Gap threshold (days)", min_value=1, max_value=30, value=3, step=1
            )
        with colH:
            topk_out = st.number_input("Top-K outliers", min_value=3, max_value=20, value=5, step=1)

    # Normalise tickers
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    # Cache key for long prices
    price_cfg = {
        "tickers": ",".join(sorted(tickers)),
        "start": str(start),
        "end": str(end),
        "interval": "1d",
        "adjust": True,
        "schema": "prices_v1",
    }

    # ──────────────────────────────────────────────────────────────────────────────
    # Action (compute) – store in session_state
    # ──────────────────────────────────────────────────────────────────────────────
    if st.button("Load & Preview", type="primary", use_container_width=True):
        if not tickers:
            st.error("Please provide at least one ticker.")
            st.stop()

        payload = _run_data_pipeline(
            tickers=tickers,
            start=start,
            end=end,
            freq_prices=freq_prices,
            ret_kind=ret_kind,
            winsor_p=winsor_p,
            freq_returns=freq_returns,
            force_refresh=force_refresh,
            invalidate_old=invalidate_old,
            price_cfg=price_cfg,
            gap_thr=int(gap_thr),
            topk_out=int(topk_out),
        )
        st.session_state["data_payload"] = (
            payload  # Keep original name for consistency with render block
        )

        # Save tickers and dates for auto-population in Quant Metrics
        st.session_state["loaded_tickers"] = ",".join(tickers)
        st.session_state["loaded_start_date"] = start
        st.session_state["loaded_end_date"] = end

        st.success(f"Loaded {len(tickers)} tickers")
        st.session_state["data_ready"] = True

# ──────────────────────────────────────────────────────────────────────────────
# Render (if we have data in session_state)
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.get("data_ready"):
    p = st.session_state["data_payload"]

    # Guard: if pipeline returned empty returns → show message + stop
    if p["df_ret_wide"] is None or (
        isinstance(p["df_ret_wide"], pl.DataFrame) and p["df_ret_wide"].height == 0
    ):
        dropped = p.get("dropped_tickers", [])
        if dropped:
            st.error(
                "Ningún ticker tiene datos suficientes en el periodo seleccionado. "
                "Ajusta el rango de fechas o el universo. "
                f"(Excluidos: {', '.join(dropped)})"
            )
        else:
            st.error("No hay datos suficientes para construir retornos en el rango elegido.")

        cov = p.get("coverage")
        if isinstance(cov, pl.DataFrame) and cov.height > 0:
            with st.expander("Cobertura por ticker", expanded=False):
                st.dataframe(cov.to_pandas(), width="stretch")

        st.stop()

    # At this point, we have valid data
    df_prices = p["df_prices"]
    df_ret_raw_long = p["df_ret_raw_long"]
    df_ret_wide = p["df_ret_wide"]
    mr = p["mr"]
    gaps = p["gaps"]
    out_top = p["out_top"]
    stats = p["stats"]
    eff = p["eff"]
    meta = p["meta"]

    # Warn about excluded tickers (if any)
    dropped_tickers = p.get("dropped_tickers", [])
    if dropped_tickers:
        st.warning(
            "Excluidos por falta de datos en el periodo seleccionado: " + ", ".join(dropped_tickers)
        )
        with st.expander("Cobertura por ticker", expanded=False):
            st.dataframe(p["coverage"].to_pandas(), width="stretch")

    # Previews
    st.subheader("Prices (tail)")
    with st.container(border=True):
        st.caption(f"Rows: {df_prices.height:,}")
        st.dataframe(df_prices.tail(10).to_pandas(), width="stretch")

    st.subheader("Returns (tail)")
    with st.container(border=True):
        st.caption(f"Returns shape: {df_ret_wide.shape[0]} x {df_ret_wide.shape[1] - 1}")
        st.dataframe(df_ret_wide.tail(10).to_pandas().round(6), width="stretch")

    # Data Health - Apple-style metric grid
    st.markdown(
        section_header("Data Health", "Overview of dataset completeness and freshness", "🩺"),
        unsafe_allow_html=True,
    )

    n_rows = df_prices.height
    n_tickers = df_prices.select(pl.col("ticker").n_unique()).item()
    n_dates = df_prices.select(pl.col("date").n_unique()).item()
    missing_prices = df_prices.filter(pl.col("price").is_null()).height
    data_age = age_seconds("prices_long", price_cfg)

    st.markdown(
        metric_grid(
            [
                {"label": "Assets in Universe", "value": n_tickers},
                {"label": "Trading Days", "value": f"{n_dates:,}"},
                {"label": "Total Observations", "value": f"{n_rows:,}"},
                {
                    "label": "Missing Data Points",
                    "value": missing_prices,
                    "negative": missing_prices > 0,
                },
                {"label": "Data Freshness", "value": _fmt_age(data_age)},
            ],
            columns=5,
        ),
        unsafe_allow_html=True,
    )

    # Universe snapshot
    uni = df_prices.group_by("ticker").agg(pl.len().alias("n_obs")).sort("n_obs", descending=True)
    st.write("Universe snapshot (observations per ticker)")
    fig_uni = px.bar(
        uni.to_pandas(),
        x="n_obs",
        y="ticker",
        orientation="h",
        title="Number of Observations per Ticker",
        labels={"n_obs": "Observations", "ticker": "Ticker"},
    )
    fig_uni = apply_gammaedge_theme(fig_uni)
    show_plot(fig_uni, config={"scrollZoom": True, "displayModeBar": True})

    # Missing report (returns, wide)
    st.write("Missing report (returns, wide)")
    st.dataframe(mr.sort("missing_pct", descending=True).to_pandas(), width="stretch")

    # Gaps & Calendar
    st.subheader("Gaps & Calendar")
    st.dataframe(gaps.to_pandas(), width="stretch")

    # Outliers (pre-winsor)
    st.subheader("Outliers (pre-winsor)")
    col_prev, col_k = st.columns([3, 1])
    with col_k:
        st.caption(f"Top-{int(topk_out)} por ticker")
    st.dataframe(out_top.to_pandas().round(6), width="stretch")
    with col_prev:
        if st.checkbox("Preview non-winsorized returns (wide)", value=False, key="prev_nowinsor"):
            prev_wide = long_to_wide(df_ret_raw_long, value_col="ret")
            st.dataframe(prev_wide.tail(10).to_pandas().round(6), width="stretch")

    # Summary stats (per asset)
    if st.checkbox("Show summary stats", value=True, key="show_stats"):
        st.markdown(
            section_header(
                "Summary Statistics",
                "Risk-return metrics computed from historical data",
            ),
            unsafe_allow_html=True,
        )
        st.dataframe(
            stats.sort("sharpe", nulls_last=True, descending=True).to_pandas(),
            width="stretch",
        )

    # Metadata
    st.markdown(
        section_header(
            "Metadata",
            "Data pipeline execution details",
        ),
        unsafe_allow_html=True,
    )
    st.dataframe(eff.to_pandas(), width="stretch")

    # Export
    st.subheader("Export")
    colP, colR, colJ = st.columns(3)
    with colP:
        buf_p = io.BytesIO()
        df_prices.write_parquet(buf_p)
        st.download_button(
            "Download Prices (parquet)",
            data=buf_p.getvalue(),
            file_name="prices_long.parquet",
            mime="application/octet-stream",
            width="stretch",
        )
    with colR:
        buf_r = io.BytesIO()
        df_ret_wide.write_parquet(buf_r)
        st.download_button(
            "Download Returns (parquet)",
            data=buf_r.getvalue(),
            file_name="returns_wide.parquet",
            mime="application/octet-stream",
            width="stretch",
        )
    with colJ:
        st.download_button(
            "Download data_config.json",
            data=json.dumps(
                meta, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default
            ).encode("utf-8"),
            file_name="data_config.json",
            mime="application/json",
            width="stretch",
        )

    # Excel Export with Quant Metrics
    st.subheader("Quant Metrics Export (Excel)")
    st.caption(
        "Export log-returns, relative volume, and intraday volatility to Excel. "
        "Data is dividend-adjusted (availability depends on ticker history)."
    )

    # Ticker input for Quant Metrics (auto-populated from Data module if available)
    default_quant_tickers = st.session_state.get("loaded_tickers", "AAPL,MSFT,GOOGL")
    quant_tickers_input = st.text_input(
        "Tickers for Quant Metrics (comma-separated)",
        value=default_quant_tickers,
        key="quant_tickers",
        help="Enter ticker symbols separated by commas. Auto-populated from Data module when you load data.",
    )

    col_dates, col_params = st.columns([2, 2])

    with col_dates:
        col_start, col_end = st.columns(2)
        with col_start:
            # Use loaded dates from Data module if available
            default_start = st.session_state.get("loaded_start_date", date(2010, 1, 1))
            quant_start_date = st.date_input(
                "Start date",
                value=default_start,
                min_value=date(1900, 1, 1),
                max_value=date.today(),
                key="quant_start_date",
                help="Auto-populated from Data module. Select start date for metrics calculation.",
            )
        with col_end:
            default_end = st.session_state.get("loaded_end_date", date.today())
            quant_end_date = st.date_input(
                "End date",
                value=default_end,
                min_value=date(1900, 1, 1),
                max_value=date.today(),
                key="quant_end_date",
                help="Select end date (up to today).",
            )

    with col_params:
        col_vol, col_method = st.columns(2)
        with col_vol:
            vol_lookback = st.number_input(
                "Volume SMA window",
                min_value=5,
                max_value=60,
                value=20,
                step=5,
                key="vol_lookback",
                help="Lookback period for relative volume calculation.",
            )
        with col_method:
            vol_method = st.selectbox(
                "Volatility Estimator",
                ["parkinson", "garman_klass"],
                key="quant_vol_method",
                help="Parkinson: Uses High/Low. Garman-Klass: Uses OHLC.",
            )

            rf_annual = (
                st.number_input(
                    "Risk-Free Rate (Annual %)",
                    min_value=0.0,
                    max_value=10.0,
                    value=4.2,
                    step=0.1,
                    format="%.1f",
                    key="quant_rf_annual",
                    help="Annual risk-free rate for Sharpe, Sortino, and Treynor ratios. Default: 2% (US 10Y Treasury ~2024)",
                )
                / 100.0
            )  # Convert percentage to decimal

        # Benchmark selector
        st.markdown("**Benchmark Selection**")
        col_bench, col_custom = st.columns([2, 2])
        with col_bench:
            benchmark_preset = st.selectbox(
                "Benchmark",
                options=[
                    "^GSPC (S&P 500)",
                    "^NDX (Nasdaq 100)",
                    "^RUT (Russell 2000)",
                    "^TNX (US 10Y Treasury Yield)",
                    "^XAU (Gold/Silver Index)",
                    "Custom",
                ],
                index=0,
                key="benchmark_preset",
                help="Benchmarks to use Indices/Yields for maximum historical data availability.",
            )

        with col_custom:
            if "Custom" in benchmark_preset:
                benchmark_ticker = st.text_input(
                    "Custom Benchmark Ticker",
                    value="^GSPC",
                    key="custom_benchmark",
                    help="Enter custom benchmark ticker symbol",
                )
            else:
                # Extract ticker from preset (e.g., "SPY (S&P 500)" -> "SPY")
                benchmark_ticker = benchmark_preset.split(" ")[0]
                st.text_input(
                    "Selected Benchmark",
                    value=benchmark_ticker,
                    disabled=True,
                    key="selected_benchmark_display",
                )

    col_preview, col_download = st.columns([1, 1])

    with col_preview:
        if st.button("Preview Quant Metrics", key="preview_quant"):
            with st.spinner("Downloading data and calculating metrics..."):
                try:
                    # Parse tickers from input
                    quant_tickers = [
                        t.strip().upper() for t in quant_tickers_input.split(",") if t.strip()
                    ]

                    if not quant_tickers:
                        st.warning("Please enter at least one ticker.")
                    else:
                        # Generate metrics summary (with selected benchmark)
                        result = generate_metrics_summary(
                            tickers=quant_tickers,
                            start=str(quant_start_date),
                            end=str(quant_end_date),
                            benchmark=benchmark_ticker,  # Use selected benchmark
                            rf_annual=0.02,
                        )

                        # Display summary table
                        st.subheader("Metrics Summary")
                        summary_df = result["summary_df"]

                        # Format for display (convert to pandas for better formatting)
                        summary_pd = summary_df.to_pandas()

                        # Format numeric columns to 4 decimal places
                        numeric_cols = [
                            "total_return",
                            "volatility",
                            "sharpe_ratio",
                            "sortino_ratio",
                            "treynor_ratio",
                            "information_ratio",
                            "beta",
                            "alpha",
                            "max_drawdown",
                            "ulcer_index",
                            "cagr",
                            "calmar",
                            "up_capture",
                            "down_capture",
                            "tail_ratio",
                            "win_rate",
                            "profit_factor",
                            "best_month",
                            "worst_month",
                            "skewness",
                            "kurtosis",
                        ]
                        for col in numeric_cols:
                            if col in summary_pd.columns:
                                summary_pd[col] = summary_pd[col].apply(
                                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                                )

                        st.dataframe(summary_pd, width="stretch")

                        # Display data quality table
                        st.subheader("Data Quality Report")
                        data_quality_df = result["data_quality_df"]
                        st.dataframe(data_quality_df.to_pandas(), width="stretch")

                        # Display warnings if any
                        if result["warnings"]:
                            st.warning("**Warnings:**")
                            for warning in result["warnings"]:
                                st.caption(f"• {warning}")
                        else:
                            st.success("No data quality issues detected")

                        # Visual Analysis Section
                        st.markdown(
                            section_header(
                                "Visual Analysis",
                                "Interactive charts for risk-return exploration",
                            ),
                            unsafe_allow_html=True,
                        )

                        # Apply tab styling
                        st.markdown(
                            f"""
                        <style>
                        .stTabs [data-baseweb="tab-list"] {{
                            gap: 8px;
                            background-color: {COLORS['bg_secondary']};
                            border-radius: 12px;
                            padding: 4px;
                        }}

                        .stTabs [data-baseweb="tab"] {{
                            padding: 12px 24px;
                            border-radius: 8px;
                            font-weight: 500;
                            transition: background-color 0.2s ease;
                            color: {COLORS['text_secondary']};
                        }}

                        .stTabs [aria-selected="true"] {{
                            background-color: {COLORS['accent_primary']} !important;
                            color: {COLORS['text_primary']} !important;
                        }}
                        </style>
                        """,
                            unsafe_allow_html=True,
                        )

                        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
                            [
                                "Equity Curves",
                                "Drawdown",
                                "Rolling Vol",
                                "Rolling Sharpe",
                                "Returns Dist",
                                "Monthly Returns",
                                "Correlation",
                            ]
                        )

                        with tab1:
                            try:
                                fig = generate_equity_curve_normalized(
                                    result["df_prices"], quant_tickers, benchmark=benchmark_ticker
                                )
                                show_plot(fig, key="quant_equity")
                            except Exception as e:
                                st.error(f"Error generating equity curve: {e}")

                        with tab2:
                            try:
                                fig = generate_drawdown_chart(result["df_prices"], quant_tickers)
                                show_plot(fig, key="quant_drawdown")
                            except Exception as e:
                                st.error(f"Error generating drawdown chart: {e}")

                        with tab3:
                            try:
                                fig = generate_rolling_volatility(
                                    result["df_returns"], quant_tickers, window=30
                                )
                                show_plot(fig, key="quant_rolling_vol")
                            except Exception as e:
                                st.error(f"Error generating rolling volatility: {e}")

                        with tab4:
                            try:
                                fig = generate_rolling_sharpe(
                                    result["df_returns"], quant_tickers, window=252, rf_annual=0.02
                                )
                                show_plot(fig, key="quant_rolling_sharpe")
                            except Exception as e:
                                st.error(f"Error generating rolling Sharpe: {e}")

                        with tab5:
                            try:
                                fig = generate_returns_distribution(
                                    result["df_returns"], quant_tickers
                                )
                                show_plot(fig, key="quant_returns_dist")
                            except Exception as e:
                                st.error(f"Error generating returns distribution: {e}")

                        with tab6:
                            try:
                                fig = generate_monthly_returns_bar(
                                    result["df_returns"], quant_tickers
                                )
                                show_plot(fig, key="quant_monthly_returns")
                            except Exception as e:
                                st.error(f"Error generating monthly returns: {e}")

                        with tab7:
                            try:
                                fig = generate_correlation_heatmap_preview(
                                    result["df_returns"], quant_tickers
                                )
                                show_plot(fig, key="quant_corr_heatmap")
                            except Exception as e:
                                st.error(f"Error generating correlation heatmap: {e}")

                except ValueError as e:
                    st.error(f"Error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    import traceback

                    st.code(traceback.format_exc())

    with col_download:
        if st.button("Generate Excel Export", type="primary", key="gen_excel"):
            with st.spinner("Downloading OHLCV and calculating metrics..."):
                try:
                    # Parse tickers from input
                    quant_tickers = [
                        t.strip().upper() for t in quant_tickers_input.split(",") if t.strip()
                    ]

                    if not quant_tickers:
                        st.warning("Please enter at least one ticker.")
                    else:
                        # Call new export function with simplified signature
                        excel_bytes = export_quant_metrics_to_excel(
                            tickers=quant_tickers,
                            start=str(quant_start_date),
                            end=str(quant_end_date),
                            benchmark=benchmark_ticker,
                            rf_annual=rf_annual,  # Use user-selected RF rate
                            vol_lookback=int(vol_lookback),
                            vol_method=vol_method,
                        )
                        st.session_state["quant_excel_bytes"] = excel_bytes
                        st.success("Excel file generated successfully!")
                except Exception as e:
                    st.error(f"Error generating Excel: {e}")
                    # Show full traceback for debugging
                    import traceback

                    st.code(traceback.format_exc(), language="python")

    # Show download button if Excel was generated
    if "quant_excel_bytes" in st.session_state:
        st.download_button(
            "Download Quant Metrics (Excel)",
            data=st.session_state["quant_excel_bytes"],
            file_name="quant_metrics.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_quant_excel",
        )

    st.success(f"Data loaded in {p['t_elapsed']:.2f}s and stored in session_state.")

    # Explicit handoff for other modules (RiskModel, Optimizer, Backtest, Attribution)
    st.session_state["returns_wide"] = df_ret_wide
    st.session_state["df_ret_wide"] = df_ret_wide
    st.session_state["df_prices"] = df_prices
    st.session_state["tickers"] = [c for c in df_ret_wide.columns if c != "date"]
    st.session_state["data_meta"] = meta
