# app/pages/05_Attribution.py
from __future__ import annotations

import os
import sys

import polars as pl
import streamlit as st

# ---------------------------------------------------------------------
# Repo root for local imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Core modules
from portfolio.backtest import attribution as bt_attr
from portfolio.backtest.brinson_utils import (
    coerce_brinson_timeseries_to_long as _coerce_brinson_ts_to_long,
)
from portfolio.backtest.brinson_utils import (
    ensure_datetime as _ensure_datetime,
)
from portfolio.viz import plot_utils as viz
from portfolio.viz.plot_utils import show_plot


# ---------------------------------------------------------------------
# Helpers (UI-only)
# ---------------------------------------------------------------------
def _csv_bytes(df: pl.DataFrame) -> bytes:
    """Return CSV bytes from a Polars DF, evitando dtypes Object/List para descargas robustas."""
    try:
        cols = []
        for name, dtype in df.schema.items():
            if dtype == pl.Datetime:
                cols.append(pl.col(name).dt.to_string().alias(name))
            elif dtype == pl.Object or dtype == pl.List:
                cols.append(pl.col(name).cast(pl.String, strict=False).alias(name))
            else:
                cols.append(pl.col(name))
        return df.select(cols).write_csv().encode()
    except Exception:
        # Fallback universal (requiere pandas en deps, ya incluida en pyproject)
        return df.to_pandas().to_csv(index=False).encode()


# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------
st.set_page_config(page_title="Attribution Analysis", layout="wide")
st.title("📊 Performance Attribution Dashboard")
st.caption("Vectorized Brinson–Fachler decomposition and return contribution analysis.")

# ---------------------------------------------------------------------
# Defensive handoff from previous pages
# ---------------------------------------------------------------------
bt = st.session_state.get("bt")
df_ret_wide = st.session_state.get("df_ret_wide", st.session_state.get("returns_wide"))
asset_meta = st.session_state.get("asset_meta")

if bt is None or df_ret_wide is None:
    st.warning("⚠️ Run pages 02–04 first, then export to Attribution.")
    st.stop()

# Normalize inputs to Polars + Datetime
try:
    if not isinstance(df_ret_wide, pl.DataFrame):
        df_ret_wide = pl.from_pandas(df_ret_wide)
    df_ret_wide = _ensure_datetime(df_ret_wide, "date")
except Exception as e:
    st.error(f"Input normalization failed: {e}")
    st.stop()

# ---------------------------------------------------------------------
# 1) Alignment & IPO safety
# ---------------------------------------------------------------------
try:
    aln = bt_attr._daily_alignment_from_bt(bt, df_ret_wide)
    aln_ipo = bt_attr.align_with_ipo_mask(aln)
    st.success("✅ Alignment successful (IPO-safe).")
except Exception as e:
    st.error(f"Alignment failed: {e}")
    st.stop()

# ---------------------------------------------------------------------
# 2) Grouping setup
# ---------------------------------------------------------------------
try:
    meta_df = asset_meta
    if meta_df is not None and not isinstance(meta_df, pl.DataFrame):
        meta_df = pl.DataFrame(meta_df)

    if meta_df is not None and "sector" in meta_df.columns:
        groups_idx, group_labels, groups_map = bt_attr.build_groups_idx(
            aln_ipo.tickers, meta_df, col="sector"
        )
    elif meta_df is not None and "country" in meta_df.columns:
        groups_idx, group_labels, groups_map = bt_attr.build_groups_idx(
            aln_ipo.tickers, meta_df, col="country"
        )
    else:
        groups_idx = list(range(len(aln_ipo.tickers)))
        group_labels = aln_ipo.tickers
        groups_map = {tk: tk for tk in aln_ipo.tickers}
except Exception:
    # Fallback sin metadatos
    groups_idx = list(range(len(aln_ipo.tickers)))
    group_labels = aln_ipo.tickers
    groups_map = {tk: tk for tk in aln_ipo.tickers}

# ---------------------------------------------------------------------
# 3) Benchmark construction
# ---------------------------------------------------------------------
try:
    Wb_daily = bt_attr.coerce_benchmark_weights(
        None, len(aln_ipo.dates), len(aln_ipo.tickers), scheme="EW"
    )
except Exception as e:
    st.error(f"Benchmark generation failed: {e}")
    st.stop()

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Asset-level", "🏗️ Group-level", "🧩 Brinson–Fachler"])

# ─────────────────────────────────────────────────────────────────────
# TAB 1 – Asset-level
# ─────────────────────────────────────────────────────────────────────
with tab1:
    try:
        df_asset = bt_attr.contributions_by_asset(aln_ipo)
        df_cum = (
            df_asset.group_by("ticker")
            .agg(pl.col("contrib").sum().alias("contrib_total"))
            .sort("contrib_total", descending=True)
            .with_columns(pl.col("contrib_total").cast(pl.Float64, strict=False))
        )

        col1, col2 = st.columns(2)
        show_plot(
            viz.plot_top_contributors(df_cum.head(10), title="Top 10 Contributors"),
            key="asset_top10",
            st_obj=col1,
        )
        show_plot(
            viz.plot_top_contributors(df_cum.tail(10), title="Bottom 10 Contributors"),
            key="asset_bottom10",
            st_obj=col2,
        )
        st.dataframe(df_cum.head(15))
    except Exception as e:
        st.error(f"Asset-level attribution failed: {e}")

# ─────────────────────────────────────────────────────────────────────
# TAB 2 – Group attribution
# ─────────────────────────────────────────────────────────────────────
with tab2:
    try:
        df_total, df_daily = bt_attr.group_contrib(bt, df_ret_wide, groups_map=groups_map)

        df_total = df_total.with_columns(
            [
                pl.col("contrib_total").cast(pl.Float64, strict=False),
                pl.col("weight_avg").cast(pl.Float64, strict=False),
            ]
        )

        show_plot(
            viz.plot_group_contrib_area(df_daily, title="Group Contributions Over Time"),
            key="group_area",
        )
        show_plot(viz.plot_group_contrib(df_total), key="group_bar")
        st.dataframe(df_total)
    except Exception as e:
        st.error(f"Group attribution failed: {e}")

# ─────────────────────────────────────────────────────────────────────
# TAB 3 – Brinson–Fachler
# ─────────────────────────────────────────────────────────────────────
with tab3:
    try:
        # 3A) Global cumulative series (alloc/select/interact/total)
        result = bt_attr.brinson_fachler_vectorized(aln_ipo, Wb_daily, groups_idx, cumulative=True)

        df_brinson = pl.DataFrame(
            {
                "date": pl.Series("date", list(pl.Series(result.date).to_list())),
                "alloc": pl.Series("alloc", result.alloc, dtype=pl.Float64),
                "select": pl.Series("select", result.select, dtype=pl.Float64),
                "interact": pl.Series("interact", result.interact, dtype=pl.Float64),
                "total": pl.Series("total", result.total, dtype=pl.Float64),
            }
        )
        df_brinson = _ensure_datetime(df_brinson, "date")

        show_plot(
            viz.plot_brinson_cumulative(df_brinson, title="Brinson–Fachler Cumulative Attribution"),
            key="brinson_cum",
        )
        show_plot(viz.plot_brinson_cumulative_components(df_brinson), key="brinson_comp")

        # 3B) Timeseries por grupo (normaliza a formato largo para plotting)
        df_brinson_g_raw = bt_attr.brinson_fachler_timeseries(
            aln_ipo, Wb_daily, groups_idx, cumulative=True, by_group=True
        )
        df_brinson_g = _coerce_brinson_ts_to_long(df_brinson_g_raw).with_columns(
            [
                pl.col("group_id").cast(pl.Int64, strict=False),
                pl.col("alloc").cast(pl.Float64, strict=False),
                pl.col("select").cast(pl.Float64, strict=False),
                pl.col("interact").cast(pl.Float64, strict=False),
                pl.col("total").cast(pl.Float64, strict=False),
            ]
        )
        df_brinson_g = _ensure_datetime(df_brinson_g, "date")

        # Etiquetas para grupos
        gser = df_brinson_g.select(pl.col("group_id").cast(pl.Int64)).get_column("group_id")
        unique_gids = sorted({int(x) for x in gser.unique().to_list() if x is not None})
        if len(unique_gids) == 1 and unique_gids[0] == 0:
            labels = ["Total"]
        else:
            labels = (
                list(group_labels) if group_labels is not None else [f"G{i}" for i in unique_gids]
            )

        show_plot(
            viz.plot_brinson_by_group_area(
                df_brinson_g,
                group_labels=labels,
                component="total",
                title="Brinson by Group (Total)",
            ),
            key="brinson_group",
        )
        st.dataframe(df_brinson.tail(10))

    except Exception as e:
        st.error(f"Brinson–Fachler attribution failed: {e}")

# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("📤 Export results")

if "df_asset" in locals():
    st.download_button(
        "Download asset daily contributions (CSV)",
        _csv_bytes(df_asset),
        file_name="contrib_asset_daily.csv",
        mime="text/csv",
    )

if "df_total" in locals():
    st.download_button(
        "Download group totals (CSV)",
        _csv_bytes(df_total),
        file_name="contrib_group_total.csv",
        mime="text/csv",
    )

if "df_brinson" in locals():
    st.download_button(
        "Download Brinson–Fachler results (CSV)",
        _csv_bytes(df_brinson),
        file_name="brinson_fachler_cumulative.csv",
        mime="text/csv",
    )
