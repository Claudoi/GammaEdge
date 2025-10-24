# app/pages/05_Attribution.py
from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st

# ---------------------------------------------------------------------
# Repo root for local imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Core modules
from portfolio.backtest import attribution as bt_attr
from portfolio.viz import plot_utils as viz

# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------
st.set_page_config(page_title="Attribution Analysis", layout="wide")
st.title("📊 Performance Attribution Dashboard")
st.caption("Vectorized Brinson–Fachler decomposition and return contribution analysis.")

# ---------------------------------------------------------------------
# Defensive handoff from previous pages
# ---------------------------------------------------------------------
bt = st.session_state.get("bt", None)
df_ret_wide = st.session_state.get("df_ret_wide", st.session_state.get("returns_wide", None))
asset_meta = st.session_state.get("asset_meta", None)

if bt is None or df_ret_wide is None:
    st.warning("⚠️ Run pages 02–04 first, then export to Attribution.")
    st.stop()

# ---------------------------------------------------------------------
# Normalize inputs
# ---------------------------------------------------------------------
if isinstance(df_ret_wide, pd.DataFrame):
    df_ret_wide = pl.from_pandas(df_ret_wide)
if df_ret_wide.schema.get("date") != pl.Datetime:
    df_ret_wide = df_ret_wide.with_columns(pl.col("date").cast(pl.Datetime))

# ---------------------------------------------------------------------
# 1️⃣ Alignment & IPO safety
# ---------------------------------------------------------------------
try:
    aln = bt_attr._daily_alignment_from_bt(bt, df_ret_wide)
    aln_ipo = bt_attr.align_with_ipo_mask(aln)
    st.success("✅ Alignment successful (IPO-safe).")
except Exception as e:
    st.error(f"Alignment failed: {e}")
    st.stop()

# ---------------------------------------------------------------------
# 2️⃣ Grouping setup
# ---------------------------------------------------------------------
try:
    meta_df = asset_meta
    if isinstance(meta_df, pd.DataFrame):
        meta_df = pl.from_pandas(meta_df)
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
    groups_idx = list(range(len(aln_ipo.tickers)))
    group_labels = aln_ipo.tickers
    groups_map = {tk: tk for tk in aln_ipo.tickers}

# ---------------------------------------------------------------------
# 3️⃣ Benchmark construction
# ---------------------------------------------------------------------
try:
    Wb_daily = bt_attr.coerce_benchmark_weights(
        Wb=None,
        T=len(aln_ipo.dates),
        N=len(aln_ipo.tickers),
        scheme="EW",
    )
except Exception as e:
    st.error(f"Benchmark generation failed: {e}")
    st.stop()

# ---------------------------------------------------------------------
# Tabs for interactive exploration
# ---------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Asset-level", "🏗️ Group-level", "🧩 Brinson–Fachler"])

# ---------------------------------------------------------------------
# TAB 1 – Asset-level contributions
# ---------------------------------------------------------------------
with tab1:
    try:
        df_asset = bt_attr.contributions_by_asset(aln_ipo)
        df_cum = (
            df_asset.group_by("ticker")
            .agg(pl.col("contrib").sum().alias("contrib_total"))
            .sort("contrib_total", descending=True)
        )
        col1, col2 = st.columns(2)
        col1.plotly_chart(
            viz.plot_top_contributors(df_cum.head(10), title="Top 10 Contributors"),
            use_container_width=True,
        )
        col2.plotly_chart(
            viz.plot_top_contributors(df_cum.tail(10), title="Bottom 10 Contributors"),
            use_container_width=True,
        )
        st.dataframe(df_cum.head(15))
    except Exception as e:
        st.error(f"Asset-level attribution failed: {e}")

# ---------------------------------------------------------------------
# TAB 2 – Group attribution
# ---------------------------------------------------------------------
with tab2:
    try:
        df_total, df_daily = bt_attr.group_contrib(bt, df_ret_wide, groups_map=groups_map)
        st.plotly_chart(
            viz.plot_group_contrib_area(df_daily, title="Group Contributions Over Time"),
            use_container_width=True,
        )
        st.plotly_chart(
            viz.plot_group_contrib_bar_total(df_total, k=min(10, df_total.height), orientation="h"),
            use_container_width=True,
        )
        st.dataframe(df_total)
    except Exception as e:
        st.error(f"Group attribution failed: {e}")

# ---------------------------------------------------------------------
# TAB 3 – Brinson–Fachler attribution
# ---------------------------------------------------------------------
with tab3:
    try:
        # Vectorized version
        result = bt_attr.brinson_fachler_vectorized(aln_ipo, Wb_daily, groups_idx, cumulative=True)
        df_brinson = pl.DataFrame(
            {
                "date": result.date,
                "alloc": result.alloc,
                "select": result.select,
                "interact": result.interact,
                "total": result.total,
            }
        )

        st.plotly_chart(
            viz.plot_brinson_cumulative(df_brinson, title="Brinson–Fachler Cumulative Attribution"),
            use_container_width=True,
        )

        st.plotly_chart(
            viz.plot_brinson_cumulative_components(df_brinson),
            use_container_width=True,
        )

        # Per-group breakdown
        df_brinson_g = bt_attr.brinson_fachler_timeseries(
            aln_ipo, Wb_daily, groups_idx, cumulative=True, by_group=True
        )

        try:
            st.plotly_chart(
                viz.plot_brinson_by_group_area(
                    df_brinson_g, component="total", title="Brinson by Group (Total)"
                ),
                use_container_width=True,
            )
        except Exception:
            pdf = df_brinson_g.to_pandas()
            fig = px.area(pdf, x="date", y="total", color="group_id", title="Brinson by Group")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_brinson.tail(10))
    except Exception as e:
        st.error(f"Brinson–Fachler attribution failed: {e}")

# ---------------------------------------------------------------------
# 6️⃣ Export section
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("📤 Export results")

if "df_asset" in locals():
    st.download_button(
        "Download asset daily contributions (CSV)",
        df_asset.write_csv(),
        file_name="contrib_asset_daily.csv",
    )
if "df_total" in locals():
    st.download_button(
        "Download group totals (CSV)",
        df_total.write_csv(),
        file_name="contrib_group_total.csv",
    )
if "df_brinson" in locals():
    st.download_button(
        "Download Brinson–Fachler results (CSV)",
        df_brinson.write_csv(),
        file_name="brinson_fachler_cumulative.csv",
    )
