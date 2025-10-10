# app/pages/05_Attribution.py
from __future__ import annotations

# --- stdlib ---
import os
import sys

# --- third-party ---
import numpy as np
import polars as pl
import streamlit as st

# ---------------------------------------------------------------------
# Repo root for local imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# --- core attribution ---
from portfolio.backtest import attribution as bt_attr

# --- plots (keep existing + add-on if available) ---
from portfolio.viz.plot_utils import (
    # Existing, already used elsewhere
    plot_top_contributors,
    plot_group_contrib_area,
    plot_group_contrib_bar_total,
    plot_brinson_cumulative,
)

# Optional extras (wrapped behind try so we never break older installs)
try:
    from portfolio.viz.plot_utils import (
        plot_contrib_heatmap_daily,
        plot_top_contributors_waterfall,
        plot_group_share_area_from_share,
        plot_brinson_by_group_area,
    )
    _HAS_EXTRAS = True
except Exception:
    _HAS_EXTRAS = False


# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Attribution", layout="wide")
st.title("🔎 Attribution")
st.caption("Return contribution by asset and group, plus Brinson–Fachler.")


# ─────────────────────────────────────────────────────────────────────
# Defensive handoff: require artifacts from 04_Backtest
# ─────────────────────────────────────────────────────────────────────
needed = ("bt", "df_ret_wide")
missing = [k for k in needed if k not in st.session_state]
if missing:
    st.warning(
        "Run pages **02 → 04** first so we can retrieve: "
        + ", ".join(missing)
        + ". (RiskModel → Optimizer → Backtest)"
    )
    st.stop()

bt: dict = st.session_state["bt"]
df_ret_wide: pl.DataFrame = st.session_state["df_ret_wide"]

asset_meta: pl.DataFrame | None = st.session_state.get("asset_meta", None)
Wb_daily_ss = st.session_state.get("bench_weights_daily", None)  # optional precomputed benchmark

# Basic guards
if not isinstance(df_ret_wide, pl.DataFrame):
    # Accept pandas too, but normalize to Polars for the pipeline
    try:
        import pandas as pd
        if isinstance(df_ret_wide, pd.DataFrame):
            df_ret_wide = pl.from_pandas(df_ret_wide)
        else:
            raise TypeError
    except Exception:
        st.error("`df_ret_wide` must be a Polars/Pandas DataFrame.")
        st.stop()

# Ensure 'date' dtype is consistent
if df_ret_wide.schema.get("date") != pl.Datetime:
    df_ret_wide = df_ret_wide.with_columns(pl.col("date").cast(pl.Datetime))


# ─────────────────────────────────────────────────────────────────────
# 1) Build daily alignment (IPO-safe); robust to incomplete histories
# ─────────────────────────────────────────────────────────────────────
st.subheader("Data alignment")

try:
    # a) Force the wide returns to include *all* tickers from the backtest (missing → nulls)
    tickers_bt = list(bt["tickers"])
    have_cols = set(df_ret_wide.columns)
    add_cols = [tk for tk in tickers_bt if tk not in have_cols]
    if add_cols:
        df_ret_wide = df_ret_wide.with_columns(**{c: pl.lit(None, dtype=pl.Float64) for c in add_cols})

    # b) Filter to bt dates and exact column order
    dates_bt = list(bt["dates"])
    df_ret_bt = (
        df_ret_wide
        .filter(pl.col("date").is_in(dates_bt))
        .unique(subset=["date"])
        .sort("date")
        .select(["date", *tickers_bt])
    )

    # c) Expand rebalance weights to daily
    W_reb = np.asarray(bt["weights"], dtype=float)   # (K, N)
    rb_dates = list(bt.get("rebalance_dates", []))
    if W_reb.size == 0 or len(rb_dates) != W_reb.shape[0]:
        # Approximate schedule if necessary (should rarely happen with the latest engine)
        K = W_reb.shape[0]
        step = max(1, len(dates_bt) // max(1, K))
        rb_dates = dates_bt[::step][:K]

    W_daily = bt_attr.expand_rebalance_weights(
        dates=df_ret_bt.get_column("date").to_list(),
        rb_dates=rb_dates,
        W_reb=W_reb,
    )

    # d) Align returns & weights (NaN-safe returns)
    aln = bt_attr.align_returns_and_weights(df_ret_bt, W_daily)

    # e) IPO-safe masking: zero weights before each asset's first valid return, then renormalize row-wise
    R = np.asarray(aln.returns, dtype=float)
    W = np.asarray(aln.weights, dtype=float).copy()
    T, N = R.shape

    valid = np.isfinite(R) & (np.abs(R) > 1e-15)
    inception_idx = np.full(N, T, dtype=int)
    for j in range(N):
        nz = np.nonzero(valid[:, j])[0]
        inception_idx[j] = int(nz[0]) if nz.size > 0 else T

    # zero pre-IPO weights
    for j in range(N):
        t0 = inception_idx[j]
        if t0 > 0:
            W[:t0, j] = 0.0

    row_sum = W.sum(axis=1, keepdims=True)
    avail = np.arange(T)[:, None] >= inception_idx[None, :]
    avail_count = avail.sum(axis=1, keepdims=True).astype(float)

    zero_rows = (np.abs(row_sum) < 1e-15).ravel()
    if np.any(zero_rows):
        W[zero_rows, :] = 0.0
        W[zero_rows, :] = np.where(avail[zero_rows, :], 1.0, 0.0)
        W[zero_rows, :] /= np.clip(avail_count[zero_rows, :], 1.0, None)

    nz_rows = ~zero_rows
    if np.any(nz_rows):
        W[nz_rows, :] = W[nz_rows, :] / np.clip(row_sum[nz_rows, :], 1e-15, None)

    aln_ipo = bt_attr.DailyAlignment(
        dates=aln.dates,
        tickers=aln.tickers,
        returns=R,
        weights=W,
    )

    st.success("Alignment OK (IPO-safe).")
except Exception as e:
    st.error(f"Alignment failed: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────
# 2) Asset-level attribution
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Asset-level contributions")

try:
    df_asset_daily = bt_attr.contributions_by_asset(aln_ipo)

    # Cumulative by ticker (keep your existing plot signature)
    df_cum = (
        df_asset_daily.group_by("ticker")
        .agg(pl.col("contrib").sum().alias("contrib_total"))
        .sort("contrib_total", descending=True)
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_top_contributors(df_cum.head(10), title="Top 10 Contributors"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            plot_top_contributors(df_cum.tail(10), title="Bottom 10 Contributors"),
            use_container_width=True,
        )

    # Optional: daily heatmap + waterfall (only if extra plotters exist)
    if _HAS_EXTRAS:
        with st.expander("More asset diagnostics", expanded=False):
            st.plotly_chart(plot_contrib_heatmap_daily(df_asset_daily), use_container_width=True)
            st.plotly_chart(plot_top_contributors_waterfall(df_cum, k=12), use_container_width=True)
except Exception as e:
    st.info(f"Asset-level attribution unavailable: {e}")


# ─────────────────────────────────────────────────────────────────────
# 3) Group attribution (sector/country or identity fallback)
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Group attribution")

try:
    # Default: identity (each ticker is its own group). If metadata exists, prefer sector or country.
    groups_map = {tk: tk for tk in aln_ipo.tickers}
    if asset_meta is not None:
        cols = asset_meta.columns
        if "ticker" in cols and "sector" in cols:
            lut = dict(zip(asset_meta["ticker"].to_list(), asset_meta["sector"].to_list()))
            groups_map = {tk: lut.get(tk, tk) for tk in aln_ipo.tickers}
        elif "ticker" in cols and "country" in cols:
            lut = dict(zip(asset_meta["ticker"].to_list(), asset_meta["country"].to_list()))
            groups_map = {tk: lut.get(tk, tk) for tk in aln_ipo.tickers}

    df_group_daily = bt_attr.contributions_by_group(aln_ipo, groups_map)

    # Ensure stable dtypes for downstream sorting/plotting
    if df_group_daily.schema.get("date") != pl.Datetime:
        df_group_daily = df_group_daily.with_columns(pl.col("date").cast(pl.Datetime))
    if df_group_daily.schema.get("group") != pl.Utf8:
        df_group_daily = df_group_daily.with_columns(pl.col("group").cast(pl.Utf8))

    df_group_total = (
        df_group_daily.group_by("group")
        .agg([
            pl.col("contrib").sum().alias("contrib_total"),
            pl.col("weight").mean().alias("avg_weight"),
        ])
        .sort("contrib_total", descending=True)
    )

    # Existing plots (safe, already in your project)
    st.plotly_chart(
        plot_group_contrib_area(df_group_daily, mode="share"),
        use_container_width=True,
    )
    st.plotly_chart(
        plot_group_contrib_bar_total(df_group_total, k=min(12, df_group_total.height)),
        use_container_width=True,
    )

    # Optional: share-of-total area using new helper
    if _HAS_EXTRAS and hasattr(bt_attr, "contributions_share_by_group"):
        df_share = bt_attr.contributions_share_by_group(df_group_daily)
        st.plotly_chart(
            plot_group_share_area_from_share(df_share),
            use_container_width=True,
        )
except Exception as e:
    st.info(f"Group attribution unavailable: {e}")


# ─────────────────────────────────────────────────────────────────────
# 4) Brinson–Fachler (total + by-group)
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Brinson–Fachler")

try:
    T, N = aln_ipo.returns.shape

    # Benchmark: use session’s `bench_weights_daily` if provided; else EW
    if isinstance(Wb_daily_ss, np.ndarray) and Wb_daily_ss.shape == (T, N):
        Wb_daily = Wb_daily_ss
    else:
        Wb_daily = np.full((T, N), 1.0 / max(N, 1), dtype=float)

    # Group indices for Brinson by group
    group_labels = list(groups_map.values())
    if hasattr(bt_attr, "build_groups_idx"):
        groups_idx, group_labels, groups_map_used = bt_attr.build_groups_idx(
            tickers=aln_ipo.tickers,
            meta_df=asset_meta,
            col=("sector" if (asset_meta is not None and "sector" in asset_meta.columns) else None),
            other="OTHER",
            fallback_identity=True,
        )
    else:
        # Identity fallback
        groups_idx = list(range(N))
        # keep previously built group_labels if available, else ticker names
        if not group_labels:
            group_labels = aln_ipo.tickers

    # Overall Brinson cumulative (existing plot)
    df_brinson = bt_attr.brinson_fachler_cumulative(
        aln=aln_ipo,
        bench_weights_daily=Wb_daily,
        groups_idx=groups_idx,
    )
    st.plotly_chart(
        plot_brinson_cumulative(df_brinson, as_percent=True),
        use_container_width=True,
    )

    # Optional: by-group time series (alloc/select/interact/total per group)
    if _HAS_EXTRAS and hasattr(bt_attr, "brinson_fachler_timeseries"):
        df_brinson_g = bt_attr.brinson_fachler_timeseries(
            aln=aln_ipo,
            bench_weights_daily=Wb_daily,
            groups_idx=groups_idx,
            cumulative=True,
            by_group=True,
        )
        st.plotly_chart(
            plot_brinson_by_group_area(df_brinson_g, group_labels, component="total"),
            use_container_width=True,
        )

except Exception as e:
    st.info(f"Brinson attribution unavailable: {e}")


# ─────────────────────────────────────────────────────────────────────
# 5) Downloadable tables (nice-to-have)
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Exports")

try:
    # Reuse variables if they exist in scope
    if "df_asset_daily" in locals():
        st.download_button(
            "Download asset daily contributions (CSV)",
            df_asset_daily.write_csv(),
            file_name="contrib_asset_daily.csv",
            mime="text/csv",
        )
    if "df_cum" in locals():
        st.download_button(
            "Download asset cumulative contributions (CSV)",
            df_cum.write_csv(),
            file_name="contrib_asset_total.csv",
            mime="text/csv",
        )
    if "df_group_daily" in locals():
        st.download_button(
            "Download group daily contributions (CSV)",
            df_group_daily.write_csv(),
            file_name="contrib_group_daily.csv",
            mime="text/csv",
        )
    if "df_group_total" in locals():
        st.download_button(
            "Download group totals (CSV)",
            df_group_total.write_csv(),
            file_name="contrib_group_total.csv",
            mime="text/csv",
        )
except Exception:
    pass