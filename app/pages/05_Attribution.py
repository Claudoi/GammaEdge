# app/pages/05_Attribution.py
from __future__ import annotations

import os
import sys
import numpy as np
import polars as pl
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------
# Repo root for local imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# --- core attribution ---
from portfolio.backtest import attribution as bt_attr

# --- plots guaranteed to exist in your repo ---
from portfolio.viz.plot_utils import (
    plot_top_contributors,
    plot_group_contrib_area,   # existing signature in your repo
    plot_brinson_cumulative,   # existing signature in your repo
)

# Optional extras (safe to miss)
_HAS_EXTRAS = False
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

# Try to import plot_group_contrib_bar_total; if missing, define a local shim
try:
    from portfolio.viz.plot_utils import plot_group_contrib_bar_total  # type: ignore
    _HAS_BAR_TOTAL = True
except Exception:
    _HAS_BAR_TOTAL = False

    def plot_group_contrib_bar_total(
        df_group_total: pl.DataFrame,
        k: int = 12,
        orientation: str = "v",
        title: str = "Group Total Contribution",
    ) -> go.Figure:
        """
        Local fallback if plot_group_contrib_bar_total isn't available.
        Expects df_group_total with columns: ['group','contrib_total'].
        """
        req = {"group", "contrib_total"}
        if not req.issubset(set(df_group_total.columns)):
            raise ValueError(f"Missing columns for bar plot: {req}")

        pdf = (
            df_group_total
            .select(["group", "contrib_total"])
            .to_pandas()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values("contrib_total", ascending=False)
            .head(max(1, int(k)))
        )

        if orientation == "h":
            fig = go.Figure(go.Bar(y=pdf["group"], x=pdf["contrib_total"], orientation="h"))
            fig.update_layout(
                title=title, xaxis_title="Total Contribution", yaxis_title="Group", template="plotly_white"
            )
        else:
            fig = go.Figure(go.Bar(x=pdf["group"], y=pdf["contrib_total"]))
            fig.update_layout(
                title=title, xaxis_title="Group", yaxis_title="Total Contribution", template="plotly_white"
            )
        return fig


# ─────────────────────────────────────────────────────────────────────
# Helpers (local fallbacks)
# ─────────────────────────────────────────────────────────────────────

def _group_share_from_daily(df_group_daily: pl.DataFrame) -> pl.DataFrame:
    """
    Build a per-date share series from df_group_daily without using pl.abs module-level.
    Returns DF: ['date','group','share'] where share = |contrib_g| / Σ_g |contrib_g|.
    """
    req = {"date", "group", "contrib"}
    if not req.issubset(set(df_group_daily.columns)):
        raise ValueError(f"df_group_daily must contain {req}")

    # |contrib| per (date, group)
    df_abs = df_group_daily.with_columns(pl.col("contrib").abs().alias("abs_contrib"))

    # total |contrib| per date
    df_tot = (
        df_abs.group_by("date")
        .agg(pl.col("abs_contrib").sum().alias("abs_total"))
    )

    # join & compute share with safe divide
    df_share = (
        df_abs.join(df_tot, on="date", how="left")
        .with_columns(
            (pl.col("abs_contrib") / pl.when(pl.col("abs_total") > 1e-15).then(pl.col("abs_total")).otherwise(1.0))
            .alias("share")
        )
        .select(["date", "group", "share"])
        .sort(["date", "group"])
    )
    return df_share


def _plot_group_share_area(df_share: pl.DataFrame, title: str = "Group Contribution Share (%)") -> go.Figure:
    """
    Simple stacked area plot for group share over time (0–100%).
    """
    req = {"date", "group", "share"}
    if not req.issubset(set(df_share.columns)):
        raise ValueError(f"df_share must contain {req}")

    pdf = (
        df_share
        .to_pandas()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["date", "group", "share"])
        .sort_values("date")
    )
    fig = px.area(pdf, x="date", y="share", color="group", title=title,
                  labels={"share": "Share"})
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Share",
        yaxis_tickformat=".0%",
        legend_title="Group",
        margin=dict(l=60, r=40, t=60, b=60)
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Attribution", layout="wide")
st.title("🔎 Attribution")
st.caption("Return contribution by asset and group, plus Brinson–Fachler.")


# ─────────────────────────────────────────────────────────────────────
# Defensive handoff: require artifacts from 04_Backtest (with fallback)
# ─────────────────────────────────────────────────────────────────────
bt = st.session_state.get("bt", None)
df_ret_wide = st.session_state.get("df_ret_wide", st.session_state.get("returns_wide", None))
asset_meta = st.session_state.get("asset_meta", None)
Wb_daily_ss = st.session_state.get("bench_weights_daily", None)

if bt is None or df_ret_wide is None:
    with st.expander("Debug session_state keys", expanded=False):
        st.write(sorted(list(st.session_state.keys())))
    st.warning(
        "Run pages **02 → 04** first so we can retrieve `bt` and `returns_wide`.\n"
        "Tip: in 04, click **Export to 05_Attribution** after running the backtest."
    )
    st.stop()

# Normalize dataframe to Polars and date dtype
try:
    if isinstance(df_ret_wide, pd.DataFrame):
        df_ret_wide = pl.from_pandas(df_ret_wide)
except Exception:
    pass

if not isinstance(df_ret_wide, pl.DataFrame):
    st.error("`df_ret_wide` must be a Polars/Pandas DataFrame.")
    st.stop()

if df_ret_wide.schema.get("date") != pl.Datetime:
    df_ret_wide = df_ret_wide.with_columns(pl.col("date").cast(pl.Datetime))


# ─────────────────────────────────────────────────────────────────────
# 1) Build daily alignment (IPO-safe)
# ─────────────────────────────────────────────────────────────────────
st.subheader("Data alignment")
try:
    tickers_bt = list(bt["tickers"])
    dates_bt = list(bt["dates"])

    # ensure all tickers exist as columns (for IPO / missing history)
    have_cols = set(df_ret_wide.columns)
    add_cols = [tk for tk in tickers_bt if tk not in have_cols]
    if add_cols:
        df_ret_wide = df_ret_wide.with_columns(**{c: pl.lit(None, dtype=pl.Float64) for c in add_cols})

    # filter/select in the exact order
    df_ret_bt = (
        df_ret_wide
        .filter(pl.col("date").is_in(dates_bt))
        .unique(subset=["date"])
        .sort("date")
        .select(["date", *tickers_bt])
    )

    # expand weights to daily grid
    W_reb = np.asarray(bt["weights"], dtype=float)  # (K, N)
    rb_dates = list(bt.get("rebalance_dates", []))
    if W_reb.size == 0 or len(rb_dates) != W_reb.shape[0]:
        K = W_reb.shape[0]
        step = max(1, len(dates_bt) // max(1, K))
        rb_dates = dates_bt[::step][:K]

    W_daily = bt_attr.expand_rebalance_weights(
        dates=df_ret_bt.get_column("date").to_list(),
        rb_dates=rb_dates,
        W_reb=W_reb,
    )

    # align returns & weights
    aln = bt_attr.align_returns_and_weights(df_ret_bt, W_daily)

    # IPO-safe renormalization: zero weights pre-inception per asset
    R = np.asarray(aln.returns, dtype=float)
    W = np.asarray(aln.weights, dtype=float).copy()
    T, N = R.shape

    valid = np.isfinite(R) & (np.abs(R) > 1e-15)
    inception_idx = np.full(N, T, dtype=int)
    for j in range(N):
        nz = np.nonzero(valid[:, j])[0]
        inception_idx[j] = int(nz[0]) if nz.size > 0 else T

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
# 2) Asset-level contributions
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Asset-level contributions")
try:
    df_asset_daily = bt_attr.contributions_by_asset(aln_ipo)

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

    if _HAS_EXTRAS:
        with st.expander("More asset diagnostics", expanded=False):
            st.plotly_chart(plot_contrib_heatmap_daily(df_asset_daily), use_container_width=True)
            st.plotly_chart(plot_top_contributors_waterfall(df_cum, k=12), use_container_width=True)
except Exception as e:
    st.info(f"Asset-level attribution unavailable: {e}")


# ─────────────────────────────────────────────────────────────────────
# 3) Group attribution
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Group attribution")
try:
    # Default: identity grouping (each asset is its own group)
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

    # dtype hygiene
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

    st.plotly_chart(
        plot_group_contrib_area(df_group_daily, title="Group Contributions Over Time"),
        use_container_width=True,
    )
    st.plotly_chart(
        plot_group_contrib_bar_total(df_group_total, k=min(12, df_group_total.height), orientation="h"),
        use_container_width=True,
    )

    # Share (% of absolute contribution) — robust local fallback
    try:
        if _HAS_EXTRAS and hasattr(bt_attr, "contributions_share_by_group"):
            df_share = bt_attr.contributions_share_by_group(df_group_daily)  # your advanced helper
            st.plotly_chart(
                plot_group_share_area_from_share(df_share),
                use_container_width=True,
            )
        else:
            df_share = _group_share_from_daily(df_group_daily)
            st.plotly_chart(
                _plot_group_share_area(df_share),
                use_container_width=True,
            )
    except Exception:
        # If anything fails here, keep the core plots working
        pass

except Exception as e:
    st.info(f"Group attribution unavailable: {e}")


# ─────────────────────────────────────────────────────────────────────
# 4) Brinson–Fachler
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Brinson–Fachler")
try:
    T, N = aln_ipo.returns.shape

    # Benchmark: use session’s daily weights if present; else equal-weight
    if isinstance(Wb_daily_ss, np.ndarray) and Wb_daily_ss.shape == (T, N):
        Wb_daily = Wb_daily_ss
    else:
        Wb_daily = np.full((T, N), 1.0 / max(N, 1), dtype=float)

    # ---------- Robust group index builder ----------
    def _make_groups_idx_local(tickers, meta_df):
        """Fallback: sector -> country -> identity."""
        if meta_df is not None and "ticker" in meta_df.columns:
            if "sector" in meta_df.columns:
                col = "sector"
            elif "country" in meta_df.columns:
                col = "country"
            else:
                col = None
            if col:
                lut = dict(zip(meta_df["ticker"].to_list(),
                               meta_df[col].to_list()))
                labels = [lut.get(tk, "OTHER") for tk in tickers]
                uniq = {}
                idx = []
                for lab in labels:
                    if lab not in uniq:
                        uniq[lab] = len(uniq)
                    idx.append(uniq[lab])
                return idx, list(uniq.keys()), col
        # identity
        return list(range(len(tickers))), list(tickers), None

    # Try bt_attr.build_groups_idx with multiple signatures
    if hasattr(bt_attr, "build_groups_idx"):
        col_pref = None
        if asset_meta is not None:
            if "sector" in asset_meta.columns:
                col_pref = "sector"
            elif "country" in asset_meta.columns:
                col_pref = "country"

        try:
            # Newer signature
            groups_idx, group_labels, _ = bt_attr.build_groups_idx(
                tickers=aln_ipo.tickers,
                meta_df=asset_meta,
                col=col_pref,
                other="OTHER",
                fallback_identity=True,
            )
        except TypeError:
            # No fallback_identity
            try:
                groups_idx, group_labels, _ = bt_attr.build_groups_idx(
                    tickers=aln_ipo.tickers,
                    meta_df=asset_meta,
                    col=col_pref,
                    other="OTHER",
                )
            except TypeError:
                # Very old: without 'other'
                try:
                    groups_idx, group_labels, _ = bt_attr.build_groups_idx(
                        tickers=aln_ipo.tickers,
                        meta_df=asset_meta,
                        col=col_pref,
                    )
                except Exception:
                    groups_idx, group_labels, _ = _make_groups_idx_local(aln_ipo.tickers, asset_meta)
        except Exception:
            groups_idx, group_labels, _ = _make_groups_idx_local(aln_ipo.tickers, asset_meta)
    else:
        groups_idx, group_labels, _ = _make_groups_idx_local(aln_ipo.tickers, asset_meta)
    # ---------- end robust builder ----------

    df_brinson = bt_attr.brinson_fachler_cumulative(
        aln=aln_ipo,
        bench_weights_daily=Wb_daily,
        groups_idx=groups_idx,
    )
    st.plotly_chart(
        plot_brinson_cumulative(df_brinson, title="Brinson-Fachler Attribution"),
        use_container_width=True,
    )

    # Optional “by group” timeseries if your repo lo tiene
    if _HAS_EXTRAS and hasattr(bt_attr, "brinson_fachler_timeseries"):
        df_brinson_g = bt_attr.brinson_fachler_timeseries(
            aln=aln_ipo,
            bench_weights_daily=Wb_daily,
            groups_idx=groups_idx,
            cumulative=True,
            by_group=True,
        )
        st.plotly_chart(
            plot_brinson_by_group_area(df_brinson_g, group_labels=None, component="total"),
            use_container_width=True,
        )
except Exception as e:
    st.info(f"Brinson attribution unavailable: {e}")


# ─────────────────────────────────────────────────────────────────────
# 5) Exports
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Exports")
try:
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