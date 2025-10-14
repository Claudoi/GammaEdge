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

# --- plots (core) ---
from portfolio.viz.plot_utils import (
    plot_top_contributors,
    plot_group_contrib_area,
    plot_brinson_cumulative,
    plot_brinson_cumulative_components,
    plot_brinson_final_bar,
    plot_brinson_by_group_area,
)

# --- extras optionals ---
_HAS_EXTRAS = False
try:
    from portfolio.viz.plot_utils import (
        plot_contrib_heatmap_daily,
        plot_top_contributors_waterfall,
        plot_group_share_area_from_share,
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


#  helper: build daily benchmark weights automatically (no user params) ---
def _make_benchmark_daily(aln: bt_attr.DailyAlignment,
                          bt_dict: dict,
                          asset_meta: pl.DataFrame | None) -> np.ndarray:
    """
    Returns Wb_daily with shape (T, N), fully automated:
      1) If session benchmark exists and matches shape -> use it
      2) Else build a buy-and-hold benchmark from w0 and cum returns:
         w_b(t) ∝ w0 * Π(1+r)  (row-normalized each day)
         where w0 = bt['weights'][0] if available else equal-weight
      3) If asset_meta has 'cap0'/'mcap0' (optional), it will override w0.
    """
    R = np.asarray(aln.returns, dtype=float)          # (T, N)
    T, N = R.shape

    # 2a) base w0 from bt first rebalance or equal-weight
    w0 = None
    try:
        W_reb = np.asarray(bt_dict.get("weights", []), dtype=float)
        if W_reb.ndim == 2 and W_reb.shape[1] == N and W_reb.shape[0] >= 1:
            w0 = W_reb[0].astype(float, copy=False)
    except Exception:
        pass
    if w0 is None:
        w0 = np.full(N, 1.0 / max(N, 1), dtype=float)

    # 2b) optional override: use initial market-cap proxy if available
    if asset_meta is not None:
        cols = set(asset_meta.columns)
        # look for a plausible initial cap column name
        for cap_col in ("cap0", "mcap0", "market_cap0", "mcap_init"):
            if cap_col in cols:
                try:
                    lut = dict(zip(asset_meta["ticker"].to_list(),
                                   asset_meta[cap_col].to_list()))
                    w0 = np.array([float(lut.get(tk, 0.0)) for tk in aln.tickers], dtype=float)
                    s = float(np.sum(w0))
                    w0 = (w0 / s) if s > 1e-12 else np.full(N, 1.0 / max(N, 1))
                except Exception:
                    pass
                break

    # 2c) buy-and-hold from w0 using cumulated simple returns
    # cum_growth[t, j] = Π_{τ≤t} (1 + r_{τ,j})
    cum_growth = np.cumprod(1.0 + np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0), axis=0)
    # Wb_daily[t, j] ∝ w0[j] * cum_growth[t, j]
    Wb_daily = cum_growth * w0.reshape(1, -1)
    row_sum = np.sum(Wb_daily, axis=1, keepdims=True)
    Wb_daily = Wb_daily / np.clip(row_sum, 1e-15, None)

    return Wb_daily



def _brinson_group_timeseries_local(
    aln: bt_attr.DailyAlignment,
    Wb_daily: np.ndarray,           # shape (T, N)
    groups_idx: list[int],          # asset -> group index [0..G-1]
    group_labels: list[str] | None = None,
    cumulative: bool = True,
) -> pl.DataFrame:
    """
    Build a by-group Brinson–Fachler time series:
    returns Polars DF with columns ['date','group','alloc','select','interact','total'].
    If cumulative=True, components are cum-summed over time.
    """
    R = np.asarray(aln.returns, float)     # (T, N)
    Wp = np.asarray(aln.weights, float)    # (T, N)
    Wb = np.asarray(Wb_daily, float)
    T, N = R.shape

    gi = np.asarray(groups_idx, int)
    G = int(gi.max()) + 1 if gi.size else 0
    if group_labels is None or len(group_labels) != G:
        group_labels = [f"G{g}" for g in range(G)]

    # Accumulators (if cumulative)
    A_c = np.zeros((T, G), float)
    S_c = np.zeros((T, G), float)
    I_c = np.zeros((T, G), float)

    for t in range(T):
        for g in range(G):
            idx = (gi == g)
            if not np.any(idx):
                continue
            wp_g = float(np.sum(Wp[t, idx]))
            wb_g = float(np.sum(Wb[t, idx]))
            rb_g = float(np.sum(Wb[t, idx] * R[t, idx])) / wb_g if wb_g > 1e-16 else 0.0
            rp_g = float(np.sum(Wp[t, idx] * R[t, idx])) / wp_g if wp_g > 1e-16 else 0.0

            diff_w = wp_g - wb_g
            diff_r = rp_g - rb_g

            a = diff_w * rb_g
            s = wb_g   * diff_r
            inter = diff_w * diff_r

            if t == 0 or not cumulative:
                A_c[t, g] = a
                S_c[t, g] = s
                I_c[t, g] = inter
            else:
                A_c[t, g] = A_c[t-1, g] + a
                S_c[t, g] = S_c[t-1, g] + s
                I_c[t, g] = I_c[t-1, g] + inter

    rows = []
    for t in range(T):
        for g in range(G):
            tot = A_c[t, g] + S_c[t, g] + I_c[t, g]
            rows.append((aln.dates[t], group_labels[g], A_c[t, g], S_c[t, g], I_c[t, g], tot))

    return pl.DataFrame(rows, schema=["date", "group", "alloc", "select", "interact", "total"]) \
             .with_columns(pl.col("date").cast(pl.Datetime))


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
# 4) Brinson–Fachler Attribution
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Brinson–Fachler")

try:
    # --- 1) Setup basic shapes -----------------------------------------------
    T, N = aln_ipo.returns.shape

    # --- 2) Benchmark weights (automatic) ------------------------------------
    Wb_session = st.session_state.get("bench_weights_daily", None)
    if isinstance(Wb_session, np.ndarray) and Wb_session.shape == (T, N):
        Wb_daily = Wb_session
    else:
        Wb_daily = _make_benchmark_daily(aln_ipo, bt, asset_meta)

    # --- 3) Group indices and labels -----------------------------------------
    meta_df = asset_meta
    try:
        if isinstance(meta_df, pd.DataFrame):
            meta_df = pl.from_pandas(meta_df)
    except Exception:
        pass

    if hasattr(bt_attr, "build_groups_idx") and (meta_df is not None):
        if "sector" in meta_df.columns:
            groups_idx, group_labels, _ = bt_attr.build_groups_idx(
                tickers=aln_ipo.tickers, meta_df=meta_df, col="sector", other="OTHER"
            )
        elif "country" in meta_df.columns:
            groups_idx, group_labels, _ = bt_attr.build_groups_idx(
                tickers=aln_ipo.tickers, meta_df=meta_df, col="country", other="OTHER"
            )
        else:
            groups_idx = list(range(N))
            group_labels = [f"A{i}" for i in range(N)]
    else:
        groups_idx = list(range(N))
        group_labels = [f"A{i}" for i in range(N)]

    # --- 4) Cumulative Brinson ------------------------------------------------
    df_brinson = bt_attr.brinson_fachler_cumulative(
        aln=aln_ipo,
        bench_weights_daily=Wb_daily,
        groups_idx=groups_idx,
    )

    # Main cumulative line chart
    st.plotly_chart(
        plot_brinson_cumulative(df_brinson, title="Brinson-Fachler Attribution (Total)"),
        use_container_width=True,
    )

    # --- 5) Optional advanced component charts -------------------------------
    # If your repo has these functions, display detailed and summary breakdowns.
    try:
        if "plot_brinson_cumulative_components" in globals() or hasattr(
            sys.modules.get("portfolio.viz.plot_utils"), "plot_brinson_cumulative_components"
        ):
            st.plotly_chart(
                plot_brinson_cumulative_components(df_brinson),
                use_container_width=True,
            )

        if "plot_brinson_final_bar" in globals() or hasattr(
            sys.modules.get("portfolio.viz.plot_utils"), "plot_brinson_final_bar"
        ):
            st.plotly_chart(
                plot_brinson_final_bar(df_brinson),
                use_container_width=True,
            )
    except Exception as e:
        st.info(f"Optional Brinson component plots skipped: {e}")

    # --- 6) By-group cumulative timeseries -----------------------------------
    df_brinson_g = None
    if hasattr(bt_attr, "brinson_fachler_timeseries") and _HAS_EXTRAS:
        try:
            tmp = bt_attr.brinson_fachler_timeseries(
                aln=aln_ipo,
                bench_weights_daily=Wb_daily,
                groups_idx=groups_idx,
                cumulative=True,
                by_group=True,
            )
            rename_map = {
                "allocation": "alloc", "selection": "select", "interaction": "interact",
                "cum_total": "total", "group_name": "group"
            }
            for old, new in rename_map.items():
                if (old in tmp.columns) and (new not in tmp.columns):
                    tmp = tmp.rename({old: new})
            required = {"date", "group", "alloc", "select", "interact", "total"}
            if required.issubset(set(tmp.columns)):
                df_brinson_g = tmp
        except Exception:
            df_brinson_g = None

    # Local fallback helper if repo helper is missing
    if df_brinson_g is None:
        df_brinson_g = _brinson_group_timeseries_local(
            aln=aln_ipo,
            Wb_daily=Wb_daily,
            groups_idx=groups_idx,
            group_labels=group_labels,
            cumulative=True,
        )

    # --- 7) By-group cumulative plots ----------------------------------------
    try:
        if _HAS_EXTRAS:
            try:
                glabels = df_brinson_g.get_column("group").unique().to_list()
            except Exception:
                glabels = []

            try:
                st.plotly_chart(
                    plot_brinson_by_group_area(
                        df_brinson_g,
                        group_labels=glabels,
                        component="total",
                        title="Brinson by Group – Total (Cumulative)",
                    ),
                    use_container_width=True,
                )
            except TypeError:
                st.plotly_chart(
                    plot_brinson_by_group_area(
                        df_brinson_g,
                        component="total",
                        title="Brinson by Group – Total (Cumulative)",
                    ),
                    use_container_width=True,
                )
        else:
            import plotly.express as px
            pdf = df_brinson_g.to_pandas()
            fig = px.area(
                pdf,
                x="date",
                y="total",
                color="group",
                title="Brinson by Group – Total (Cumulative)",
                labels={"total": "Attribution"},
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Brinson by-group plot skipped: {e}")

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