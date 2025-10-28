# app/pages/05_Attribution.py
from __future__ import annotations

import os
import sys

import pandas as pd
import polars as pl
import streamlit as st

# ---------------------------------------------------------------------
# Repo root for local imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Core modules
from portfolio.backtest import attribution as bt_attr
from portfolio.viz import plot_utils as viz
from portfolio.viz.plot_utils import show_plot


def _csv_bytes(df: pl.DataFrame) -> bytes:
    """Return CSV bytes from a Polars DF, evitando dtypes 'object'."""
    try:
        cols = []
        for name, dtype in df.schema.items():
            if dtype == pl.Datetime:
                cols.append(pl.col(name).dt.to_string().alias(name))
            elif dtype == pl.Object or dtype == pl.List:
                cols.append(pl.col(name).cast(pl.String, strict=False).alias(name))
            else:
                cols.append(pl.col(name))
        df2 = df.select(cols)
        return df2.write_csv().encode()
    except Exception:
        # Fallback universal method
        return df.to_pandas().to_csv(index=False).encode()


def _ensure_datetime(df: pl.DataFrame, col: str = "date") -> pl.DataFrame:
    """Force `col` to pl.Datetime from common incoming dtypes (Object, Utf8, date, np.datetime64)."""
    dt = df.schema.get(col)
    if dt == pl.Datetime:
        return df

    try:
        df1 = df.with_columns(pl.col(col).cast(pl.Datetime, strict=False))
        if df1.schema.get(col) == pl.Datetime:
            return df1
    except Exception:
        pass

    try:
        df2 = df.with_columns(
            pl.col(col).cast(pl.Utf8, strict=False).str.strptime(pl.Datetime, strict=False)
        )
        if df2.schema.get(col) == pl.Datetime:
            return df2
    except Exception:
        pass

    try:
        s = df.get_column(col)
        as_pd = pd.to_datetime(s.to_list(), errors="coerce")
        df3 = df.with_columns(pl.Series(col, as_pd.to_pydatetime()).cast(pl.Datetime, strict=False))
        return df3
    except Exception:
        return df


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
    groups_idx = list(range(len(aln_ipo.tickers)))
    group_labels = aln_ipo.tickers
    groups_map = {tk: tk for tk in aln_ipo.tickers}

# ---------------------------------------------------------------------
# 3️⃣ Benchmark construction
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

# ---------------------------------------------------------------------
# TAB 1 – Asset-level
# ---------------------------------------------------------------------
with tab1:
    try:
        df_asset = bt_attr.contributions_by_asset(aln_ipo)
        # top/bottom per global contribution
        df_cum = (
            df_asset.group_by("ticker")
            .agg(pl.col("contrib").sum().alias("contrib_total"))
            .sort("contrib_total", descending=True)
        ).with_columns(pl.col("contrib_total").cast(pl.Float64, strict=False))

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


# ---------------------------------------------------------------------
# TAB 2 – Group attribution
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# TAB 3 – Brinson–Fachler
# ---------------------------------------------------------------------
with tab3:
    try:
        # — small helper: enforce pl.Datetime —
        def _force_pl_datetime(df: pl.DataFrame, col: str = "date") -> pl.DataFrame:
            if df.schema.get(col) == pl.Datetime:
                return df
            try:
                df1 = df.with_columns(pl.col(col).cast(pl.Datetime, strict=False))
                if df1.schema.get(col) == pl.Datetime:
                    return df1
            except Exception:
                pass
            try:
                df2 = df.with_columns(
                    pl.col(col).cast(pl.Utf8, strict=False).str.strptime(pl.Datetime, strict=False)
                )
                if df2.schema.get(col) == pl.Datetime:
                    return df2
            except Exception:
                pass
            s = df.get_column(col)
            dt = pd.to_datetime(s.to_list(), errors="coerce")
            return df.with_columns(
                pl.Series(col, dt.to_pydatetime()).cast(pl.Datetime, strict=False)
            )

        # — coercer: cualquier salida del backend -> formato largo estándar —
        def _coerce_brinson_ts_to_long(df: pl.DataFrame) -> pl.DataFrame:
            cols = set(df.columns)
            metrics = ["alloc", "select", "interact", "total"]
            need = {"date", *metrics}

            # 0) NUEVO: agregado sin grupo (solo date + métricas)
            if need.issubset(cols) and "group_id" not in cols and "group" not in cols:
                return (
                    df.select(list(need))
                    .with_columns(pl.lit(0).alias("group_id"))  # grupo sintético
                    .select(["date", "group_id"] + metrics)
                )

            # 1) largo con group_id
            if {"group_id", *need}.issubset(cols):
                return df.select(["date", "group_id"] + metrics)

            # 2) largo con group
            if {"group", *need}.issubset(cols):
                uniq = sorted(
                    [g for g in df.get_column("group").unique().to_list() if g is not None]
                )
                mapping = {g: i for i, g in enumerate(uniq)}
                df2 = df.with_columns(
                    pl.col("group").map_elements(lambda g: mapping.get(g, -1)).alias("group_id")
                )
                return df2.select(["date", "group_id"] + metrics)

            # 3) ancho: alloc_0/select_0/interact_0/total_0 …
            pattern_cols: dict[str, list[str]] = {}
            for m in metrics:
                cs = [c for c in df.columns if c.startswith(m + "_") or c.startswith(m + "_g")]
                if not cs:
                    cs = [
                        c
                        for c in df.columns
                        if c.startswith(m) and any(ch.isdigit() for ch in c[len(m) :])
                    ]
                pattern_cols[m] = cs

            if any(pattern_cols[m] for m in metrics):

                def _parse_gid(name: str) -> int:
                    tail = (
                        name.split("_", 1)[-1]
                        if "_" in name
                        else name[len(name.rstrip("0123456789")) :]
                    )
                    tail = tail[1:] if tail.startswith(("g", "G")) else tail
                    digits = "".join(ch for ch in tail if ch.isdigit())
                    return int(digits) if digits else -1

                long_parts: list[pl.DataFrame] = []
                for m in metrics:
                    cs = pattern_cols[m]
                    if not cs:
                        continue
                    melted = df.select(["date"] + cs).melt(
                        id_vars="date", variable_name="col", value_name=m
                    )
                    melted = melted.with_columns(
                        pl.col(m).cast(pl.Float64, strict=False),
                        pl.col("col").map_elements(_parse_gid).alias("group_id"),
                    ).drop("col")
                    long_parts.append(melted)

                out = long_parts[0]
                for part in long_parts[1:]:
                    out = out.join(part, on=["date", "group_id"], how="outer")
                return out.select(["date", "group_id"] + metrics)

            raise ValueError(
                "Unsupported Brinson timeseries format: neither {group_id, group} nor wide metric_* columns found. "
                f"Available columns: {sorted(df.columns)}"
            )

        # ===== 3A) Serie acumulada (global) =====
        result = bt_attr.brinson_fachler_vectorized(aln_ipo, Wb_daily, groups_idx, cumulative=True)
        try:
            date_py = pd.to_datetime(result.date, errors="coerce").to_pydatetime()
        except Exception:
            date_py = result.date

        df_brinson = pl.DataFrame(
            {
                "date": pl.Series("date", list(date_py)),
                "alloc": pl.Series("alloc", result.alloc, dtype=pl.Float64),
                "select": pl.Series("select", result.select, dtype=pl.Float64),
                "interact": pl.Series("interact", result.interact, dtype=pl.Float64),
                "total": pl.Series("total", result.total, dtype=pl.Float64),
            }
        )
        df_brinson = _force_pl_datetime(df_brinson, "date")

        show_plot(
            viz.plot_brinson_cumulative(df_brinson, title="Brinson–Fachler Cumulative Attribution"),
            key="brinson_cum",
        )
        show_plot(viz.plot_brinson_cumulative_components(df_brinson), key="brinson_comp")

        # ===== 3B) Time series por grupo =====
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
        df_brinson_g = _force_pl_datetime(df_brinson_g, "date")

        # labels: si solo hay un grupo sintético, usa ["Total"]; si no, usa group_labels existentes
        unique_gids = sorted(
            [
                int(x)
                for x in pl.select(df_brinson_g["group_id"]).unique().to_series().to_list()
                if x is not None
            ]
        )
        if len(unique_gids) == 1 and unique_gids[0] == 0:
            _labels = ["Total"]
        else:
            _labels = (
                group_labels
                if "group_labels" in locals() and group_labels is not None
                else [f"G{i}" for i in unique_gids]
            )

        show_plot(
            viz.plot_brinson_by_group_area(
                df_brinson_g,
                group_labels=_labels,
                component="total",
                title="Brinson by Group (Total)",
            ),
            key="brinson_group",
        )

        st.dataframe(df_brinson.tail(10))

    except Exception as e:
        st.error(f"Brinson–Fachler attribution failed: {e}")


# ---------------------------------------------------------------------
# Export section
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
