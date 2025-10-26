# Backtest reporting
# portfolio/backtest/reporting.py
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import polars as pl

from portfolio.core.compat import dataclass_compat as dataclass

from .attribution import (
    DailyAlignment,
    align_returns_and_weights,
    contributions_by_asset,
    contributions_by_group,
)

# ──────────────────────────────────────────────────────────────────────────────
# Plotly figures (robust versions)
# ──────────────────────────────────────────────────────────────────────────────


def _ensure_same_length(
    x: Iterable[Any],
    y: np.ndarray,
    name_x: str = "x",
    name_y: str = "y",
) -> None:
    x_len = len(list(x)) if not isinstance(x, list) else len(x)
    if x_len != len(y):
        raise ValueError(f"{name_x} and {name_y} must have the same length ({x_len} vs {len(y)})")


def fig_equity(
    dates: Sequence[Any],
    equity: np.ndarray,
    title: str = "Equity Curve",
) -> go.Figure:
    eq = np.asarray(equity, dtype=float)
    _ensure_same_length(dates, eq, "dates", "equity")
    eq = np.where(np.isfinite(eq), eq, np.nan)

    fig = go.Figure(
        go.Scatter(
            x=list(dates),
            y=eq,
            mode="lines",
            name="Equity",
            hovertemplate="%{x}<br>NAV: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="NAV", template="plotly_white")
    return fig


def fig_drawdown(
    dates: Sequence[Any],
    equity: np.ndarray,
    title: str = "Drawdown",
) -> go.Figure:
    eq = np.asarray(equity, dtype=float)
    _ensure_same_length(dates, eq, "dates", "equity")
    eq = np.where(np.isfinite(eq), eq, np.nan)

    # Drawdown (ratio, negative in drawdowns)
    cummax = np.maximum.accumulate(np.nan_to_num(eq, nan=-np.inf))
    base = np.where(np.isfinite(cummax) & (cummax > 0), cummax, np.nan)
    dd = (eq / base) - 1.0

    fig = go.Figure(
        go.Scatter(
            x=list(dates),
            y=dd,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            hovertemplate="%{x}<br>DD: %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        yaxis=dict(tickformat=".0%"),
        template="plotly_white",
    )
    return fig


def fig_weights_heatmap(
    dates: Sequence[Any],
    tickers: Sequence[str],
    W: np.ndarray,
    title: str = "Weights",
) -> go.Figure:
    W = np.asarray(W, dtype=float)
    T, N = W.shape
    if len(dates) != T:
        raise ValueError("len(dates) must equal W.shape[0]")
    if len(tickers) != N:
        raise ValueError("len(tickers) must equal W.shape[1]")

    W = np.where(np.isfinite(W), W, np.nan)
    avg_w = np.nanmean(W, axis=0)
    order = np.argsort(-avg_w)
    tickers_ord = [tickers[i] for i in order]
    W_ord = W[:, order]  # (T, N)

    fig = go.Figure(
        data=go.Heatmap(
            z=W_ord.T,
            x=list(dates),
            y=tickers_ord,
            coloraxis="coloraxis",
            zmin=0.0,
            zmax=1.0,  # weights assumed in [0,1]
            hovertemplate="Ticker: %{y}<br>Date: %{x}<br>Weight: %{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(title=title, coloraxis_colorscale="Blues", template="plotly_white")
    return fig


def fig_bar_contrib(df_top: pl.DataFrame, title: str = "Top Contributors") -> go.Figure:
    # expects ['ticker','contrib_total']
    pdf = (
        df_top.select(["ticker", "contrib_total"])
        .to_pandas()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    pdf = pdf.sort_values("contrib_total", ascending=False)
    fig = go.Figure(go.Bar(x=pdf["ticker"], y=pdf["contrib_total"]))
    fig.update_layout(
        title=title, xaxis_title="Ticker", yaxis_title="Total Contribution", template="plotly_white"
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Report Builder (tables + figures)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BacktestReport:
    """
    Lightweight container for Streamlit:
    - tables: dict of Polars DataFrames
    - figures: dict of Plotly figures
    - meta: metadata dict
    """

    tables: dict[str, pl.DataFrame]
    figures: dict[str, go.Figure]
    meta: dict[str, Any]


def _top_contributors_from_df(
    df_asset_contrib: pl.DataFrame, k: int = 10, sign: str = "both"
) -> pl.DataFrame:
    """
    Local top contributors aggregator compatible with this module.
    df_asset_contrib must contain ['ticker','contrib'].
    """
    agg = (
        df_asset_contrib.group_by("ticker")
        .agg(pl.col("contrib").sum().alias("contrib_total"))
        .sort("contrib_total", descending=True)
    )
    if sign == "pos":
        return agg.filter(pl.col("contrib_total") > 0).head(k)
    if sign == "neg":
        return (
            agg.filter(pl.col("contrib_total") < 0).sort("contrib_total", descending=False).head(k)
        )
    # both
    pos = agg.filter(pl.col("contrib_total") > 0).head(k)
    neg = agg.filter(pl.col("contrib_total") < 0).sort("contrib_total", descending=False).head(k)
    return pl.concat([pos, neg])


def build_backtest_report(
    df_ret_wide: pl.DataFrame,  # ['date', tickers...]
    daily_weights: np.ndarray,  # (T, N) daily weights
    equity: np.ndarray,  # (T,) equity on the same daily grid
    *,
    group_map: dict[str, str] | None = None,
    title: str = "Backtest Report",
) -> BacktestReport:
    """
    Assemble common reporting tables and figures: equity, drawdown, weights heatmap,
    attribution (asset & group), and top contributors (+/-).
    """
    aln: DailyAlignment = align_returns_and_weights(df_ret_wide, daily_weights)
    dates = aln.dates
    tickers = aln.tickers
    W = aln.weights

    # Attribution tables
    df_contrib_asset = contributions_by_asset(aln)
    tables: dict[str, pl.DataFrame] = {
        "contrib_asset_daily": df_contrib_asset,
        "contrib_asset_total": (
            df_contrib_asset.group_by("ticker")
            .agg(pl.col("contrib").sum().alias("contrib_total"))
            .sort("contrib_total", descending=True)
        ),
    }

    if group_map:
        df_contrib_group = contributions_by_group(aln, group_map)
        tables["contrib_group_daily"] = df_contrib_group
        tables["contrib_group_total"] = (
            df_contrib_group.group_by("group")
            .agg(
                [
                    pl.col("contrib").sum().alias("contrib_total"),
                    pl.col("weight").mean().alias("avg_weight"),
                ]
            )
            .sort("contrib_total", descending=True)
        )

    # Top contributors (local aggregator to avoid signature mismatch)
    df_top_both = _top_contributors_from_df(df_contrib_asset, k=10, sign="both")
    tables["top_contributors"] = df_top_both

    # Figures
    figures: dict[str, go.Figure] = {
        "equity": fig_equity(dates, equity, title=f"{title} — Equity"),
        "drawdown": fig_drawdown(dates, equity, title=f"{title} — Drawdown"),
        "weights": fig_weights_heatmap(dates, tickers, W, title=f"{title} — Weights"),
        "top_contrib": fig_bar_contrib(df_top_both, title=f"{title} — Top Contributors"),
    }

    meta = {
        "title": title,
        "n_days": len(dates),
        "n_assets": len(tickers),
    }
    return BacktestReport(tables=tables, figures=figures, meta=meta)


# ──────────────────────────────────────────────────────────────────────────────
# HTML report (for Streamlit download button)
# ──────────────────────────────────────────────────────────────────────────────


def _to_html_table(df_any: Any) -> str:
    """Convert Polars/Pandas DataFrame to a compact HTML table string."""
    pdf = df_any.to_pandas() if hasattr(df_any, "to_pandas") else df_any
    html = cast(str, pdf.to_html(index=False, border=0))
    # Simple styling; Streamlit will embed this HTML as-is.
    return html.replace('class="dataframe"', 'class="dataframe" style="font-size:12px"')


def render_html_report(
    bt: Mapping[str, Any] | dict[str, Any],
    df_metrics: pl.DataFrame | Any,
) -> str:
    """
    Minimal HTML report builder from the `bt` dict produced by the engine
    and a metrics table (Polars or Pandas).
    - Embeds Equity, Drawdown, and Weights heatmap
    - Includes a simple metrics table
    """
    # Extract essentials
    dates = bt.get("dates", [])
    equity = np.asarray(bt.get("equity", []), dtype=float)
    tickers = bt.get("tickers", [])
    W = np.asarray(bt.get("weights", []), dtype=float)
    rb_dates = bt.get("rebalance_dates")  # may be None

    # Build figures
    fig_eq = fig_equity(dates, equity, title="Equity")
    fig_dd = fig_drawdown(dates, equity, title="Drawdown")

    # Weights heatmap expects (T_w, N) + dates_w of same length
    if W.size > 0:
        if rb_dates is None or len(rb_dates) != W.shape[0]:
            # Fallback: trim daily dates to number of rebalance steps
            dates_w = dates[: W.shape[0]]
        else:
            dates_w = rb_dates
        fig_w = fig_weights_heatmap(dates_w, tickers, W, title="Weights (rebalance steps)")
        w_html = fig_w.to_html(full_html=False, include_plotlyjs="cdn")
    else:
        w_html = "<p>No weights available.</p>"

    # Metrics table
    metrics_html = _to_html_table(
        df_metrics.to_pandas() if hasattr(df_metrics, "to_pandas") else df_metrics
    )

    # Compose HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Backtest Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
h1, h2 {{ margin: 0.2rem 0 0.6rem; }}
.section {{ margin: 24px 0; }}
.card {{ background: #fff; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); padding: 16px; }}
.dataframe th, .dataframe td {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
</style>
</head>
<body>
  <h1>Backtest Report</h1>

  <div class="section card">
    <h2>Metrics</h2>
    {metrics_html}
  </div>

  <div class="section card">
    <h2>Equity</h2>
    {fig_eq.to_html(full_html=False, include_plotlyjs="cdn")}
  </div>

  <div class="section card">
    <h2>Drawdown</h2>
    {fig_dd.to_html(full_html=False, include_plotlyjs=False)}
  </div>

  <div class="section card">
    <h2>Weights</h2>
    {w_html}
  </div>
</body>
</html>
"""
    return html
