# Backtest reporting
# portfolio/backtest/reporting.py
from __future__ import annotations

from portfolio.core.compat import dataclass_compat as dataclass
from typing import Any

import numpy as np
import polars as pl
import plotly.graph_objects as go

from .attribution import (
    DailyAlignment,
    align_returns_and_weights,
    contributions_by_asset,
    contributions_by_group,
    top_contributors,
)


# ──────────────────────────────────────────────────────────────────────────────
# Figuras rápidas (Plotly)
# ──────────────────────────────────────────────────────────────────────────────

def fig_equity(dates: list, equity: np.ndarray, title: str = "Equity Curve") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=equity, mode="lines", name="Equity"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="NAV", template="plotly_white")
    return fig


def fig_drawdown(dates: list, equity: np.ndarray, title: str = "Drawdown") -> go.Figure:
    cummax = np.maximum.accumulate(equity)
    dd = (equity / np.maximum(cummax, 1e-12)) - 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=dd, mode="lines", name="Drawdown"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="DD", template="plotly_white")
    return fig


def fig_weights_heatmap(dates: list, tickers: list[str], W: np.ndarray, title: str = "Weights"):
    # muestra los K nombres con mayor peso medio
    avg_w = W.mean(axis=0)
    order = np.argsort(-avg_w)
    tickers_ord = [tickers[i] for i in order]
    W_ord = W[:, order]

    fig = go.Figure(
        data=go.Heatmap(
            z=W_ord.T,
            x=dates,
            y=tickers_ord,
            coloraxis="coloraxis",
            hovertemplate="Ticker: %{y}<br>Date: %{x}<br>Weight: %{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(title=title, coloraxis_colorscale="Blues", template="plotly_white")
    return fig


def fig_bar_contrib(df_top: pl.DataFrame, title: str = "Top Contributors") -> go.Figure:
    # df_top: ['ticker','contrib_total']
    tick = df_top.get_column("ticker").to_list()
    val = df_top.get_column("contrib_total").to_list()
    fig = go.Figure(go.Bar(x=tick, y=val))
    fig.update_layout(title=title, xaxis_title="Ticker", yaxis_title="Total Contribution", template="plotly_white")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Report Builder
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class BacktestReport:
    """
    Contenedor ligero para pasar a Streamlit:
    - tables: dict de Polars DF
    - figures: dict de Plotly figs
    - meta: dict con metadatos
    """
    tables: dict[str, pl.DataFrame]
    figures: dict[str, go.Figure]
    meta: dict[str, Any]


def build_backtest_report(
    df_ret_wide: pl.DataFrame,             # ['date', tickers...]
    daily_weights: np.ndarray,             # (T, N) pesos diarios
    equity: np.ndarray,                    # (T,) equity
    *,
    group_map: dict[str, str] | None = None,
    title: str = "Backtest Report",
) -> BacktestReport:
    """
    Ensambla tablas y gráficos típicos de reporting: equity, drawdown, heatmap W,
    atribución por activo y por grupo, y top contributors ±.
    """
    aln: DailyAlignment = align_returns_and_weights(df_ret_wide, daily_weights)
    dates = aln.dates
    tickers = aln.tickers
    W = aln.weights
    R = aln.returns

    # Tablas de atribución
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

    # Top contributors
    df_top_both = top_contributors(df_contrib_asset, k=10, sign="both")
    tables["top_contributors"] = df_top_both

    # Figuras
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
