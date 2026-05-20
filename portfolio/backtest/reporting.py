# portfolio/backtest/reporting.py
from __future__ import annotations

import base64
import io
import os
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from typing import Any, Literal, cast

import numpy as np
import plotly.graph_objects as go
import polars as pl

from portfolio.backtest.attribution_reporting import (
    BrinsonReport,
    build_brinson_attribution_report,
)
from portfolio.core.compat import dataclass_compat as dataclass

from .attribution import (
    DailyAlignment,
    align_returns_and_weights,
    contributions_by_asset,
    contributions_by_group,
    expand_rebalance_weights,
)

# ──────────────────────────────────────────────────────────────────────────────
# Optional soft dependencies for PDF
# ──────────────────────────────────────────────────────────────────────────────
try:  # pragma: no cover - optional
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
except Exception:  # pragma: no cover
    A4 = None
    canvas = None
    ImageReader = None

# ──────────────────────────────────────────────────────────────────────────────
# Plotly figures (robust versions)
# ──────────────────────────────────────────────────────────────────────────────


def _ensure_same_length(
    x: Iterable[Any],
    y: np.ndarray,
    name_x: str = "x",
    name_y: str = "y",
) -> None:
    """Check that x and y have the same length."""
    x_len = len(list(x)) if not isinstance(x, list) else len(x)
    if x_len != len(y):
        raise ValueError(f"{name_x} and {name_y} must have the same length ({x_len} vs {len(y)})")


def fig_equity(
    dates: Sequence[Any],
    equity: np.ndarray,
    title: str = "Equity Curve",
) -> go.Figure:
    """Equity line chart with NaN-safe handling."""
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
    """Drawdown chart from equity series (as ratio)."""
    eq = np.asarray(equity, dtype=float)
    _ensure_same_length(dates, eq, "dates", "equity")
    eq = np.where(np.isfinite(eq), eq, np.nan)

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
    """Heatmap of weights (T, N). Dates on x, tickers on y, ordered by avg weight desc."""
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
    W_ord = W[:, order]

    fig = go.Figure(
        data=go.Heatmap(
            z=W_ord.T,
            x=list(dates),
            y=tickers_ord,
            coloraxis="coloraxis",
            zmin=0.0,
            zmax=1.0,
            hovertemplate="Ticker: %{y}<br>Date: %{x}<br>Weight: %{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(title=title, coloraxis_colorscale="Blues", template="plotly_white")
    return fig


def fig_bar_contrib(df_top: pl.DataFrame, title: str = "Top Contributors") -> go.Figure:
    """
    Bar plot for top contributors.
    Expects DataFrame with columns ['ticker','contrib_total'].
    """
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
# Backtest Report container (for Streamlit pages)
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
    Aggregate top contributors (+/-) from daily asset contributions.
    Requires columns: ['ticker','contrib'].
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
    pos = agg.filter(pl.col("contrib_total") > 0).head(k)
    neg = agg.filter(pl.col("contrib_total") < 0).sort("contrib_total", descending=False).head(k)
    return pl.concat([pos, neg])


def build_backtest_report(
    df_ret_wide: pl.DataFrame,  # ['date', tickers...]
    daily_weights: np.ndarray,  # (T, N)
    equity: np.ndarray,  # (T,)
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

    # Top contributors (local aggregator to avoid signature mismatches)
    df_top_both = _top_contributors_from_df(df_contrib_asset, k=10, sign="both")
    tables["top_contributors"] = df_top_both

    # Figures
    figures: dict[str, go.Figure] = {
        "equity": fig_equity(dates, equity, title=f"{title} — Equity"),
        "drawdown": fig_drawdown(dates, equity, title=f"{title} — Drawdown"),
        "weights": fig_weights_heatmap(dates, tickers, W, title=f"{title} — Weights"),
        "top_contrib": fig_bar_contrib(df_top_both, title=f"{title} — Top Contributors"),
    }

    meta = {"title": title, "n_days": len(dates), "n_assets": len(tickers)}
    return BacktestReport(tables=tables, figures=figures, meta=meta)


def build_brinson_attribution_section(
    timeseries: Any,
    how: Literal["sum", "mean"] = "sum",
) -> BrinsonReport:
    """
    Build a Brinson-style attribution section for a backtest report.

    This is a thin wrapper over ``build_brinson_attribution_report`` to keep
    a consistent API within :mod:`portfolio.backtest.reporting`.
    """
    return build_brinson_attribution_report(timeseries, how=how)


# ──────────────────────────────────────────────────────────────────────────────
# Minimal HTML report (legacy helper, kept for compatibility)
# ──────────────────────────────────────────────────────────────────────────────


def _to_html_table(df_any: Any) -> str:
    """Convert Polars/Pandas DataFrame to a compact HTML table string."""
    pdf = df_any.to_pandas() if hasattr(df_any, "to_pandas") else df_any
    html = cast(str, pdf.to_html(index=False, border=0))
    return html.replace('class="dataframe"', 'class="dataframe" style="font-size:12px"')


def render_html_report(
    bt: Mapping[str, Any] | dict[str, Any],
    df_metrics: pl.DataFrame | Any,
) -> str:
    """
    Minimal HTML report builder from the `bt` dict produced by the engine
    and a metrics table (Polars or Pandas).
    """
    dates = bt.get("dates", [])
    equity = np.asarray(bt.get("equity", []), dtype=float)
    tickers = bt.get("tickers", [])
    W = np.asarray(bt.get("weights", []), dtype=float)
    rb_dates = bt.get("rebalance_dates")

    fig_eq = fig_equity(dates, equity, title="Equity")
    fig_dd = fig_drawdown(dates, equity, title="Drawdown")

    if W.size > 0:
        if rb_dates is None or len(rb_dates) != W.shape[0]:
            dates_w = dates[: W.shape[0]]
        else:
            dates_w = rb_dates
        fig_w = fig_weights_heatmap(dates_w, tickers, W, title="Weights (rebalance steps)")
        w_html = fig_w.to_html(full_html=False, include_plotlyjs="cdn")
    else:
        w_html = "<p>No weights available.</p>"

    metrics_html = _to_html_table(
        df_metrics.to_pandas() if hasattr(df_metrics, "to_pandas") else df_metrics
    )

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


# ──────────────────────────────────────────────────────────────────────────────
# New: Rich reporting (HTML with inline images + PDF via reportlab)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ReportFigure:
    """Container for exporting figures in consistent size."""

    title: str
    fig: go.Figure
    width: int = 1200
    height: int = 700


def _kaleido_available() -> bool:
    """Check if kaleido can render (requires Chrome on newer versions)."""
    try:
        import kaleido  # noqa: F401

        # Quick smoke-test: create a tiny figure and try to render it
        _test_fig = go.Figure(go.Scatter(x=[0], y=[0]))
        _test_fig.to_image(format="png", width=50, height=50, engine="kaleido")
        return True
    except Exception:
        return False


def _fig_to_png_bytes(fig: go.Figure, width: int = 1100, height: int = 600) -> bytes:
    """Render a Plotly figure to PNG bytes using kaleido (returns bytes).

    Raises RuntimeError if kaleido/Chrome is not available.
    """
    try:
        png_any = fig.to_image(format="png", width=width, height=height, engine="kaleido")
    except Exception as exc:
        raise RuntimeError(
            "Cannot render figures to PNG (kaleido requires Chrome). "
            "HTML export will use interactive charts instead."
        ) from exc
    if isinstance(png_any, bytes):
        return png_any
    if isinstance(png_any, bytearray):
        return bytes(png_any)
    if isinstance(png_any, memoryview):
        return png_any.tobytes()
    return cast(bytes, png_any)


def _png_bytes_to_b64_img_src(png: bytes) -> str:
    """Return a data URI suitable for <img src='...'>."""
    b64 = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _ensure_datetime(df: pl.DataFrame, col: str = "date") -> pl.DataFrame:
    """Force a column to Datetime if possible; otherwise return df unchanged."""
    dt = df.schema.get(col)
    if dt in (pl.Datetime, pl.Date):
        return df
    try:
        return df.with_columns(pl.col(col).cast(pl.Datetime, strict=False))
    except Exception:
        return df


def _safe_metrics_to_dict(metrics_df_obj: Any) -> dict[str, Any] | None:
    """
    Best effort conversion of a metrics table (Polars/Pandas) to a simple
    dict[name -> value] using the first two columns.
    """
    try:
        pdf = metrics_df_obj.to_pandas() if hasattr(metrics_df_obj, "to_pandas") else metrics_df_obj
        return {str(k): float(v) for k, v in zip(pdf.iloc[:, 0], pdf.iloc[:, 1], strict=False)}
    except Exception:
        return None


def build_context(
    *,
    portfolio_name: str,
    period: tuple[str, str],
    df_asset_cum: pl.DataFrame | None,
    df_group_total: pl.DataFrame | None,
    df_brinson: pl.DataFrame | None,
    extra_metrics: dict[str, Any] | None = None,
    euler_last_day: list[dict[str, Any]] | None = None,
    factor_rc: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build a serialisable context for the report.
    Keeps small, meaningful tables to avoid huge PDFs.
    """
    ctx: dict[str, Any] = {
        "portfolio_name": portfolio_name,
        "period_start": period[0],
        "period_end": period[1],
    }
    if extra_metrics:
        ctx["metrics"] = extra_metrics

    if df_asset_cum is not None:
        df_asset_cum = _ensure_datetime(df_asset_cum, "date")
        if "contrib_total" in df_asset_cum.columns:
            top = (
                df_asset_cum.sort("contrib_total", descending=True)
                .select(["ticker", "contrib_total"])
                .head(10)
            )
        else:
            top = (
                df_asset_cum.group_by("ticker")
                .agg(pl.col("contrib").sum().alias("contrib_total"))
                .sort("contrib_total", descending=True)
                .head(10)
            )
        ctx["top_contributors"] = top.to_dicts()

    if df_group_total is not None:
        keep = [
            c
            for c in ["group", "group_id", "contrib_total", "weight_avg", "avg_weight"]
            if c in df_group_total.columns
        ]
        ctx["group_totals"] = df_group_total.select(keep).to_dicts()

    if df_brinson is not None:
        df_brinson = _ensure_datetime(df_brinson, "date")
        tail = (
            df_brinson.tail(1)
            .select(
                [c for c in ["alloc", "select", "interact", "total"] if c in df_brinson.columns]
            )
            .to_dicts()
        )
        ctx["brinson_last"] = tail[0] if tail else {}

    if euler_last_day:
        ctx["euler_last_day"] = euler_last_day

    if factor_rc:
        ctx["factor_rc"] = factor_rc

    return ctx


_HTML_SHELL = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
    h1 {{ margin: 0 0 8px 0; }}
    h2 {{ margin-top: 28px; border-bottom: 1px solid #eee; padding-bottom: 4px; }}
    h3 {{ margin-top: 20px; margin-bottom: 6px; }}
    .meta {{ color: #555; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f7f7f7; }}
    .imgwrap {{ margin: 14px 0; border: 1px solid #eee; padding: 8px; border-radius: 8px; }}
    .caption {{ font-size: 13px; color: #666; margin-top: 6px; }}
    .kpi {{ display: inline-block; margin-right: 24px; }}
    .footer {{ margin-top: 28px; font-size: 12px; color: #777; }}
  </style>
</head>
<body>
  <h1>{h1}</h1>
  <div class="meta">Portfolio: <b>{portfolio_name}</b> &middot; Period: <b>{period_start}</b> → <b>{period_end}</b></div>

  {metrics_block}

  {tables_block}

  <h2>Figures</h2>
  {figures_block}

  <div class="footer">Generated by GammaEdge Reporting</div>
</body>
</html>
"""


def _render_metrics_block(ctx: dict[str, Any]) -> str:
    metrics = ctx.get("metrics")
    if not metrics:
        return ""
    items = []
    for k, v in metrics.items():
        items.append(f'<div class="kpi"><b>{k}</b><br/>{v}</div>')
    return "<h2>Key Metrics</h2>\n" + "\n".join(items)


def _render_table_dicts(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{row.get(c, '')}</td>" for c in cols) + "</tr>" for row in rows
    )
    return f"<h2>{title}</h2>\n<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _render_tables_block(ctx: dict[str, Any]) -> str:
    blocks: list[str] = []
    if ctx.get("top_contributors"):
        blocks.append(_render_table_dicts("Top Contributors", ctx["top_contributors"]))
    if ctx.get("group_totals"):
        blocks.append(_render_table_dicts("Group Totals", ctx["group_totals"]))
    if ctx.get("brinson_last"):
        blocks.append(_render_table_dicts("Brinson (last cumulative)", [ctx["brinson_last"]]))
    if ctx.get("euler_last_day"):
        blocks.append(_render_table_dicts("Euler RC (last day)", ctx["euler_last_day"]))
    if ctx.get("factor_rc"):
        blocks.append(_render_table_dicts("Factor RC (PCA)", ctx["factor_rc"]))
    return "\n".join(blocks)


# ──────────────────────────────────────────────────────────────────────────────
# Figure sectioning helpers (Performance / Brinson / Euler / Factors)
# ──────────────────────────────────────────────────────────────────────────────

_SECTION_ORDER: list[str] = [
    "Performance Overview",
    "Performance Attribution (Brinson)",
    "Risk Attribution (Euler)",
    "Factor Decomposition (PCA)",
    "Other Figures",
]


def _classify_figure_section(title: str) -> str:
    lt = title.lower()

    if any(k in lt for k in ("equity", "drawdown", "weight", "top contrib", "top contributor")):
        return "Performance Overview"

    if "brinson" in lt:
        return "Performance Attribution (Brinson)"

    if "euler" in lt:
        return "Risk Attribution (Euler)"

    if any(k in lt for k in ("factor", "pca")):
        return "Factor Decomposition (PCA)"

    return "Other Figures"


def _render_figures_block(fig_srcs: list[tuple[str, str]]) -> str:
    """
    Render figures grouped by logical sections.

    fig_srcs: list of (title, img_src_data_uri).
    """
    if not fig_srcs:
        return "<p>No figures.</p>"

    # Group by section name, preserving input order
    grouped: dict[str, list[tuple[str, str]]] = {name: [] for name in _SECTION_ORDER}
    for title, src in fig_srcs:
        section = _classify_figure_section(title)
        if section not in grouped:
            grouped[section] = []
        grouped[section].append((title, src))

    parts: list[str] = []
    for section in _SECTION_ORDER:
        figs = grouped.get(section) or []
        if not figs:
            continue
        parts.append(f"<h3>{section}</h3>")
        for title, src in figs:
            parts.append(
                f'<div class="imgwrap"><img alt="{title}" src="{src}" '
                f'style="max-width:100%;height:auto;"/>'
                f'<div class="caption">{title}</div></div>'
            )

    return "\n".join(parts) if parts else "<p>No figures.</p>"


def _render_figures_block_interactive(
    figures: list[ReportFigure],
) -> str:
    """
    Render figures as interactive Plotly charts (no kaleido/Chrome needed).
    Uses Plotly.js CDN for the first figure, then reuses it for subsequent ones.
    Grouped by logical sections.
    """
    if not figures:
        return "<p>No figures.</p>"

    # Group by section name, preserving input order
    grouped: dict[str, list[ReportFigure]] = {name: [] for name in _SECTION_ORDER}
    for rf in figures:
        section = _classify_figure_section(rf.title)
        if section not in grouped:
            grouped[section] = []
        grouped[section].append(rf)

    parts: list[str] = []
    first = True
    for section in _SECTION_ORDER:
        figs = grouped.get(section) or []
        if not figs:
            continue
        parts.append(f"<h3>{section}</h3>")
        for rf in figs:
            # Include plotly.js only with first figure, reuse for rest
            plotly_js = "cdn" if first else False
            first = False
            fig_html = rf.fig.to_html(
                full_html=False,
                include_plotlyjs=plotly_js,
                config={"responsive": True, "displayModeBar": True},
            )
            parts.append(
                f'<div class="imgwrap">'
                f"{fig_html}"
                f'<div class="caption">{rf.title}</div></div>'
            )

    return "\n".join(parts) if parts else "<p>No figures.</p>"


def render_html(
    ctx: dict[str, Any],
    figures: list[ReportFigure],
    *,
    page_title: str = "GammaEdge Report",
    h1: str = "Portfolio Report",
) -> bytes:
    """
    Render a complete HTML report.

    Strategy:
    - If kaleido/Chrome is available: inline static PNG images (self-contained).
    - Otherwise: interactive Plotly charts via CDN (richer experience, needs internet).

    Figures are automatically grouped into logical sections:
    - Performance Overview
    - Performance Attribution (Brinson)
    - Risk Attribution (Euler)
    - Factor Decomposition (PCA)
    - Other Figures
    """
    use_static = _kaleido_available()

    if use_static:
        fig_srcs: list[tuple[str, str]] = []
        for rf in figures:
            png = _fig_to_png_bytes(rf.fig, rf.width, rf.height)
            src = _png_bytes_to_b64_img_src(png)
            fig_srcs.append((rf.title, src))
        figures_block = _render_figures_block(fig_srcs)
    else:
        figures_block = _render_figures_block_interactive(figures)

    html = _HTML_SHELL.format(
        title=page_title,
        h1=h1,
        portfolio_name=ctx.get("portfolio_name", "—"),
        period_start=ctx.get("period_start", "—"),
        period_end=ctx.get("period_end", "—"),
        metrics_block=_render_metrics_block(ctx),
        tables_block=_render_tables_block(ctx),
        figures_block=figures_block,
    )
    return html.encode("utf-8")


def render_pdf(
    ctx: dict[str, Any],
    figures: list[ReportFigure],
    *,
    title: str = "GammaEdge Report",
) -> bytes:
    """
    Render a PDF using reportlab.

    Strategy:
    - If kaleido/Chrome is available: embed static PNG figures.
    - Otherwise: generate a metrics-only PDF (tables + metadata, no charts).
    - If reportlab is missing: raises RuntimeError.
    """
    if canvas is None or A4 is None or ImageReader is None:
        raise RuntimeError("reportlab is not installed; please `pip install reportlab`")

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w_page, h_page = A4

    margin = 36  # points
    max_w = w_page - 2 * margin
    max_h = h_page - 2 * margin - 40  # leave space for header

    # First page header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, h_page - margin, title)
    meta = (
        f"Portfolio: {ctx.get('portfolio_name', '—')}   "
        f"Period: {ctx.get('period_start', '—')} → {ctx.get('period_end', '—')}"
    )
    c.setFont("Helvetica", 10)
    c.drawString(margin, h_page - margin - 18, meta)

    # Metrics on first page
    y_cursor = h_page - margin - 50
    metrics = ctx.get("metrics", {})
    if metrics:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y_cursor, "Key Metrics")
        y_cursor -= 18
        c.setFont("Helvetica", 9)
        for k, v in metrics.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            c.drawString(margin + 8, y_cursor, f"{k}: {val_str}")
            y_cursor -= 14
            if y_cursor < margin + 20:
                c.showPage()
                y_cursor = h_page - margin

    # Try to render figures with kaleido
    use_static = _kaleido_available()

    if use_static:
        rendered: list[tuple[str, bytes]] = []
        for rf in figures:
            try:
                png = _fig_to_png_bytes(rf.fig, rf.width, rf.height)
                rendered.append((rf.title, png))
            except Exception:
                continue

        for _i, (fig_title, png_bytes) in enumerate(rendered):
            c.showPage()
            # Page header
            c.setFont("Helvetica-Bold", 16)
            c.drawString(margin, h_page - margin, title)
            c.setFont("Helvetica", 10)
            c.drawString(margin, h_page - margin - 18, meta)

            img = ImageReader(io.BytesIO(png_bytes))
            iw, ih = img.getSize()
            scale = min(max_w / iw, max_h / ih)
            draw_w, draw_h = iw * scale, ih * scale
            x = margin
            y = (h_page - margin - 40) - draw_h
            c.drawImage(
                img,
                x,
                y,
                width=draw_w,
                height=draw_h,
                preserveAspectRatio=True,
                mask="auto",
            )
            c.setFont("Helvetica", 10)
            c.drawString(margin, y - 14, fig_title)
    else:
        # Fallback: note in the PDF that figures need the HTML export
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, h_page - margin, "Figures")
        c.setFont("Helvetica", 10)
        c.drawString(
            margin,
            h_page - margin - 20,
            "Interactive charts are available in the HTML export.",
        )
        c.drawString(
            margin,
            h_page - margin - 36,
            "(Static PNG rendering requires Chrome/kaleido, not available in this environment.)",
        )
        # List figure titles
        y_cursor = h_page - margin - 60
        for rf in figures:
            c.drawString(margin + 8, y_cursor, f"• {rf.title}")
            y_cursor -= 16
            if y_cursor < margin + 20:
                c.showPage()
                y_cursor = h_page - margin

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# High-level helper to export a backtest report without Streamlit
# ──────────────────────────────────────────────────────────────────────────────


def export_backtest_report(
    *,
    bt: Mapping[str, Any],
    df_ret_wide: pl.DataFrame | Any,
    group_map: dict[str, str] | None = None,
    metrics_df: Any | None = None,
    df_brinson: pl.DataFrame | None = None,
    bench_meta: Mapping[str, Any] | None = None,
    portfolio_name: str = "Portfolio",
    output_dir: str | os.PathLike[str] | None = None,
    export_html: bool = True,
    export_pdf: bool = False,
    title: str = "GammaEdge Backtest",
) -> dict[str, bytes]:
    """
    High-level helper to export a backtest report to HTML/PDF without Streamlit.
    """
    # 1) Normalise returns frame
    if not isinstance(df_ret_wide, pl.DataFrame):
        df_ret_wide = pl.from_pandas(df_ret_wide)
    df_ret_wide = _ensure_datetime(df_ret_wide, "date")

    dates_bt_any = list(bt.get("dates", []))
    equity = np.asarray(bt.get("equity", []), dtype=float)
    tickers = list(bt.get("tickers", []))
    W_reb = np.asarray(bt.get("weights", []), dtype=float)
    rb_dates_any = list(bt.get("rebalance_dates", []))

    if not dates_bt_any or equity.size == 0 or W_reb.size == 0 or not tickers:
        raise ValueError("bt must contain non-empty 'dates', 'equity', 'weights' and 'tickers'.")

    dates_bt = list(dates_bt_any)

    # Align returns to backtest calendar
    df_ret_bt = (
        df_ret_wide.filter(pl.col("date").is_in(dates_bt)).unique(subset=["date"]).sort("date")
    )

    # Robust handling of weight dimensions and rebalance dates
    if W_reb.ndim == 2:
        K, N = W_reb.shape
    else:
        K = W_reb.shape[0]
        N = len(tickers)

    if len(tickers) != N:
        raise ValueError(f"weights columns ({N}) do not match tickers ({len(tickers)})")

    if len(rb_dates_any) != K:
        step = max(len(dates_bt) // max(K, 1), 1)
        rb_dates_list = dates_bt[::step][:K]
    else:
        rb_dates_list = rb_dates_any

    # Expand rebalance weights to daily grid
    daily_W = expand_rebalance_weights(
        dates=df_ret_bt.get_column("date").to_list(),
        rb_dates=list(rb_dates_list),
        W_reb=W_reb,
    )

    # 2) Build BacktestReport (tables + core figures)
    report = build_backtest_report(
        df_ret_wide=df_ret_bt,
        daily_weights=daily_W,
        equity=equity,
        group_map=group_map,
        title=title,
    )

    df_asset_total = report.tables.get("contrib_asset_total")
    df_group_total = report.tables.get("contrib_group_total")

    # 3) Normalise Brinson timeseries (if available)
    df_brinson_ctx: pl.DataFrame | None = None
    if isinstance(df_brinson, pl.DataFrame) and "date" in df_brinson.columns:
        df_brinson_ctx = _ensure_datetime(df_brinson, "date")
        df_brinson_ctx = df_brinson_ctx.filter(pl.col("date").is_in(dates_bt)).sort("date")

    # 4) Metrics context
    extra_metrics: dict[str, Any] | None = (
        _safe_metrics_to_dict(metrics_df) if metrics_df is not None else None
    )
    if extra_metrics is None:
        from portfolio.backtest.kpis import compute_kpis

        extra_metrics = compute_kpis(equity, rf_daily=0.0, periods_per_year=252)

    # Append benchmark metadata if present
    if bench_meta and "scheme" in bench_meta:
        scheme_value = bench_meta.get("scheme")
        if scheme_value is not None:
            with suppress(TypeError, ValueError):
                extra_metrics["Benchmark Scheme"] = float(scheme_value)

    period_start = str(dates_bt[0]) if dates_bt else "—"
    period_end = str(dates_bt[-1]) if dates_bt else "—"

    ctx = build_context(
        portfolio_name=portfolio_name,
        period=(period_start, period_end),
        df_asset_cum=df_asset_total,
        df_group_total=df_group_total,
        df_brinson=df_brinson_ctx,
        extra_metrics=extra_metrics,
    )

    # 5) Figures for export (core pack)
    figures: list[ReportFigure] = [
        ReportFigure("Equity", report.figures["equity"]),
        ReportFigure("Drawdown", report.figures["drawdown"]),
        ReportFigure("Weights", report.figures["weights"]),
        ReportFigure("Top Contributors", report.figures["top_contrib"]),
    ]

    # Optional Brinson cumulative figure if data is available and plotting works
    if df_brinson_ctx is not None:
        try:
            from portfolio.viz import plot_utils as viz

            br_cum_fig = viz.plot_brinson_cumulative(
                df_brinson_ctx,
                title="Brinson–Fachler (Cumulative)",
            )
            figures.append(ReportFigure("Brinson (Cumulative)", br_cum_fig))
        except Exception:
            # Keep the rest of the report even if Brinson plotting fails
            pass

    # 6) Export artefacts
    outputs: dict[str, bytes] = {}

    if export_html:
        outputs["html"] = render_html(
            ctx,
            figures,
            page_title="GammaEdge Report",
            h1="Backtest Report",
        )

    if export_pdf:
        outputs["pdf"] = render_pdf(ctx, figures, title="GammaEdge Report")

    # 7) Optional write to disk
    if output_dir is not None and outputs:
        os.makedirs(output_dir, exist_ok=True)
        if "html" in outputs:
            with open(os.path.join(output_dir, "backtest_report.html"), "wb") as f:
                f.write(outputs["html"])
        if "pdf" in outputs:
            with open(os.path.join(output_dir, "backtest_report.pdf"), "wb") as f:
                f.write(outputs["pdf"])

    return outputs
