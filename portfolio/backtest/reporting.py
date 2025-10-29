# portfolio/backtest/reporting.py
from __future__ import annotations

import base64
import io
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


def _fig_to_png_bytes(fig: go.Figure, width: int = 1100, height: int = 600) -> bytes:
    """Render a Plotly figure to PNG bytes using kaleido (returns bytes)."""
    png_any = fig.to_image(format="png", width=width, height=height, engine="kaleido")
    if isinstance(png_any, bytes):
        return png_any
    if isinstance(png_any, bytearray):
        return bytes(png_any)
    if isinstance(png_any, memoryview):
        return png_any.tobytes()
    # Last resort: cast so mypy is satisfied; will error at runtime if incompatible
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


def build_context(
    *,
    portfolio_name: str,
    period: tuple[str, str],
    df_asset_cum: pl.DataFrame | None,
    df_group_total: pl.DataFrame | None,
    df_brinson: pl.DataFrame | None,
    extra_metrics: dict[str, Any] | None = None,
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
    return "\n".join(blocks)


def _render_figures_block(fig_srcs: list[tuple[str, str]]) -> str:
    """fig_srcs: list of (title, img_src_data_uri)."""
    if not fig_srcs:
        return "<p>No figures.</p>"
    parts = []
    for title, src in fig_srcs:
        parts.append(
            f'<div class="imgwrap"><img alt="{title}" src="{src}" style="max-width:100%;height:auto;"/><div class="caption">{title}</div></div>'
        )
    return "\n".join(parts)


def render_html(
    ctx: dict[str, Any],
    figures: list[ReportFigure],
    *,
    page_title: str = "GammaEdge Report",
    h1: str = "Portfolio Report",
) -> bytes:
    """
    Render a complete HTML with inline base64 PNG figures (no external assets).
    """
    fig_srcs: list[tuple[str, str]] = []
    for rf in figures:
        png = _fig_to_png_bytes(rf.fig, rf.width, rf.height)
        src = _png_bytes_to_b64_img_src(png)
        fig_srcs.append((rf.title, src))

    html = _HTML_SHELL.format(
        title=page_title,
        h1=h1,
        portfolio_name=ctx.get("portfolio_name", "—"),
        period_start=ctx.get("period_start", "—"),
        period_end=ctx.get("period_end", "—"),
        metrics_block=_render_metrics_block(ctx),
        tables_block=_render_tables_block(ctx),
        figures_block=_render_figures_block(fig_srcs),
    )
    return html.encode("utf-8")


def render_pdf(
    ctx: dict[str, Any],
    figures: list[ReportFigure],
    *,
    title: str = "GammaEdge Report",
) -> bytes:
    """
    Render a simple PDF using reportlab and the same figures.
    If reportlab is not available, raises a RuntimeError.
    """
    if canvas is None or A4 is None or ImageReader is None:
        raise RuntimeError("reportlab is not installed; please `pip install reportlab`")

    # Render figures first
    rendered: list[tuple[str, bytes]] = []
    for rf in figures:
        png = _fig_to_png_bytes(rf.fig, rf.width, rf.height)
        rendered.append((rf.title, png))

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w_page, h_page = A4

    margin = 36  # points
    max_w = w_page - 2 * margin
    max_h = h_page - 2 * margin - 40  # leave space for header

    # First page header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, h_page - margin, title)
    meta = f"Portfolio: {ctx.get('portfolio_name','—')}   Period: {ctx.get('period_start','—')} → {ctx.get('period_end','—')}"
    c.setFont("Helvetica", 10)
    c.drawString(margin, h_page - margin - 18, meta)

    for i, (fig_title, png_bytes) in enumerate(rendered):
        if i > 0:
            c.showPage()
        img = ImageReader(io.BytesIO(png_bytes))
        iw, ih = img.getSize()
        scale = min(max_w / iw, max_h / ih)
        draw_w, draw_h = iw * scale, ih * scale
        x = margin
        y = (h_page - margin - 40) - draw_h  # below header
        c.drawImage(img, x, y, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
        c.setFont("Helvetica", 10)
        c.drawString(margin, y - 14, fig_title)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()
