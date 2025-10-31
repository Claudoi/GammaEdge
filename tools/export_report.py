# tools/export_report.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Añade repo root para imports locales
sys.path.append(str(Path(__file__).resolve().parents[1]))

from portfolio.backtest import attribution as bt_attr
from portfolio.backtest.kpis import compute_kpis
from portfolio.backtest.reporting import (
    BacktestReport,
    ReportFigure,
    build_backtest_report,
    build_context,
    render_html,
    render_pdf,
)


def _read_returns_wide(path: str) -> pl.DataFrame:
    """Lee returns wide con columna 'date'. Soporta .csv y .parquet."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".csv":
        df = pl.read_csv(path, try_parse_dates=True)
    elif ext in {".parquet", ".pq"}:
        df = pl.read_parquet(path)
    else:
        raise ValueError(f"Unsupported returns file extension: {ext}")
    if "date" not in df.columns:
        raise ValueError("returns_wide must contain a 'date' column")
    # normaliza a Datetime
    if df.schema.get("date") != pl.Datetime:
        df = df.with_columns(pl.col("date").cast(pl.Datetime, strict=False))
    return df


def _read_bt_json(path: str) -> dict[str, Any]:
    """
    JSON esperado:
      - dates: list[str] o list[date-like]
      - equity: list[float]
      - tickers: list[str]
      - weights: list[list[float]] (K×N en rebalances)
      - rebalance_dates: optional list[str] (K)
    """
    with open(path, encoding="utf-8") as f:
        bt = json.load(f)
    for k in ["dates", "equity", "tickers", "weights"]:
        if k not in bt:
            raise ValueError(f"bt JSON is missing required key: {k}")
    return bt


def _read_group_map(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    return {str(k): str(v) for k, v in obj.items()}


def _read_metrics_csv(path: str | None) -> dict[str, float] | None:
    """CSV de dos columnas <metric, value> -> dict (best effort)."""
    if not path:
        return None
    df = pl.read_csv(path)
    if df.width < 2 or df.height == 0:
        return None
    kcol, vcol = df.columns[:2]
    keys = df.get_column(kcol).cast(pl.Utf8, strict=False).to_list()
    vals_any = df.get_column(vcol).to_list()
    out: dict[str, float] = {}
    for k, v in zip(keys, vals_any, strict=False):
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out or None


def _ensure_numpy(x: Any, dtype=float) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def export_report(
    *,
    bt_path: str,
    returns_path: str,
    out_dir: str,
    group_map_path: str | None = None,
    metrics_path: str | None = None,
    portfolio_name: str = "Portfolio",
    page_title: str = "GammaEdge Report",
    h1_title: str = "Backtest Report",
    rf_daily: float = 0.0,
    periods_per_year: int = 252,
    no_pdf: bool = False,
) -> None:
    # 1) Carga insumos
    bt = _read_bt_json(bt_path)
    df_ret_wide = _read_returns_wide(returns_path)
    group_map = _read_group_map(group_map_path)
    extra_metrics = _read_metrics_csv(metrics_path)

    # 2) Extrae BT básicos
    dates_any = list(bt.get("dates", []))
    equity = _ensure_numpy(bt.get("equity", []), dtype=float)
    tickers = list(bt.get("tickers", []))
    W_reb = _ensure_numpy(bt.get("weights", []), dtype=float)  # K×N (rebalances)
    rb_dates = list(bt.get("rebalance_dates", []))

    if equity.size == 0 or W_reb.size == 0 or not dates_any or not tickers:
        raise ValueError("bt JSON must contain non-empty dates, equity, weights and tickers")

    # 3) Parse robusto de fechas del JSON a Datetime (evita mismatch str vs Datetime)
    s_dates = pl.Series(dates_any).cast(pl.Utf8, strict=False)
    try:
        # Polars recientes: usa 'format'
        dates_bt = s_dates.str.strptime(pl.Datetime, format=None, strict=False).to_list()
    except TypeError:
        # Polars anteriores: firma sin 'format'
        dates_bt = s_dates.str.strptime(pl.Datetime, strict=False).to_list()

    # 4) Alinea returns al grid del backtest
    df_ret_bt = (
        df_ret_wide.filter(pl.col("date").is_in(dates_bt)).unique(subset=["date"]).sort("date")
    )
    dates_bt = df_ret_bt.get_column("date").to_list()  # final, por si se redujo con unique
    T = len(dates_bt)
    N = len(tickers)

    # 5) Coherencia de W_reb -> expande a diario
    if W_reb.ndim != 2:
        raise ValueError("weights must be a 2D array (K×N) of rebalance weights")
    if W_reb.shape[1] != N:
        raise ValueError(f"weights second dim must match tickers (got {W_reb.shape[1]} vs {N})")

    K = W_reb.shape[0]
    # si no hay rb_dates o no coinciden con K, infiere K fechas equiespaciadas sobre dates_bt
    if len(rb_dates) != K:
        step = max(T // max(K, 1), 1)
        rb_dates = [dates_bt[i] for i in range(0, T, step)][:K]

    daily_W = bt_attr.expand_rebalance_weights(
        dates=dates_bt,
        rb_dates=rb_dates,
        W_reb=W_reb,
    )

    # 6) Construye BacktestReport
    report: BacktestReport = build_backtest_report(
        df_ret_wide=df_ret_bt,
        daily_weights=daily_W,
        equity=equity,
        group_map=group_map,
        title=h1_title,
    )

    # 7) KPIs (fallback si no se pasan)
    if extra_metrics is None:
        extra_metrics = compute_kpis(equity, rf_daily=rf_daily, periods_per_year=periods_per_year)

    # 8) Contexto para export
    period_start = str(dates_bt[0]) if dates_bt else "—"
    period_end = str(dates_bt[-1]) if dates_bt else "—"

    df_asset_total = report.tables.get("contrib_asset_total")
    df_group_total = report.tables.get("contrib_group_total")
    df_brinson = None  # opcional: inyectable en el futuro

    ctx = build_context(
        portfolio_name=portfolio_name,
        period=(period_start, period_end),
        df_asset_cum=df_asset_total,
        df_group_total=df_group_total,
        df_brinson=df_brinson,
        extra_metrics=extra_metrics,
    )

    # 9) Export
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    figs = [
        ReportFigure("Equity", report.figures["equity"]),
        ReportFigure("Drawdown", report.figures["drawdown"]),
        ReportFigure("Weights", report.figures["weights"]),
        ReportFigure("Top Contributors", report.figures["top_contrib"]),
    ]

    html_bytes = render_html(ctx, figs, page_title=page_title, h1=h1_title)
    (out_dir_p / "backtest_report.html").write_bytes(html_bytes)
    print(f"[export_report] HTML written: {out_dir_p / 'backtest_report.html'}")

    if not no_pdf:
        try:
            pdf_bytes = render_pdf(ctx, figs, title=page_title)
            (out_dir_p / "backtest_report.pdf").write_bytes(pdf_bytes)
            print(f"[export_report] PDF written:  {out_dir_p / 'backtest_report.pdf'}")
        except Exception as e:
            print(f"[export_report] PDF export skipped: {e}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="export_report",
        description="Generate HTML/PDF backtest report from saved results.",
    )
    p.add_argument("--bt", dest="bt_path", required=True, help="Path to bt.json")
    p.add_argument(
        "--returns", dest="returns_path", required=True, help="Path to returns_wide (.csv/.parquet)"
    )
    p.add_argument("--out", dest="out_dir", required=True, help="Output directory")
    p.add_argument("--groups", dest="group_map_path", default=None, help="Optional group_map.json")
    p.add_argument(
        "--metrics", dest="metrics_path", default=None, help="Optional metrics.csv (metric,value)"
    )
    p.add_argument("--portfolio-name", dest="portfolio_name", default="Portfolio")
    p.add_argument("--page-title", dest="page_title", default="GammaEdge Report")
    p.add_argument("--h1", dest="h1_title", default="Backtest Report")
    p.add_argument("--rf-daily", dest="rf_daily", type=float, default=0.0)
    p.add_argument("--ppy", dest="periods_per_year", type=int, default=252)
    p.add_argument("--no-pdf", dest="no_pdf", action="store_true", help="Skip PDF export")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    export_report(
        bt_path=args.bt_path,
        returns_path=args.returns_path,
        out_dir=args.out_dir,
        group_map_path=args.group_map_path,
        metrics_path=args.metrics_path,
        portfolio_name=args.portfolio_name,
        page_title=args.page_title,
        h1_title=args.h1_title,
        rf_daily=args.rf_daily,
        periods_per_year=args.periods_per_year,
        no_pdf=args.no_pdf,
    )


if __name__ == "__main__":
    main()
