#!/usr/bin/env python3
"""
Golden Dataset Export v2: QQQ/VOO/BIL +20 Years
===============================================

Genera Excel golden con dos modos:
- common_window: desde inception del más joven (VOO 2010) → limpio para ML
- max_history: máximo histórico por ticker → para research de régimen

REGLAS:
- VOO inception: 2010-09-09
- BIL inception: 2007-05-30
- QQQ inception: 1999-03-10

Usage:
    # Common window (2010-hoy, los 3 activos completos)
    python scripts/export_golden_dataset.py --mode common_window

    # Max history (máximo por ticker, panel irregular)
    python scripts/export_golden_dataset.py --mode max_history

    # Con Massive
    export MASSIVE_API_KEY=your_key
    python scripts/export_golden_dataset.py --provider massive --mode common_window
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Ticker Inception Dates (source of truth)
# =============================================================================

TICKER_INCEPTION = {
    "QQQ": date(1999, 3, 10),
    "VOO": date(2010, 9, 9),
    "BIL": date(2007, 5, 30),
    "SPY": date(1993, 1, 29),
    "IWM": date(2000, 5, 26),
}


def get_common_start(tickers: list[str]) -> date:
    """Retorna la fecha más tardía de inception (common window start)."""
    inceptions = [TICKER_INCEPTION.get(t, date(2000, 1, 1)) for t in tickers]
    return max(inceptions)


def get_max_start(tickers: list[str]) -> date:
    """Retorna la fecha más temprana de inception (max history start)."""
    inceptions = [TICKER_INCEPTION.get(t, date(2000, 1, 1)) for t in tickers]
    return min(inceptions)


# =============================================================================
# Data Download
# =============================================================================


def download_yahoo(tickers: list[str], start_date: date, end_date: date) -> pl.DataFrame:
    """Descarga desde Yahoo Finance."""
    import pandas as pd
    import yfinance as yf

    all_dfs = []
    for ticker in tickers:
        # Usar inception date si start_date es anterior
        ticker_start = max(start_date, TICKER_INCEPTION.get(ticker, start_date))
        logger.info(f"  {ticker}: {ticker_start} → {end_date}")

        df = yf.download(
            ticker, start=ticker_start, end=end_date, progress=False, auto_adjust=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.reset_index()
        df["ticker"] = ticker
        all_dfs.append(df)

    df_combined = pd.concat(all_dfs, ignore_index=True)
    df = pl.from_pandas(df_combined)

    # Normalize columns
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower == "date":
            col_map[col] = "date"
        elif col_lower == "open":
            col_map[col] = "open"
        elif col_lower == "high":
            col_map[col] = "high"
        elif col_lower == "low":
            col_map[col] = "low"
        elif col_lower == "close":
            col_map[col] = "close"
        elif col_lower == "volume":
            col_map[col] = "volume"
        elif col_lower == "ticker":
            col_map[col] = "ticker"
        elif col_lower in ["adj close", "adj_close"]:
            col_map[col] = "adj_close"

    df = df.rename(col_map)

    if "date" in df.columns and df["date"].dtype == pl.Datetime:
        df = df.with_columns(pl.col("date").dt.date())

    return df


# =============================================================================
# Metadata Builder (audit-ready)
# =============================================================================


def build_professional_metadata(
    df: pl.DataFrame,
    config: dict,
    mode: str,
    content_hash: str,
) -> dict:
    """Construye metadata profesional nivel 'dataset card'."""
    from datetime import datetime

    # Actual range por ticker
    actual_start_by_ticker = {}
    actual_end_by_ticker = {}
    rows_by_ticker = {}

    for ticker in config["tickers"]:
        df_ticker = df.filter(pl.col("ticker") == ticker)
        if df_ticker.height > 0:
            actual_start_by_ticker[ticker] = str(df_ticker["date"].min())
            actual_end_by_ticker[ticker] = str(df_ticker["date"].max())
            rows_by_ticker[ticker] = df_ticker.height

    # Common window
    if actual_start_by_ticker:
        common_start = max(actual_start_by_ticker.values())
        common_end = min(actual_end_by_ticker.values())
    else:
        common_start = common_end = None

    return {
        # Identity
        "dataset_id": config.get("dataset_id"),
        "description": config.get("description"),
        "mode": mode,
        # Content
        "content_hash": content_hash,
        "n_rows": df.height,
        "n_columns": len(df.columns),
        "columns": df.columns,
        # Time Range
        "requested_start": config.get("requested_start"),
        "requested_end": config.get("requested_end"),
        "actual_start_by_ticker": actual_start_by_ticker,
        "actual_end_by_ticker": actual_end_by_ticker,
        "common_start": common_start,
        "common_end": common_end,
        # Coverage
        "tickers": config["tickers"],
        "ticker_inception_dates": {
            t: str(TICKER_INCEPTION.get(t, "unknown")) for t in config["tickers"]
        },
        "rows_by_ticker": rows_by_ticker,
        # Features
        "features": config.get("features", []),
        # Versions
        "provider": config.get("provider"),
        "adjustment_version": config.get("adjustment_version", "2.0.0"),
        "feature_set_version": config.get("feature_set_version", "1.0.0"),
        "calendar_id": config.get("calendar_id", "NYSE"),
        # Audit
        "created_at": datetime.utcnow().isoformat() + "Z",
        "gammaedge_version": "1.0.0",
        "export_script": "scripts/export_golden_dataset.py",
        # Warnings
        "warnings": [],
    }


# =============================================================================
# Main Export
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Export Golden Dataset v2")
    parser.add_argument(
        "--provider", choices=["yahoo"], default="yahoo", help="Data provider (Only Yahoo enabled)"
    )
    parser.add_argument(
        "--mode",
        choices=["common_window", "max_history"],
        default="common_window",
        help="Export mode: common_window (desde VOO 2010) o max_history (máximo por ticker)",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=["QQQ", "VOO", "BIL"], help="Tickers to include"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). If not set, uses mode default.",
    )
    parser.add_argument(
        "--end", type=str, default=None, help="End date (YYYY-MM-DD). Default: today"
    )
    parser.add_argument("--output", default=None, help="Output Excel path")
    args = parser.parse_args()

    tickers = args.tickers
    end_date = date.fromisoformat(args.end) if args.end else date.today()

    # Determine start date based on mode
    if args.start:
        start_date = date.fromisoformat(args.start)
    elif args.mode == "common_window":
        start_date = get_common_start(tickers)
    else:  # max_history
        start_date = get_max_start(tickers)

    # Default output filename
    if args.output:
        output_path = args.output
    else:
        ticker_str = "_".join(t.lower() for t in tickers)
        output_path = f"golden_dataset_{args.mode}_{ticker_str}.xlsx"

    logger.info("=" * 60)
    logger.info("GOLDEN DATASET EXPORT v2")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Provider: {args.provider}")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Requested: {start_date} → {end_date}")
    logger.info(f"Output: {output_path}")
    logger.info("")

    # =========================================================================
    # Download
    # =========================================================================
    logger.info("📥 Downloading data...")

    if args.provider == "massive":
        raise ValueError("Massive provider has been removed. Use yahoo.")
    else:
        df_raw = download_yahoo(tickers, start_date, end_date)

    logger.info(f"  Total: {df_raw.height} rows")
    logger.info("")

    # =========================================================================
    # Add instrument_id
    # =========================================================================
    df_raw = df_raw.with_columns(pl.col("ticker").alias("instrument_id"))

    # =========================================================================
    # Build Features
    # =========================================================================
    logger.info("🔧 Building features...")

    from portfolio.trading.v1_features import V1FeatureBuilder

    builder = V1FeatureBuilder()
    df_features = builder.build(df_raw)

    logger.info(f"  {len(df_features.columns)} features")
    logger.info("")

    # =========================================================================
    # Merge
    # =========================================================================
    df_combined = df_raw.join(df_features, on="date", how="inner")

    # =========================================================================
    # Filter by mode
    # =========================================================================
    if args.mode == "common_window":
        # Solo mantener filas donde todos los tickers tienen datos
        common_start = get_common_start(tickers)
        df_combined = df_combined.filter(pl.col("date") >= common_start)
        logger.info(f"  Filtered to common window: {common_start} onwards")
        logger.info(f"  Rows after filter: {df_combined.height}")

    # =========================================================================
    # Export
    # =========================================================================
    logger.info("")
    logger.info("📊 Exporting Excel...")

    import hashlib

    from portfolio.io.export import GoldenDatasetConfig, GoldenExcelExporter

    # Compute hash
    buffer = df_combined.write_csv().encode("utf-8")
    content_hash = hashlib.sha256(buffer).hexdigest()[:16]

    config = GoldenDatasetConfig(
        dataset_id=f"{args.mode}_{args.provider}_{'_'.join(t.lower() for t in tickers)}",
        description=f"{'/'.join(tickers)} dataset - {args.mode} mode",
        tickers=tickers,
        provider=args.provider,
        features=[c for c in df_features.columns if c != "date"],
    )

    # Build professional metadata
    config_dict = {
        "dataset_id": config.dataset_id,
        "description": config.description,
        "tickers": tickers,
        "provider": args.provider,
        "requested_start": str(start_date),
        "requested_end": str(end_date),
        "features": config.features,
        "adjustment_version": "2.0.0",
        "feature_set_version": "1.0.0",
        "calendar_id": "NYSE",
    }

    metadata = build_professional_metadata(
        df_combined,
        config_dict,
        args.mode,
        content_hash,
    )

    # Add warnings for max_history mode
    if args.mode == "max_history":
        metadata["warnings"].append(
            "max_history mode: Panel is IRREGULAR. Some tickers have missing data before their inception."
        )
        for ticker in tickers:
            inception = TICKER_INCEPTION.get(ticker)
            if inception and inception > date(2000, 1, 1):
                metadata["warnings"].append(
                    f"{ticker} inception: {inception}. No data before this date."
                )

    # Export
    exporter = GoldenExcelExporter(config)
    exporter._content_hash = content_hash
    exporter.export(df_combined, output_path=output_path)

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ EXPORT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"File: {output_path}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Content hash: {content_hash}")
    logger.info("")
    logger.info("Coverage by ticker:")
    for ticker, start in metadata["actual_start_by_ticker"].items():
        end = metadata["actual_end_by_ticker"].get(ticker)
        rows = metadata["rows_by_ticker"].get(ticker, 0)
        logger.info(f"  {ticker}: {start} → {end} ({rows} rows)")

    logger.info("")
    logger.info(f"Common window: {metadata['common_start']} → {metadata['common_end']}")
    logger.info(f"Total rows: {metadata['n_rows']}")
    logger.info(f"Total columns: {metadata['n_columns']}")

    if metadata["warnings"]:
        logger.info("")
        logger.info("⚠️  Warnings:")
        for w in metadata["warnings"]:
            logger.info(f"  - {w}")


if __name__ == "__main__":
    main()
