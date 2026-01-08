#!/usr/bin/env python3
"""
Freeze Dataset: Create production-ready frozen dataset
======================================================

Creates a complete, auditable, frozen dataset:

datasets/
└── {dataset_id}_v{version}/
    ├── data.parquet              # truth
    ├── data.xlsx                 # presentation
    ├── metadata.json             # structured metadata
    ├── dataset_card.md           # human documentation
    └── quality_report.json       # quality certificate

Usage:
    # Common window (balanced panel)
    python scripts/freeze_dataset.py --mode common_window

    # Max history (unbalanced panel)
    python scripts/freeze_dataset.py --mode max_history

    # With Massive
    export MASSIVE_API_KEY=your_key
    python scripts/freeze_dataset.py --provider massive --mode common_window
"""

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Ticker inception dates
TICKER_INCEPTION = {
    "QQQ": date(1999, 3, 10),
    "VOO": date(2010, 9, 9),
    "BIL": date(2007, 5, 30),
    "SPY": date(1993, 1, 29),
}


def download_data(
    provider: str, tickers: list[str], start_date: date, end_date: date, method: str = "rest"
) -> pl.DataFrame:
    """Download data from provider."""
    if provider == "yahoo":
        import pandas as pd
        import yfinance as yf

        all_dfs = []
        for ticker in tickers:
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
        col_map = {c: c.lower().replace(" ", "_") for c in df.columns}
        col_map["adj_close"] = "adj_close"  # ensure this
        df = df.rename(col_map)

        if "date" in df.columns and df["date"].dtype == pl.Datetime:
            df = df.with_columns(pl.col("date").dt.date())

        return df

    elif provider == "massive":
        # Note: Deprecated / Blocked by Plan
        api_key = os.environ.get("MASSIVE_API_KEY")
        if not api_key:
            raise ValueError("MASSIVE_API_KEY not set")
        from portfolio.io.providers import MassiveProvider

        prov = MassiveProvider(api_key=api_key)

        all_dfs = []
        for ticker in tickers:
            ticker_start = max(start_date, TICKER_INCEPTION.get(ticker, start_date))
            logger.info(f"  {ticker}: {ticker_start} → {end_date} (method={method})")

            # Forced REST only, S3 path deprecated
            df = prov.get_bars_daily([ticker], str(ticker_start), str(end_date))

            all_dfs.append(df)
        return pl.concat(all_dfs)

    else:
        raise ValueError(f"Provider {provider} not supported.")


def main():
    parser = argparse.ArgumentParser(description="Freeze Dataset")
    parser.add_argument("--provider", choices=["massive", "yahoo"], default="yahoo")
    # parser.add_argument("--method", choices=["rest", "s3"], default="rest", help="Download method (massive only)")
    parser.add_argument("--mode", choices=["common_window", "max_history"], default="common_window")
    parser.add_argument("--tickers", nargs="+", default=["QQQ", "VOO", "BIL"])
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument("--output-dir", default="datasets")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    tickers = args.tickers
    end_date = date.fromisoformat(args.end) if args.end else date.today()

    # Determine start date
    if args.start:
        start_date = date.fromisoformat(args.start)
        # Assuming manual date means we treat it as is, usually implies balanced intention for that window
        panel_type = "custom_window"
    elif args.mode == "common_window":
        start_date = max(TICKER_INCEPTION.get(t, date(2000, 1, 1)) for t in tickers)
        panel_type = "balanced"
    else:
        start_date = min(TICKER_INCEPTION.get(t, date(2000, 1, 1)) for t in tickers)
        panel_type = "unbalanced"

    dataset_id = f"{'_'.join(t.lower() for t in tickers)}_{args.mode}_{args.provider}_wide"

    logger.info("=" * 60)
    logger.info("DATASET FREEZE")
    logger.info("=" * 60)
    logger.info(f"Dataset ID: {dataset_id}")
    logger.info(f"Version: {args.version}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Panel Type: {panel_type}")
    logger.info(f"Provider: {args.provider}")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Requested: {start_date} → {end_date}")
    logger.info("")

    # Download
    logger.info("📥 Downloading data...")
    # Download
    logger.info("📥 Downloading data...")
    df_raw = download_data(args.provider, tickers, start_date, end_date)
    # df_raw = download_data(args.provider, tickers, start_date, end_date, method=args.method)
    df_raw = df_raw.with_columns(pl.col("ticker").alias("instrument_id"))
    logger.info(f"  Total: {df_raw.height} rows")

    # Build features (This produces Wide features like ret_1d_QQQ based on v1_features logic)
    logger.info("")
    logger.info("🔧 Building features...")
    from portfolio.trading.v1_features import V1FeatureBuilder

    builder = V1FeatureBuilder()
    df_features = builder.build(df_raw)
    logger.info(f"  {len(df_features.columns)} features")

    # Transform RAW to WIDE (to match features)
    logger.info("  Pivoting RAW to Wide...")

    # Pivot logic for OHLCV
    dfs_wide = []
    base_cols = ["open", "high", "low", "close", "volume", "adj_close"]

    for ticker in tickers:
        df_ticker = df_raw.filter(pl.col("ticker") == ticker)

        # Select and rename
        cols_to_select = ["date"] + [c for c in base_cols if c in df_ticker.columns]
        df_t = df_ticker.select(cols_to_select)

        # Rename cols: close -> close_QQQ
        name_map = {c: f"{c}_{ticker}" for c in cols_to_select if c != "date"}
        df_t = df_t.rename(name_map)

        dfs_wide.append(df_t)

    # Join all wide dfs
    df_wide_raw = dfs_wide[0]
    for df_w in dfs_wide[1:]:
        df_wide_raw = df_wide_raw.join(df_w, on="date", how="full", coalesce=True)

    # Merge with Features
    # Features are already Wide (ret_1d_QQQ, etc)
    df = df_wide_raw.join(df_features, on="date", how="inner")

    # Add quality flag and event day (global placeholders or per asset)
    # For Wide, we can add a global flag or just skip
    df = df.with_columns(
        [
            pl.lit("OK").alias("quality_flag"),
            pl.lit(False).alias("is_event_day"),
        ]
    )

    # Filter for common_window
    if args.mode == "common_window":
        common_start = max(TICKER_INCEPTION.get(t, date(2000, 1, 1)) for t in tickers)
        df = df.filter(pl.col("date") >= common_start)
        logger.info(f"  Filtered to common window: {df.height} rows")

    # Freeze
    logger.info("")
    logger.info("🧊 Freezing dataset...")

    from portfolio.io.export import DatasetDefinition, DatasetFreezer

    definition = DatasetDefinition(
        dataset_id=dataset_id,
        dataset_version=args.version,
        panel_type=panel_type,
        description=f"{'/'.join(tickers)} historical dataset (Wide Format). Mode: {args.mode}.",
        provider=args.provider,
        tickers=tickers,
        requested_start=start_date,
        requested_end=end_date,
    )

    freezer = DatasetFreezer(base_path=args.output_dir)
    folder = freezer.freeze(df, definition)

    # Verify
    logger.info("")
    logger.info("🔍 Verifying...")
    if freezer.verify(folder):
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ DATASET FROZEN SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Location: {folder}")
        logger.info("")
        logger.info("Contents:")
        for f in folder.iterdir():
            logger.info(f"  - {f.name}")
        logger.info("")
        logger.info("Sheets:")
        logger.info("  - DATA")
        logger.info("  - METADATA")
        logger.info("  - DATA_DICTIONARY")
        logger.info("  - QUALITY_REPORT")
    else:
        logger.error("❌ Verification failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
