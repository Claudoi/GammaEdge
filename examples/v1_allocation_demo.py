#!/usr/bin/env python3
"""
V1 Allocation Demo: QQQ/VOO/BIL
==============================

Script de ejemplo que descarga datos reales y ejecuta backtest V1.

Usage:
    python examples/v1_allocation_demo.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl


def download_data():
    """Descarga datos de QQQ, VOO, BIL vía Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        print("Error: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    tickers = ["QQQ", "VOO", "BIL"]
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 3)  # 3 años

    print(f"📥 Downloading {tickers} from {start_date} to {end_date}...")

    import pandas as pd

    all_dfs = []
    for ticker in tickers:
        try:
            # Descargar individualmente
            df = yf.download(
                ticker, start=start_date, end=end_date, progress=False, auto_adjust=False
            )

            # Aplanar MultiIndex si existe
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            df = df.reset_index()
            df["ticker"] = ticker
            all_dfs.append(df)
        except Exception as e:
            print(f"   Warning: Could not download {ticker}: {e}")

    if not all_dfs:
        print("Error: No data downloaded")
        sys.exit(1)

    # Combinar
    df_combined = pd.concat(all_dfs, ignore_index=True)

    # Convertir a Polars
    df = pl.from_pandas(df_combined)

    # Normalizar nombres de columnas
    col_map = {}
    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()
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
        elif col_lower in ["adj close", "adj_close"]:
            col_map[col] = "adj_close"
        elif col_lower == "volume":
            col_map[col] = "volume"
        elif col_lower == "ticker":
            col_map[col] = "ticker"

    df = df.rename(col_map)

    # Convertir fecha si es datetime
    if "date" in df.columns:
        if df["date"].dtype == pl.Datetime:
            df = df.with_columns(pl.col("date").dt.date())
        elif df["date"].dtype == pl.Date:
            pass  # Ya está bien

    print(f"   Downloaded {df.height} rows, columns: {df.columns}")
    return df


def run_backtest():
    """Ejecuta backtest V1 completo."""
    from portfolio.trading import (
        AllocationBacktest,
        AllocationLabelBuilder,
        SampleValidityChecker,
        V1FeatureBuilder,
        get_v1_allocation_config,
    )

    # 1. Descargar datos
    df = download_data()
    print()

    # 2. Configuración
    config = get_v1_allocation_config()
    print(f"📋 Config: {config.config_version} (hash: {config.compute_hash()})")
    print(f"   Assets: {config.assets}")
    print(f"   Round-trip cost: {config.round_trip_bps} bps")
    print()

    # 3. Construir features
    print("🔧 Building features...")
    feature_builder = V1FeatureBuilder()
    df_features = feature_builder.build(df)
    print(f"   {len(df_features.columns)} features, {df_features.height} rows")

    # 4. Construir forward returns (labels)
    print("🔧 Building labels (forward returns)...")
    label_builder = AllocationLabelBuilder(config)
    df_returns = label_builder.build(df)
    print(f"   {df_returns.height} rows with labels")

    # 5. Merge features + returns
    df_merged = df_features.join(df_returns, on="date", how="inner")
    print(f"   {df_merged.height} rows after merge")

    # 6. Sample validity
    print("✅ Checking sample validity...")
    checker = SampleValidityChecker(config)
    df_merged = checker.add_validity_column(df_merged)
    valid_count = df_merged.filter(pl.col("sample_valid")).height
    print(
        f"   {valid_count}/{df_merged.height} valid samples ({100 * valid_count / df_merged.height:.1f}%)"
    )

    df_valid = df_merged.filter(pl.col("sample_valid"))

    # 7. Backtest
    print()
    print("🚀 Running backtest...")
    backtest = AllocationBacktest(config)
    result = backtest.run(df_valid)

    # 8. Resultados
    print()
    print("=" * 50)
    print("📊 V1 ALLOCATION BACKTEST RESULTS")
    print("=" * 50)
    print(f"   Period: {df_valid['date'].min()} to {df_valid['date'].max()}")
    print(f"   Days: {len(result.dates)}")
    print()
    print(f"   📈 Total Return:    {result.total_return * 100:+.1f}%")
    print(f"   📈 Sharpe Ratio:    {result.sharpe_annual:.2f}")
    print(f"   📉 Max Drawdown:    {result.max_drawdown * 100:.1f}%")
    print(f"   🔄 Turnover (year): {result.turnover_annual * 100:.0f}%")
    print(f"   🎯 Hit Rate vs VOO: {result.hit_rate_vs_voo * 100:.1f}%")
    print(f"   📊 Volatility:      {result.volatility_annual * 100:.1f}%")
    print()

    # Targets
    print("Targets V1:")
    sharpe_ok = "✅" if result.sharpe_annual > 1.0 else "❌"
    dd_ok = "✅" if result.max_drawdown > -0.15 else "❌"
    turnover_ok = "✅" if result.turnover_annual < 3.0 else "❌"
    hit_ok = "✅" if result.hit_rate_vs_voo > 0.55 else "❌"

    print(f"   {sharpe_ok} Sharpe > 1.0:     {result.sharpe_annual:.2f}")
    print(f"   {dd_ok} MaxDD > -15%:    {result.max_drawdown * 100:.1f}%")
    print(f"   {turnover_ok} Turnover < 300%: {result.turnover_annual * 100:.0f}%")
    print(f"   {hit_ok} Hit Rate > 55%:  {result.hit_rate_vs_voo * 100:.1f}%")
    print()

    # Sample weights
    print("📊 Sample final weights:")
    if result.weights:
        final_weights = result.weights[-1]
        for asset, w in final_weights.items():
            print(f"   {asset}: {w:.1%}")

    print()
    print(f"Config hash: {result.config_hash}")

    return result


if __name__ == "__main__":
    run_backtest()
