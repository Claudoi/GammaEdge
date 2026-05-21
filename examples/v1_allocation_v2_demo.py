#!/usr/bin/env python3
"""
V1 Allocation v2: Turnover Controlled + Predictive Model
========================================================

Ejecuta:
1. Grid search para encontrar γ, τ, α óptimos (turnover < 300%)
2. Backtest con controles ajustados
3. Walk-forward validation con modelo predictivo Ridge

Usage:
    python examples/v1_allocation_v2_demo.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl


def download_data():
    """Descarga datos de QQQ, VOO, BIL."""
    try:
        import yfinance as yf
    except ImportError:
        print("Error: pip install yfinance")
        sys.exit(1)

    tickers = ["QQQ", "VOO", "BIL"]
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 3)

    print(f"📥 Downloading {tickers}...")

    import pandas as pd

    all_dfs = []
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.reset_index()
        df["ticker"] = ticker
        all_dfs.append(df)

    df_combined = pd.concat(all_dfs, ignore_index=True)
    df = pl.from_pandas(df_combined)

    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower == "date":
            col_map[col] = "date"
        elif col_lower == "open":
            col_map[col] = "open"
        elif col_lower == "close":
            col_map[col] = "close"
        elif col_lower == "high":
            col_map[col] = "high"
        elif col_lower == "low":
            col_map[col] = "low"
        elif col_lower == "volume":
            col_map[col] = "volume"
        elif col_lower == "ticker":
            col_map[col] = "ticker"

    df = df.rename(col_map)

    if "date" in df.columns and df["date"].dtype == pl.Datetime:
        df = df.with_columns(pl.col("date").dt.date())

    print(f"   Downloaded {df.height} rows")
    return df


def run_demo():
    """Demo completo: grid search + backtest v2 + walk-forward."""
    from portfolio.trading.predictor import (
        PredictorConfig,
        WalkForwardConfig,
        WalkForwardValidator,
    )
    from portfolio.trading.v1_allocation import AllocationLabelBuilder
    from portfolio.trading.v1_allocation_v2 import (
        AllocationBacktestV2,
        V1AllocationConfigV2,
        grid_search_turnover_controls,
        select_best_config,
    )
    from portfolio.trading.v1_features import V1FeatureBuilder

    # 1. Descargar datos
    df = download_data()
    print()

    # 2. Construir features
    print("🔧 Building features...")
    feature_builder = V1FeatureBuilder()
    df_features = feature_builder.build(df)
    print(f"   {len(df_features.columns)} features, {df_features.height} rows")

    # 3. Construir returns
    print("🔧 Building returns...")
    # Usamos la V1 config solo para construir returns
    from portfolio.trading.v1_allocation import V1AllocationConfig

    label_config = V1AllocationConfig()
    label_builder = AllocationLabelBuilder(label_config)
    df_returns = label_builder.build(df)
    print(f"   {df_returns.height} rows with returns")

    # Merge
    df_merged = df_features.join(df_returns, on="date", how="inner").drop_nulls()

    # =========================================================================
    # PASO 1: Grid Search para encontrar controles óptimos
    # =========================================================================
    print()
    print("=" * 60)
    print("PASO 1: Grid Search Turnover Controls")
    print("=" * 60)
    print()

    print("🔍 Running grid search (γ, τ, α)...")
    print("   γ (turnover_penalty): [0.10, 0.20, 0.35, 0.50]")
    print("   τ (rebalance_threshold): [0.05, 0.10, 0.15]")
    print("   α (partial_adjustment): [0.25, 0.50, 0.75]")
    print()

    results = grid_search_turnover_controls(
        df_merged,
        gammas=[0.02, 0.05, 0.10, 0.15],  # Más bajos
        taus=[0.02, 0.05, 0.08],  # Más bajos
        alphas=[0.30, 0.50, 0.70],
        max_turnover=3.0,
        min_sharpe=0.5,
    )

    # Mostrar top 5
    print("📊 Top 5 configurations:")
    print(f"{'γ':>6} {'τ':>6} {'α':>6} {'Sharpe':>8} {'TO%':>8} {'DD%':>8} {'#Reb':>6} {'OK':>4}")
    print("-" * 60)

    for r in results[:5]:
        ok = "✅" if r["feasible"] else "❌"
        print(
            f"{r['gamma']:>6.2f} {r['tau']:>6.2f} {r['alpha']:>6.2f} "
            f"{r['sharpe']:>8.2f} {r['turnover'] * 100:>7.0f}% {r['max_dd'] * 100:>7.1f}% "
            f"{r['n_rebalances']:>6} {ok:>4}"
        )

    # Seleccionar mejor config
    best_config = select_best_config(results)

    if best_config is None:
        print("\n❌ No feasible config found. Using default with higher controls.")
        best_config = V1AllocationConfigV2(
            turnover_penalty=0.50,
            rebalance_threshold=0.15,
            partial_adjustment=0.25,
        )

    print("\n✅ Selected config:")
    print(f"   γ = {best_config.turnover_penalty}")
    print(f"   τ = {best_config.rebalance_threshold}")
    print(f"   α = {best_config.partial_adjustment}")
    print(f"   hash = {best_config.compute_hash()}")

    # =========================================================================
    # PASO 2: Backtest con controles óptimos
    # =========================================================================
    print()
    print("=" * 60)
    print("PASO 2: Backtest con Controles Óptimos")
    print("=" * 60)
    print()

    backtest = AllocationBacktestV2(best_config)
    result = backtest.run(df_merged)

    print("📊 V1 ALLOCATION v2 RESULTS")
    print("-" * 40)
    print(f"   📈 Total Return:     {result.total_return * 100:+.1f}%")
    print(f"   📈 Sharpe Ratio:     {result.sharpe_annual:.2f}")
    print(f"   📉 Max Drawdown:     {result.max_drawdown * 100:.1f}%")
    print(f"   🔄 Turnover (year):  {result.turnover_annual * 100:.0f}%")
    print(f"   🔁 # Rebalances:     {result.n_rebalances}")
    print(f"   🎯 Hit Rate vs VOO:  {result.hit_rate_vs_voo * 100:.1f}%")
    print(f"   📉 Downside Capture: {result.downside_capture:.2f}")
    print()

    # Check targets
    print("Targets V1:")
    sharpe_ok = "✅" if result.sharpe_annual > 1.0 else "❌"
    dd_ok = "✅" if result.max_drawdown > -0.15 else "❌"
    turnover_ok = "✅" if result.turnover_annual < 3.0 else "❌"

    print(f"   {sharpe_ok} Sharpe > 1.0:      {result.sharpe_annual:.2f}")
    print(f"   {dd_ok} MaxDD > -15%:     {result.max_drawdown * 100:.1f}%")
    print(f"   {turnover_ok} Turnover < 300%:  {result.turnover_annual * 100:.0f}%")

    # =========================================================================
    # PASO 3: Walk-Forward con Modelo Predictivo
    # =========================================================================
    print()
    print("=" * 60)
    print("PASO 3: Walk-Forward Validation (Ridge Predictor)")
    print("=" * 60)
    print()

    print("📊 Walk-forward config:")
    print("   Train: 252 days (1 year)")
    print("   Test: 21 days (1 month)")
    print("   Step: 21 days")
    print()

    # Verificar si tenemos sklearn
    try:
        from sklearn.linear_model import Ridge  # noqa: F401

        has_sklearn = True
    except ImportError:
        has_sklearn = False
        print("⚠️  sklearn not installed. Skipping walk-forward.")

    if has_sklearn:
        print("🚀 Running walk-forward...")

        validator = WalkForwardValidator(
            predictor_config=PredictorConfig(model_type="ridge", alpha=1.0),
            allocation_config=best_config,
            wf_config=WalkForwardConfig(train_days=252, test_days=21, step_days=21),
        )

        wf_results = validator.run(df_features, df_returns)
        summary = validator.summarize(wf_results)

        print()
        print("📊 Walk-Forward Summary:")
        print("-" * 40)
        print(f"   Windows:           {summary.get('n_windows', 0)}")
        print(
            f"   Sharpe (mean):     {summary.get('sharpe_mean', 0):.2f} ± {summary.get('sharpe_std', 0):.2f}"
        )
        print(
            f"   Sharpe (min/max):  {summary.get('sharpe_min', 0):.2f} / {summary.get('sharpe_max', 0):.2f}"
        )
        print(f"   Return (mean):     {summary.get('return_mean', 0) * 100:.1f}%")
        print(f"   Turnover (mean):   {summary.get('turnover_mean', 0) * 100:.0f}%")
        print(f"   Worst DD:          {summary.get('max_dd_worst', 0) * 100:.1f}%")
        print(f"   % Positive Sharpe: {summary.get('pct_positive_sharpe', 0) * 100:.0f}%")

    print()
    print("=" * 60)
    print("✅ V1 Allocation v2 Complete")
    print("=" * 60)

    return result


if __name__ == "__main__":
    run_demo()
