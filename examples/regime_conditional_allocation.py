"""
Regime-Conditional Allocation Example

Demonstrates how to adjust portfolio allocation based on detected market regimes.

Strategy:
- Bull Market: 120% allocation (leveraged)
- Bear Market: 80% allocation (reduced exposure)
- Crisis: 50% allocation (defensive)
"""

import numpy as np
import polars as pl
import yfinance as yf

from portfolio.features.regime_detection import RegimeDetector


def regime_conditional_backtest(
    df_returns: pl.DataFrame,
    regime_col: str = "regime_label",
    base_weights: np.ndarray | None = None,
) -> tuple[pl.DataFrame, dict]:
    """
    Backtest with regime-conditional allocation scaling.
    
    Args:
        df_returns: Returns with regime labels
        regime_col: Column name for regime labels
        base_weights: Base portfolio weights (default: equal weight)
        
    Returns:
        (performance_df, statistics)
    """
    if base_weights is None:
        n_assets = len([c for c in df_returns.columns if c.startswith("ret_")])
        base_weights = np.ones(n_assets) / n_assets
    
    # Regime scaling rules
    regime_scales = {
        "Bull": 1.2,      # Leverage in bull markets
        "Bear": 0.8,      # Reduce exposure in bear
        "Crisis": 0.5,    # Very defensive in crisis
    }
    
    # Apply scaling
    results = []
    
    for row in df_returns.iter_rows(named=True):
        regime = row.get(regime_col, "Bull")
        scale = regime_scales.get(regime, 1.0)
        
        # Scale weights
        scaled_weights = base_weights * scale
        scaled_weights = scaled_weights / scaled_weights.sum()  # Renormalize
        
        # Compute portfolio return
        asset_returns = [v for k, v in row.items() if k.startswith("ret_")]
        port_return = np.dot(scaled_weights, asset_returns)
        
        results.append({
            "date": row.get("date"),
            "regime": regime,
            "scale": scale,
            "portfolio_return": port_return,
        })
    
    df_results = pl.DataFrame(results)
    
    # Compute cumulative returns
    df_results = df_results.with_columns(
        (1 + pl.col("portfolio_return")).cum_prod().alias("equity")
    )
    
    # Statistics
    total_return = df_results["equity"][-1] - 1.0
    sharpe = (
        df_results["portfolio_return"].mean()
        / df_results["portfolio_return"].std()
        * np.sqrt(252)
    )
    
    # Performance by regime
    perf_by_regime = (
        df_results.group_by("regime")
        .agg([
            pl.col("portfolio_return").mean().alias("avg_return"),
            pl.col("portfolio_return").std().alias("volatility"),
            pl.col("portfolio_return").count().alias("n_days"),
        ])
        .with_columns(
            (pl.col("avg_return") / pl.col("volatility") * np.sqrt(252)).alias("sharpe")
        )
    )
    
    stats = {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "by_regime": perf_by_regime,
    }
    
    return df_results, stats


def main():
    print("=" * 80)
    print("REGIME-CONDITIONAL ALLOCATION EXAMPLE")
    print("=" * 80)
    
    # Download data
    print("\n[1/4] Downloading data...")
    tickers = ["SPY", "QQQ", "TLT"]
    data = yf.download(tickers, start="2020-01-01", end="2024-12-31", progress=False)
    
    # Compute returns
    returns_dict = {"date": data.index}
    for ticker in tickers:
        returns_dict[f"ret_{ticker}"] = data["Adj Close"][ticker].pct_change()
    
    df_returns = pl.DataFrame(returns_dict).drop_nulls()
    print(f"   Downloaded {len(df_returns)} days")
    
    # Detect regimes
    print("\n[2/4] Detecting market regimes...")
    spy_df = df_returns.select(["date", "ret_SPY"]).rename({"ret_SPY": "returns"})
    
    detector = RegimeDetector(n_regimes=3, random_state=42)
    detector.fit(spy_df, returns_col="returns")
    df_regimes = detector.predict(spy_df, returns_col="returns")
    
    # Merge with returns
    df_full = df_returns.join(
        df_regimes.select(["date", "regime_label"]),
        on="date",
        how="inner",
    )
    
    print("   Regime distribution:")
    regime_dist = (
        df_regimes.group_by("regime_label")
        .agg(pl.col("regime_label").count().alias("count"))
    )
    print(regime_dist)
    
    # Run regime-conditional backtest
    print("\n[3/4] Running regime-conditional backtest...")
    
    df_results, stats = regime_conditional_backtest(df_full)
    
    # Compare with static allocation
    print("\n[4/4] Comparing with static allocation...")
    
    # Static allocation (no regime adjustment)
    static_weights = np.ones(3) / 3
    static_returns = []
    
    for row in df_full.iter_rows(named=True):
        asset_returns = [row[f"ret_{t}"] for t in tickers]
        port_return = np.dot(static_weights, asset_returns)
        static_returns.append(port_return)
    
    static_equity = np.cumprod(1 + np.array(static_returns))
    static_total_return = static_equity[-1] - 1.0
    static_sharpe = (
        np.mean(static_returns) / np.std(static_returns) * np.sqrt(252)
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print("\n📊 Overall Performance:")
    print(f"  Regime-Conditional:")
    print(f"    Total Return: {stats['total_return']:.2%}")
    print(f"    Sharpe Ratio: {stats['sharpe']:.2f}")
    
    print(f"\n  Static Allocation:")
    print(f"    Total Return: {static_total_return:.2%}")
    print(f"    Sharpe Ratio: {static_sharpe:.2f}")
    
    print(f"\n  🎯 Improvement: {(stats['total_return'] - static_total_return) / abs(static_total_return) * 100:+.1f}% return")
    print(f"     Sharpe gain: {stats['sharpe'] - static_sharpe:+.2f}")
    
    print("\n📈 Performance by Regime:")
    print(stats["by_regime"])
    
    print("\n💡 Regime Scaling Rules:")
    print("  Bull Market  → 120% allocation (leverage)")
    print("  Bear Market  →  80% allocation (reduce)")
    print("  Crisis       →  50% allocation (defensive)")
    
    print("\n✅ Regime-conditional strategy completed!")


if __name__ == "__main__":
    main()
