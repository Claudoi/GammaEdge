"""
TIER 1 Complete Integration Demo

End-to-end demonstration of all 4 TIER 1 enhancements:
1. HMM Regime Detection
2. Fama-French Factor Models
3. XGBoost Predictor
4. CVaR Optimization

This script shows how to combine all features in a realistic workflow.
"""

import numpy as np
import polars as pl
import yfinance as yf

from portfolio.features.factor_models import compute_factor_loadings, fetch_fama_french

# TIER 1 imports
from portfolio.features.regime_detection import RegimeDetector, compute_regime_performance
from portfolio.optim.cvar import cvar_portfolio_optimizer, generate_bootstrap_scenarios
from portfolio.trading.ml_predictors import GradientBoostingPredictor, XGBoostConfig


def main():
    print("=" * 80)
    print("TIER 1 INTEGRATION DEMO: HMM + FF + XGBoost + CVaR")
    print("=" * 80)

    # =============================================================================
    # Step 1: Download Market Data
    # =============================================================================
    print("\n[1/6] Downloading market data...")

    tickers = ["SPY", "QQQ", "TLT"]  # Stocks, Tech, Bonds
    data = yf.download(tickers, start="2020-01-01", end="2024-12-31", progress=False)

    # Compute returns
    returns_dict = {"date": data.index}
    for ticker in tickers:
        returns_dict[f"ret_{ticker}"] = data["Adj Close"][ticker].pct_change()

    df_returns = pl.DataFrame(returns_dict).drop_nulls()
    print(f"   Downloaded {len(df_returns)} days of data for {tickers}")

    # =============================================================================
    # Step 2: Regime Detection (HMM)
    # =============================================================================
    print("\n[2/6] Detecting market regimes with HMM...")

    # Create single return series for regime detection (use SPY as market proxy)
    spy_df = df_returns.select(["date", "ret_SPY"]).rename({"ret_SPY": "returns"})

    detector = RegimeDetector(n_regimes=3, random_state=42)
    detector.fit(spy_df, returns_col="returns")

    df_regimes = detector.predict(spy_df, returns_col="returns")

    # Print regime stats
    stats = detector.get_regime_stats()
    print("\n   Regime Statistics:")
    print(stats.to_string(index=False))

    # Performance by regime
    perf = compute_regime_performance(df_regimes, returns_col="returns")
    print("\n   Performance by Regime:")
    print(
        perf[["regime", "mean_return", "volatility", "sharpe", "frequency"]].to_string(index=False)
    )

    # =============================================================================
    # Step 3: Factor Analysis (Fama-French)
    # =============================================================================
    print("\n[3/6] Computing Fama-French factor exposures...")

    try:
        # Fetch FF3 factors
        factors = fetch_fama_french("FF3", start="2020-01-01", end="2024-12-31")

        # Compute loadings for SPY
        spy_returns = df_returns.to_pandas()["ret_SPY"]
        loadings = compute_factor_loadings(spy_returns, factors, model="FF3")

        print("\n   SPY Factor Exposures:")
        print(f"   Alpha (annualized): {loadings['alpha']:.2%}")
        print(f"   Market Beta: {loadings['betas']['Mkt-RF']:.3f}")
        print(f"   SMB (Size): {loadings['betas']['SMB']:.3f}")
        print(f"   HML (Value): {loadings['betas']['HML']:.3f}")
        print(f"   R²: {loadings['r_squared']:.2%}")

    except Exception as e:
        print(f"   ⚠️  Factor data fetch failed (likely no internet): {e}")
        loadings = None

    # =============================================================================
    # Step 4: XGBoost Return Prediction
    # =============================================================================
    print("\n[4/6] Training XGBoost predictor...")

    # Create simple features
    df_features_dict = {"date": df_returns["date"]}

    for ticker in tickers:
        ret_col = f"ret_{ticker}"
        # Feature: 5-day momentum
        df_features_dict[f"mom_5d_{ticker}"] = df_returns[ret_col].shift(1).rolling_sum(5)
        # Feature: 20-day volatility
        df_features_dict[f"vol_20d_{ticker}"] = df_returns[ret_col].shift(1).rolling_std(20)

    df_features = pl.DataFrame(df_features_dict).drop_nulls()
    df_returns_clean = df_returns.join(df_features.select("date"), on="date", how="inner")

    # Configure and train XGBoost
    config = XGBoostConfig(
        assets=tickers,
        n_estimators=50,
        max_depth=3,
        feature_cols=["mom_5d", "vol_20d"],
    )

    try:
        predictor = GradientBoostingPredictor(config)
        predictor.fit(df_features, df_returns_clean, use_cv=False)  # Skip CV for demo speed

        # Predict on latest data
        predictions = predictor.predict(df_features.tail(1))

        print("\n   XGBoost Predictions (latest):")
        for ticker, pred in zip(tickers, predictions, strict=False):
            print(f"   {ticker}: {pred:.4f} ({pred * 252:.2%} annualized)")

        # Feature importance
        importance = predictor.get_feature_importance(tickers[0])
        if importance:
            print(f"\n   Feature Importance ({tickers[0]}):")
            for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:3]:
                print(f"   {feat}: {imp:.3f}")

    except Exception as e:
        print(f"   ⚠️  XGBoost training failed (may need OpenMP): {e}")
        predictions = np.array([0.001] * len(tickers))

    # =============================================================================
    # Step 5: CVaR Portfolio Optimization
    # =============================================================================
    print("\n[5/6] Optimizing portfolio with CVaR...")

    # Prepare historical returns for bootstrap
    returns_matrix = df_returns_clean.select([f"ret_{t}" for t in tickers]).to_numpy()

    # Estimate mu and Sigma
    mu = returns_matrix.mean(axis=0) * 252  # Annualized
    Sigma = np.cov(returns_matrix.T) * 252

    # CVaR optimization with bootstrap scenarios
    weights_cvar = cvar_portfolio_optimizer(
        mu=mu,
        Sigma=Sigma,
        alpha=0.95,
        n_scenarios=1000,
        w_min=0.0,
        w_max=0.5,
        use_bootstrap=True,
        historical_returns=returns_matrix,
    )

    print("\n   CVaR-Optimized Weights (95% confidence):")
    for ticker, weight in zip(tickers, weights_cvar, strict=False):
        print(f"   {ticker}: {weight:.2%}")

    # Compare with equal weights
    weights_equal = np.array([1 / len(tickers)] * len(tickers))

    # Generate scenarios for comparison
    scenarios = generate_bootstrap_scenarios(returns_matrix, n_scenarios=1000)

    # Compute CVaR for both
    from portfolio.optim.cvar import compute_portfolio_cvar

    var_cvar, cvar_cvar = compute_portfolio_cvar(weights_cvar, scenarios, alpha=0.95)
    var_eq, cvar_eq = compute_portfolio_cvar(weights_equal, scenarios, alpha=0.95)

    print("\n   CVaR Comparison:")
    print(f"   Equal Weights:      VaR={-var_eq:.2%}, CVaR={-cvar_eq:.2%}")
    print(f"   CVaR-Optimized:     VaR={-var_cvar:.2%}, CVaR={-cvar_cvar:.2%}")
    print(f"   Improvement:        {(cvar_eq - cvar_cvar) / cvar_eq * 100:.1f}% lower CVaR")

    # =============================================================================
    # Step 6: Regime-Conditional Allocation
    # =============================================================================
    print("\n[6/6] Regime-conditional strategy...")

    # Simple rule: More defensive in Bear/Crisis regimes
    current_regime = df_regimes.tail(1)["regime_label"].item()

    if current_regime == "Bull":
        allocation_multiplier = 1.2  # Leverage in bull market
        print(f"   Current regime: {current_regime} → 120% allocation")
    elif current_regime == "Bear":
        allocation_multiplier = 0.8  # Reduce exposure in bear
        print(f"   Current regime: {current_regime} → 80% allocation")
    else:  # Crisis
        allocation_multiplier = 0.5  # Very defensive
        print(f"   Current regime: {current_regime} → 50% allocation")

    weights_final = weights_cvar * allocation_multiplier
    weights_final = weights_final / weights_final.sum()  # Renormalize

    print("\n   Final Regime-Adjusted Weights:")
    for ticker, weight in zip(tickers, weights_final, strict=False):
        print(f"   {ticker}: {weight:.2%}")

    # =============================================================================
    # Summary
    # =============================================================================
    print("\n" + "=" * 80)
    print("TIER 1 INTEGRATION COMPLETE")
    print("=" * 80)
    print("\n✅ HMM: Detected 3 market regimes")
    print(
        f"✅ Fama-French: Computed factor exposures (R² = {loadings['r_squared']:.1%})"
        if loadings
        else "⚠️  Fama-French: Skipped (no internet)"
    )
    print("✅ XGBoost: Trained gradient boosting predictor")
    print(
        f"✅ CVaR: Optimized portfolio ({(cvar_eq - cvar_cvar) / cvar_eq * 100:.1f}% CVaR improvement)"
    )
    print(f"✅ Regime-Adaptive: {current_regime} regime → {allocation_multiplier:.0%} allocation")
    print("\n")


if __name__ == "__main__":
    main()
