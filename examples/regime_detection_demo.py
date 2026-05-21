"""
Example usage of HMM Regime Detection

This script demonstrates:
1. Downloading market data
2. Detecting regimes (Bull/Bear/Crisis)
3. Visualizing regime transitions
4. Computing performance metrics by regime
"""

import polars as pl
import yfinance as yf

from portfolio.features.regime_detection import (
    RegimeDetector,
    compute_regime_performance,
)
from portfolio.viz.regime_plots import (
    plot_regime_duration_histogram,
    plot_regime_performance,
    plot_regime_probabilities,
    plot_regime_states,
    plot_regime_transitions,
)


def main():
    print("=" * 80)
    print("HMM Regime Detection Demo")
    print("=" * 80)

    # Step 1: Download SPY data (S&P 500 ETF)
    print("\n[1/5] Downloading SPY data...")
    spy = yf.download("SPY", start="2020-01-01", end="2024-12-31", progress=False)

    # Compute returns
    spy["returns"] = spy["Adj Close"].pct_change()
    spy = spy.dropna()

    # Convert to Polars
    df = pl.DataFrame(
        {
            "date": spy.index,
            "returns": spy["returns"].values,
        }
    )

    print(f"   Downloaded {len(df)} days of data")

    # Step 2: Fit regime detector
    print("\n[2/5] Fitting HMM regime detector (3 states)...")
    detector = RegimeDetector(n_regimes=3, random_state=42, n_iter=200)
    detector.fit(df, returns_col="returns")

    print("   Regime statistics:")
    stats = detector.get_regime_stats()
    print(stats.to_string(index=False))

    # Step 3: Predict regimes
    print("\n[3/5] Predicting regimes...")
    df_regimes = detector.predict(df, returns_col="returns")

    # Show regime distribution
    regime_counts = df_regimes.group_by("regime_label").count()
    print("\n   Regime distribution:")
    print(regime_counts)

    # Step 4: Compute performance by regime
    print("\n[4/5] Computing performance metrics by regime...")
    perf = compute_regime_performance(df_regimes, returns_col="returns")
    print(perf.to_string(index=False))

    # Step 5: Visualizations
    print("\n[5/5] Creating visualizations...")

    # Plot 1: Equity curve with regime bands
    fig1 = plot_regime_states(df_regimes, title="SPY Regimes 2020-2024")
    fig1.write_html("regime_states.html")
    print("   - Saved: regime_states.html")

    # Plot 2: Regime probabilities over time
    fig2 = plot_regime_probabilities(df_regimes, n_regimes=3)
    fig2.write_html("regime_probabilities.html")
    print("   - Saved: regime_probabilities.html")

    # Plot 3: Transition matrix
    trans_mat = detector.get_transition_matrix()
    fig3 = plot_regime_transitions(trans_mat)
    fig3.write_html("regime_transitions.html")
    print("   - Saved: regime_transitions.html")

    # Plot 4: Performance scatter
    fig4 = plot_regime_performance(perf)
    fig4.write_html("regime_performance.html")
    print("   - Saved: regime_performance.html")

    # Plot 5: Duration histogram
    fig5 = plot_regime_duration_histogram(df_regimes)
    fig5.write_html("regime_durations.html")
    print("   - Saved: regime_durations.html")

    print("\n" + "=" * 80)
    print("Demo complete! Open the HTML files to view interactive charts.")
    print("=" * 80)

    # Bonus: Show transition probabilities
    print("\nTransition Probability Matrix:")
    print("(Rows = From state, Columns = To state)")
    print(trans_mat)

    # Bonus: Recent regime
    recent_regime = df_regimes.tail(1).select("regime_label").item()
    print(f"\nCurrent regime (most recent): {recent_regime}")

    return df_regimes, detector


if __name__ == "__main__":
    df_regimes, detector = main()
