"""
Visualization utilities for regime detection analysis.

Provides interactive Plotly charts for regime states, transitions,
and performance metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    import polars as pl


def plot_regime_states(
    df: pl.DataFrame,
    regime_col: str = "regime_label",
    returns_col: str = "returns",
    date_col: str = "date",
    title: str = "Market Regimes Over Time",
) -> go.Figure:
    """
    Plot equity curve colored by regime state.
    
    Args:
        df: DataFrame with date, returns, and regime labels
        regime_col: Column name for regime labels
        returns_col: Column name for returns
        date_col: Column name for dates
        title: Chart title
        
    Returns:
        Plotly Figure with colored regime bands
        
    Example:
        >>> fig = plot_regime_states(df_regimes)
        >>> fig.show()
    """
    df_pd = df.to_pandas()
    df_pd[date_col] = pd.to_datetime(df_pd[date_col])
    
    # Compute cumulative returns (equity curve)
    df_pd["equity"] = (1 + df_pd[returns_col]).cumprod()
    
    # Regime colors (supports up to 4 regimes)
    regime_colors = {
        "Bull": "rgba(0, 255, 0, 0.2)",      # Green
        "Bear": "rgba(255, 165, 0, 0.2)",    # Orange
        "Crisis": "rgba(255, 0, 0, 0.2)",    # Red
        "Volatile": "rgba(128, 0, 128, 0.2)", # Purple
        "Regime 0": "rgba(0, 255, 0, 0.2)",
        "Regime 1": "rgba(255, 165, 0, 0.2)",
        "Regime 2": "rgba(255, 0, 0, 0.2)",
        "Regime 3": "rgba(128, 0, 128, 0.2)",
    }
    
    fig = go.Figure()
    
    # Add equity curve
    fig.add_trace(go.Scatter(
        x=df_pd[date_col],
        y=df_pd["equity"],
        mode="lines",
        name="Equity",
        line=dict(color="navy", width=2),
    ))
    
    # Add regime background shading
    regimes = df_pd[regime_col].unique()
    for regime in regimes:
        regime_data = df_pd[df_pd[regime_col] == regime]
        
        # Find continuous regime periods
        regime_periods = []
        start = None
        
        for i, (idx, row) in enumerate(df_pd.iterrows()):
            if row[regime_col] == regime:
                if start is None:
                    start = row[date_col]
            else:
                if start is not None:
                    regime_periods.append((start, df_pd.loc[idx - 1, date_col]))
                    start = None
        
        # Handle case where regime continues to end
        if start is not None:
            regime_periods.append((start, df_pd[date_col].iloc[-1]))
        
        # Add shaded regions
        for start_date, end_date in regime_periods:
            fig.add_vrect(
                x0=start_date,
                x1=end_date,
                fillcolor=regime_colors.get(regime, "rgba(128, 128, 128, 0.2)"),
                layer="below",
                line_width=0,
                annotation_text=regime if start_date == regime_periods[0][0] else "",
                annotation_position="top left",
            )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Equity (Cumulative Return)",
        hovermode="x unified",
        height=500,
    )
    
    return fig


def plot_regime_probabilities(
    df: pl.DataFrame,
    n_regimes: int = 3,
    date_col: str = "date",
    title: str = "Regime Probabilities Over Time",
) -> go.Figure:
    """
    Plot regime probabilities as stacked area chart.
    
    Args:
        df: DataFrame with date and prob_0, prob_1, prob_2 columns
        n_regimes: Number of regimes
        date_col: Column name for dates
        title: Chart title
        
    Returns:
        Plotly Figure with stacked probabilities
    """
    df_pd = df.to_pandas()
    df_pd[date_col] = pd.to_datetime(df_pd[date_col])
    
    fig = go.Figure()
    
    # Dynamic regime names and colors based on n_regimes
    regime_names_full = ["Bull", "Bear", "Crisis", "Volatile"]
    colors_full = ["green", "orange", "red", "purple"]
    
    regime_names = regime_names_full[:n_regimes]
    colors = colors_full[:n_regimes]
    
    for i in range(n_regimes):
        fig.add_trace(go.Scatter(
            x=df_pd[date_col],
            y=df_pd[f"prob_{i}"],
            mode="lines",
            name=regime_names[i],
            stackgroup="one",
            fillcolor=colors[i],
            line=dict(width=0),
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Probability",
        hovermode="x unified",
        height=400,
        yaxis=dict(range=[0, 1]),
    )
    
    return fig


def plot_regime_transitions(
    transition_matrix: np.ndarray,
    regime_labels: list[str] | None = None,
    title: str = "Regime Transition Probabilities",
) -> go.Figure:
    """
    Plot regime transition matrix as heatmap.
    
    Args:
        transition_matrix: (n_regimes, n_regimes) transition probabilities
        regime_labels: Labels for regimes
        title: Chart title
        
    Returns:
        Plotly Figure heatmap
        
    Example:
        >>> detector = RegimeDetector(n_regimes=3)
        >>> detector.fit(df)
        >>> fig = plot_regime_transitions(detector.get_transition_matrix())
        >>> fig.show()
    """
    n_regimes = transition_matrix.shape[0]
    
    if regime_labels is None:
        # Dynamic labels based on actual number of regimes
        default_labels = ["Bull", "Bear", "Crisis", "Volatile"]
        regime_labels = default_labels[:n_regimes]
    
    # Annotate with percentages
    annotations = []
    for i in range(n_regimes):
        for j in range(n_regimes):
            annotations.append(
                dict(
                    text=f"{transition_matrix[i, j]:.2%}",
                    x=regime_labels[j],
                    y=regime_labels[i],
                    xref="x",
                    yref="y",
                    showarrow=False,
                    font=dict(color="white" if transition_matrix[i, j] > 0.5 else "black"),
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=transition_matrix,
        x=regime_labels,
        y=regime_labels,
        colorscale="RdYlGn",
        text=transition_matrix,
        texttemplate="%{text:.2%}",
        textfont={"size": 12},
        hovertemplate="From %{y} → %{x}: %{z:.2%}<extra></extra>",
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="To Regime",
        yaxis_title="From Regime",
        height=400,
        width=500,
    )
    
    return fig


def plot_regime_performance(
    regime_stats: pd.DataFrame,
    title: str = "Performance by Regime",
) -> go.Figure:
    """
    Plot regime performance metrics (return vs volatility).
    
    Args:
        regime_stats: DataFrame with regime, mean_return, volatility
        title: Chart title
        
    Returns:
        Plotly Figure scatter plot
        
    Example:
        >>> from portfolio.features.regime_detection import compute_regime_performance
        >>> perf = compute_regime_performance(df_regimes)
        >>> fig = plot_regime_performance(perf)
        >>> fig.show()
    """
    # Dynamic colors supporting up to 4 regimes
    colors = {
        "Bull": "green",
        "Bear": "orange",
        "Crisis": "red",
        "Volatile": "purple",
        "Regime 0": "green",
        "Regime 1": "orange",
        "Regime 2": "red",
        "Regime 3": "purple",
    }
    
    fig = go.Figure()
    
    for _, row in regime_stats.iterrows():
        regime = row["regime"]
        color = colors.get(regime, "gray")
        
        fig.add_trace(go.Scatter(
            x=[row["volatility"] * 100],  # Convert to %
            y=[row["mean_return"] * 252 * 100],  # Annualized %
            mode="markers+text",
            name=regime,
            text=[regime],
            textposition="top center",
            marker=dict(
                size=row.get("frequency", 0.33) * 500,  # Size by frequency
                color=color,
                opacity=0.7,
            ),
            hovertemplate=(
                f"<b>{regime}</b><br>"
                "Volatility: %{x:.2f}%<br>"
                "Return: %{y:.2f}%<br>"
                f"Frequency: {row.get('frequency', 0):.1%}<extra></extra>"
            ),
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Volatility (Daily %)",
        yaxis_title="Return (Annualized %)",
        showlegend=False,
        height=500,
        width=600,
    )
    
    # Add quadrant lines (zero return, median vol)
    if len(regime_stats) > 0:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


def plot_regime_duration_histogram(
    df: pl.DataFrame,
    regime_col: str = "regime_label",
    title: str = "Regime Duration Distribution",
) -> go.Figure:
    """
    Plot histogram of regime stay durations.
    
    Args:
        df: DataFrame with regime labels
        regime_col: Column name for regime labels
        title: Chart title
        
    Returns:
        Plotly Figure histogram
    """
    df_pd = df.to_pandas()
    
    # Compute regime durations
    regime_durations = {regime: [] for regime in df_pd[regime_col].unique()}
    
    current_regime = None
    current_duration = 0
    
    for regime in df_pd[regime_col]:
        if regime == current_regime:
            current_duration += 1
        else:
            if current_regime is not None:
                regime_durations[current_regime].append(current_duration)
            current_regime = regime
            current_duration = 1
    
    # Add final regime
    if current_regime is not None:
        regime_durations[current_regime].append(current_duration)
    
    fig = go.Figure()
    
    # Dynamic colors supporting up to 4 regimes
    colors = {
        "Bull": "green",
        "Bear": "orange",
        "Crisis": "red",
        "Volatile": "purple",
        "Regime 0": "green",
        "Regime 1": "orange",
        "Regime 2": "red",
        "Regime 3": "purple",
    }
    
    for regime, durations in regime_durations.items():
        if durations:
            regime_str = str(regime)  # Convert to string for Plotly
            fig.add_trace(go.Histogram(
                x=durations,
                name=regime_str,  # Use string version
                marker_color=colors.get(regime_str, "gray"),
                opacity=0.7,
                hovertemplate=f"<b>{regime_str}</b><br>Duration: %{{x}} days<br>Count: %{{y}}<extra></extra>",
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Duration (Days)",
        yaxis_title="Frequency",
        barmode="overlay",
        height=400,
    )
    
    return fig
