"""
Quantitative Metrics Chart Generation Module

This module provides chart generation functions for the Quant Metrics preview,
reusing existing plot_utils functions while maintaining consistent styling.
"""

from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
import polars as pl

from portfolio.viz.plot_utils import (
    corr_heatmap,
    show_plot,
    apply_fig_defaults,
)


def generate_equity_curve_normalized(
    df_prices: pl.DataFrame,
    tickers: list[str],
    benchmark: str = "SPY",
    price_col_prefix: str = "adj_close_",
) -> go.Figure:
    """
    Generate normalized equity curves (base 100).
    
    Args:
        df_prices: DataFrame with date and price columns
        tickers: List of ticker symbols
        benchmark: Benchmark ticker (default: SPY)
        price_col_prefix: Prefix for price columns
    
    Returns:
        Plotly figure with normalized price series
    """
    if df_prices.is_empty():
        return _placeholder_figure("Equity Curve (Normalized to 100)")
    
    fig = go.Figure()
    
    # Add ticker curves
    for ticker in tickers:
        price_col = f"{price_col_prefix}{ticker}"
        if price_col not in df_prices.columns:
            continue
        
        prices = df_prices[price_col].to_numpy()
        dates = df_prices["date"].to_list()
        
        # Normalize to 100
        first_valid_idx = np.where(np.isfinite(prices))[0]
        if len(first_valid_idx) == 0:
            continue
        
        first_price = prices[first_valid_idx[0]]
        if first_price == 0:
            continue
        
        normalized = (prices / first_price) * 100
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=normalized,
                mode="lines",
                name=ticker,
                hovertemplate=f"{ticker}\u003cbr\u003e%{{x}}\u003cbr\u003e%{{y:.2f}}\u003cextra\u003e\u003c/extra\u003e",
            )
        )
    
    # Add benchmark (in gray)
    benchmark_col = f"{price_col_prefix}{benchmark}"
    if benchmark_col in df_prices.columns and benchmark not in tickers:
        prices = df_prices[benchmark_col].to_numpy()
        dates = df_prices["date"].to_list()
        
        first_valid_idx = np.where(np.isfinite(prices))[0]
        if len(first_valid_idx) > 0:
            first_price = prices[first_valid_idx[0]]
            if first_price != 0:
                normalized = (prices / first_price) * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=normalized,
                        mode="lines",
                        name=f"{benchmark} (benchmark)",
                        line=dict(color="rgba(150,150,150,0.6)", dash="dash"),
                        hovertemplate=f"{benchmark}\u003cbr\u003e%{{x}}\u003cbr\u003e%{{y:.2f}}\u003cextra\u003e\u003c/extra\u003e",
                    )
                )
    
    fig.update_layout(
        title="Equity Curve (Normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Indexed Value (Base 100)",
        margin=dict(l=60, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        template="plotly_white",
    )
    
    return fig


def generate_drawdown_chart(
    df_prices: pl.DataFrame,
    tickers: list[str],
    price_col_prefix: str = "adj_close_",
) -> go.Figure:
    """
    Generate standalone drawdown chart.
    
    Args:
        df_prices: DataFrame with date and price columns
        tickers: List of ticker symbols
        price_col_prefix: Prefix for price columns
    
    Returns:
        Plotly figure with drawdown curves
    """
    if df_prices.is_empty():
        return _placeholder_figure("Drawdown Analysis")
    
    fig = go.Figure()
    
    for ticker in tickers:
        price_col = f"{price_col_prefix}{ticker}"
        if price_col not in df_prices.columns:
            continue
        
        prices = df_prices[price_col].to_numpy()
        dates = df_prices["date"].to_list()
        
        # Calculate drawdown
        drawdown = np.full_like(prices, np.nan, dtype=float)
        peak = -np.inf
        
        for i, price in enumerate(prices):
            if np.isfinite(price):
                if price > peak:
                    peak = price
                if np.isfinite(peak) and peak > 0:
                    drawdown[i] = (price / peak - 1.0) * 100  # Convert to percentage
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown,
                mode="lines",
                name=ticker,
                fill="tozeroy",
                fillcolor=f"rgba(255,100,100,0.2)",
                hovertemplate=f"{ticker}\u003cbr\u003e%{{x}}\u003cbr\u003e%{{y:.2f}}%\u003cextra\u003e\u003c/extra\u003e",
            )
        )
        
        # Add MDD horizontal line
        mdd = np.nanmin(drawdown)
        if np.isfinite(mdd):
            fig.add_hline(
                y=mdd,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{ticker} MDD: {mdd:.2f}%",
                annotation_position="right",
            )
    
    fig.update_layout(
        title="Drawdown Analysis",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        margin=dict(l=60, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        template="plotly_white",
        yaxis=dict(ticksuffix="%"),
    )
    
    return fig


def generate_rolling_volatility(
    df_returns: pl.DataFrame,
    tickers: list[str],
    window: int = 30,
    return_col_prefix: str = "ret_",
) -> go.Figure:
    """
    Generate rolling volatility chart (annualized).
    
    Args:
        df_returns: DataFrame with date and return columns
        tickers: List of ticker symbols
        window: Rolling window size (default: 30 days)
        return_col_prefix: Prefix for return columns
    
    Returns:
        Plotly figure with rolling volatility series
    """
    if df_returns.is_empty():
        return _placeholder_figure("Rolling Volatility (30d)")
    
    fig = go.Figure()
    
    for ticker in tickers:
        ret_col = f"{return_col_prefix}{ticker}"
        if ret_col not in df_returns.columns:
            continue
        
        returns = df_returns[ret_col].to_numpy()
        dates = df_returns["date"].to_list()
        
        # Calculate rolling volatility (annualized)
        rolling_vol = np.full(len(returns), np.nan)
        
        for i in range(window - 1, len(returns)):
            window_returns = returns[i - window + 1 : i + 1]
            window_returns = window_returns[np.isfinite(window_returns)]
            
            if len(window_returns) >= window // 2:  # At least half the window
                vol = np.std(window_returns, ddof=1) * np.sqrt(252)  # Annualize
                rolling_vol[i] = vol * 100  # Convert to percentage
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rolling_vol,
                mode="lines",
                name=ticker,
                hovertemplate=f"{ticker}\u003cbr\u003e%{{x}}\u003cbr\u003e%{{y:.2f}}%\u003cextra\u003e\u003c/extra\u003e",
            )
        )
    
    fig.update_layout(
        title=f"Rolling Volatility ({window}d, Annualized)",
        xaxis_title="Date",
        yaxis_title="Volatility (% p.a.)",
        margin=dict(l=60, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        template="plotly_white",
        yaxis=dict(ticksuffix="%"),
    )
    
    return fig


def generate_correlation_heatmap_preview(
    df_returns: pl.DataFrame,
    tickers: list[str],
    return_col_prefix: str = "ret_",
) -> go.Figure:
    """
    Generate correlation heatmap for preview.
    
    Wrapper around plot_utils.corr_heatmap() with data preparation.
    
    Args:
        df_returns: DataFrame with date and return columns
        tickers: List of ticker symbols
        return_col_prefix: Prefix for return columns
    
    Returns:
        Plotly figure with correlation heatmap
    """
    if df_returns.is_empty() or len(tickers) < 2:
        return _placeholder_figure(
            "Correlation Heatmap",
            subtitle="Requires at least 2 tickers"
        )
    
    # Extract return columns
    ret_cols = [f"{return_col_prefix}{t}" for t in tickers]
    available_cols = [col for col in ret_cols if col in df_returns.columns]
    
    if len(available_cols) < 2:
        return _placeholder_figure(
            "Correlation Heatmap",
            subtitle="Insufficient data for correlation"
        )
    
    # Build correlation matrix
    returns_matrix = df_returns.select(available_cols).to_numpy()
    
    # Remove rows with any NaN
    mask = np.all(np.isfinite(returns_matrix), axis=1)
    returns_clean = returns_matrix[mask]
    
    if returns_clean.shape[0] < 10:
        return _placeholder_figure(
            "Correlation Heatmap",
            subtitle="Insufficient overlapping data"
        )
    
    # Calculate correlation
    corr_matrix = np.corrcoef(returns_clean.T)
    
    # Extract ticker names from column names
    ticker_labels = [col.replace(return_col_prefix, "") for col in available_cols]
    
    # Use existing corr_heatmap function
    fig = corr_heatmap(
        corr_matrix,
        labels=ticker_labels,
        is_cov=False,  # Already correlation matrix
        title="Correlation Heatmap (Returns)",
    )
    
    return fig


def _placeholder_figure(title: str, subtitle: str = "No data available") -> go.Figure:
    """Generate placeholder figure when data is unavailable."""
    fig = go.Figure()
    fig.add_annotation(
        text=subtitle,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        align="center",
        font=dict(size=14),
    )
    fig.update_layout(
        title=title,
        margin=dict(l=60, r=20, t=60, b=60),
        showlegend=False,
        template="plotly_white",
    )
    return fig
