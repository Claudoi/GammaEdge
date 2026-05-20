"""
Quantitative Metrics Chart Generation Module

This module provides chart generation functions for the Quant Metrics preview,
reusing existing plot_utils functions while maintaining consistent styling.
"""

import numpy as np
import plotly.graph_objects as go
import polars as pl

from portfolio.viz.plot_utils import (
    corr_heatmap,
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
                fillcolor="rgba(255,100,100,0.2)",
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


# ============================================================================
# Rolling Sharpe Ratio (Regime Detection)
# ============================================================================


def generate_rolling_sharpe(
    df_returns: pl.DataFrame,
    tickers: list[str],
    window: int = 252,  # 1 year
    rf_annual: float = 0.02,
    return_col_prefix: str = "ret_",
) -> go.Figure:
    """
    Generate Rolling Sharpe Ratio chart with regime bands.

    Args:
        df_returns: DataFrame with date and ret_{ticker} columns
        tickers: List of ticker symbols
        window: Rolling window size (default: 252 = 1 year)
        rf_annual: Annual risk-free rate
        return_col_prefix: Prefix for return columns

    Returns:
        Plotly figure with rolling Sharpe ratios

    Features:
        - Green zone: Sharpe > 1.0 (excellent)
        - Yellow zone: 0.5 < Sharpe < 1.0 (good)
        - Red zone: Sharpe < 0.5 (poor)
        - Detects regime changes
    """
    if df_returns.is_empty():
        return _placeholder_figure("Rolling Sharpe Ratio (252d)")

    fig = go.Figure()

    # Calculate daily risk-free rate
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1

    # Add rolling Sharpe for each ticker
    for ticker in tickers:
        ret_col = f"{return_col_prefix}{ticker}"
        if ret_col not in df_returns.columns:
            continue

        returns = df_returns[ret_col].to_numpy()
        dates = df_returns["date"].to_list()

        # Calculate rolling Sharpe
        rolling_sharpe = []
        for i in range(len(returns)):
            if i < window - 1:
                rolling_sharpe.append(np.nan)
            else:
                window_returns = returns[i - window + 1 : i + 1]
                window_returns = window_returns[~np.isnan(window_returns)]

                if len(window_returns) < window * 0.8:  # Require 80% data
                    rolling_sharpe.append(np.nan)
                else:
                    excess = window_returns - rf_daily
                    mean_excess = np.mean(excess)
                    std_returns = np.std(window_returns, ddof=1)

                    if std_returns < 1e-10:
                        rolling_sharpe.append(np.nan)
                    else:
                        sharpe = (mean_excess / std_returns) * np.sqrt(252)
                        rolling_sharpe.append(sharpe)

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rolling_sharpe,
                mode="lines",
                name=ticker,
                hovertemplate=f"{ticker}<br>%{{x}}<br>Sharpe: %{{y:.2f}}<extra></extra>",
            )
        )

    # Add regime bands
    fig.add_hrect(
        y0=1.0,
        y1=3.0,
        fillcolor="green",
        opacity=0.1,
        layer="below",
        line_width=0,
        annotation_text="Excellent (>1.0)",
        annotation_position="top right",
    )
    fig.add_hrect(
        y0=0.5,
        y1=1.0,
        fillcolor="yellow",
        opacity=0.1,
        layer="below",
        line_width=0,
        annotation_text="Good (0.5-1.0)",
        annotation_position="top right",
    )
    fig.add_hrect(
        y0=-2.0,
        y1=0.5,
        fillcolor="red",
        opacity=0.1,
        layer="below",
        line_width=0,
        annotation_text="Poor (<0.5)",
        annotation_position="bottom right",
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=f"Rolling Sharpe Ratio ({window}d window)",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=80, b=60),
    )

    return fig


# ============================================================================
# Returns Distribution (Histogram + Normal Fit)
# ============================================================================


def generate_returns_distribution(
    df_returns: pl.DataFrame,
    tickers: list[str],
    return_col_prefix: str = "ret_",
) -> go.Figure:
    """
    Generate Returns Distribution with histogram and normal fit.

    Args:
        df_returns: DataFrame with date and ret_{ticker} columns
        tickers: List of ticker symbols
        return_col_prefix: Prefix for return columns

    Returns:
        Plotly figure with returns distribution

    Features:
        - Histogram of actual returns
        - Normal distribution overlay
        - Skewness and Kurtosis annotations
        - Shows if returns are normally distributed
    """
    if df_returns.is_empty() or len(tickers) == 0:
        return _placeholder_figure("Returns Distribution")

    # Use first ticker for single distribution
    ticker = tickers[0]
    ret_col = f"{return_col_prefix}{ticker}"

    if ret_col not in df_returns.columns:
        return _placeholder_figure("Returns Distribution")

    returns = df_returns[ret_col].drop_nulls().to_numpy()

    if len(returns) < 30:
        return _placeholder_figure("Returns Distribution (insufficient data)")

    # Calculate statistics
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    # Skewness and Kurtosis
    from scipy import stats

    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns))  # Excess kurtosis

    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=returns * 100,  # Convert to percentage
            name="Actual",
            nbinsx=50,
            marker_color="rgba(0, 123, 255, 0.7)",
            hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        )
    )

    # Normal distribution overlay
    x_range = np.linspace(returns.min(), returns.max(), 100)
    normal_pdf = stats.norm.pdf(x_range, mean_ret, std_ret)

    # Scale to match histogram
    hist_counts, _ = np.histogram(returns, bins=50)
    scale_factor = len(returns) * (returns.max() - returns.min()) / 50
    normal_pdf_scaled = normal_pdf * scale_factor

    fig.add_trace(
        go.Scatter(
            x=x_range * 100,
            y=normal_pdf_scaled,
            mode="lines",
            name="Normal Fit",
            line=dict(color="red", width=2, dash="dash"),
            hovertemplate="Return: %{x:.2f}%<extra></extra>",
        )
    )

    # Add annotations
    annotation_text = (
        f"<b>Statistics</b><br>"
        f"Mean: {mean_ret*100:.2f}%<br>"
        f"Std: {std_ret*100:.2f}%<br>"
        f"Skew: {skew:.2f}<br>"
        f"Kurt: {kurt:.2f}"
    )

    fig.add_annotation(
        x=0.98,
        y=0.98,
        xref="paper",
        yref="paper",
        text=annotation_text,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="gray",
        borderwidth=1,
        xanchor="right",
        yanchor="top",
    )

    fig.update_layout(
        title=f"Returns Distribution - {ticker}",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=60, b=60),
        barmode="overlay",
    )

    return fig


# ============================================================================
# Monthly Returns Bar Chart (Consistency View)
# ============================================================================


def generate_monthly_returns_bar(
    df_returns: pl.DataFrame,
    tickers: list[str],
    return_col_prefix: str = "ret_",
) -> go.Figure:
    """
    Generate Monthly Returns Bar Chart.

    Args:
        df_returns: DataFrame with date and ret_{ticker} columns
        tickers: List of ticker symbols
        return_col_prefix: Prefix for return columns

    Returns:
        Plotly figure with monthly returns bars

    Features:
        - Green bars: Positive months
        - Red bars: Negative months
        - Shows consistency visually
        - Easier to interpret than numbers
    """
    if df_returns.is_empty() or len(tickers) == 0:
        return _placeholder_figure("Monthly Returns")

    # Use first ticker
    ticker = tickers[0]
    ret_col = f"{return_col_prefix}{ticker}"

    if ret_col not in df_returns.columns:
        return _placeholder_figure("Monthly Returns")

    # Convert to monthly returns
    df = df_returns.select(["date", ret_col])
    df = df.drop_nulls()

    if df.height == 0:
        return _placeholder_figure("Monthly Returns (no data)")

    # Add year-month columns
    df = df.with_columns(
        [
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        ]
    )

    # Group by year-month
    monthly = (
        df.group_by(["year", "month"])
        .agg([((1 + pl.col(ret_col)).product() - 1).alias("monthly_ret")])
        .sort(["year", "month"])
    )

    if monthly.height == 0:
        return _placeholder_figure("Monthly Returns (no monthly data)")

    # Create date labels
    monthly = monthly.with_columns(
        [
            (pl.col("year").cast(str) + "-" + pl.col("month").cast(str).str.zfill(2)).alias(
                "month_label"
            )
        ]
    )

    monthly_rets = monthly["monthly_ret"].to_numpy() * 100  # Convert to percentage
    month_labels = monthly["month_label"].to_list()

    # Color bars based on sign
    colors = ["green" if r > 0 else "red" for r in monthly_rets]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=month_labels,
            y=monthly_rets,
            marker_color=colors,
            name=ticker,
            hovertemplate="%{x}<br>Return: %{y:.2f}%<extra></extra>",
        )
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

    # Calculate average
    avg_monthly = np.mean(monthly_rets)
    fig.add_hline(
        y=avg_monthly,
        line_dash="dash",
        line_color="blue",
        line_width=2,
        annotation_text=f"Avg: {avg_monthly:.2f}%",
        annotation_position="right",
    )

    fig.update_layout(
        title=f"Monthly Returns - {ticker}",
        xaxis_title="Month",
        yaxis_title="Return (%)",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=60, r=20, t=60, b=60),
        xaxis=dict(tickangle=-45),
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
        return _placeholder_figure("Correlation Heatmap", subtitle="Requires at least 2 tickers")

    # Extract return columns
    ret_cols = [f"{return_col_prefix}{t}" for t in tickers]
    available_cols = [col for col in ret_cols if col in df_returns.columns]

    if len(available_cols) < 2:
        return _placeholder_figure(
            "Correlation Heatmap", subtitle="Insufficient data for correlation"
        )

    # Build correlation matrix
    returns_matrix = df_returns.select(available_cols).to_numpy()

    # Remove rows with any NaN
    mask = np.all(np.isfinite(returns_matrix), axis=1)
    returns_clean = returns_matrix[mask]

    if returns_clean.shape[0] < 10:
        return _placeholder_figure("Correlation Heatmap", subtitle="Insufficient overlapping data")

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
