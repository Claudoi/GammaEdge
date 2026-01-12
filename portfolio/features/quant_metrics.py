"""
Production-Grade Quantitative Metrics Module

This module provides institutional-quality financial metrics calculations
with rigorous definitions, proper data alignment, and comprehensive edge case handling.

All metrics are based on adjusted close prices (dividend and split adjusted).
Returns are simple returns (not log returns) calculated as: r_t = (P_t - P_{t-1}) / P_{t-1}

Standards:
    - Annualization: 252 trading days
    - Risk-free rate: 2% annual → 0.00007926 daily
    - Benchmark: SPY
    - Calendar: NYSE (proxied by SPY dates)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl


# ============================================================================
# Constants
# ============================================================================

TRADING_DAYS_PER_YEAR = 252
RF_ANNUAL_DEFAULT = 0.02
RF_DAILY_DEFAULT = (1 + RF_ANNUAL_DEFAULT) ** (1 / TRADING_DAYS_PER_YEAR) - 1  # 0.00007926


# ============================================================================
# Internal Helpers
# ============================================================================


def _as_pl_date(s: pl.Series) -> pl.Series:
    """
    Normalize any date-like Series to Polars Date.
    
    Handles:
      - pl.Date (already correct)
      - pl.Datetime (cast to Date)
      - Python date/datetime stored as object
      - ISO strings
    
    This prevents "Hash Inner Join between object and object" errors.
    """
    if s.dtype == pl.Date:
        return s

    if s.dtype == pl.Datetime:
        return s.cast(pl.Date)

    # If it's already Utf8 (strings), parse
    if s.dtype == pl.Utf8:
        return s.str.strptime(pl.Date, strict=False)

    # Object or other: try to stringify then parse
    # This avoids "Hash Inner Join between object and object"
    return s.cast(pl.Utf8).str.strptime(pl.Date, strict=False)


# ============================================================================
# Returns Calculation
# ============================================================================


def calculate_returns(
    prices_wide: pl.DataFrame,
    price_col_prefix: str = "adj_close_",
) -> pl.DataFrame:
    """
    Calculate simple returns from adjusted close prices.

    Formula: r_t = (P_t - P_{t-1}) / P_{t-1}

    Args:
        prices_wide: DataFrame with date and adj_close_{ticker} columns
        price_col_prefix: Prefix for price columns

    Returns:
        DataFrame with date and ret_{ticker} columns

    Notes:
        - First row will have NaN returns
        - Returns are NOT annualized
        - Uses simple returns, not log returns

    Example:
        >>> prices = pl.DataFrame({
        ...     "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        ...     "adj_close_AAPL": [100.0, 102.0, 101.0]
        ... })
        >>> returns = calculate_returns(prices)
        >>> # Returns: [NaN, 0.02, -0.0098]
    """
    price_cols = [col for col in prices_wide.columns if col.startswith(price_col_prefix)]

    if not price_cols:
        raise ValueError(f"No columns found with prefix '{price_col_prefix}'")

    # Calculate returns for each ticker
    returns_exprs = [pl.col("date")]

    for price_col in price_cols:
        ticker = price_col.replace(price_col_prefix, "")
        ret_col = f"ret_{ticker}"

        # Simple return: (P_t - P_{t-1}) / P_{t-1}
        returns_exprs.append(
            (pl.col(price_col) / pl.col(price_col).shift(1) - 1.0).alias(ret_col)
        )

    return prices_wide.select(returns_exprs)


# ============================================================================
# Beta & Alpha (vs Benchmark)
# ============================================================================


def calculate_beta_alpha(
    returns: pl.Series,
    benchmark_returns: pl.Series,
    dates: pl.Series,
    benchmark_dates: pl.Series,
    min_obs: int = 30,
) -> dict[str, float | None]:
    """
    Calculate Beta and Alpha via OLS regression.

    Model: r_i = alpha + beta * r_m + epsilon

    Args:
        returns: Ticker returns (daily)
        benchmark_returns: Benchmark returns (daily)
        dates: Dates for ticker
        benchmark_dates: Dates for benchmark
        min_obs: Minimum observations required

    Returns:
        {
            "beta": float | None,
            "alpha_daily": float | None,
            "alpha_annual": float | None,  # alpha_daily * 252
            "r_squared": float | None,
            "n_obs": int,
            "data_loss_pct": float,  # % lost in inner join
        }

    Notes:
        - Inner join on dates (aligned)
        - If Var(r_m) < 1e-10 → all None
        - If n_obs < min_obs → all None
        - alpha_annual = alpha_daily * 252 (approximation)
        - Uses closed-form OLS: beta = Cov(r_i, r_m) / Var(r_m)

    Example:
        >>> result = calculate_beta_alpha(
        ...     returns=ticker_returns,
        ...     benchmark_returns=spy_returns,
        ...     dates=ticker_dates,
        ...     benchmark_dates=spy_dates,
        ... )
        >>> print(result["beta"])  # e.g., 1.15
    """
    # Normalize dates to pl.Date to avoid join errors
    d_ticker = _as_pl_date(dates).alias("date")
    d_bench = _as_pl_date(benchmark_dates).alias("date")

    # Create DataFrames
    df_ticker = pl.DataFrame({"date": d_ticker, "r_i": returns})
    df_benchmark = pl.DataFrame({"date": d_bench, "r_m": benchmark_returns})

    # Drop null returns (first row is often null)
    df_ticker = df_ticker.drop_nulls(["date", "r_i"])
    df_benchmark = df_benchmark.drop_nulls(["date", "r_m"])

    n_ticker = df_ticker.height

    # Inner join on dates
    df_aligned = df_ticker.join(df_benchmark, on="date", how="inner")

    n_aligned = df_aligned.height
    data_loss_pct = 100.0 * (1.0 - (n_aligned / n_ticker)) if n_ticker > 0 else 0.0

    # Check minimum observations
    if n_aligned < min_obs:
        return {
            "beta": None,
            "alpha_daily": None,
            "alpha_annual": None,
            "r_squared": None,
            "n_obs": n_aligned,
            "data_loss_pct": data_loss_pct,
        }

    # Convert to numpy for calculations
    r_i = df_aligned["r_i"].to_numpy()
    r_m = df_aligned["r_m"].to_numpy()

    n_obs = len(r_i)

    # Calculate variance of benchmark
    var_m = float(r_m.var(ddof=0))

    # Guard against zero variance
    if var_m < 1e-10:
        return {
            "beta": None,
            "alpha_daily": None,
            "alpha_annual": None,
            "r_squared": None,
            "n_obs": n_obs,
            "data_loss_pct": data_loss_pct,
        }

    # OLS: beta = Cov(r_i, r_m) / Var(r_m)
    cov_im = float(((r_i - r_i.mean()) * (r_m - r_m.mean())).mean())
    beta = cov_im / var_m

    # alpha = mean(r_i) - beta * mean(r_m)
    alpha_daily = float(r_i.mean() - beta * r_m.mean())
    alpha_annual = alpha_daily * TRADING_DAYS_PER_YEAR

    # R-squared
    y_pred = alpha_daily + beta * r_m
    ss_res = float(((r_i - y_pred) ** 2).sum())
    ss_tot = float(((r_i - r_i.mean()) ** 2).sum())
    r_squared = None if ss_tot < 1e-20 else 1.0 - (ss_res / ss_tot)

    return {
        "beta": float(beta),
        "alpha_daily": float(alpha_daily),
        "alpha_annual": float(alpha_annual),
        "r_squared": None if r_squared is None else float(r_squared),
        "n_obs": int(n_obs),
        "data_loss_pct": float(data_loss_pct),
    }


# ============================================================================
# Sharpe Ratio
# ============================================================================


def calculate_sharpe_ratio(
    returns: pl.Series,
    rf_annual: float = RF_ANNUAL_DEFAULT,
    min_obs: int = 60,
    annualize: bool = True,
) -> dict[str, float | None]:
    """
    Calculate Sharpe Ratio.

    Formula: SR = mean(r - rf) / std(r) * sqrt(252)

    Args:
        returns: Daily returns
        rf_annual: Annual risk-free rate
        min_obs: Minimum observations
        annualize: If True, multiply by sqrt(252)

    Returns:
        {
            "sharpe_ratio": float | None,
            "mean_excess_return": float,  # Daily
            "volatility": float,  # Daily std
            "n_obs": int,
            "warning": str | None,
        }

    Notes:
        - rf_daily = (1 + rf_annual)**(1/252) - 1
        - If std(r) < 1e-10 → None
        - If n_obs < min_obs → None + warning
        - Uses std(returns), not std(excess_returns)

    Example:
        >>> result = calculate_sharpe_ratio(returns)
        >>> print(result["sharpe_ratio"])  # e.g., 0.82 (annualized)
    """
    # Convert to numpy and remove NaN
    r = returns.to_numpy()
    r = r[~np.isnan(r)]

    n_obs = len(r)
    warning = None

    # Check minimum observations
    if n_obs < min_obs:
        warning = f"Sample size ({n_obs}) < minimum ({min_obs}), Sharpe unreliable"
        return {
            "sharpe_ratio": None,
            "mean_excess_return": None,
            "volatility": None,
            "n_obs": n_obs,
            "warning": warning,
        }

    # Calculate daily risk-free rate
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    # Excess returns
    excess_returns = r - rf_daily
    mean_excess = np.mean(excess_returns)

    # Volatility (std of returns, not excess returns)
    volatility = np.std(r, ddof=1)

    # Guard against zero volatility
    if volatility < 1e-10:
        return {
            "sharpe_ratio": None,
            "mean_excess_return": float(mean_excess),
            "volatility": float(volatility),
            "n_obs": n_obs,
            "warning": "Zero volatility",
        }

    # Sharpe ratio
    sharpe = mean_excess / volatility

    if annualize:
        sharpe *= np.sqrt(TRADING_DAYS_PER_YEAR)

    return {
        "sharpe_ratio": float(sharpe),
        "mean_excess_return": float(mean_excess),
        "volatility": float(volatility),
        "n_obs": n_obs,
        "warning": warning,
    }


# ============================================================================
# Maximum Drawdown
# ============================================================================


def calculate_max_drawdown(
    prices: pl.Series,
    dates: pl.Series | None = None,
) -> dict[str, float | str | None]:
    """
    Calculate Maximum Drawdown (negative).

    Formula: MDD = min((equity_t / peak_t) - 1)

    Args:
        prices: Price series (adjusted close)
        dates: Optional date series for tracking peak/trough dates

    Returns:
        {
            "max_drawdown": float,  # Negative (e.g., -0.35 for -35%)
            "peak_date": str | None,
            "trough_date": str | None,
            "recovery_date": str | None,
            "underwater_days": int,
        }

    Notes:
        - Always ≤ 0
        - If prices monotonically increase → 0.0
        - Calculated on equity curve (cumulative returns)

    Example:
        >>> prices = pl.Series([100, 120, 90, 110])
        >>> result = calculate_max_drawdown(prices)
        >>> print(result["max_drawdown"])  # -0.25 (-25%)
    """
    # Convert to numpy
    p = prices.to_numpy()
    p = p[~np.isnan(p)]

    if len(p) == 0:
        return {
            "max_drawdown": 0.0,
            "peak_date": None,
            "trough_date": None,
            "recovery_date": None,
            "underwater_days": 0,
        }

    # Calculate running maximum (peak)
    running_max = np.maximum.accumulate(p)

    # Drawdown at each point: (equity_t / peak_t) - 1
    drawdown = (p / running_max) - 1.0

    # Maximum drawdown (most negative)
    mdd = np.min(drawdown)

    # Find peak and trough indices
    mdd_idx = np.argmin(drawdown)
    peak_idx = np.argmax(running_max[:mdd_idx + 1]) if mdd_idx > 0 else 0

    # Find recovery (if any)
    recovery_idx = None
    if mdd_idx < len(p) - 1:
        future_prices = p[mdd_idx + 1:]
        peak_price = running_max[mdd_idx]
        recovery_mask = future_prices >= peak_price
        if np.any(recovery_mask):
            recovery_idx = mdd_idx + 1 + np.argmax(recovery_mask)

    # Calculate underwater days
    underwater_days = mdd_idx - peak_idx

    # Get dates if provided
    peak_date = None
    trough_date = None
    recovery_date = None

    if dates is not None:
        dates_arr = dates.to_list()
        peak_date = str(dates_arr[peak_idx]) if peak_idx < len(dates_arr) else None
        trough_date = str(dates_arr[mdd_idx]) if mdd_idx < len(dates_arr) else None
        if recovery_idx is not None and recovery_idx < len(dates_arr):
            recovery_date = str(dates_arr[recovery_idx])

    return {
        "max_drawdown": float(mdd),
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "underwater_days": int(underwater_days),
    }


# ============================================================================
# CAGR
# ============================================================================


def calculate_cagr(
    prices: pl.Series,
    dates: pl.Series,
    min_days: int = 252,
) -> dict[str, float | None]:
    """
    Calculate Compound Annual Growth Rate.

    Formula: CAGR = (P_end / P_start)**(252 / n_trading_days) - 1

    Args:
        prices: Price series
        dates: Date series
        min_days: Minimum trading days required

    Returns:
        {
            "cagr": float | None,
            "n_days": int,
            "start_price": float,
            "end_price": float,
            "total_return": float,
        }

    Notes:
        - n_days = count of returns (len(prices) - 1)
        - If n_days < min_days → None

    Example:
        >>> result = calculate_cagr(prices, dates)
        >>> print(result["cagr"])  # e.g., 0.185 (18.5% annual)
    """
    # Convert to numpy and remove NaN
    p = prices.to_numpy()
    mask = ~np.isnan(p)
    p = p[mask]

    n_obs = len(p)
    n_days = n_obs - 1  # Number of returns

    if n_days < min_days:
        return {
            "cagr": None,
            "n_days": n_days,
            "start_price": None,
            "end_price": None,
            "total_return": None,
        }

    start_price = p[0]
    end_price = p[-1]
    total_return = (end_price / start_price) - 1.0

    # CAGR = (P_end / P_start)**(252 / n_days) - 1
    cagr = (end_price / start_price) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0

    return {
        "cagr": float(cagr),
        "n_days": n_days,
        "start_price": float(start_price),
        "end_price": float(end_price),
        "total_return": float(total_return),
    }


# ============================================================================
# Calmar Ratio
# ============================================================================


def calculate_calmar(
    cagr: float | None,
    mdd: float,
    min_days: int = 252,
) -> float | None:
    """
    Calculate Calmar Ratio.

    Formula: Calmar = CAGR / |MDD|

    Args:
        cagr: CAGR value
        mdd: Maximum drawdown (negative)
        min_days: Minimum days (for guard)

    Returns:
        Calmar ratio or None

    Notes:
        - If |MDD| < 1e-6 → None
        - If CAGR is None → None

    Example:
        >>> calmar = calculate_calmar(cagr=0.185, mdd=-0.35)
        >>> print(calmar)  # 0.53
    """
    if cagr is None:
        return None

    abs_mdd = abs(mdd)

    # Guard against zero or near-zero drawdown
    if abs_mdd < 1e-6:
        return None

    return cagr / abs_mdd


# ============================================================================
# Distribution Moments
# ============================================================================


def calculate_moments(
    returns: pl.Series,
) -> dict[str, float]:
    """
    Calculate Skewness and Kurtosis.

    Args:
        returns: Return series

    Returns:
        {
            "skewness": float,  # Fisher-Pearson adjusted
            "kurtosis": float,  # Excess kurtosis (normal = 0)
            "n_obs": int,
        }

    Notes:
        - Skewness: Fisher-Pearson adjusted (unbiased)
        - Kurtosis: Excess kurtosis (normal distribution = 0)
        - Uses numpy implementation (no scipy dependency)

    Example:
        >>> result = calculate_moments(returns)
        >>> print(result["skewness"])  # e.g., -0.12
        >>> print(result["kurtosis"])  # e.g., 3.2 (excess)
    """
    # Convert to numpy and remove NaN
    r = returns.to_numpy()
    r = r[~np.isnan(r)]

    n = len(r)

    if n < 3:
        return {"skewness": np.nan, "kurtosis": np.nan, "n_obs": n}

    # Calculate moments
    mean = np.mean(r)
    std = np.std(r, ddof=1)

    if std < 1e-10:
        return {"skewness": np.nan, "kurtosis": np.nan, "n_obs": n}

    # Skewness (Fisher-Pearson adjusted)
    m3 = np.mean(((r - mean) / std) ** 3)
    skewness = m3 * (n * (n - 1)) ** 0.5 / (n - 2) if n > 2 else m3

    # Kurtosis (excess)
    m4 = np.mean(((r - mean) / std) ** 4)
    kurtosis = m4 - 3.0  # Excess kurtosis

    # Adjust for sample bias
    if n > 3:
        kurtosis = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * kurtosis + 6)

    return {
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "n_obs": n,
    }


# ============================================================================
# Correlation Matrix
# ============================================================================


def calculate_correlation_matrix(
    returns_wide: pl.DataFrame,
    return_col_prefix: str = "ret_",
) -> dict[str, pl.DataFrame | None]:
    """
    Calculate pairwise correlation matrix on returns.

    Args:
        returns_wide: DataFrame with date and ret_{ticker} columns
        return_col_prefix: Prefix for return columns

    Returns:
        {
            "correlation_matrix": pl.DataFrame | None,  # None if single ticker
            "sample_sizes": pl.DataFrame | None,  # N per pair
            "method": str,  # "pearson"
        }

    Notes:
        - Inner join on dates per pair
        - Diagonal = 1.0
        - If only 1 ticker → None
        - Calculated on returns, not prices

    Example:
        >>> result = calculate_correlation_matrix(returns_wide)
        >>> print(result["correlation_matrix"])
        #        AAPL   MSFT  GOOGL
        # AAPL   1.00   0.85   0.78
        # MSFT   0.85   1.00   0.82
        # GOOGL  0.78   0.82   1.00
    """
    ret_cols = [col for col in returns_wide.columns if col.startswith(return_col_prefix)]

    if len(ret_cols) < 2:
        return {
            "correlation_matrix": None,
            "sample_sizes": None,
            "method": "pearson",
        }

    tickers = [col.replace(return_col_prefix, "") for col in ret_cols]

    # Calculate correlation matrix
    n_tickers = len(tickers)
    corr_matrix = np.ones((n_tickers, n_tickers))
    sample_sizes = np.zeros((n_tickers, n_tickers), dtype=int)

    for i, ticker_i in enumerate(tickers):
        for j, ticker_j in enumerate(tickers):
            if i == j:
                sample_sizes[i, j] = len(returns_wide)
                continue

            # Get returns for both tickers
            col_i = f"{return_col_prefix}{ticker_i}"
            col_j = f"{return_col_prefix}{ticker_j}"

            df_pair = returns_wide.select(["date", col_i, col_j]).drop_nulls()

            r_i = df_pair[col_i].to_numpy()
            r_j = df_pair[col_j].to_numpy()

            n_obs = len(r_i)
            sample_sizes[i, j] = n_obs

            if n_obs < 2:
                corr_matrix[i, j] = np.nan
            else:
                corr_matrix[i, j] = np.corrcoef(r_i, r_j)[0, 1]

    # Create DataFrames
    corr_df = pl.DataFrame(corr_matrix, schema=tickers)
    corr_df = corr_df.with_columns(pl.Series("ticker", tickers)).select(
        ["ticker"] + tickers
    )

    sample_df = pl.DataFrame(sample_sizes, schema=tickers)
    sample_df = sample_df.with_columns(pl.Series("ticker", tickers)).select(
        ["ticker"] + tickers
    )

    return {
        "correlation_matrix": corr_df,
        "sample_sizes": sample_df,
        "method": "pearson",
    }


# ============================================================================
# Data Quality Metrics
# ============================================================================


def calculate_data_quality(
    dates: pl.Series,
    benchmark_dates: pl.Series,
    ticker: str,
) -> dict[str, any]:
    """
    Calculate data coverage and gaps.

    Args:
        dates: Ticker dates
        benchmark_dates: Benchmark dates (calendar proxy, e.g., SPY)
        ticker: Ticker symbol

    Returns:
        {
            "ticker": str,
            "first_date": str,
            "last_date": str,
            "n_obs": int,
            "expected_obs": int,  # From benchmark calendar
            "coverage_pct": float,
            "max_gap_days": int,
            "missing_blocks": int,
            "warnings": list[str],
        }

    Notes:
        - Expected obs = len(benchmark_dates in range)
        - Missing block = consecutive missing trading days
        - Uses benchmark (SPY) as NYSE calendar proxy

    Example:
        >>> result = calculate_data_quality(aapl_dates, spy_dates, "AAPL")
        >>> print(result["coverage_pct"])  # 99.8
    """
    # Convert to lists
    ticker_dates = set(dates.to_list())
    bench_dates = benchmark_dates.to_list()

    if not ticker_dates:
        return {
            "ticker": ticker,
            "first_date": None,
            "last_date": None,
            "n_obs": 0,
            "expected_obs": 0,
            "coverage_pct": 0.0,
            "max_gap_days": 0,
            "missing_blocks": 0,
            "warnings": ["No data available"],
        }

    # Date range
    first_date = min(ticker_dates)
    last_date = max(ticker_dates)

    # Expected dates from benchmark in range
    expected_dates = [d for d in bench_dates if first_date <= d <= last_date]
    expected_obs = len(expected_dates)

    n_obs = len(ticker_dates)
    coverage_pct = (n_obs / expected_obs * 100) if expected_obs > 0 else 0.0

    # Find gaps
    missing_dates = [d for d in expected_dates if d not in ticker_dates]

    # Calculate max gap and missing blocks
    max_gap = 0
    missing_blocks = 0

    if missing_dates:
        missing_dates_sorted = sorted(missing_dates)
        current_block = 1
        missing_blocks = 1

        for i in range(1, len(missing_dates_sorted)):
            # Check if consecutive in benchmark calendar
            prev_idx = bench_dates.index(missing_dates_sorted[i - 1])
            curr_idx = bench_dates.index(missing_dates_sorted[i])

            if curr_idx == prev_idx + 1:
                current_block += 1
            else:
                max_gap = max(max_gap, current_block)
                missing_blocks += 1
                current_block = 1

        max_gap = max(max_gap, current_block)

    # Warnings
    warnings = []
    if coverage_pct < 80:
        warnings.append(f"Low coverage: {coverage_pct:.1f}%")
    if max_gap > 5:
        warnings.append(f"Large gap detected: {max_gap} days")

    return {
        "ticker": ticker,
        "first_date": str(first_date),
        "last_date": str(last_date),
        "n_obs": n_obs,
        "expected_obs": expected_obs,
        "coverage_pct": float(coverage_pct),
        "max_gap_days": max_gap,
        "missing_blocks": missing_blocks,
        "warnings": warnings,
    }


# ============================================================================
# Sortino Ratio (Downside Risk Focus)
# ============================================================================


def calculate_sortino_ratio(
    returns: pl.Series,
    rf_annual: float = RF_ANNUAL_DEFAULT,
    target_return: float = 0.0,
    min_obs: int = 60,
    annualize: bool = True,
) -> dict[str, float | None]:
    """
    Calculate Sortino Ratio (penalizes only downside volatility).

    Formula: Sortino = (mean_return - target) / downside_deviation

    Args:
        returns: Return series
        rf_annual: Annual risk-free rate
        target_return: Target return threshold (default: 0)
        min_obs: Minimum observations required
        annualize: Whether to annualize the ratio

    Returns:
        {
            "sortino_ratio": float | None,
            "downside_deviation": float | None,
            "warning": str | None,
        }

    Notes:
        - Only penalizes volatility below target return
        - Better than Sharpe for assets with positive skew
        - Downside deviation uses only returns < target

    Example:
        >>> result = calculate_sortino_ratio(returns, rf_annual=0.02)
        >>> print(result["sortino_ratio"])  # 1.45
    """
    r = returns.drop_nulls().to_numpy()

    if r.size < min_obs:
        return {
            "sortino_ratio": None,
            "downside_deviation": None,
            "warning": f"Insufficient data for Sortino ({r.size} < {min_obs})",
        }

    # Calculate excess returns
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess_returns = r - rf_daily

    # Downside deviation (only negative excess returns)
    downside_returns = excess_returns[excess_returns < target_return]

    if downside_returns.size < 2:
        return {
            "sortino_ratio": None,
            "downside_deviation": None,
            "warning": "No downside returns for Sortino calculation",
        }

    downside_dev = np.std(downside_returns, ddof=1)

    if downside_dev < 1e-10:
        return {
            "sortino_ratio": None,
            "downside_deviation": 0.0,
            "warning": "Zero downside deviation",
        }

    # Calculate Sortino
    mean_excess = np.mean(excess_returns)
    sortino = mean_excess / downside_dev

    if annualize:
        sortino *= np.sqrt(TRADING_DAYS_PER_YEAR)
        downside_dev_ann = downside_dev * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        downside_dev_ann = downside_dev

    return {
        "sortino_ratio": float(sortino),
        "downside_deviation": float(downside_dev_ann),
        "warning": None,
    }


# ============================================================================
# Information Ratio (vs Benchmark)
# ============================================================================


def calculate_information_ratio(
    returns: pl.Series,
    benchmark_returns: pl.Series,
    dates: pl.Series,
    benchmark_dates: pl.Series,
    min_obs: int = 30,
) -> dict[str, float | None]:
    """
    Calculate Information Ratio (excess return / tracking error).

    Formula: IR = (R_p - R_b) / σ(R_p - R_b)

    Args:
        returns: Portfolio/ticker returns
        benchmark_returns: Benchmark returns
        dates: Portfolio dates
        benchmark_dates: Benchmark dates
        min_obs: Minimum observations required

    Returns:
        {
            "information_ratio": float | None,
            "tracking_error": float | None (annualized),
            "excess_return_annual": float | None,
            "data_loss_pct": float,
            "warning": str | None,
        }

    Notes:
        - Measures manager skill vs benchmark
        - IR > 0.5 is good, > 1.0 is excellent
        - Tracking error is annualized std of excess returns

    Example:
        >>> result = calculate_information_ratio(aapl_ret, spy_ret, dates, bench_dates)
        >>> print(result["information_ratio"])  # 0.82
    """
    # Align dates (inner join)
    d_ticker = _as_pl_date(dates).alias("date")
    d_bench = _as_pl_date(benchmark_dates).alias("date")

    df_ticker = pl.DataFrame({"date": d_ticker, "ret": returns})
    df_benchmark = pl.DataFrame({"date": d_bench, "ret_bench": benchmark_returns})

    df_aligned = df_ticker.join(df_benchmark, on="date", how="inner")

    n_aligned = df_aligned.height
    n_original = len(dates)
    data_loss_pct = 100.0 * (1.0 - n_aligned / max(n_original, 1))

    if n_aligned < min_obs:
        return {
            "information_ratio": None,
            "tracking_error": None,
            "excess_return_annual": None,
            "data_loss_pct": data_loss_pct,
            "warning": f"Insufficient aligned data ({n_aligned} < {min_obs})",
        }

    # Calculate excess returns
    r_p = df_aligned["ret"].to_numpy()
    r_b = df_aligned["ret_bench"].to_numpy()
    excess = r_p - r_b

    # Tracking error (annualized)
    tracking_error = np.std(excess, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

    if tracking_error < 1e-10:
        return {
            "information_ratio": None,
            "tracking_error": 0.0,
            "excess_return_annual": float(np.mean(excess) * TRADING_DAYS_PER_YEAR),
            "data_loss_pct": data_loss_pct,
            "warning": "Zero tracking error (identical to benchmark)",
        }

    # Information Ratio
    excess_return_annual = np.mean(excess) * TRADING_DAYS_PER_YEAR
    information_ratio = excess_return_annual / tracking_error

    return {
        "information_ratio": float(information_ratio),
        "tracking_error": float(tracking_error),
        "excess_return_annual": float(excess_return_annual),
        "data_loss_pct": data_loss_pct,
        "warning": None,
    }


# ============================================================================
# Tail Ratio (Upside/Downside Tails)
# ============================================================================


def calculate_tail_ratio(
    returns: pl.Series,
    percentile: float = 0.95,
) -> dict[str, float]:
    """
    Calculate Tail Ratio (upside tail / downside tail).

    Formula: Tail Ratio = abs(95th percentile) / abs(5th percentile)

    Args:
        returns: Return series
        percentile: Percentile for tail definition (default: 0.95)

    Returns:
        {
            "tail_ratio": float,
            "upside_tail": float (95th percentile),
            "downside_tail": float (5th percentile),
        }

    Notes:
        - Tail Ratio > 1: Upside tails dominate (good)
        - Tail Ratio < 1: Downside tails dominate (bad)
        - Complements skewness with interpretable metric

    Example:
        >>> result = calculate_tail_ratio(returns)
        >>> print(result["tail_ratio"])  # 1.18 (upside tails 18% larger)
    """
    r = returns.drop_nulls().to_numpy()

    if r.size < 20:
        return {
            "tail_ratio": np.nan,
            "upside_tail": np.nan,
            "downside_tail": np.nan,
        }

    # Calculate percentiles
    upside_tail = float(np.percentile(r, percentile * 100))
    downside_tail = float(np.percentile(r, (1 - percentile) * 100))

    # Tail ratio
    if abs(downside_tail) < 1e-10:
        tail_ratio = np.nan
    else:
        tail_ratio = abs(upside_tail) / abs(downside_tail)

    return {
        "tail_ratio": float(tail_ratio),
        "upside_tail": upside_tail,
        "downside_tail": downside_tail,
    }


# ============================================================================
# Win Metrics (Win Rate, Profit Factor, Avg Win/Loss)
# ============================================================================


def calculate_win_metrics(
    returns: pl.Series,
) -> dict[str, float]:
    """
    Calculate win-based metrics (popular in trading systems).

    Metrics:
        - Win Rate: % of positive return days
        - Profit Factor: sum(gains) / abs(sum(losses))
        - Avg Win/Loss Ratio: mean(wins) / abs(mean(losses))

    Args:
        returns: Return series

    Returns:
        {
            "win_rate": float (0-100%),
            "profit_factor": float,
            "avg_win_loss_ratio": float,
            "total_wins": int,
            "total_losses": int,
        }

    Notes:
        - Win Rate > 50%: More winning days than losing
        - Profit Factor > 1: Profitable overall
        - Avg Win/Loss > 1: Wins larger than losses on average

    Example:
        >>> result = calculate_win_metrics(returns)
        >>> print(result["win_rate"])  # 54.2%
        >>> print(result["profit_factor"])  # 1.35
    """
    r = returns.drop_nulls().to_numpy()

    if r.size == 0:
        return {
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "avg_win_loss_ratio": np.nan,
            "total_wins": 0,
            "total_losses": 0,
        }

    # Separate wins and losses
    wins = r[r > 0]
    losses = r[r < 0]

    total_wins = int(wins.size)
    total_losses = int(losses.size)
    total_days = r.size

    # Win Rate
    win_rate = (total_wins / total_days * 100) if total_days > 0 else 0.0

    # Profit Factor
    sum_wins = np.sum(wins) if wins.size > 0 else 0.0
    sum_losses = abs(np.sum(losses)) if losses.size > 0 else 0.0

    if sum_losses < 1e-10:
        profit_factor = np.inf if sum_wins > 0 else np.nan
    else:
        profit_factor = sum_wins / sum_losses

    # Avg Win/Loss Ratio
    avg_win = np.mean(wins) if wins.size > 0 else 0.0
    avg_loss = abs(np.mean(losses)) if losses.size > 0 else 0.0

    if avg_loss < 1e-10:
        avg_win_loss_ratio = np.inf if avg_win > 0 else np.nan
    else:
        avg_win_loss_ratio = avg_win / avg_loss

    return {
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_win_loss_ratio": float(avg_win_loss_ratio),
        "total_wins": total_wins,
        "total_losses": total_losses,
    }
