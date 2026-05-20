"""
Quantitative Metrics Preview Module

This module provides UI-friendly preview functionality for quantitative metrics,
generating summary DataFrames for display in Streamlit before full Excel export.
"""

import pandas as pd
import polars as pl
import yfinance as yf

from portfolio.features.quant_metrics import (
    calculate_beta_alpha,
    calculate_cagr,
    calculate_calmar,
    calculate_capture_ratios,
    calculate_data_quality,
    calculate_information_ratio,
    calculate_max_drawdown,
    calculate_moments,
    calculate_monthly_extremes,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_tail_ratio,
    calculate_treynor_ratio,
    calculate_ulcer_index,
    calculate_win_metrics,
)


def _download_prices(tickers: list[str], start: str, end: str) -> pl.DataFrame:
    """
    Download adjusted close prices for multiple tickers.

    Reuses logic from excel_export.py for consistency.

    Args:
        tickers: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)

    Returns:
        Polars DataFrame with columns: date, adj_close_{ticker}
    """
    if not tickers:
        return pl.DataFrame()

    # Download data (set auto_adjust=False to avoid warning)
    data = yf.download(
        tickers, start=start, end=end, progress=False, group_by="ticker", auto_adjust=False
    )

    # Check if data is empty
    if data is None or len(data) == 0:
        return pl.DataFrame()

    # Handle single vs multiple tickers
    if len(tickers) == 1:
        # Single ticker - columns are simple: Open, High, Low, Close, Adj Close, Volume
        df = data.reset_index()

        # Check available columns
        if "Adj Close" in df.columns:
            df = df[["Date", "Adj Close"]]
            df.columns = ["date", f"adj_close_{tickers[0]}"]
            return pl.from_pandas(df)
        elif "Close" in df.columns:
            # Fallback to Close if Adj Close not available
            df = df[["Date", "Close"]]
            df.columns = ["date", f"adj_close_{tickers[0]}"]
            return pl.from_pandas(df)
        else:
            return pl.DataFrame()
    else:
        # Multiple tickers - MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            dfs = []
            for ticker in tickers:
                try:
                    if ticker in data.columns.levels[0]:
                        df_ticker = data[ticker].reset_index()

                        # Try Adj Close first, then Close
                        if "Adj Close" in df_ticker.columns:
                            df_ticker = df_ticker[["Date", "Adj Close"]]
                        elif "Close" in df_ticker.columns:
                            df_ticker = df_ticker[["Date", "Close"]]
                        else:
                            continue

                        df_ticker.columns = ["date", f"adj_close_{ticker}"]
                        dfs.append(pl.from_pandas(df_ticker))
                except (KeyError, AttributeError):
                    continue

            if not dfs:
                return pl.DataFrame()

            # Get all unique dates
            all_dates = pl.concat([df.select("date") for df in dfs]).unique().sort("date")

            # Join each ticker's data
            result = all_dates
            for df in dfs:
                # Join price column (adj_close_TICKER) for this ticker
                result = result.join(df, on="date", how="left")

            return result
        else:
            # Fallback for unexpected format
            return pl.DataFrame()


def generate_metrics_summary(
    tickers: list[str],
    start: str,
    end: str,
    benchmark: str = "SPY",
    rf_annual: float = 0.02,
) -> dict:
    """
    Generate a summary DataFrame of quant metrics for preview.

    Args:
        tickers: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        benchmark: Benchmark ticker (default: SPY)
        rf_annual: Annual risk-free rate (default: 0.02)

    Returns:
        {
            "summary_df": pl.DataFrame,  # Metrics table (ticker x metrics)
            "data_quality_df": pl.DataFrame,  # Data quality report
            "warnings": list[str],  # Any issues encountered
        }

    Raises:
        ValueError: If no data can be downloaded
    """
    warnings = []

    # 1. Download prices
    all_tickers = list(set(tickers + [benchmark]))
    df_prices = _download_prices(all_tickers, start, end)

    if df_prices.is_empty():
        raise ValueError(
            f"No price data downloaded for any ticker. "
            f"Please check ticker symbols and date range ({start} to {end})."
        )

    # Validate that we have data for requested tickers
    missing_tickers = [t for t in tickers if f"adj_close_{t}" not in df_prices.columns]
    if missing_tickers:
        raise ValueError(
            f"No data found for ticker(s): {', '.join(missing_tickers)}. "
            f"Please verify ticker symbols are correct."
        )

    # 2. Calculate returns
    df_returns = calculate_returns(df_prices, price_col_prefix="adj_close_")

    # 3. Build summary for each ticker
    summary_rows = []
    data_quality_rows = []

    for ticker in tickers:
        price_col = f"adj_close_{ticker}"
        ret_col = f"ret_{ticker}"

        # Extract series
        prices = df_prices[price_col]
        returns = df_returns[ret_col]
        dates = df_prices["date"]

        # Benchmark data
        benchmark_returns = df_returns[f"ret_{benchmark}"]
        benchmark_dates = df_prices["date"]

        # Calculate metrics
        beta_alpha = calculate_beta_alpha(
            returns=returns,
            benchmark_returns=benchmark_returns,
            dates=dates,
            benchmark_dates=benchmark_dates,
        )

        sharpe = calculate_sharpe_ratio(returns, rf_annual=rf_annual)
        mdd = calculate_max_drawdown(prices, dates)
        cagr_result = calculate_cagr(prices, dates)
        calmar = calculate_calmar(cagr_result.get("cagr"), mdd["max_drawdown"])
        moments = calculate_moments(returns)
        data_quality = calculate_data_quality(dates, benchmark_dates, ticker)

        # NEW: Calculate additional metrics (Sprint UI v2)
        sortino = calculate_sortino_ratio(returns, rf_annual=rf_annual)
        info_ratio = calculate_information_ratio(
            returns=returns,
            benchmark_returns=benchmark_returns,
            dates=dates,
            benchmark_dates=benchmark_dates,
        )
        tail_ratio_result = calculate_tail_ratio(returns)
        win_metrics = calculate_win_metrics(returns)

        # NEW: Calculate 10/10 metrics
        ulcer = calculate_ulcer_index(prices, dates)
        treynor = calculate_treynor_ratio(
            returns=returns,
            beta=beta_alpha.get("beta"),
            rf_annual=rf_annual,
        )
        capture = calculate_capture_ratios(
            returns=returns,
            benchmark_returns=benchmark_returns,
            dates=dates,
            benchmark_dates=benchmark_dates,
        )
        monthly_extremes = calculate_monthly_extremes(returns, dates)

        # Total return
        total_return = cagr_result.get("total_return")

        # Volatility (annualized)
        returns_clean = returns.drop_nulls()
        volatility = float(returns_clean.std() * (252**0.5)) if len(returns_clean) > 1 else None

        # Build summary row (now with 22 metrics)
        summary_rows.append(
            {
                "ticker": ticker,
                "total_return": total_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe.get("sharpe_ratio"),
                "sortino_ratio": sortino.get("sortino_ratio"),
                "treynor_ratio": treynor.get("treynor_ratio"),  # NEW 10/10
                "information_ratio": info_ratio.get("information_ratio"),
                "beta": beta_alpha.get("beta"),
                "alpha": beta_alpha.get("alpha_annual"),
                "max_drawdown": mdd["max_drawdown"],
                "ulcer_index": ulcer.get("ulcer_index"),  # NEW 10/10
                "cagr": cagr_result.get("cagr"),
                "calmar": calmar,
                "up_capture": capture.get("up_capture"),  # NEW 10/10
                "down_capture": capture.get("down_capture"),  # NEW 10/10
                "tail_ratio": tail_ratio_result.get("tail_ratio"),
                "win_rate": win_metrics.get("win_rate"),
                "profit_factor": win_metrics.get("profit_factor"),
                "best_month": monthly_extremes.get("best_month"),  # NEW 10/10
                "worst_month": monthly_extremes.get("worst_month"),  # NEW 10/10
                "skewness": moments.get("skewness"),
                "kurtosis": moments.get("kurtosis"),
            }
        )

        # Build data quality row
        data_quality_rows.append(
            {
                "ticker": ticker,
                "total_obs": data_quality["n_obs"],
                "missing_values": data_quality["expected_obs"] - data_quality["n_obs"],
                "data_loss_pct": 100.0 - data_quality["coverage_pct"],
                "warnings": "; ".join(data_quality["warnings"]) if data_quality["warnings"] else "",
            }
        )

        # Collect warnings
        if data_quality["warnings"]:
            warnings.extend([f"{ticker}: {w}" for w in data_quality["warnings"]])
        if sharpe.get("warning"):
            warnings.append(f"{ticker}: {sharpe['warning']}")
        if beta_alpha.get("warning"):
            warnings.append(f"{ticker}: {beta_alpha['warning']}")

    # Convert to DataFrames
    summary_df = pl.DataFrame(summary_rows)
    data_quality_df = pl.DataFrame(data_quality_rows)

    return {
        "summary_df": summary_df,
        "data_quality_df": data_quality_df,
        "warnings": warnings,
        "df_prices": df_prices,  # NEW: for charts
        "df_returns": df_returns,  # NEW: for charts
    }
