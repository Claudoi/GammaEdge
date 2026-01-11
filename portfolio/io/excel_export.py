"""
Production-Grade Excel Export for Quant Metrics

Exports quantitative metrics to audit-ready Excel workbook with 5 sheets:
- DATA: Time series (date, ticker, adj_close, ret_1d)
- SUMMARY: Period metrics per ticker (beta, alpha, sharpe, etc.)
- METADATA: Standards and definitions
- DATA_QUALITY: Coverage, gaps, warnings
- CORRELATION: Correlation matrix + sample sizes (if ≥2 tickers)
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl
import yfinance as yf
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows

from portfolio.features.quant_metrics import (
    RF_ANNUAL_DEFAULT,
    RF_DAILY_DEFAULT,
    TRADING_DAYS_PER_YEAR,
    _as_pl_date,
    calculate_beta_alpha,
    calculate_cagr,
    calculate_calmar,
    calculate_correlation_matrix,
    calculate_data_quality,
    calculate_max_drawdown,
    calculate_moments,
    calculate_returns,
    calculate_sharpe_ratio,
)


def export_quant_metrics_to_excel(
    tickers: list[str],
    start: str,
    end: str,
    benchmark: str = "SPY",
    rf_annual: float = RF_ANNUAL_DEFAULT,
    output_format: Literal["bytes", "path"] = "bytes",
    output_path: str | None = None,
) -> bytes | str:
    """
    Export quant metrics to audit-grade Excel workbook.

    Args:
        tickers: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        benchmark: Benchmark ticker for beta calculation (default: SPY)
        rf_annual: Annual risk-free rate (default: 0.02)
        output_format: 'bytes' or 'path'
        output_path: Path to save file (required if output_format='path')

    Returns:
        bytes or path depending on output_format

    Raises:
        ValueError: If no valid data found or output_path missing

    Example:
        >>> excel_bytes = export_quant_metrics_to_excel(
        ...     tickers=["AAPL", "MSFT"],
        ...     start="2020-01-01",
        ...     end="2024-12-31",
        ... )
    """
    if output_format == "path" and not output_path:
        raise ValueError("output_path required when output_format='path'")

    # 1. Download data
    all_tickers = list(set(tickers + [benchmark]))
    df_prices = _download_prices(all_tickers, start, end)
    
    # Debug: print what we got
    if not df_prices.is_empty():

    if df_prices.is_empty():
        raise ValueError(
            f"No price data downloaded for tickers: {tickers}. "
            f"Please check ticker symbols and date range ({start} to {end})."
        )

    # 2. Calculate returns
    df_returns = calculate_returns(df_prices, price_col_prefix="adj_close_")

    # 3. Get benchmark data
    bench_prices = df_prices.select(["date", f"adj_close_{benchmark}"]).rename(
        {f"adj_close_{benchmark}": "adj_close"}
    )
    bench_returns = df_returns.select(["date", f"ret_{benchmark}"]).rename(
        {f"ret_{benchmark}": "ret"}
    )
    bench_dates = bench_returns["date"]

    # 4. Calculate metrics for each ticker
    summary_rows = []
    data_quality_rows = []

    for ticker in tickers:
        price_col = f"adj_close_{ticker}"
        ret_col = f"ret_{ticker}"

        if price_col not in df_prices.columns or ret_col not in df_returns.columns:
            continue

        ticker_prices = df_prices.select(["date", price_col]).rename({price_col: "adj_close"})
        ticker_returns = df_returns.select(["date", ret_col]).rename({ret_col: "ret"})
        ticker_dates = ticker_returns["date"]

        # Beta & Alpha
        beta_result = calculate_beta_alpha(
            returns=ticker_returns["ret"],
            benchmark_returns=bench_returns["ret"],
            dates=ticker_dates,
            benchmark_dates=bench_dates,
        )

        # Sharpe
        sharpe_result = calculate_sharpe_ratio(
            returns=ticker_returns["ret"],
            rf_annual=rf_annual,
        )

        # Max Drawdown
        mdd_result = calculate_max_drawdown(
            prices=ticker_prices["adj_close"],
            dates=ticker_dates,
        )

        # CAGR
        cagr_result = calculate_cagr(
            prices=ticker_prices["adj_close"],
            dates=ticker_dates,
        )

        # Calmar
        calmar = calculate_calmar(
            cagr=cagr_result["cagr"],
            mdd=mdd_result["max_drawdown"],
        )

        # Moments
        moments_result = calculate_moments(ticker_returns["ret"])

        # Data Quality
        dq_result = calculate_data_quality(
            dates=ticker_dates,
            benchmark_dates=bench_dates,
            ticker=ticker,
        )

        # Summary row
        summary_rows.append({
            "ticker": ticker,
            "beta": beta_result["beta"],
            "alpha_daily": beta_result["alpha_daily"],
            "alpha_annual": beta_result["alpha_annual"],
            "r_squared": beta_result["r_squared"],
            "sharpe_ratio": sharpe_result["sharpe_ratio"],
            "max_drawdown": mdd_result["max_drawdown"],
            "cagr": cagr_result["cagr"],
            "calmar_ratio": calmar,
            "skewness": moments_result["skewness"],
            "kurtosis": moments_result["kurtosis"],
            "n_obs": beta_result["n_obs"],
        })

        # Data quality row
        data_quality_rows.append(dq_result)

    # 5. Create DataFrames
    df_summary = pl.DataFrame(summary_rows)
    df_data_quality = pl.DataFrame(data_quality_rows)

    # 6. Prepare DATA sheet (long format)
    df_data_long = _prepare_data_sheet(df_prices, df_returns, tickers)

    # 7. Calculate correlation (if ≥2 tickers)
    corr_result = None
    if len(tickers) >= 2:
        corr_result = calculate_correlation_matrix(df_returns, return_col_prefix="ret_")

    # 8. Create Excel workbook
    wb = Workbook()
    wb.remove(wb.active)  # Remove default sheet

    # Generate sheets
    _generate_data_sheet(wb, df_data_long)
    
    _generate_summary_sheet(wb, df_summary)
    
    _generate_metadata_sheet(wb, start, end, benchmark, rf_annual, tickers)
    
    _generate_data_quality_sheet(wb, df_data_quality)

    if corr_result and corr_result["correlation_matrix"] is not None:
        _generate_correlation_sheet(
            wb,
            corr_result["correlation_matrix"],
            corr_result["sample_sizes"],
        )
    else:

    # 9. Save or return bytes
    if output_format == "path":
        wb.save(output_path)
        return output_path
    else:
        buffer = io.BytesIO()
        wb.save(buffer)
        return buffer.getvalue()


def _download_prices(tickers: list[str], start: str, end: str) -> pl.DataFrame:
    """Download adjusted close prices for multiple tickers."""
    import pandas as pd
    
    
    # Download data (set auto_adjust=False to avoid warning)
    data = yf.download(tickers, start=start, end=end, progress=False, group_by="ticker", auto_adjust=False)
    
    
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
            # logger.warning(f"'{tickers[0]}' - 'Adj Close' not found, using 'Close' price.")
            df = df[["Date", "Close"]]
            df.columns = ["date", f"adj_close_{tickers[0]}"]
            return pl.from_pandas(df)
        else:
            # logger.warning(f"'{tickers[0]}' - Neither 'Adj Close' nor 'Close' price found.")
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
                            # logger.warning(f"'{ticker}' - 'Adj Close' not found, using 'Close' price.")
                            df_ticker = df_ticker[["Date", "Close"]]
                        else:
                            # logger.warning(f"'{ticker}' - Neither 'Adj Close' nor 'Close' price found. Skipping ticker.")
                            continue
                            
                        df_ticker.columns = ["date", f"adj_close_{ticker}"]
                        dfs.append(pl.from_pandas(df_ticker))
                except (KeyError, AttributeError) as e:
                    # logger.warning(f"Error processing data for ticker {ticker}: {e}. Skipping.")
                    continue

            if not dfs:
                return pl.DataFrame()

            # Concatenate all dataframes and ensure no duplicate date columns
            # Strategy: collect all data, then pivot wide
            if len(dfs) == 1:
                return dfs[0]
            
            # Get all unique dates
            all_dates = pl.concat([df.select("date") for df in dfs]).unique().sort("date")
            
            # Join each ticker's data
            result = all_dates
            for df in dfs:
                # Get the price column name (should be adj_close_TICKER)
                price_col = [c for c in df.columns if c.startswith("adj_close_")][0]
                result = result.join(df, on="date", how="left")

            return result
        else:
            # Fallback for unexpected format
            # logger.warning("Unexpected data format from yfinance for multiple tickers. Returning empty DataFrame.")
            return pl.DataFrame()


def _prepare_data_sheet(
    df_prices: pl.DataFrame,
    df_returns: pl.DataFrame,
    tickers: list[str],
) -> pl.DataFrame:
    """Prepare DATA sheet in long format."""
    rows = []

    for ticker in tickers:
        price_col = f"adj_close_{ticker}"
        ret_col = f"ret_{ticker}"

        if price_col not in df_prices.columns:
            continue

        df_ticker = df_prices.select(["date", price_col]).rename({price_col: "adj_close"})

        if ret_col in df_returns.columns:
            df_ticker = df_ticker.join(
                df_returns.select(["date", ret_col]).rename({ret_col: "ret_1d"}),
                on="date",
                how="left",
            )
        else:
            df_ticker = df_ticker.with_columns(pl.lit(None).alias("ret_1d"))

        df_ticker = df_ticker.with_columns(pl.lit(ticker).alias("ticker"))
        rows.append(df_ticker)

    if not rows:
        return pl.DataFrame()

    return pl.concat(rows).select(["date", "ticker", "adj_close", "ret_1d"]).sort(["ticker", "date"])


def _generate_data_sheet(wb: Workbook, df: pl.DataFrame) -> None:
    """Generate DATA sheet with time series."""
    ws = wb.create_sheet("DATA", 0)

    # Check if DataFrame is empty
    if df.is_empty():
        ws.append(["date", "ticker", "adj_close", "ret_1d"])
        _style_header(ws, 1)
        ws.append(["No data available"])
        return

    # Header
    ws.append(["date", "ticker", "adj_close", "ret_1d"])
    _style_header(ws, 1)

    # Data
    for row in df.iter_rows():
        ws.append(row)

    # Format
    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 10
    ws.column_dimensions["C"].width = 12
    ws.column_dimensions["D"].width = 12


def _generate_summary_sheet(wb: Workbook, df: pl.DataFrame) -> None:
    """Generate SUMMARY sheet with period metrics."""
    ws = wb.create_sheet("SUMMARY", 1)

    # Header
    headers = [
        "ticker",
        "beta",
        "alpha_daily",
        "alpha_annual",
        "r_squared",
        "sharpe_ratio",
        "max_drawdown",
        "cagr",
        "calmar_ratio",
        "skewness",
        "kurtosis",
        "n_obs",
    ]
    ws.append(headers)
    _style_header(ws, 1)

    # Data
    for row in df.iter_rows():
        ws.append(row)

    # Format
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 15


def _generate_metadata_sheet(
    wb: Workbook,
    start: str,
    end: str,
    benchmark: str,
    rf_annual: float,
    tickers: list[str],
) -> None:
    """Generate METADATA sheet with standards and definitions."""
    ws = wb.create_sheet("METADATA", 2)

    metadata = [
        ["# Data Source", ""],
        ["provider", "yahoo"],
        ["price_field", "adj_close"],
        ["timezone", "America/New_York"],
        ["calendar_proxy", benchmark],
        ["", ""],
        ["# Returns", ""],
        ["returns_definition", "adj_close_to_adj_close_simple"],
        ["returns_formula", "(adj_close_t - adj_close_{t-1}) / adj_close_{t-1}"],
        ["", ""],
        ["# Annualization", ""],
        ["trading_days_per_year", TRADING_DAYS_PER_YEAR],
        ["rf_annual", rf_annual],
        ["rf_daily_value", RF_DAILY_DEFAULT],
        ["rf_daily_formula", "(1 + rf_annual)**(1/252) - 1"],
        ["", ""],
        ["# Benchmark", ""],
        ["benchmark_ticker", benchmark],
        ["", ""],
        ["# Statistical Methods", ""],
        ["correlation_method", "pearson"],
        ["skewness_method", "fisher_pearson_adjusted"],
        ["kurtosis_type", "excess"],
        ["", ""],
        ["# MDD", ""],
        ["mdd_sign", "negative"],
        ["mdd_formula", "min((equity_t / peak_t) - 1)"],
        ["", ""],
        ["# Sharpe", ""],
        ["sharpe_std", "std(returns)"],
        ["sharpe_formula", "mean(r - rf) / std(r) * sqrt(252)"],
        ["", ""],
        ["# Sample Sizes", ""],
        ["min_obs_sharpe", 60],
        ["min_obs_beta", 30],
        ["min_obs_cagr", 252],
        ["", ""],
        ["# Export Info", ""],
        ["date_range", f"{start} to {end}"],
        ["tickers", ", ".join(tickers)],
        ["generated_at", datetime.now().isoformat()],
        ["gammaedge_version", "1.0.0"],
    ]

    for row in metadata:
        ws.append(row)

    # Style headers (rows with #)
    for row_idx, row in enumerate(ws.iter_rows(min_row=1, max_row=len(metadata)), start=1):
        if row[0].value and str(row[0].value).startswith("#"):
            row[0].font = Font(bold=True, size=12)
            row[0].fill = PatternFill(start_color="E0E0E0", end_color="E0E0E0", fill_type="solid")

    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 50


def _generate_data_quality_sheet(wb: Workbook, df: pl.DataFrame) -> None:
    """Generate DATA_QUALITY sheet."""
    ws = wb.create_sheet("DATA_QUALITY", 3)

    # Header
    headers = [
        "ticker",
        "first_date",
        "last_date",
        "n_obs",
        "expected_obs",
        "coverage_pct",
        "max_gap_days",
        "missing_blocks",
        "warnings",
    ]
    ws.append(headers)
    _style_header(ws, 1)

    # Data
    for row in df.iter_rows():
        # Convert warnings list to string
        row_list = list(row)
        if row_list[-1]:  # warnings column
            if isinstance(row_list[-1], list):
                row_list[-1] = "; ".join(row_list[-1]) if row_list[-1] else ""
        else:
            row_list[-1] = ""
        ws.append(row_list)

    # Format
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 15


def _generate_correlation_sheet(
    wb: Workbook,
    corr_df: pl.DataFrame,
    sample_df: pl.DataFrame,
) -> None:
    """Generate CORRELATION sheet with matrix and sample sizes."""
    ws = wb.create_sheet("CORRELATION", 4)

    # Correlation matrix
    ws.append(["Correlation Matrix"])
    ws["A1"].font = Font(bold=True, size=12)

    for row in corr_df.iter_rows():
        ws.append(row)

    _style_header(ws, 2)

    # Sample sizes
    start_row = len(corr_df) + 4
    ws.cell(row=start_row, column=1, value="Sample Sizes")
    ws.cell(row=start_row, column=1).font = Font(bold=True, size=12)

    for row in sample_df.iter_rows():
        ws.append(row)

    _style_header(ws, start_row + 1)

    # Format
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 12


def _style_header(ws, row_num: int) -> None:
    """Apply header styling to a row."""
    for cell in ws[row_num]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        cell.font = Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(horizontal="center")
