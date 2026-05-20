"""
Fama-French Factor Models

Implements factor exposure analysis and attribution using:
- Fama-French 3-factor (MKT, SMB, HML)
- Fama-French 5-factor (+ RMW, CMA)
- Carhart 4-factor (+ MOM)

Data source: Ken French Data Library (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

Based on:
- Fama & French (1993): "Common risk factors in the returns on stocks and bonds"
- Fama & French (2015): "A five-factor asset pricing model"
- Carhart (1997): "On Persistence in Mutual Fund Performance"

Author: GammaEdge TIER 1 Enhancement
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression

# NOTE: ``pandas_datareader`` is imported lazily inside ``fetch_fama_french`` to
# avoid an import-time crash with recent pandas releases. ``pandas_datareader``
# v0.10.0 (latest, unmaintained) calls ``pd.util._decorators.deprecate_kwarg``
# with an old signature, which raises ``TypeError`` at import. Lazy importing
# keeps this module collectable by pytest and only impacts code paths that
# actually fetch Fama-French data.


FactorModel = Literal["FF3", "FF5", "Carhart4"]


def fetch_fama_french(
    model: FactorModel = "FF3",
    start: str = "2020-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch Fama-French factors from Ken French Data Library.

    Args:
        model: Which factor model to fetch
            - "FF3": Fama-French 3-factor (MKT-RF, SMB, HML)
            - "FF5": Fama-French 5-factor (+ RMW, CMA)
            - "Carhart4": 3-factor + Momentum (MOM)
        start: Start date (YYYY-MM-DD)
        end: End date (default: today)

    Returns:
        DataFrame with daily factors + RF (risk-free rate)
        All returns in decimal format (0.01 = 1%)

    Example:
        >>> factors = fetch_fama_french("FF3", start="2020-01-01")
        >>> print(factors.head())
        >>> # columns: ['Mkt-RF', 'SMB', 'HML', 'RF']

    Note:
        Requires internet connection. Data is free from Ken French library.
    """
    # Map model to dataset name
    dataset_map = {
        "FF3": "F-F_Research_Data_Factors_daily",
        "FF5": "F-F_Research_Data_5_Factors_2x3_daily",
        "Carhart4": "F-F_Research_Data_Factors_daily",  # We'll add MOM separately
    }

    dataset_name = dataset_map[model]

    # Lazy import: ``pandas_datareader`` 0.10.0 fails at import time with newer
    # pandas versions (TypeError in ``deprecate_kwarg``). Import here so the
    # module remains collectable by pytest and only this code path is affected.
    from pandas_datareader import data as pdr

    try:
        # Fetch main factors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pdr.DataReader(dataset_name, "famafrench", start=start, end=end)[0]

        # Convert from percentage to decimal (Ken French uses percentages)
        df = df / 100.0

        # If Carhart, also fetch Momentum
        if model == "Carhart4":
            mom_df = pdr.DataReader(
                "F-F_Momentum_Factor_daily", "famafrench", start=start, end=end
            )[0]
            mom_df = mom_df / 100.0
            df = df.join(mom_df)

        return df

    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch {model} factors from Ken French library. "
            f"Check internet connection or date range. Error: {e}"
        ) from e


def compute_factor_loadings(
    returns: pl.Series | pd.Series,
    factors: pd.DataFrame,
    model: FactorModel = "FF3",
) -> dict:
    """
    Compute factor loadings (betas) via OLS regression.

    Model: r_i,t - RF_t = alpha + beta_MKT * (MKT-RF)_t + beta_SMB * SMB_t + ... + epsilon_t

    Args:
        returns: Asset returns (daily)
        factors: DataFrame with factor returns (from fetch_fama_french)
        model: Which factor model to use

    Returns:
        Dictionary with:
        - alpha: Jensen's alpha (annualized)
        - betas: Dict of factor exposures
        - r_squared: R² of regression
        - residual_vol: Std dev of residuals (idiosyncratic risk)
        - t_stats: T-statistics for significan testing

    Example:
        >>> factors = fetch_fama_french("FF3", start="2020-01-01")
        >>> loadings = compute_factor_loadings(spy_returns, factors)
        >>> print(f"Market beta: {loadings['betas']['Mkt-RF']:.2f}")
        >>> print(f"Alpha: {loadings['alpha']:.2%} (annual)")
    """
    # Convert to pandas if needed
    if isinstance(returns, pl.Series):
        returns = returns.to_pandas()

    # Align dates
    returns = pd.Series(returns, name="returns")
    returns.index = pd.to_datetime(returns.index)
    factors.index = pd.to_datetime(factors.index)

    # Join returns with factors
    df = pd.DataFrame({"returns": returns}).join(factors, how="inner")
    df = df.dropna()

    if len(df) < 30:
        raise ValueError(f"Insufficient observations after alignment: {len(df)} days (need >= 30)")

    # Compute excess returns
    excess_returns = df["returns"] - df["RF"]

    # Select factor columns based on model
    if model == "FF3":
        factor_cols = ["Mkt-RF", "SMB", "HML"]
    elif model == "FF5":
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    elif model == "Carhart4":
        factor_cols = [
            "Mkt-RF",
            "SMB",
            "HML",
            "Mom   ",
        ]  # Note: Ken French uses "Mom   " with spaces
        # Standardize to "MOM"
        if "Mom   " in df.columns:
            df["MOM"] = df["Mom   "]
            factor_cols[-1] = "MOM"
    else:
        raise ValueError(f"Unknown model: {model}")

    # Check columns exist
    missing = [col for col in factor_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing factor columns: {missing}. Available: {list(df.columns)}")

    # Prepare regression
    X = df[factor_cols].values
    y = excess_returns.values

    # OLS regression
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)

    # Extract results
    alpha_daily = reg.intercept_
    betas_array = reg.coef_

    # R-squared
    y_pred = reg.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Residuals (idiosyncratic risk)
    residuals = y - y_pred
    residual_vol = np.std(residuals, ddof=len(factor_cols) + 1)

    # T-statistics (for alpha and betas)
    n = len(y)
    k = len(factor_cols)
    se_residual = np.sqrt(ss_res / (n - k - 1))

    # Standard error of alpha
    X_var = np.var(X, axis=0, ddof=1)
    se_alpha = se_residual * np.sqrt(1 / n + np.sum(X.mean(axis=0) ** 2 / (n * X_var)))

    # Standard errors of betas
    se_betas = se_residual / np.sqrt(n * X_var)

    t_alpha = alpha_daily / se_alpha if se_alpha > 0 else 0.0
    t_betas = betas_array / se_betas

    # Package results
    betas_dict = dict(zip(factor_cols, betas_array, strict=False))
    t_stats_dict = dict(zip(factor_cols, t_betas, strict=False))
    t_stats_dict["alpha"] = t_alpha

    return {
        "alpha": alpha_daily * 252,  # Annualize
        "betas": betas_dict,
        "r_squared": r_squared,
        "residual_vol": residual_vol * np.sqrt(252),  # Annualize
        "t_stats": t_stats_dict,
        "n_obs": n,
    }


def factor_attribution(
    returns: pl.Series | pd.Series,
    factors: pd.DataFrame,
    model: FactorModel = "FF3",
) -> pd.DataFrame:
    """
    Decompose returns into factor contributions.

    For each time period:
    Return = Alpha + sum(beta_f * factor_f) + Residual

    Args:
        returns: Asset returns (daily)
        factors: Factor returns
        model: Factor model to use

    Returns:
        DataFrame with columns:
        - date
        - total_return: Actual return
        - alpha_contrib: Alpha contribution
        - {factor}_contrib: Contribution from each factor
        - residual: Unexplained return

    Example:
        >>> attr = factor_attribution(spy_returns, factors)
        >>> print(attr[['date', 'total_return', 'Mkt-RF_contrib', 'alpha_contrib']].head())
    """
    # First compute loadings
    loadings = compute_factor_loadings(returns, factors, model)

    # Convert to pandas if needed
    if isinstance(returns, pl.Series):
        returns = returns.to_pandas()

    # Align data
    returns = pd.Series(returns, name="returns")
    returns.index = pd.to_datetime(returns.index)
    factors.index = pd.to_datetime(factors.index)

    df = pd.DataFrame({"returns": returns}).join(factors, how="inner")
    df = df.dropna()

    # Select factors
    if model == "FF3":
        factor_cols = ["Mkt-RF", "SMB", "HML"]
    elif model == "FF5":
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    elif model == "Carhart4":
        if "Mom   " in df.columns:
            df["MOM"] = df["Mom   "]
        factor_cols = ["Mkt-RF", "SMB", "HML", "MOM"]

    # Compute contributions
    result = pd.DataFrame({"date": df.index})
    result["total_return"] = df["returns"].values

    # Alpha contribution (daily alpha)
    result["alpha_contrib"] = loadings["alpha"] / 252

    # Factor contributions
    for factor in factor_cols:
        beta = loadings["betas"][factor]
        result[f"{factor}_contrib"] = beta * df[factor].values

    # Residual
    total_factor_contrib = sum(result[f"{fc}_contrib"] for fc in factor_cols)
    result["residual"] = result["total_return"] - result["alpha_contrib"] - total_factor_contrib

    return result


def factor_adjusted_returns(
    returns: pl.DataFrame,
    factors: pd.DataFrame,
    returns_col_prefix: str = "ret_",
    model: FactorModel = "FF3",
) -> pl.DataFrame:
    """
    Compute factor-adjusted returns (alphas) for multiple assets.

    Args:
        returns: Wide DataFrame with date + ret_{ticker} columns
        factors: Factor returns
        returns_col_prefix: Prefix for return columns
        model: Factor model

    Returns:
        DataFrame with:
        - date
        - alpha_{ticker}: Factor-adjusted return for each asset

    Example:
        >>> factors = fetch_fama_french("FF3", start="2020-01-01")
        >>> alphas = factor_adjusted_returns(returns_wide_df, factors)
        >>> # alphas contains only idiosyncratic component
    """
    returns_pd = returns.to_pandas()
    ret_cols = [c for c in returns_pd.columns if c.startswith(returns_col_prefix)]

    alpha_dict = {"date": returns_pd["date"]}

    for col in ret_cols:
        ticker = col.replace(returns_col_prefix, "")

        try:
            loadings = compute_factor_loadings(returns_pd[col], factors, model)

            # Reconstruct alpha series
            # alpha_t = r_t - sum(beta_f * factor_f,t)
            returns_pd = returns_pd.copy()
            returns_pd.index = pd.to_datetime(returns_pd["date"])

            aligned = returns_pd[[col]].join(factors, how="inner").dropna()

            if model == "FF3":
                factor_cols = ["Mkt-RF", "SMB", "HML"]
            elif model == "FF5":
                factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
            elif model == "Carhart4":
                if "Mom   " in aligned.columns:
                    aligned["MOM"] = aligned["Mom   "]
                factor_cols = ["Mkt-RF", "SMB", "HML", "MOM"]

            # Compute factor-adjusted return
            factor_contrib = sum(loadings["betas"][fc] * aligned[fc] for fc in factor_cols)

            alpha_series = aligned[col] - factor_contrib - loadings["alpha"] / 252

            # Align back to original dates
            alpha_dict[f"alpha_{ticker}"] = returns_pd[col].copy()
            mask = returns_pd.index.isin(alpha_series.index)
            alpha_dict[f"alpha_{ticker}"][mask] = alpha_series.values

        except Exception as e:
            # If factor regression fails, just use raw returns
            print(f"Warning: Factor adjustment failed for {ticker}: {e}")
            alpha_dict[f"alpha_{ticker}"] = returns_pd[col]

    return pl.DataFrame(alpha_dict)


def compute_exposures_wide(
    returns_wide: pl.DataFrame,
    factors: pd.DataFrame,
    returns_col_prefix: str = "ret_",
    model: FactorModel = "FF3",
) -> pd.DataFrame:
    """
    Compute factor exposures for all assets in wide format.

    Args:
        returns_wide: DataFrame with date + ret_{ticker} columns
        factors: Factor returns
        returns_col_prefix: Prefix for return columns
        model: Factor model

    Returns:
        DataFrame with one row per asset:
        - ticker
        - alpha (annualized)
        - beta_{factor}
        - r_squared
        - residual_vol

    Example:
        >>> exposures = compute_exposures_wide(returns_df, factors, model="FF3")
        >>> print(exposures)
        ticker     alpha  beta_Mkt-RF  beta_SMB  beta_HML  r_squared  residual_vol
        AAPL       0.05         1.15     -0.20      0.10       0.65          0.25
    """
    returns_pd = returns_wide.to_pandas()
    ret_cols = [c for c in returns_pd.columns if c.startswith(returns_col_prefix)]

    results = []

    for col in ret_cols:
        ticker = col.replace(returns_col_prefix, "")

        try:
            loadings = compute_factor_loadings(returns_pd[col], factors, model)

            row = {
                "ticker": ticker,
                "alpha": loadings["alpha"],
                "r_squared": loadings["r_squared"],
                "residual_vol": loadings["residual_vol"],
                "n_obs": loadings["n_obs"],
            }

            # Add beta columns
            for factor, beta in loadings["betas"].items():
                row[f"beta_{factor}"] = beta

            # Add t-stats
            for factor, t_stat in loadings["t_stats"].items():
                row[f"t_{factor}"] = t_stat

            results.append(row)

        except Exception as e:
            print(f"Warning: Failed to compute exposures for {ticker}: {e}")
            continue

    return pd.DataFrame(results)
