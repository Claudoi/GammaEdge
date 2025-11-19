# portfolio/attribution/euler.py
from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
import polars as pl

from portfolio.attribution.engine import compute_portfolio_contributions


def euler_risk_contributions(
    weights: Sequence[float] | pd.Series,
    cov: np.ndarray | pd.DataFrame,
) -> pd.Series:
    """
    Compute Euler risk contributions for a portfolio with volatility:

        σ_p = sqrt(wᵀ Σ w)

    Using Euler decomposition:

        RC_i = w_i * (Σ w)_i / σ_p

    where Σ is the covariance matrix.
    """
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    sigma = np.asarray(cov, dtype=float)

    n = w.shape[0]
    if sigma.shape != (n, n):
        msg = f"Covariance matrix must be {n}x{n}, got {sigma.shape}."
        raise ValueError(msg)

    port_var = float(w.T @ sigma @ w)
    if port_var <= 0.0:
        msg = "Portfolio variance must be positive for Euler decomposition."
        raise ValueError(msg)
    port_sigma = float(np.sqrt(port_var))

    # Σ w → marginal contributions
    marginal = sigma @ w  # (n, 1)

    # Euler RC = w_i * MC_i / σ_p
    contrib = (w * marginal) / port_sigma
    contrib = contrib.ravel()

    # Build index consistent with the input
    if isinstance(weights, pd.Series):
        idx: pd.Index = weights.index
    elif isinstance(cov, pd.DataFrame):
        idx = cov.index
    else:
        # RangeIndex always yields a valid pd.Index
        idx = pd.RangeIndex(n)

    series = pd.Series(contrib, index=idx, name="risk_contribution")
    # Pandas constructor is typed as Any; cast explicitly for mypy.
    return cast(pd.Series, series)


def run_euler_engine(
    df: pl.DataFrame,
    weights_col: str = "w",
    returns_col: str = "r",
    date_col: str = "date",
) -> pl.DataFrame:
    """
    Use the same contribution engine to compute contributions
    per factor (or asset) in a long-format df.

    Expects a df with columns: date, asset, weights_col, returns_col.
    """
    if "asset" not in df.columns:
        msg = "Expected a long dataframe with an 'asset' column."
        raise ValueError(msg)

    weights_wide = (
        df.select([date_col, "asset", weights_col])
        .pivot(values=weights_col, index=date_col, on="asset")
        .sort(date_col)
    )

    returns_wide = (
        df.select([date_col, "asset", returns_col])
        .pivot(values=returns_col, index=date_col, on="asset")
        .sort(date_col)
    )

    contrib_wide = compute_portfolio_contributions(
        weights=weights_wide,
        returns=returns_wide,
        date_col=date_col,
        method="euler",
    ).contributions

    contrib_long = contrib_wide.melt(
        id_vars=date_col,
        variable_name="asset",
        value_name="contribution",
    )

    return contrib_long
