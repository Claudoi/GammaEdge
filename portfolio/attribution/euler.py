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
    Calcula las Euler risk contributions para una cartera con volatilidad:

        σ_p = sqrt(wᵀ Σ w)

    Usando la descomposición de Euler:

        RC_i = w_i * (Σ w)_i / σ_p

    donde Σ es la matriz de covarianzas.
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

    # Índice bien tipado para mypy
    if isinstance(weights, pd.Series):
        idx: pd.Index = weights.index
    elif isinstance(cov, pd.DataFrame):
        idx = cov.index
    else:
        # RangeIndex produce siempre un pd.Index compatible
        idx = pd.RangeIndex(n)

    series = pd.Series(contrib, index=idx, name="risk_contribution")
    # El constructor de pandas está tipado como Any; hacemos cast explícito.
    return cast(pd.Series, series)


def run_euler_engine(
    df: pl.DataFrame,
    weights_col: str = "w",
    returns_col: str = "r",
    date_col: str = "date",
) -> pl.DataFrame:
    """
    Usa el mismo engine de contribuciones para construir contribuciones
    por factor (o activo) en un df largo.

    Espera un df con columnas: date, asset, weights_col, returns_col.
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
    ).contributions

    contrib_long = contrib_wide.melt(
        id_vars=date_col,
        variable_name="asset",
        value_name="contribution",
    )

    return contrib_long
