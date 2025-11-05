# portfolio/attribution/euler.py

from __future__ import annotations

from collections.abc import Sequence

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

    if isinstance(weights, pd.Series):
        index = weights.index
    elif isinstance(cov, pd.DataFrame):
        index = cov.index
    else:
        index = np.arange(n)

    return pd.Series(contrib, index=index, name="risk_contribution")


def _extract_contrib_frame(result: object) -> pl.DataFrame:
    """
    Igual que en brinson: normaliza la salida del engine a un DataFrame.
    """
    if isinstance(result, pl.DataFrame):
        return result

    for attr in ("contributions", "frame", "df", "data"):
        if hasattr(result, attr):
            candidate = getattr(result, attr)
            if isinstance(candidate, pl.DataFrame):
                return candidate

    msg = (
        "compute_portfolio_contributions(...) must return either a Polars "
        "DataFrame, or an object with a 'contributions'/'frame'/'df'/'data' "
        "attribute holding a Polars DataFrame."
    )
    raise TypeError(msg)


def _call_portfolio_contributions_from_long(
    df: pl.DataFrame,
    weights_col: str,
    returns_col: str,
    date_col: str,
) -> pl.DataFrame:
    """
    Parte de df *largo* (date, asset, w, r) y usa el engine de atribución
    en formato ancho por factor/asset. Devuelve:

        ['date', 'asset', 'contribution']
    """
    if "asset" not in df.columns:
        msg = "Expected a long dataframe with an 'asset' column."
        raise ValueError(msg)

    weights_wide = (
        df.select([date_col, "asset", weights_col])
        .pivot(index=date_col, columns="asset", values=weights_col)
        .sort(date_col)
    )

    returns_wide = (
        df.select([date_col, "asset", returns_col])
        .pivot(index=date_col, columns="asset", values=returns_col)
        .sort(date_col)
    )

    result = compute_portfolio_contributions(
        weights=weights_wide,
        returns=returns_wide,
        date_col=date_col,
    )
    contrib_wide = _extract_contrib_frame(result)

    contrib_long = contrib_wide.melt(
        id_vars=date_col,
        variable_name="asset",
        value_name="contribution",
    )

    return contrib_long


def run_euler_engine(
    df: pl.DataFrame,
    factor_weights_col: str = "w",
    factor_returns_col: str = "r",
    date_col: str = "date",
) -> pl.DataFrame:
    """
    Wrapper fino para reutilizar el engine de atribución en un contexto
    de factores/Euler a partir de un df largo.
    """
    contributions = _call_portfolio_contributions_from_long(
        df=df,
        weights_col=factor_weights_col,
        returns_col=factor_returns_col,
        date_col=date_col,
    )
    return df.join(contributions, on=[date_col, "asset"], how="left")
