# portfolio/attribution/euler.py
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


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

    Parameters
    ----------
    weights:
        Pesos de la cartera (long-only o no). Puede ser sequence o pd.Series.
    cov:
        Matriz de covarianzas (numpy o pandas DataFrame) de dimensión NxN.

    Returns
    -------
    pandas.Series
        Serie con las contribuciones al riesgo por activo, que suman σ_p.
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
