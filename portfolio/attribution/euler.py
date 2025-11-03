# portfolio/attribution/euler.py

from __future__ import annotations

import numpy as np


def euler_risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Euler risk contributions for a covariance risk model.

    RC_i = w_i * (Sigma w)_i
    Sum(RC) = w^T Sigma w   (portfolio variance)

    Parameters
    ----------
    weights : array-like, shape (n,)
        Portfolio weights.
    cov : array-like, shape (n, n)
        Covariance matrix (symmetric PSD).

    Returns
    -------
    rc : ndarray, shape (n,)
        Euler risk contributions per asset.
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    S = np.asarray(cov, dtype=float)
    if S.shape[0] != S.shape[1] or S.shape[0] != w.shape[0]:
        raise ValueError("Shape mismatch between weights and covariance")

    marg = S @ w  # marginal contributions
    rc = w * marg  # Euler contributions
    return rc
