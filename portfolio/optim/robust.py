# portfolio/optim/robust.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def apply_ridge(Sigma: FloatArray, eps: float) -> FloatArray:
    """
    Añade una ridge (eps * I) a la covarianza.
    Garantiza dtype float64 y salida simétrica.
    """
    S = np.asarray(Sigma, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("Sigma must be a square (n x n) array.")
    if eps <= 0:
        return S
    n = S.shape[0]
    # np.eye devuelve float64; aseguramos float64 final
    return (S + np.eye(n, dtype=np.float64) * float(eps)).astype(np.float64, copy=False)


def shrink_mu(mu: FloatArray, target: FloatArray, lam: float) -> FloatArray:
    """
    Encoge el vector de medias hacia `target` con factor lam in [0,1].
    """
    m = np.asarray(mu, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    if m.shape != t.shape:
        raise ValueError(f"`mu` and `target` must have same shape, got {m.shape} vs {t.shape}.")
    lam_f = float(np.clip(lam, 0.0, 1.0))
    return ((1.0 - lam_f) * m + lam_f * t).astype(np.float64, copy=False)
