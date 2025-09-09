# portfolio/optim/robust.py
from __future__ import annotations
import numpy as np

def apply_ridge(Sigma: np.ndarray, eps: float) -> np.ndarray:
    if eps <= 0:
        return Sigma
    n = Sigma.shape[0]
    return Sigma + np.eye(n) * float(eps)

def shrink_mu(mu: np.ndarray, target: np.ndarray, lam: float) -> np.ndarray:
    lam = float(np.clip(lam, 0.0, 1.0))
    return (1.0 - lam) * mu + lam * target
