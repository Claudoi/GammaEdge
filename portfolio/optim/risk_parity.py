# portfolio/optim/risk_parity.py
from __future__ import annotations
import numpy as np
from .mean_variance import ensure_psd, project_to_box_simplex

def risk_parity(
    Sigma: np.ndarray,
    *,
    w_min: float = 0.0,
    w_max: float = 1.0,
    iters: int = 500,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Risk parity long-only con caja+simplex (heurístico multiplicativo + proyección).
    Minimiza discrepancias de RC relativas.
    """
    N = Sigma.shape[0]
    Sigma = ensure_psd(Sigma)
    w = np.full(N, 1.0 / N)
    for _ in range(iters):
        Sw = Sigma @ w
        g = w * Sw  # RC
        target = float(np.sum(g)) / N
        # update multiplicativa hacia target RC
        scale = np.where(g > 0, target / np.maximum(g, 1e-18), 1.0)
        w = w * np.sqrt(scale)
        w = project_to_box_simplex(w, w_min, w_max)
        if np.linalg.norm(g - target) < tol:
            break
    return w
