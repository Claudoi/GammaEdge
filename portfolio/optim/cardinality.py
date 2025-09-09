# portfolio/optim/cardinality.py
from __future__ import annotations
import numpy as np
from .mean_variance import project_to_box_simplex

def topk_soft(w: np.ndarray, k: int, w_min: float = 0.0, w_max: float = 1.0) -> np.ndarray:
    """
    Mantiene top‑k pesos y reproyecta a caja+simplex. Suave (no MILP).
    """
    if k is None or k <= 0 or k >= w.size:
        return project_to_box_simplex(w, w_min, w_max)
    idx = np.argsort(w)[::-1]
    keep = idx[:k]
    w2 = np.zeros_like(w)
    w2[keep] = w[keep]
    return project_to_box_simplex(w2, w_min, w_max)
