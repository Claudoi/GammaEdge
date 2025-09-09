from __future__ import annotations
import numpy as np

def box_feasible(n: int, w_min: float, w_max: float) -> bool:
    """
    Comprueba si existe w con sum=1 y w_min ≤ w ≤ w_max.
    """
    return (n * w_min <= 1.0) and (n * w_max >= 1.0) and (w_min <= w_max)

def validate_weights(w: np.ndarray, w_min: float = 0.0, w_max: float = 1.0, tol: float = 1e-8) -> None:
    """
    Valida suma presupuesto y caja con tolerancia; levanta AssertionError si falla.
    """
    assert np.isfinite(w).all(), "Weights contain NaN/Inf"
    assert abs(float(w.sum()) - 1.0) < tol, "Weights do not sum to 1"
    assert (w >= w_min - tol).all() and (w <= w_max + tol).all(), "Weights out of bounds"

def has_min_coverage(n_obs: int, min_obs: int = 2) -> bool:
    return n_obs >= min_obs
