# portfolio/optim/costs_turnover.py
from __future__ import annotations
import numpy as np

def l2_turnover_penalty(w: np.ndarray, w_ref: np.ndarray, lam: float) -> float:
    return 0.5 * float(lam) * float(np.sum((w - w_ref) ** 2))

def l1_turnover_penalty(w: np.ndarray, w_ref: np.ndarray, lam: float) -> float:
    return float(lam) * float(np.sum(np.abs(w - w_ref)))
