from __future__ import annotations
import numpy as np
from typing import Sequence

from portfolio.core.utils import project_to_box_simplex
# Asumimos que ya tienes estos solvers; si sus nombres/firma difieren, ajusta aquí:
from portfolio.optim.mean_variance import pgd_box_simplex_l2  # MV con L2/PGD
from portfolio.optim.cvar import cvar_minimization           # CVaR (LP u otro)

def solve_cvar_with_fallback(
    R: np.ndarray,
    cols_used: Sequence[str],
    mu: np.ndarray,
    Sigma: np.ndarray,
    names: Sequence[str],
    w_bench: np.ndarray,
    w_min: float,
    w_max: float,
    alpha: float = 0.95,
    lam_l1: float = 0.0,
    mv_gamma: float = 10.0
) -> np.ndarray:
    """
    Resuelve CVaR sobre (R, cols_used). Si no hay datos suficientes o falla el solver,
    fallback a Mean-Variance L2. Devuelve pesos en el UNIVERSO COMPLETO (orden 'names'),
    proyectados con caja+simplex.
    """
    names = list(names)
    cols_used = list(cols_used)
    mu = np.asarray(mu, float).reshape(-1)
    Sigma = np.asarray(Sigma, float)
    w_bench = np.asarray(w_bench, float).reshape(-1)

    N = len(names)
    if R.size == 0 or R.shape[0] < 2 or len(cols_used) < 2:
        # Fallback directo al universo completo (robusto)
        w_mv = pgd_box_simplex_l2(
            mu, Sigma, gamma=mv_gamma,
            w_min=w_min, w_max=w_max,
            lam_turnover=0.0, w_ref=w_bench.copy()
        )
        return project_to_box_simplex(w_mv, w_min, w_max)

    # Subconjunto consistente μ, Σ, w_ref
    idx = [names.index(c) for c in cols_used]
    mu_sub = mu[idx]
    Sigma_sub = Sigma[np.ix_(idx, idx)]
    w_ref_sub = w_bench[idx]

    try:
        w_sub = cvar_minimization(
            R, alpha=alpha, w_min=w_min, w_max=w_max, budget=1.0,
            lam_l1_turnover=lam_l1, w_ref=w_ref_sub
        )
    except Exception:
        # Fallback: MV (L2) en el subuniverso
        w_sub = pgd_box_simplex_l2(
            mu_sub, Sigma_sub, gamma=mv_gamma,
            w_min=w_min, w_max=w_max,
            lam_turnover=0.0, w_ref=np.full(len(idx), 1.0 / len(idx))
        )

    # Map back al universo completo
    w_full = np.zeros(N, dtype=float)
    for i, c in enumerate(cols_used):
        w_full[names.index(c)] = w_sub[i]
    return project_to_box_simplex(w_full, w_min, w_max)


def stack_Ws(
    Ws_list: Sequence[np.ndarray],
    N: int
) -> np.ndarray:
    """
    Convierte una lista de carteras en una matriz (nG x N), filtrando entradas
    vacías o con tamaño incorrecto. Lanza ValueError si no queda ninguna.
    """
    good = []
    for w in Ws_list:
        if w is None:
            continue
        w = np.asarray(w, float).reshape(-1)
        if w.size != N:
            continue
        if not np.all(np.isfinite(w)):
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        good.append(w)
    if not good:
        raise ValueError("No valid weight vectors to stack (after filtering).")
    return np.vstack(good)
