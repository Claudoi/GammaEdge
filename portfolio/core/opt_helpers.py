# portfolio/core/opt_helpers.py
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from portfolio.optim.cvar import cvar_minimization
from portfolio.optim.mean_variance import pgd_box_simplex_l2, project_to_box_simplex


def safe_project(w: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    """Proyecta con higiene numérica y normaliza. Fallback: equal-weight."""
    w = np.asarray(w, float).reshape(-1)
    if not np.all(np.isfinite(w)) or w.size == 0:
        return np.array([], dtype=float) if w.size == 0 else np.full(w.size, 1.0 / w.size)
    n = w.size
    # caja infactible → equal-weight
    if (n * w_min) > 1.0 or (n * w_max) < 1.0:
        return np.full(n, 1.0 / n, dtype=float)
    x = project_to_box_simplex(w, w_min=w_min, w_max=w_max)
    s = float(x.sum())
    return (x / s) if s > 1e-12 else np.full(n, 1.0 / n, dtype=float)


def stack_Ws(Ws_list: list[np.ndarray], N: int) -> np.ndarray:
    """Apila carteras (lista → matriz nGxN) filtrando None, tamaños malos y NaN/Inf."""
    good: list[np.ndarray] = []
    for w in Ws_list:
        if w is None:
            continue
        ww = np.asarray(w, float).reshape(-1)
        if ww.size != N:
            continue
        if not np.all(np.isfinite(ww)):
            ww = np.nan_to_num(ww, nan=0.0, posinf=0.0, neginf=0.0)
        good.append(ww)
    if not good:
        raise ValueError("No valid weight vectors to stack (after filtering).")
    return np.vstack(good)



def solve_cvar_with_fallback(
    R: np.ndarray,
    cols_used: Sequence[str],
    mu: np.ndarray,
    Sigma: np.ndarray,
    names: Sequence[str],
    w_bench: np.ndarray,
    *,
    w_min: float,
    w_max: float,
    alpha: float,
    lam_l1: float,
    mv_gamma: float = 10.0,
) -> np.ndarray:
    """
    CVaR con limpieza de subuniverso y fallback a MV-L2 si no hay datos suficientes o falla el LP.
    Devuelve w en el universo completo (|names|).
    """
    N = len(names)
    if R.size == 0 or R.shape[0] < 2 or len(cols_used) < 2:
        w_mv = pgd_box_simplex_l2(mu, Sigma, gamma=mv_gamma, w_min=w_min, w_max=w_max, lam_turnover=0.0, w_ref=w_bench)
        return project_to_box_simplex(w_mv, w_min, w_max)

    idx = [names.index(c) for c in cols_used]
    mu_sub = mu[idx]
    S_sub = Sigma[np.ix_(idx, idx)]
    w_ref_sub = w_bench[idx]

    try:
        w_sub = cvar_minimization(
            R, alpha=alpha, w_min=w_min, w_max=w_max, budget=1.0,
            lam_l1_turnover=lam_l1, w_ref=w_ref_sub
        )
    except Exception:
        w_sub = pgd_box_simplex_l2(mu_sub, S_sub, gamma=mv_gamma, w_min=w_min, w_max=w_max, lam_turnover=0.0, w_ref=np.full(len(idx), 1.0/len(idx)))

    # map back
    w_full = np.zeros(N, dtype=float)
    for i, c in enumerate(cols_used):
        w_full[names.index(c)] = float(w_sub[i])
    return project_to_box_simplex(w_full, w_min, w_max)
