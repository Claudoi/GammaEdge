# portfolio/optim/te.py
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict
import numpy as np

from .mean_variance import ensure_psd, project_to_box_simplex

# -----------------------------------------------------------------------------
# TE optimizer (penalizado): 
#   Min  1/2 (w-wb)' Σ (w-wb)  - γ μ'(w-wb)  + (λ/2)||w-w_ref||^2  + ρ * expo_penalty
#   s.t. ∑ w = 1,  w_min ≤ w_i ≤ w_max
#   expo_penalty = 1/2 * ||relu(X(w-wb)-ub)||^2 + 1/2 * ||relu(lb-(X(w-wb)))||^2
#   (X es matriz F x N de exposiciones; lb/ub son cotas por factor sobre exposición ACTIVA)
# -----------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)

def te_loss_and_grad(
    w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, w_bench: np.ndarray, *,
    gamma: float,
    lam_l2: float,
    w_ref: np.ndarray,
    X: Optional[np.ndarray],
    lb: Optional[np.ndarray],
    ub: Optional[np.ndarray],
    rho_expo: float,
) -> Tuple[float, np.ndarray, Dict[str, float]]:
    """
    Devuelve (loss, grad, diag) para el objetivo TE-penalizado.
    """
    v = w - w_bench
    Swv = Sigma @ v

    # Pérdida base: 1/2 v'Σv - γ μ'v + (λ/2)||w - w_ref||^2
    te2 = 0.5 * float(v @ Swv)
    ar  = float(mu @ v)
    reg = 0.5 * float(lam_l2) * float(np.sum((w - w_ref) ** 2))
    loss = te2 - gamma * ar + reg

    grad = Swv - gamma * mu + lam_l2 * (w - w_ref)

    # Penalización por exposiciones activas
    expo_pen = 0.0
    if X is not None and (lb is not None or ub is not None) and rho_expo > 0.0:
        aexp = X @ v  # (F,)
        if ub is not None:
            viol_u = _relu(aexp - ub)
            expo_pen += 0.5 * float(np.sum(viol_u ** 2))
            grad += rho_expo * (X.T @ viol_u)  # d/ dw de 1/2||vi||^2 = X^T * viol_u
        if lb is not None:
            viol_l = _relu(lb - aexp)
            expo_pen += 0.5 * float(np.sum(viol_l ** 2))
            grad -= rho_expo * (X.T @ viol_l)  # signo opuesto

        loss += rho_expo * expo_pen

    diag = {"te2": float(v @ (Sigma @ v)), "active_ret": ar, "reg": reg, "expo_pen": float(expo_pen)}
    return loss, grad, diag


def te_active_pgd(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_bench: np.ndarray,
    *,
    gamma: float = 1.0,
    w_min: float = 0.0,
    w_max: float = 0.1,
    lam_l2: float = 0.0,
    w_ref: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,       # F x N
    lb: Optional[np.ndarray] = None,      # (F,) cotas lower sobre X(w-wb)
    ub: Optional[np.ndarray] = None,      # (F,) cotas upper sobre X(w-wb)
    rho_expo: float = 0.0,                # peso de penalización de exposición
    iters: int = 800,
    step: Optional[float] = None,
    warm_start: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Optimiza cartera activa con PGD bajo caja+simplex y penalización de exposiciones activas.
    Devuelve (w, diag).
    """
    n = mu.shape[0]
    Sigma = ensure_psd(Sigma)

    if w_ref is None:
        w_ref = w_bench.copy()

    # Paso (Lipschitz)
    if step is None:
        lam_max = float(np.max(np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))))
        L = max(lam_max, 1e-8) + lam_l2
        step = 1.0 / L

    # Initial
    if warm_start is not None and warm_start.shape == (n,):
        w = warm_start.copy()
    else:
        w = project_to_box_simplex(w_bench.copy(), w_min, w_max)  # start cerca del bench

    last = None
    for _ in range(iters):
        loss, grad, _ = te_loss_and_grad(
            w, mu, Sigma, w_bench,
            gamma=gamma, lam_l2=lam_l2, w_ref=w_ref, X=X, lb=lb, ub=ub, rho_expo=rho_expo
        )
        w = project_to_box_simplex(w - step * grad, w_min, w_max)
        last = loss

    # Diags finales
    _, _, diag = te_loss_and_grad(
        w, mu, Sigma, w_bench,
        gamma=gamma, lam_l2=lam_l2, w_ref=w_ref, X=X, lb=lb, ub=ub, rho_expo=rho_expo
    )
    v = w - w_bench
    diag.update({
        "te": float(np.sqrt(max(v @ (Sigma @ v), 0.0))),
        "active_ret": float(mu @ v),
        "gamma": float(gamma),
    })
    return w, diag


def te_frontier_sweep(
    mu: np.ndarray, Sigma: np.ndarray, w_bench: np.ndarray, gammas: Sequence[float], *,
    w_min: float = 0.0, w_max: float = 0.1,
    lam_l2: float = 0.0, w_ref: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None, lb: Optional[np.ndarray] = None, ub: Optional[np.ndarray] = None, rho_expo: float = 0.0,
    iters: int = 600, step: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Barre γ y devuelve:
      Ws (n_gamma x N), TE (n_gamma,), AR (n_gamma,), Loss (n_gamma,)
    """
    nG = len(gammas)
    N = mu.shape[0]
    Ws = np.zeros((nG, N))
    TE = np.zeros(nG)
    AR = np.zeros(nG)
    Loss = np.zeros(nG)
    w0 = None
    for i, g in enumerate(gammas):
        w, diag = te_active_pgd(
            mu, Sigma, w_bench, gamma=g, w_min=w_min, w_max=w_max,
            lam_l2=lam_l2, w_ref=w_ref, X=X, lb=lb, ub=ub, rho_expo=rho_expo,
            iters=iters, step=step, warm_start=w0
        )
        Ws[i, :] = w
        TE[i] = diag["te"]
        AR[i] = diag["active_ret"]
        # loss aproximada = 0.5 TE^2 - γ AR + reg + expo_pen
        Loss[i] = 0.5 * (diag["te2"]) - g * AR[i] + diag["reg"] + diag["expo_pen"]
        w0 = w.copy()
    return Ws, TE, AR, Loss
