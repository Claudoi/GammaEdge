# portfolio/optim/cvar.py
from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.optimize import linprog

def cvar_minimization(
    R: np.ndarray,                 # (T, N) returns (periodic)
    alpha: float = 0.95,
    *,
    w_min: float = 0.0,
    w_max: float = 1.0,
    budget: float = 1.0,
    lam_l1_turnover: float = 0.0,
    w_ref: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Minimize CVaR_α of losses L_t = -R_t·w, long-only caja+simplex, opcional L1 turnover.
    LP estándar:
      min ζ + (1/((1-α)T)) Σ u_t + λ Σ d_i
      s.t. u_t ≥ 0
           u_t ≥ -R_t w - ζ
           sum w_i = budget
           w_min ≤ w_i ≤ w_max
           d_i ≥ w_i - w_ref_i,  d_i ≥ -(w_i - w_ref_i)  (si λ>0)

    Devuelve w (N,).
    """
    R = np.asarray(R, dtype=float)
    T, N = R.shape
    if T == 0 or N == 0:
        return np.array([])

    if w_ref is None:
        w_ref = np.full(N, budget / N)

    lam = float(max(lam_l1_turnover, 0.0))
    # variables: [w(0..N-1), zeta, u(0..T-1), d(0..N-1) if lam>0]
    n_w = N
    n_z = 1
    n_u = T
    n_d = N if lam > 0 else 0
    n_var = n_w + n_z + n_u + n_d

    # Objective
    c = np.zeros(n_var)
    c[n_w] = 1.0  # zeta
    c[n_w + n_z : n_w + n_z + n_u] = 1.0 / ((1.0 - alpha) * T)  # sum u / ((1-α)T)
    if lam > 0:
        c[n_w + n_z + n_u :] = lam  # λ sum d_i

    # Inequalities A_ub x ≤ b_ub
    A = []
    b = []

    # u_t ≥ 0 ⇒ -u_t ≤ 0
    for t in range(T):
        row = np.zeros(n_var)
        row[n_w + n_z + t] = -1.0
        A.append(row); b.append(0.0)

    # u_t ≥ -R_t w - ζ ⇒ -u_t - R_t w - ζ ≤ 0
    for t in range(T):
        row = np.zeros(n_var)
        row[:n_w] = -(-R[t, :])  # = R_t
        row[n_w] = -1.0          # -ζ
        row[n_w + n_z + t] = -1.0  # -u_t
        A.append(row); b.append(0.0)

    # Caja w_min ≤ w_i ≤ w_max
    lb = np.full(n_var, -np.inf)
    ub = np.full(n_var,  np.inf)
    for i in range(N):
        lb[i] = w_min
        ub[i] = w_max

    # ζ libre (en principio puede ser negativa/positiva)
    # u_t ≥ 0 ya asegurado con desigualdades; acotamos u_t ≥ 0 vía lb
    for t in range(T):
        lb[n_w + n_z + t] = 0.0

    # Turnover L1: d_i ≥ |w_i - w_ref_i|
    if lam > 0:
        # d_i ≥  w_i - w_ref_i  ⇒ -w_i + d_i ≥ -w_ref_i
        for i in range(N):
            row = np.zeros(n_var)
            row[i] = -1.0
            row[n_w + n_z + n_u + i] = 1.0
            A.append(row); b.append(-float(w_ref[i]))
        # d_i ≥ -(w_i - w_ref_i) ⇒  w_i + d_i ≥  w_ref_i  ⇒ -w_i - d_i ≤ -w_ref_i
        for i in range(N):
            row = np.zeros(n_var)
            row[i] = -1.0
            row[n_w + n_z + n_u + i] = -1.0
            A.append(row); b.append(-float(w_ref[i]))
        # d_i ≥ 0
        for i in range(N):
            lb[n_w + n_z + n_u + i] = 0.0

    # Igualdad: sum w_i = budget
    A_eq = np.zeros((1, n_var))
    A_eq[0, :n_w] = 1.0
    b_eq = np.array([budget], dtype=float)

    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        raise RuntimeError(f"CVaR LP failed: {res.message}")

    w_opt = res.x[:N]
    # normaliza por si redondeos
    s = float(np.sum(w_opt))
    if s != 0:
        w_opt = w_opt / s * budget
    return w_opt
