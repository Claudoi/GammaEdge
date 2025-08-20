# portfolio/optim/mean_variance.py
from __future__ import annotations

from typing import Optional, Tuple, Sequence, Dict
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades numéricas
# ──────────────────────────────────────────────────────────────────────────────

def ensure_psd(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clipa autovalores negativos y re-simetriza."""
    if S.size == 0:
        return S
    A = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    R = (V @ np.diag(w) @ V.T)
    return 0.5 * (R + R.T)

def cond_number(S: np.ndarray) -> float:
    """Número de condición κ(Σ)."""
    if S.size == 0:
        return float("nan")
    Ssym = 0.5 * (S + S.T)
    w = np.linalg.eigvalsh(Ssym)
    if w.size == 0:
        return float("nan")
    lam_min = float(np.min(w))
    lam_max = float(np.max(w))
    return float(lam_max / max(lam_min, 1e-16))

# ──────────────────────────────────────────────────────────────────────────────
# Markowitz sin caja (short permitido)
# ──────────────────────────────────────────────────────────────────────────────

def markowitz_closed_form(
    mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (w_minvar, w_tan) con full-investment y short permitido.
    """
    Sigma = ensure_psd(Sigma)
    invS = np.linalg.pinv(Sigma)
    one = np.ones_like(mu)

    A = float(one @ invS @ one)
    B = float(one @ invS @ mu)

    w_mvp = (invS @ one) / A

    ex = mu - rf * one
    k = float(one @ invS @ ex)
    w_tan = (invS @ ex) / (k if abs(k) > 1e-18 else 1.0)
    return w_mvp, w_tan

def frontier_closed_form(
    mu: np.ndarray, Sigma: np.ndarray, r_min: float, r_max: float, npts: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frontera eficiente (short permitido) parametrizada por retorno objetivo.
    Devuelve (risks, rets) para npts entre r_min y r_max.
    """
    Sigma = ensure_psd(Sigma)
    invS = np.linalg.pinv(Sigma)
    one = np.ones_like(mu)
    A = float(one @ invS @ one)
    B = float(one @ invS @ mu)
    C = float(mu @ invS @ mu)
    D = A * C - B * B
    Rgrid = np.linspace(r_min, r_max, npts)
    risks = np.empty_like(Rgrid)
    for i, R in enumerate(Rgrid):
        w = ((C - B * R) / D) * (invS @ one) + ((A * R - B) / D) * (invS @ mu)
        risks[i] = np.sqrt(max(float(w @ Sigma @ w), 0.0))
    return risks, Rgrid

# ──────────────────────────────────────────────────────────────────────────────
# Proyecciones (caja + simplex) y helpers de sparsidad
# ──────────────────────────────────────────────────────────────────────────────

def project_to_box_simplex(v: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    """
    Proyección a { w_min ≤ w_i ≤ w_max, ∑ w_i = 1 } vía bisección sobre θ.
    Requiere factibilidad: N*w_min ≤ 1 ≤ N*w_max.
    """
    v = np.asarray(v, dtype=float)
    lo = float(w_min); hi = float(w_max)
    L = float(np.min(v - hi)) - 1.0
    U = float(np.max(v - lo)) + 1.0
    for _ in range(80):
        theta = 0.5 * (L + U)
        w = np.clip(v - theta, lo, hi)
        s = w.sum()
        if s > 1.0: L = theta
        else:       U = theta
    theta = 0.5 * (L + U)
    return np.clip(v - theta, lo, hi)

def soft_threshold(u: np.ndarray, t: float) -> np.ndarray:
    """Operator de soft-thresholding componente a componente."""
    return np.sign(u) * np.maximum(np.abs(u) - t, 0.0)

def sparsify_topk_and_project(w: np.ndarray, k: int, w_min: float, w_max: float) -> np.ndarray:
    """
    Cardinalidad suave: conserva top‑k por peso, pone el resto a 0 y reproyecta.
    """
    if k is None or k <= 0 or k >= w.size:
        return project_to_box_simplex(w, w_min, w_max)
    idx = np.argsort(w)[::-1]
    keep = idx[:k]
    w2 = np.zeros_like(w)
    w2[keep] = w[keep]
    return project_to_box_simplex(w2, w_min, w_max)

# ──────────────────────────────────────────────────────────────────────────────
# Proyección con caps por grupo (sector/país) — heurística estable
# ──────────────────────────────────────────────────────────────────────────────

def project_with_group_caps(
    w: np.ndarray,
    groups: Sequence[str],
    group_max: Dict[str, float],
    w_min: float,
    w_max: float,
    iters: int = 20,
) -> np.ndarray:
    """
    Proyección heurística que respeta:
      - caja + simplex (vía project_to_box_simplex)
      - caps por grupo: sum_{i∈g} w_i ≤ group_max[g]

    Algoritmo:
      1) proyecta a caja+simplex
      2) si algún grupo excede, reduce proporcionalmente pesos en ese grupo (clip a ≥ w_min)
      3) re-normaliza con project_to_box_simplex
      4) repetir hasta converger o agotar iteraciones
    """
    w = project_to_box_simplex(w, w_min, w_max)
    g = np.asarray(list(groups))
    for _ in range(iters):
        changed = False
        for grp, cap in group_max.items():
            mask = (g == grp)
            s = float(w[mask].sum())
            if s > cap + 1e-12 and mask.any():
                changed = True
                # Reducción proporcional dentro del grupo respetando w_min
                excess = s - cap
                w_grp = w[mask]
                # Evita división por cero si todo está en w_min
                headroom = np.maximum(w_grp - w_min, 0.0)
                H = float(headroom.sum())
                if H > 0:
                    w[mask] = np.maximum(w_grp - excess * (headroom / H), w_min)
                else:
                    # si no hay margen, fuerza todo a w_min
                    w[mask] = np.full(w_grp.shape, w_min)
                # Reequilibra globalmente a caja+simplex
                w = project_to_box_simplex(w, w_min, w_max)
        if not changed:
            break
    return w

# ──────────────────────────────────────────────────────────────────────────────
# PGD con penalizaciones de turnover (L2 y L1)
# ──────────────────────────────────────────────────────────────────────────────

def pgd_box_simplex_l2(
    mu: np.ndarray, Sigma: np.ndarray, gamma: float, *,
    w_min: float, w_max: float,
    lam_turnover: float = 0.0,
    w_ref: Optional[np.ndarray] = None,
    groups: Optional[Sequence[str]] = None,
    group_max: Optional[Dict[str, float]] = None,
    iters: int = 500,
    step: Optional[float] = None,
) -> np.ndarray:
    """
    Min  1/2 w'Σw − γ μ'w + (λ/2)||w − w_ref||^2
    s.t. ∑w=1,  w_min ≤ w_i ≤ w_max, y caps por grupo (opc).

    groups: etiqueta por activo ('Tech','Energy',...), group_max={'Tech':0.35,...}
    """
    n = mu.shape[0]
    Sigma = ensure_psd(Sigma)
    if w_ref is None: w_ref = np.full(n, 1.0 / n)
    lam = float(max(lam_turnover, 0.0))

    if step is None:
        lam_max = float(np.max(np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))))
        L = max(lam_max, 1e-8) + lam
        step = 1.0 / L

    w = project_to_box_simplex(np.full(n, 1.0 / n), w_min, w_max)
    for _ in range(iters):
        grad = (Sigma @ w) - gamma * mu + lam * (w - w_ref)
        w = project_to_box_simplex(w - step * grad, w_min, w_max)
        if groups is not None and group_max:
            w = project_with_group_caps(w, groups, group_max, w_min, w_max)
    return w

def pgd_box_simplex_l1(
    mu: np.ndarray, Sigma: np.ndarray, gamma: float, *,
    w_min: float, w_max: float,
    lam_l1: float,
    w_ref: Optional[np.ndarray] = None,
    groups: Optional[Sequence[str]] = None,
    group_max: Optional[Dict[str, float]] = None,
    iters: int = 600,
    step: Optional[float] = None,
) -> np.ndarray:
    """
    Min  1/2 w'Σw − γ μ'w + λ||w − w_ref||_1
    s.t. ∑w=1,  w_min ≤ w_i ≤ w_max, y caps por grupo (opc).
    """
    n = mu.shape[0]
    Sigma = ensure_psd(Sigma)
    if w_ref is None: w_ref = np.full(n, 1.0 / n)
    lam = float(max(lam_l1, 0.0))

    if step is None:
        lam_max = float(np.max(np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))))
        L = max(lam_max, 1e-8)
        step = 1.0 / L

    w = project_to_box_simplex(np.full(n, 1.0 / n), w_min, w_max)
    for _ in range(iters):
        grad = (Sigma @ w) - gamma * mu
        z = w - step * grad
        u = soft_threshold(z - w_ref, step * lam)
        w = project_to_box_simplex(w_ref + u, w_min, w_max)
        if groups is not None and group_max:
            w = project_with_group_caps(w, groups, group_max, w_min, w_max)
    return w

# ──────────────────────────────────────────────────────────────────────────────
# Tracking Error (TE) — minimización y trade‑off con alfa
# ──────────────────────────────────────────────────────────────────────────────

def min_te_pgd(
    Sigma: np.ndarray,
    w_bench: np.ndarray,
    *,
    w_min: float, w_max: float,
    groups: Optional[Sequence[str]] = None,
    group_max: Optional[Dict[str, float]] = None,
    iters: int = 500,
    step: Optional[float] = None,
) -> np.ndarray:
    """
    Minimize Tracking Error: min_w  (1/2)(w−wb)' Σ (w−wb)
    s.t. ∑w=1, box, (y opcionalmente caps por grupo).
    """
    n = w_bench.shape[0]
    Sigma = ensure_psd(Sigma)

    if step is None:
        lam_max = float(np.max(np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))))
        L = max(lam_max, 1e-8)
        step = 1.0 / L

    w = project_to_box_simplex(np.copy(w_bench), w_min, w_max)
    for _ in range(iters):
        grad = Sigma @ (w - w_bench)
        w = project_to_box_simplex(w - step * grad, w_min, w_max)
        if groups is not None and group_max:
            w = project_with_group_caps(w, groups, group_max, w_min, w_max)
    return w

def mean_te_tradeoff_pgd(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_bench: np.ndarray,
    gamma: float,
    *,
    w_min: float, w_max: float,
    groups: Optional[Sequence[str]] = None,
    group_max: Optional[Dict[str, float]] = None,
    iters: int = 600,
    step: Optional[float] = None,
) -> np.ndarray:
    """
    Trade‑off alfa vs TE:
      min_w  (1/2)(w−wb)' Σ (w−wb) − γ μ' (w−wb)
      s.t. ∑w=1, box, (y opcionalmente caps por grupo).
    """
    n = mu.shape[0]
    Sigma = ensure_psd(Sigma)

    if step is None:
        lam_max = float(np.max(np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))))
        L = max(lam_max, 1e-8)
        step = 1.0 / L

    w = project_to_box_simplex(np.copy(w_bench), w_min, w_max)
    for _ in range(iters):
        grad = Sigma @ (w - w_bench) - gamma * mu
        w = project_to_box_simplex(w - step * grad, w_min, w_max)
        if groups is not None and group_max:
            w = project_with_group_caps(w, groups, group_max, w_min, w_max)
    return w

# ──────────────────────────────────────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────────────────────────────────────

def risk_contributions(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    RC_i = w_i * (Σ w)_i  (contribución absoluta).
    """
    return w * (Sigma @ w)

__all__ = [
    "ensure_psd", "cond_number",
    "markowitz_closed_form", "frontier_closed_form",
    "project_to_box_simplex", "soft_threshold", "sparsify_topk_and_project",
    "pgd_box_simplex_l2", "pgd_box_simplex_l1",
    "project_with_group_caps",
    "min_te_pgd", "mean_te_tradeoff_pgd",
    "risk_contributions",
]
