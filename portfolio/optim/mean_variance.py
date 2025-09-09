# portfolio/optim/mean_variance.py
from __future__ import annotations

from typing import Optional, Tuple, Sequence, Dict
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades numéricas (robustas)
# ──────────────────────────────────────────────────────────────────────────────

def ensure_psd(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Saneo + proyección PSD robusta:
    - simetriza
    - reemplaza no finitos por 0
    - asegura diagonal mínima
    - si eigh falla, aplica jitter creciente en la diagonal (hasta 6 intentos)
    - último recurso: SVD con clip
    """
    if S.size == 0:
        return S

    A = 0.5 * (S + S.T)
    A = np.asarray(A, dtype=float)
    A[~np.isfinite(A)] = 0.0

    d = np.diag(A).copy()
    d[~np.isfinite(d)] = 0.0
    d = np.maximum(d, eps)
    np.fill_diagonal(A, d)

    jitter = 0.0
    for _ in range(6):
        try:
            w, V = np.linalg.eigh(A)
            w = np.clip(w, eps, None)
            R = V @ np.diag(w) @ V.T
            return 0.5 * (R + R.T)
        except np.linalg.LinAlgError:
            jitter = eps if jitter == 0.0 else jitter * 10.0
            A = A + np.eye(A.shape[0]) * jitter

    # Fallback muy raro
    U, s, VT = np.linalg.svd(0.5 * (A + A.T), full_matrices=False)
    s = np.clip(s, eps, None)
    R = (U * s) @ VT
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
    Devuelve (w_mvp, w_tan) con presupuesto ∑w=1 y short permitido.
    - w_mvp: Global Minimum Variance (GMV)
    - w_tan: cartera tangente (respecto a rf)
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    n = mu.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError("Sigma shape must be (N,N) aligned with mu")

    Sigma = ensure_psd(Sigma)
    invS = np.linalg.pinv(Sigma)
    one = np.ones(n, dtype=np.float64)

    A = float(one @ invS @ one)         # 1' S^{-1} 1
    B = float(one @ invS @ mu)          # 1' S^{-1} mu

    # GMV (si A≈0, cae a igual ponderación)
    w_mvp = (invS @ one) / (A if abs(A) > 1e-18 else max(n, 1))

    # Tangente (si k≈0, cae a proyección normalizada)
    ex = mu - rf * one
    k = float(one @ invS @ ex)
    w_tan = (invS @ ex) / (k if abs(k) > 1e-18 else max(n, 1))

    return w_mvp, w_tan


def frontier_closed_form(
    mu: np.ndarray, Sigma: np.ndarray, r_min: float, r_max: float, npts: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frontera eficiente (short permitido) parametrizada por retorno objetivo.
    Devuelve (risks, rets) para `npts` puntos entre `r_min` y `r_max`.

    Si el determinante de la cónica (D) ≈ 0, colapsa a GMV.
    Si r_min >= r_max, se crea un rango automático razonable.
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    n = mu.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError("Sigma shape must be (N,N) aligned with mu")

    Sigma = ensure_psd(Sigma)
    invS = np.linalg.pinv(Sigma)
    one = np.ones(n, dtype=np.float64)

    A = float(one @ invS @ one)         # 1' S^{-1} 1
    B = float(one @ invS @ mu)          # 1' S^{-1} mu
    C = float(mu  @ invS @ mu)          # mu' S^{-1} mu
    D = A * C - B * B                   # discriminante

    # Si rango inválido, creamos uno en torno a GMV y max‑ret
    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_min >= r_max:
        # GMV return
        w_gmv = (invS @ one) / (A if abs(A) > 1e-18 else max(n, 1))
        r_gmv = float(w_gmv @ mu)
        # direcc. “máx retorno” bajo norma 1 (no restringimos riesgo aquí)
        e_max = np.zeros_like(mu); e_max[np.argmax(mu)] = 1.0
        r_max_auto = float(e_max @ mu)
        lo, hi = np.percentile([r_gmv, r_max_auto], [10, 90]).tolist()
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = r_gmv * 0.8, r_gmv * 1.2
        r_min, r_max = lo, hi

    if abs(D) < 1e-18 or not np.isfinite(D):
        # colapso: retorna el punto GMV
        w = (invS @ one) / (A if abs(A) > 1e-18 else max(n, 1))
        r = float(w @ mu)
        s = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
        return np.array([s], dtype=np.float64), np.array([r], dtype=np.float64)

    Rgrid = np.linspace(r_min, r_max, max(2, int(npts)))
    risks = np.empty_like(Rgrid, dtype=np.float64)

    invS_one = invS @ one
    invS_mu  = invS @ mu
    for i, R in enumerate(Rgrid):
        # w(R) = a(R)*invS*1 + b(R)*invS*mu
        a = (C - B * R) / D
        b = (A * R - B) / D
        w = a * invS_one + b * invS_mu
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
    n = v.size
    if n == 0:
        return v
    if n * w_min - 1.0 > 1e-12 or 1.0 - n * w_max > 1e-12:
        raise ValueError("Infeasible box-simplex: ensure N*w_min ≤ 1 ≤ N*w_max.")

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
    """Operador de soft‑thresholding componente a componente."""
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
# Caps por grupo (sector/país) — proyección heurística estable
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
                excess = s - cap
                w_grp = w[mask]
                headroom = np.maximum(w_grp - w_min, 0.0)
                H = float(headroom.sum())
                if H > 0:
                    w[mask] = np.maximum(w_grp - excess * (headroom / H), w_min)
                else:
                    w[mask] = np.full(w_grp.shape, w_min)
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
# Tracking Error (TE)
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
    Minimize TE: min_w  (1/2)(w−wb)' Σ (w−wb)
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
# Métricas y frontera proyectada (para pintar con caja)
# ──────────────────────────────────────────────────────────────────────────────

def risk_contributions(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """RC_i = w_i * (Σ w)_i  (contribución absoluta)."""
    return w * (Sigma @ w)


def frontier_box_projected(
    mu: np.ndarray, Sigma: np.ndarray,
    w_min: float, w_max: float,
    r_min: Optional[float] = None, r_max: Optional[float] = None,
    npts: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aproximación para frontera con caja: parte de la solución cerrada para
    un target μ y proyecta a {sum=1, w_min≤w≤w_max}. Devuelve (risks, rets).
    """
    Sigma = ensure_psd(Sigma)
    invS = np.linalg.pinv(Sigma)
    one = np.ones_like(mu)

    A = float(one @ invS @ one)
    B = float(one @ invS @ mu)
    C = float(mu @ invS @ mu)
    D = A * C - B * B
    if abs(D) < 1e-18:
        w = (invS @ one) / (A if A != 0 else 1.0)
        w = project_to_box_simplex(w, w_min, w_max)
        r = float(w @ mu)
        s = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
        return np.array([s]), np.array([r])

    if r_min is None: r_min = float(np.min(mu))
    if r_max is None: r_max = float(np.max(mu))
    Rgrid = np.linspace(r_min, r_max, max(2, int(npts)))
    risks = np.empty_like(Rgrid)
    rets  = np.empty_like(Rgrid)
    for i, R in enumerate(Rgrid):
        w_free = ((C - B * R) / D) * (invS @ one) + ((A * R - B) / D) * (invS @ mu)
        w = project_to_box_simplex(w_free, w_min, w_max)
        rets[i]  = float(w @ mu)
        risks[i] = np.sqrt(max(float(w @ Sigma @ w), 0.0))
    return risks, rets

__all__ = [
    "ensure_psd", "cond_number",
    "markowitz_closed_form", "frontier_closed_form",
    "project_to_box_simplex", "soft_threshold", "sparsify_topk_and_project",
    "project_with_group_caps",
    "pgd_box_simplex_l2", "pgd_box_simplex_l1",
    "min_te_pgd", "mean_te_tradeoff_pgd",
    "risk_contributions",
]

