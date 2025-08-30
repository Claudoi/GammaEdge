from __future__ import annotations
import numpy as np
import polars as pl
from typing import Callable

# ─────────────────────────────────────────────────────────
# Matrices de covarianza: saneado + diagnóstico
# ─────────────────────────────────────────────────────────
def ensure_psd(S: np.ndarray, eps: float = 1e-10, clip: bool = True) -> np.ndarray:
    """
    Simetriza, rellena NaN/Inf y garantiza PSD. Si clip=True, hace clipping espectral (Higham-lite).
    """
    S = np.asarray(S, dtype=float)
    # NaN/Inf → 0 y simetriza
    S = np.nan_to_num(0.5 * (S + S.T), nan=0.0, posinf=0.0, neginf=0.0)

    # Bump mínimo en diagonal
    d = np.diag(S).copy()
    if d.size:
        np.fill_diagonal(S, np.maximum(d, eps))

    if clip:
        w, V = np.linalg.eigh(S)
        w = np.maximum(w, eps)
        S = (V * w) @ V.T

    # Re-simetriza por seguridad
    return 0.5 * (S + S.T)


def cond_number(S: np.ndarray) -> float:
    """
    Condición espectral de S (eigh). Devuelve NaN si S es vacía.
    """
    S = np.asarray(S, dtype=float)
    if S.size == 0:
        return float("nan")
    w = np.linalg.eigvalsh(0.5 * (S + S.T))
    if w.size == 0:
        return float("nan")
    lam_min = float(max(np.min(w), 1e-16))
    return float(np.max(w) / lam_min)


# ─────────────────────────────────────────────────────────
# Proyección caja+simplex sum=1 via bisección del multiplicador
# ─────────────────────────────────────────────────────────
def _sum_clipped(v: np.ndarray, tau: float, lo: float, hi: float) -> float:
    x = v - tau
    x = np.clip(x, lo, hi)
    return float(x.sum())


def project_to_box_simplex(
    v: np.ndarray,
    w_min: float = 0.0,
    w_max: float = 1.0,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Proyecta v a la intersección {sum=1} ∩ {w_min ≤ w_i ≤ w_max} por bisección del lagrangiano.
    Requisitos: caja factible (lo comprueba el caller con box_feasible) y w_min ≤ w_max.

    Estrategia:
      hallamos tau tal que sum(clip(v - tau, w_min, w_max)) = 1.
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    if n == 0:
        return v

    # Si no hay finitos, fallback equal-weight
    if not np.all(np.isfinite(v)):
        return np.full(n, 1.0 / n)

    # Si w_min == w_max, no hay grados de libertad → pesos fijos:
    if abs(w_max - w_min) < tol:
        return np.full(n, 1.0 / n)  # y presupone n*w_min == 1 (caller debe validarlo antes)

    # Cotas iniciales para tau: suficientemente amplias
    # f(tau) = sum(clip(v - tau, w_min, w_max)) - 1
    # Usamos un rango que garantice cambio de signo
    tau_lo = v.max() - w_max  # hace la suma lo más grande posible
    tau_hi = v.min() - w_min  # hace la suma lo más pequeña posible
    # Expandimos por si no hay cambio de signo aún
    f_lo = _sum_clipped(v, tau_lo, w_min, w_max) - 1.0
    f_hi = _sum_clipped(v, tau_hi, w_min, w_max) - 1.0
    k_expand = 0
    while f_lo * f_hi > 0.0 and k_expand < 50:
        # expandimos simétricamente
        width = (tau_hi - tau_lo) if (tau_hi - tau_lo) != 0 else 1.0
        tau_lo -= 2.0 * width
        tau_hi += 2.0 * width
        f_lo = _sum_clipped(v, tau_lo, w_min, w_max) - 1.0
        f_hi = _sum_clipped(v, tau_hi, w_min, w_max) - 1.0
        k_expand += 1

    # Bisección
    for _ in range(max_iter):
        tau = 0.5 * (tau_lo + tau_hi)
        f_tau = _sum_clipped(v, tau, w_min, w_max) - 1.0
        if abs(f_tau) <= tol:
            break
        if f_lo * f_tau > 0.0:
            tau_lo, f_lo = tau, f_tau
        else:
            tau_hi, f_hi = tau, f_tau

    x = np.clip(v - tau, w_min, w_max)
    # Normalización final (numérica) para asegurar suma 1
    s = x.sum()
    if s <= tol:
        x = np.full(n, 1.0 / n)
    else:
        x = x / s
    return x


# ─────────────────────────────────────────────────────────
# Limpieza de retornos (Polars)
# ─────────────────────────────────────────────────────────
def clean_returns_matrix(returns_wide: pl.DataFrame) -> pl.DataFrame:
    """
    Elimina columnas totalmente vacías y filas con NaN/Inf.
    Asume que si existe, la columna temporal se llama 'date'.
    """
    df = returns_wide
    # Asegura tipo float
    cols_float = [c for c in df.columns if c != "date"]
    df = df.with_columns([pl.col(c).cast(pl.Float64) for c in cols_float])

    # Quita columnas completamente nulas
    keep = ["date"] if "date" in df.columns else []
    for c in cols_float:
        n_null = df.select(pl.col(c).is_null().sum()).item()
        if n_null < df.height:
            keep.append(c)
    df = df.select([c for c in keep if c in returns_wide.columns] + ([] if "date" not in df.columns else []))

    # Filtro filas con finitos en todas las series numéricas
    numeric_cols = [c for c in df.columns if c != "date"]
    if numeric_cols:
        df = df.filter(pl.all_horizontal([pl.col(c).is_finite() for c in numeric_cols]).all())
    return df


# ─────────────────────────────────────────────────────────
# Wrappers robustos
# ─────────────────────────────────────────────────────────
def hrp_safe(hrp_func: Callable, cov: np.ndarray, **kwargs) -> np.ndarray:
    """
    Envuelve un HRP que pudiera fallar; si algo sale mal, devuelve equal-weight.
    Espera que hrp_func(cov=..., **kwargs) -> np.ndarray (pesos).
    """
    try:
        w = hrp_func(cov=cov, **kwargs)
        w = np.asarray(w, dtype=float)
        if w.ndim != 1 or not np.all(np.isfinite(w)) or w.size != cov.shape[0]:
            raise ValueError("Invalid weights shape or NaN/Inf in HRP output")
        # Proyección ligera por seguridad
        w = np.clip(w, 0.0, 1.0)
        s = w.sum()
        return w / s if s > 0 else np.full(w.size, 1.0 / w.size)
    except Exception:
        n = cov.shape[0]
        return np.full(n, 1.0 / n)
