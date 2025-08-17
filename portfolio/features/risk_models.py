# portfolio/features/risk_models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.covariance import LedoitWolf, OAS

# ──────────────────────────────────────────────────────────────────────────────
# Tipos y utilidades generales
# ──────────────────────────────────────────────────────────────────────────────

MuMethod = Literal["historical", "ema", "shrunk"]
CovMethod = Literal["sample", "lw", "oas", "ewma"]

@dataclass(frozen=True, slots=True)
class Periodicity:
    per_year: int  # 252, 52, 12 ...

PERIODICITY_DAY   = Periodicity(252)
PERIODICITY_WEEK  = Periodicity(52)
PERIODICITY_MONTH = Periodicity(12)

def _infer_periodicity(df_ret_wide: pl.DataFrame) -> Periodicity:
    """
    Infer periods/year from time deltas in 'date' col.
    """
    if "date" not in df_ret_wide.columns or df_ret_wide.height < 2:
        return PERIODICITY_DAY
    d = (
        df_ret_wide.select(pl.col("date").diff().dt.total_days())
                   .drop_nulls()
                   .to_series()
    )
    if d.is_empty():
        return PERIODICITY_DAY
    med_days = float(pl.Series(d).median())
    if med_days <= 3.0:
        return PERIODICITY_DAY
    if med_days <= 7.0:
        return PERIODICITY_WEEK
    return PERIODICITY_MONTH

def _wide_to_matrix(
    df_ret_wide: pl.DataFrame,
    fill: Literal["drop", "mean"] = "drop",
) -> Tuple[np.ndarray, list[str]]:
    """
    Convierte retornos anchos → matriz NumPy (T,N) y lista de tickers.
    Estrategias de NaN:
      - "drop": elimina filas con cualquier NaN (robusto para Σ/μ).
      - "mean": imputa NaN con la media de la columna antes de centrar.
    """
    tickers = [c for c in df_ret_wide.columns if c != "date"]
    X = df_ret_wide.select(tickers).to_numpy()  # (T, N)
    if X.size == 0:
        return X, tickers
    if fill == "drop":
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
    elif fill == "mean":
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
    else:
        raise ValueError("fill must be 'drop' or 'mean'")
    return X, tickers

def _ema_last(x: np.ndarray, span: int) -> float:
    """
    EWA de una serie 1D y devuelve el último valor (α = 2/(span+1)).
    """
    if x.size == 0:
        return np.nan
    alpha = 2.0 / (span + 1.0)
    s = 0.0
    w = 0.0
    # Inicialización estable: arranca en el primer no-NaN
    it = np.flatnonzero(~np.isnan(x))
    if it.size == 0:
        return np.nan
    s = x[it[0]]
    for i in range(it[0] + 1, x.shape[0]):
        xi = x[i]
        if np.isnan(xi):
            continue
        s = alpha * xi + (1.0 - alpha) * s
        w = 1.0  # marcador
    return s if w or it.size > 0 else np.nan

def _ensure_psd(Sigma: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Proyección PSD simple: clip de autovalores negativos y reconstrucción.
    """
    if Sigma.size == 0:
        return Sigma
    # simetriza para estabilidad numérica
    A = 0.5 * (Sigma + Sigma.T)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals_clipped = np.clip(eigvals, eps, None)
    A_psd = (eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T)
    # re-simetriza por seguridad
    return 0.5 * (A_psd + A_psd.T)

# ──────────────────────────────────────────────────────────────────────────────
# Expected returns (μ)
# ──────────────────────────────────────────────────────────────────────────────

def expected_returns(
    df_ret_wide: pl.DataFrame,
    *,
    method: MuMethod = "ema",
    span: Optional[int] = 60,
    shrink_to: Optional[np.ndarray] = None,
    annualize: bool = True,
    fill: Literal["drop", "mean"] = "drop",
) -> Tuple[np.ndarray, list[str]]:
    """
    Calcula μ con varios métodos. Devuelve (mu_vec, tickers), ambos alineados.

    - historical: media muestral.
    - ema: media exponencial, usa _ema_last por columna.
    - shrunk: shrinkage convexo hacia 'shrink_to' (vector objetivo); si no se pasa, target=0.
              Peso lam por defecto heurístico a partir de varianzas.

    Parámetros
    ----------
    df_ret_wide : pl.DataFrame   ['date', T1, T2, ...]
    span : int | None            ventana EMA (por defecto 60 periodos)
    shrink_to : np.ndarray       vector objetivo (N,)
    annualize : bool             si True, escala por periods/year inferidos
    fill : {"drop","mean"}       tratamiento de NaNs antes de μ
    """
    X, names = _wide_to_matrix(df_ret_wide, fill=fill)
    if X.size == 0:
        return np.array([]), names

    per = _infer_periodicity(df_ret_wide)
    ann = float(per.per_year) if annualize else 1.0

    if method == "historical":
        mu = np.nanmean(X, axis=0)

    elif method == "ema":
        s = int(span or 60)
        mu = np.array([_ema_last(X[:, j], s) for j in range(X.shape[1])], dtype=float)

    elif method == "shrunk":
        base = np.nanmean(X, axis=0)
        target = shrink_to if shrink_to is not None else np.zeros_like(base)
        # Heurística de λ: var media / (var media + var(base))
        var_cols = np.nanvar(X, axis=0, ddof=1)
        var_mean = float(np.nanmean(var_cols))
        base_var = float(np.var(base)) if base.size else 0.0
        lam = 0.0 if (var_mean + base_var) == 0 else var_mean / (var_mean + base_var)
        lam = float(np.clip(lam, 0.0, 1.0))
        mu = (1.0 - lam) * base + lam * target

    else:
        raise ValueError(f"Unknown expected return method: {method}")

    return mu * ann, names

# ──────────────────────────────────────────────────────────────────────────────
# Covariance (Σ)
# ──────────────────────────────────────────────────────────────────────────────

def covariance(
    df_ret_wide: pl.DataFrame,
    *,
    method: CovMethod = "oas",
    ewma_lambda: float = 0.94,
    min_periods: int = 60,
    annualize: bool = True,
    fill: Literal["drop", "mean"] = "drop",
    psd: bool = True,
) -> Tuple[np.ndarray, list[str]]:
    """
    Estima Σ con varios métodos. Devuelve (Sigma, tickers).

    - sample: covarianza muestral (ddof=1)
    - lw: Ledoit-Wolf
    - oas: Oracle Approximating Shrinkage
    - ewma: RiskMetrics-style (λ≈0.94 diaria). Centra con media muestral.

    NaNs:
      - "drop": elimina cualquier fila con NaN.
      - "mean": imputa NaN por media de columna (rápido y estable).

    Si psd=True, proyecta Σ a PSD por clipping de autovalores.
    """
    X, names = _wide_to_matrix(df_ret_wide, fill=fill)
    T, N = X.shape if X.ndim == 2 else (0, 0)
    if T < max(2, min_periods):
        # fallback seguro
        method = "sample"

    per = _infer_periodicity(df_ret_wide)
    ann = float(per.per_year) if annualize else 1.0

    if method == "sample":
        Sigma = np.cov(X, rowvar=False, ddof=1) if T > 1 else np.zeros((N, N))

    elif method == "lw":
        lw = LedoitWolf().fit(X)
        Sigma = lw.covariance_

    elif method == "oas":
        oas = OAS().fit(X)
        Sigma = oas.covariance_

    elif method == "ewma":
        lam = float(ewma_lambda)
        Xc = X - np.nanmean(X, axis=0, keepdims=True)
        # EWMA recursivo vectorizado: Σ_t = λ Σ_{t-1} + (1-λ) x_t x_t^T
        Sigma = np.zeros((N, N))
        for t in range(T):
            xt = Xc[t : t + 1, :]  # shape (1, N)
            if np.isnan(xt).any():
                # si aún quedan NaNs, sáltate o imputa 0; aquí optamos por skip
                continue
            Sigma = lam * Sigma + (1.0 - lam) * (xt.T @ xt)
    else:
        raise ValueError(f"Unknown covariance method: {method}")

    if psd:
        Sigma = _ensure_psd(Sigma)

    return Sigma * ann, names

# ──────────────────────────────────────────────────────────────────────────────
# Correlación
# ──────────────────────────────────────────────────────────────────────────────

def correlation_from_cov(Sigma: np.ndarray) -> np.ndarray:
    """
    Convierte Σ a ρ. Clipa divisiones por cero y simetriza.
    """
    if Sigma.size == 0:
        return Sigma
    d = np.sqrt(np.clip(np.diag(Sigma), 0.0, None))
    d[d == 0.0] = 1e-16
    Corr = (Sigma / d[:, None]) / d[None, :]
    Corr = np.clip(Corr, -1.0, 1.0)
    return 0.5 * (Corr + Corr.T)

# ──────────────────────────────────────────────────────────────────────────────
# Black-Litterman (prior + views)
# ──────────────────────────────────────────────────────────────────────────────

def black_litterman(
    mu_prior: np.ndarray,        # (N,)
    Sigma: np.ndarray,           # (N,N)
    P: np.ndarray,               # (K,N)  — pick matrix
    Q: np.ndarray,               # (K,)   — views
    *,
    tau: float = 0.05,           # escala de incertidumbre del prior
    Omega: Optional[np.ndarray] = None,  # (K,K) incertidumbre de vistas; si None, diag proporcional
) -> np.ndarray:
    """
    Devuelve μ_post (N,) según Black-Litterman en forma canónica (no bayes empírica).
    """
    N = mu_prior.shape[0]
    K = P.shape[0]
    Sigma_tau = tau * Sigma

    if Omega is None:
        # Heurística: Ω = diag( diag(P Σ τ P^T) )
        Omega = np.diag(np.diag(P @ Sigma_tau @ P.T))
        # Evitar ceros
        Omega = Omega + np.eye(K) * 1e-12

    inv_term = np.linalg.inv(P @ Sigma_tau @ P.T + Omega)
    middle = Sigma_tau @ P.T @ inv_term
    mu_post = mu_prior + middle @ (Q - P @ mu_prior)
    return mu_post

# ──────────────────────────────────────────────────────────────────────────────
# Wrappers convenientes para la UI
# ──────────────────────────────────────────────────────────────────────────────

def compute_mu_sigma(
    df_ret_wide: pl.DataFrame,
    *,
    mu_method: MuMethod = "ema",
    mu_span: Optional[int] = 60,
    mu_shrink_to: Optional[np.ndarray] = None,
    cov_method: CovMethod = "oas",
    ewma_lambda: float = 0.94,
    min_periods: int = 60,
    annualize: bool = True,
    fill: Literal["drop", "mean"] = "drop",
    psd: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Atajo para obtener (μ, Σ, tickers) con políticas coherentes de NaN/annualización/PSD.
    """
    mu, names = expected_returns(
        df_ret_wide,
        method=mu_method,
        span=mu_span,
        shrink_to=mu_shrink_to,
        annualize=annualize,
        fill=fill,
    )
    Sigma, names2 = covariance(
        df_ret_wide,
        method=cov_method,
        ewma_lambda=ewma_lambda,
        min_periods=min_periods,
        annualize=annualize,
        fill=fill,
        psd=psd,
    )
    if names != names2:
        raise RuntimeError("Ticker order mismatch between μ and Σ.")
    return mu, Sigma, names
