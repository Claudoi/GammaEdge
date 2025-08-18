# portfolio/features/risk_models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Sequence, Dict

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
    Infer periods/year from time deltas in 'date' column.
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
    if med_days <= 9.0:
        return PERIODICITY_WEEK
    return PERIODICITY_MONTH


def ewma_default_lambda(periodicity: Periodicity | int) -> float:
    """
    Defaults de RiskMetrics por granularidad:
      - daily  ≈ 0.94
      - weekly ≈ 0.80
      - monthly≈ 0.60
    """
    per_year = periodicity.per_year if isinstance(periodicity, Periodicity) else int(periodicity)
    if per_year >= 250:
        return 0.94
    if per_year >= 50:
        return 0.80
    return 0.60


def _wide_to_matrix(
    df_ret_wide: pl.DataFrame,
    fill: Literal["drop", "mean", "none"] = "drop",
) -> Tuple[np.ndarray, list[str]]:
    """
    Convierte retornos anchos → matriz NumPy (T,N) float64 C-contiguous + lista de tickers.
    NaN policy:
      - "drop": elimina filas con cualquier NaN.
      - "mean": imputa NaN con la media de columna.
      - "none": no toca NaNs (se espera que ya estén resueltos aguas arriba).
    También normaliza ±inf → NaN antes de aplicar la política.
    """
    tickers = [c for c in df_ret_wide.columns if c != "date"]
    if not tickers:
        return np.empty((0, 0), dtype=np.float64), tickers

    X = df_ret_wide.select(tickers).to_numpy()  # (T, N)
    # Inf → NaN para tratamiento homogéneo
    X = np.asarray(X, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan

    if X.size == 0:
        return X, tickers

    if fill == "drop":
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
    elif fill == "mean":
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        if inds[0].size:
            X[inds] = np.take(col_mean, inds[1])
    elif fill == "none":
        # Validación ligera (útil para evitar fallos silenciosos en sklearn)
        if np.isnan(X).any():
            raise ValueError("NaNs present but fill='none'. Pre-clean your data before calling.")
    else:
        raise ValueError("fill must be 'drop', 'mean' or 'none'.")

    # Garantizamos float64 C-order (sklearn evita DataOrientationWarning/eficiencia)
    X = np.ascontiguousarray(X, dtype=np.float64)
    return X, tickers


def _ema_last(x: np.ndarray, span: int) -> float:
    """
    EWA de una serie 1D y devuelve el último valor (α = 2/(span+1)).
    """
    if x.size == 0:
        return np.nan
    alpha = 2.0 / (span + 1.0)
    # Inicialización estable: primer no-NaN
    it = np.flatnonzero(~np.isnan(x))
    if it.size == 0:
        return np.nan
    s = x[it[0]]
    for i in range(it[0] + 1, x.shape[0]):
        xi = x[i]
        if np.isnan(xi):
            continue
        s = alpha * xi + (1.0 - alpha) * s
    return s


def _ensure_psd(Sigma: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Proyección PSD por clipping de autovalores.
    """
    if Sigma.size == 0:
        return Sigma
    A = 0.5 * (Sigma + Sigma.T)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, eps, None)
    A_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return 0.5 * (A_psd + A_psd.T)


def apply_ridge(Sigma: np.ndarray, eps: float) -> np.ndarray:
    """
    Ridge diagonal: Σ + ε I. Útil para conditioning.
    """
    if eps <= 0:
        return Sigma
    n = Sigma.shape[0]
    return Sigma + np.eye(n) * float(eps)

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
    fill: Literal["drop", "mean", "none"] = "drop",
) -> Tuple[np.ndarray, list[str]]:
    """
    Calcula μ con varios métodos. Devuelve (mu_vec, tickers) alineados.

    - historical: media muestral.
    - ema: media exponencial, usa _ema_last por columna.
    - shrunk: shrinkage convexo hacia 'shrink_to'; si no se pasa, target=0.
              λ heurístico a partir de varianzas.
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
    fill: Literal["drop", "mean", "none"] = "drop",
    psd: bool = True,
) -> Tuple[np.ndarray, list[str]]:
    """
    Estima Σ con varios métodos. Devuelve (Sigma, tickers).

    - sample: covarianza muestral (ddof=1)
    - lw: Ledoit-Wolf
    - oas: Oracle Approximating Shrinkage
    - ewma: RiskMetrics-style (λ≈0.94 diaria). Centra con media muestral.

    Si fill="none", se asume que no existen NaNs (validado).
    """
    X, names = _wide_to_matrix(df_ret_wide, fill=fill)
    T, N = X.shape if X.ndim == 2 else (0, 0)

    if T < max(2, min_periods) and method in ("lw", "oas", "ewma"):
        # Fallback seguro cuando no hay suficientes observaciones
        method = "sample"

    per = _infer_periodicity(df_ret_wide)
    ann = float(per.per_year) if annualize else 1.0

    if method == "sample":
        Sigma = np.cov(X, rowvar=False, ddof=1) if T > 1 else np.zeros((N, N))

    elif method == "lw":
        if np.isnan(X).any():
            raise ValueError("NaNs not allowed in lw; use fill='drop' or 'mean'.")
        Sigma = LedoitWolf().fit(X).covariance_

    elif method == "oas":
        if np.isnan(X).any():
            raise ValueError("NaNs not allowed in oas; use fill='drop' or 'mean'.")
        Sigma = OAS().fit(X).covariance_

    elif method == "ewma":
        lam = float(ewma_lambda)
        # Centro con media muestral (NaN-safe si fill!='none')
        mu = np.nanmean(X, axis=0, keepdims=True)
        Xc = X - mu
        Sigma = np.zeros((N, N), dtype=np.float64)
        for t in range(T):
            xt = Xc[t : t + 1, :]
            if np.isnan(xt).any():
                # Si aún quedan NaNs con fill='none', saltamos esa fila
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
# Black-Litterman (posterior μ) — forma canónica
# ──────────────────────────────────────────────────────────────────────────────

def black_litterman(
    mu_prior: np.ndarray,        # (N,)
    Sigma: np.ndarray,           # (N,N)
    P: np.ndarray,               # (K,N) — pick matrix
    Q: np.ndarray,               # (K,)   — views
    *,
    tau: float = 0.05,           # incertidumbre del prior
    Omega: Optional[np.ndarray] = None,  # (K,K) incertidumbre de vistas; si None, diag proporcional
) -> np.ndarray:
    """
    Devuelve μ_post (N,) según Black–Litterman en forma canónica.
    """
    K = P.shape[0]
    Sigma_tau = tau * Sigma

    if Omega is None:
        Omega = np.diag(np.diag(P @ Sigma_tau @ P.T)) + np.eye(K) * 1e-12

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
    fill: Literal["drop", "mean", "none"] = "drop",
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

# ──────────────────────────────────────────────────────────────────────────────
# CAPM μ (rf + beta_i * (E[Rm] - rf))
# ──────────────────────────────────────────────────────────────────────────────

def capm_mu(
    df_ret_wide: pl.DataFrame,
    *,
    market: str = "SPY",
    rf: float = 0.0,             # por periodo (misma frecuencia que df_ret_wide)
    fill: Literal["drop", "mean", "none"] = "drop",
    annualize: bool = True,
) -> np.ndarray:
    """
    Estima μ vía CAPM: μ_i = rf + β_i * (E[R_m] - rf).

    - Si `market` está en columnas, lo usa como índice de mercado.
    - Si NO está, usa un proxy: media cross-sectional (equal-weight) de los activos.
    - rf es por periodo; si annualize=True se escala con per_year inferido.

    Devuelve: vector μ (N,) alineado con los tickers (todas las cols != 'date').
    """
    tickers = [c for c in df_ret_wide.columns if c != "date"]
    if not tickers:
        return np.array([])

    X, _ = _wide_to_matrix(df_ret_wide.select(["date"] + tickers), fill=fill)  # (T,N)

    if market in df_ret_wide.columns:
        Rm, _ = _wide_to_matrix(
            df_ret_wide.select(["date", market]).rename({market: "MKT"}), fill=fill
        )
        Rm = Rm[:, 0] if Rm.size else np.array([])
    else:
        Rm = np.nanmean(X, axis=1) if X.size else np.array([])

    if X.size == 0 or Rm.size == 0:
        mu_hist, _ = expected_returns(df_ret_wide, method="historical", annualize=annualize, fill=fill)
        return mu_hist

    Er_m = float(np.nanmean(Rm))
    Rm_c = Rm - np.nanmean(Rm)
    var_m = float(np.nanvar(Rm_c, ddof=1))
    if var_m <= 0:
        mu_hist, _ = expected_returns(df_ret_wide, method="historical", annualize=annualize, fill=fill)
        return mu_hist

    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    cov_im = (Xc * Rm_c[:, None])
    beta = np.nanmean(cov_im, axis=0) / var_m  # (N,)

    mu_period = rf + beta * (Er_m - rf)

    per = _infer_periodicity(df_ret_wide)
    ann = float(per.per_year) if annualize else 1.0
    return mu_period * ann

# ──────────────────────────────────────────────────────────────────────────────
# Black–Litterman prior μ (equilibrium)
# ──────────────────────────────────────────────────────────────────────────────

def black_litterman_mu(
    data_or_names: pl.DataFrame | Sequence[str],
    *,
    Sigma: Optional[np.ndarray] = None,
    market_weights: Optional[np.ndarray] = None,
    delta: float = 2.5,          # aversión al riesgo típica
    annualize: bool = True,
    fill: Literal["drop", "mean", "none"] = "drop",
    cov_method: CovMethod = "oas",
) -> np.ndarray:
    """
    Devuelve el prior de BL (equilibrium returns) π = δ Σ w_mkt.

    Usos:
    - black_litterman_mu(df_ret_wide, Sigma=..., market_weights=..., delta=2.5)
    - black_litterman_mu(df_ret_wide)  → estima Σ con OAS y w_mkt = equal-weight
    - black_litterman_mu(names:list[str]) → vector cero (compatibilidad)
    """
    if not isinstance(data_or_names, pl.DataFrame):
        names = list(data_or_names)
        return np.zeros(len(names), dtype=float)

    df_ret_wide = data_or_names
    tickers = [c for c in df_ret_wide.columns if c != "date"]
    N = len(tickers)
    if N == 0:
        return np.array([])

    if Sigma is None:
        Sigma, _ = covariance(
            df_ret_wide,
            method=cov_method,
            annualize=False,
            fill=fill,
            psd=True,
        )

    if market_weights is None:
        w = np.full(N, 1.0 / N, dtype=float)
    else:
        w = np.asarray(market_weights, dtype=float)
        if w.shape != (N,):
            raise ValueError("market_weights must have shape (N,)")
        s = float(np.sum(w))
        if s <= 0:
            raise ValueError("market_weights sum must be > 0")
        w = w / s

    pi = delta * (Sigma @ w)  # por periodo

    per = _infer_periodicity(df_ret_wide)
    ann = float(per.per_year) if annualize else 1.0
    return pi * ann

# ──────────────────────────────────────────────────────────────────────────────
# PCA factor covariance (Σ ≈ V_k Λ_k V_k^T + diag(specific))
# ──────────────────────────────────────────────────────────────────────────────

def pca_factor_cov(
    df_ret_wide: pl.DataFrame,
    *,
    mu_method: MuMethod = "ema",
    mu_span: Optional[int] = 60,
    n_factors: int = 5,
    annualize: bool = False,
    fill: Literal["drop", "mean", "none"] = "drop",
) -> Tuple[np.ndarray, np.ndarray, list[str], Dict[str, object]]:
    """
    Estima μ (método indicado) y Σ vía modelo de factores PCA:
        S = V Λ V^T  (eigendecomp de la cov muestral)
        Σ ≈ V_k Λ_k V_k^T + diag( diag(S - V_k Λ_k V_k^T) )
    """
    X, names = _wide_to_matrix(df_ret_wide, fill=fill)  # (T,N)
    T, N = X.shape if X.ndim == 2 else (0, 0)
    if T < 2 or N == 0:
        mu, _ = expected_returns(df_ret_wide, method=mu_method, span=mu_span, annualize=annualize, fill=fill)
        return mu, np.zeros((N, N)), names, {"eigvals": np.array([]), "explained": 0.0, "k": 0}

    mu, _ = expected_returns(
        df_ret_wide, method=mu_method, span=mu_span, annualize=False, fill=fill
    )

    S = np.cov(X, rowvar=False, ddof=1) if T > 1 else np.zeros((N, N))
    S_sym = 0.5 * (S + S.T)
    eigvals, eigvecs = np.linalg.eigh(S_sym)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    k = int(np.clip(n_factors, 1, N))
    Vk = eigvecs[:, :k]
    Lk = np.diag(np.clip(eigvals[:k], 0.0, None))
    S_k = Vk @ Lk @ Vk.T
    specific = np.diag(np.clip(np.diag(S - S_k), 1e-12, None))
    Sigma = S_k + specific  # PSD por construcción

    if annualize:
        per = _infer_periodicity(df_ret_wide)
        a = float(per.per_year)
        mu = mu * a
        Sigma = Sigma * a

    total_var = float(np.sum(eigvals)) if eigvals.size else 0.0
    explained = float(np.sum(eigvals[:k]) / total_var) if total_var > 0 else 0.0

    Sigma = _ensure_psd(Sigma)
    info = {"eigvals": eigvals, "explained": explained, "k": k}
    return mu, Sigma, names, info

# ──────────────────────────────────────────────────────────────────────────────
# Diagnóstico / Métricas auxiliares
# ──────────────────────────────────────────────────────────────────────────────

def nan_policy_stats(
    df_ret_wide: pl.DataFrame,
    *,
    fill: Literal["drop", "mean", "none"] = "drop",
) -> Dict[str, object]:
    """
    Reporta impacto de la política de NaN:
      - drop: nº de filas eliminadas
      - mean: % imputado por ticker
      - none: solo valida que no existan NaNs (o reporta cuántos)
    """
    value_cols = [c for c in df_ret_wide.columns if c != "date"]
    X = df_ret_wide.select(value_cols).to_numpy()
    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = np.nan

    out: Dict[str, object] = {"policy": fill}
    if fill == "drop":
        before = X.shape[0]
        after = (~np.isnan(X).any(axis=1)).sum()
        out["rows_dropped"] = int(before - after)
        out["imputed_pct_by_ticker"] = {c: 0.0 for c in value_cols}
    elif fill == "mean":
        total_cells = X.size
        n_na = int(np.isnan(X).sum())
        # ratio por ticker
        imputed = {}
        for j, c in enumerate(value_cols):
            col = X[:, j]
            imputed[c] = float(np.isnan(col).sum()) / max(col.size, 1) * 100.0
        out["total_imputed_pct"] = float(n_na) / max(total_cells, 1) * 100.0
        out["imputed_pct_by_ticker"] = imputed
    elif fill == "none":
        n_na = int(np.isnan(X).sum())
        out["n_na_total"] = n_na
        if n_na > 0:
            out["warning"] = "NaNs present with fill='none'."
    else:
        raise ValueError("fill must be 'drop', 'mean' or 'none'.")
    return out


def risk_contributions(
    w: np.ndarray,
    Sigma: np.ndarray,
    *,
    normalize: bool = False,
) -> np.ndarray:
    """
    Devuelve contribución al riesgo por activo:
      RC_i = w_i * (Σ w)_i
    Si normalize=True, divide por wᵀΣw para obtener % que suma 1.
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    mrc = Sigma @ w
    rc = w * mrc
    if normalize:
        tot = float(w.T @ mrc)
        return rc / (tot if abs(tot) > 1e-16 else 1.0)
    return rc


def rolling_metrics(
    df_ret_wide: pl.DataFrame,
    *,
    window: int = 26,
    pair: Optional[Tuple[str, str]] = None,
) -> Dict[str, pl.DataFrame]:
    """
    Métricas rolling:
      - vol (por activo): std rolling
      - corr (par de activos): corr rolling de un par
    Devuelve dict con DataFrames en Polars.
    """
    if "date" not in df_ret_wide.columns:
        raise ValueError("'date' column required.")
    tickers = [c for c in df_ret_wide.columns if c != "date"]

    lf = df_ret_wide.lazy().sort("date")
    # rolling std por activo
    vol_exprs = [pl.col(t).rolling_std(window, ddof=1).alias(t) for t in tickers]
    vol_df = lf.with_columns(vol_exprs).select(["date"] + tickers).collect()

    out: Dict[str, pl.DataFrame] = {"vol": vol_df}

    if pair is not None:
        a, b = pair
        if a in tickers and b in tickers:
            # Corr rolling = cov / (std_a * std_b)
            ra = pl.col(a) - pl.col(a).rolling_mean(window)
            rb = pl.col(b) - pl.col(b).rolling_mean(window)
            cov = (ra * rb).rolling_mean(window)
            std_a = pl.col(a).rolling_std(window, ddof=1)
            std_b = pl.col(b).rolling_std(window, ddof=1)
            corr = (cov / (std_a * std_b)).alias("corr")
            corr_df = lf.select(["date", a, b]).with_columns(corr).select(["date", "corr"]).collect()
            out["corr"] = corr_df
    return out
