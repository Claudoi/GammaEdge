# portfolio/features/risk_models.py
from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
import polars as pl
from sklearn.covariance import OAS, LedoitWolf

# ──────────────────────────────────────────────────────────────────────────────
# Tipos y utilidades generales
# ──────────────────────────────────────────────────────────────────────────────

MuMethod = Literal["historical", "ema", "shrunk"]
CovMethod = Literal["sample", "lw", "oas", "ewma"]


class Periodicity(Enum):
    D = "D"
    W = "W"
    M = "M"
    Q = "Q"

    @property
    def per_year(self) -> int:
        return {"D": 252, "W": 52, "M": 12, "Q": 4}[self.value]


# Constantes prácticas (evitamos construir con argumentos “extras”)
PERIODICITY_DAY = Periodicity.D
PERIODICITY_WEEK = Periodicity.W
PERIODICITY_MONTH = Periodicity.M
PERIODICITY_QUARTER = Periodicity.Q


def _to_float(x: Any) -> float:
    """Conversión robusta a float (numpy/pandas friendly)."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, np.ndarray):
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        return float(arr[0]) if arr.size else float("nan")
    if hasattr(x, "item"):
        try:
            return float(x.item())  # numpy scalar
        except Exception:
            pass
    return float(x)


def _infer_periodicity(df_ret_wide: pl.DataFrame | pd.DataFrame) -> Periodicity:
    """
    Infer periods/year from time deltas in 'date' column (Polars o Pandas).
    """
    if isinstance(df_ret_wide, pd.DataFrame):
        if "date" not in df_ret_wide.columns or len(df_ret_wide) < 2:
            return PERIODICITY_DAY
        dates = pd.to_datetime(df_ret_wide["date"], errors="coerce").dropna()
        if len(dates) < 2:
            return PERIODICITY_DAY
        # diferencias en días (como floats)
        vals = dates.to_numpy(dtype="datetime64[ns]")
        deltas = np.diff(vals).astype("timedelta64[D]").astype(float)
        if deltas.size == 0:
            return PERIODICITY_DAY
        med_days = float(np.median(deltas))
    else:
        if "date" not in df_ret_wide.columns or df_ret_wide.height < 2:
            return PERIODICITY_DAY
        d = df_ret_wide.select(pl.col("date").diff().dt.total_days()).drop_nulls().to_series()
        if d.is_empty():
            return PERIODICITY_DAY
        m = pl.Series(d).median()
        # mypy puede tipar “m” como Any: convertimos con helper
        med_days = _to_float(0.0 if m is None else m)

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
    df_ret_wide: pl.DataFrame | pd.DataFrame,
    fill: Literal["drop", "mean", "none"] = "drop",
) -> tuple[np.ndarray, list[str]]:
    """
    Convierte retornos anchos (Polars o Pandas) → (matriz T x N float64, lista de tickers).
    NaN policy:
      - "drop": elimina filas con cualquier NaN
      - "mean": imputa con la media de col
      - "none": valida que no existan NaNs
    """
    if isinstance(df_ret_wide, pd.DataFrame):
        tickers = [c for c in df_ret_wide.columns if c != "date"]
        if not tickers:
            return np.empty((0, 0), dtype=np.float64), []
        X = df_ret_wide[tickers].to_numpy(dtype=np.float64)
    else:
        tickers = [c for c in df_ret_wide.columns if c != "date"]
        if not tickers:
            return np.empty((0, 0), dtype=np.float64), []
        X = np.asarray(df_ret_wide.select(tickers).to_numpy(), dtype=np.float64)

    X[~np.isfinite(X)] = np.nan
    if X.size == 0:
        return X, tickers

    if fill == "drop":
        X = X[~np.isnan(X).any(axis=1)]
    elif fill == "mean":
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        if inds[0].size:
            X[inds] = np.take(col_mean, inds[1])
    elif fill == "none":
        if np.isnan(X).any():
            raise ValueError("NaNs present but fill='none'. Pre-clean your data before calling.")
    else:
        raise ValueError("fill must be 'drop', 'mean' or 'none'.")

    return np.ascontiguousarray(X, dtype=np.float64), tickers


def _ema_last(x: np.ndarray, span: int) -> float:
    """
    EWA de una serie 1D y devuelve el último valor (α = 2/(span+1)).
    """
    if x.size == 0:
        return float("nan")
    alpha = 2.0 / (span + 1.0)
    it = np.flatnonzero(~np.isnan(x))
    if it.size == 0:
        return float("nan")
    s: float = float(x[it[0]])
    for i in range(it[0] + 1, x.shape[0]):
        xi = x[i]
        if np.isnan(xi):
            continue
        s = alpha * float(xi) + (1.0 - alpha) * s
    return s


def _ensure_psd(Sigma: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Proyección PSD por clipping de autovalores.
    """
    S = np.asarray(Sigma, dtype=np.float64)
    if S.size == 0:
        return S
    A = np.asarray(0.5 * (S + S.T), dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.asarray(np.clip(eigvals, eps, None), dtype=np.float64)
    eigvecs = np.asarray(eigvecs, dtype=np.float64)
    A_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    A_psd = np.asarray(A_psd, dtype=np.float64)
    return np.asarray(0.5 * (A_psd + A_psd.T), dtype=np.float64)


def apply_ridge(Sigma: np.ndarray, eps: float) -> np.ndarray:
    """
    Ridge diagonal: Σ + ε I.
    """
    # Aseguramos ndarray bien tipado siempre
    S = np.asarray(Sigma, dtype=np.float64)
    if eps <= 0:
        return S
    n = int(S.shape[0])
    ridge = S + np.eye(n, dtype=np.float64) * float(eps)
    return np.asarray(ridge, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Expected returns (μ)
# ──────────────────────────────────────────────────────────────────────────────


def expected_returns(
    df_ret_wide: pl.DataFrame | pd.DataFrame,
    *,
    method: MuMethod = "ema",
    span: int | None = 60,
    shrink_to: np.ndarray | None = None,
    annualize: bool = True,
    fill: Literal["drop", "mean", "none"] = "drop",
) -> pd.Series:
    """
    Calcula el vector de retornos esperados μ y lo devuelve como un pd.Series.
    """
    X, names = _wide_to_matrix(df_ret_wide, fill=fill)
    if X.size == 0:
        return pd.Series([], index=names, dtype=float, name="expected_return")

    per = _infer_periodicity(df_ret_wide)
    ann = float(per.per_year) if annualize else 1.0

    if method == "historical":
        mu = np.nanmean(X, axis=0)

    elif method == "ema":
        s = int(span or 60)
        mu = np.array([_ema_last(X[:, j], s) for j in range(X.shape[1])], dtype=np.float64)

    elif method == "shrunk":
        base = np.nanmean(X, axis=0)
        target = shrink_to if shrink_to is not None else np.zeros_like(base)
        var_cols = np.nanvar(X, axis=0, ddof=1)
        var_mean = _to_float(np.nanmean(var_cols))
        base_var = _to_float(np.var(base)) if base.size else 0.0
        lam = 0.0 if (var_mean + base_var) == 0 else var_mean / (var_mean + base_var)
        lam = float(np.clip(lam, 0.0, 1.0))
        mu = (1.0 - lam) * base + lam * target

    else:
        raise ValueError(f"Unknown expected return method: {method}")

    return pd.Series(np.asarray(mu * ann, dtype=np.float64), index=names, name="expected_return")


# ──────────────────────────────────────────────────────────────────────────────
# Covariance (Σ) — overloads para retorno tipado
# ──────────────────────────────────────────────────────────────────────────────


@overload
def covariance(
    df_ret_wide: pl.DataFrame | pd.DataFrame,
    *,
    method: CovMethod = "oas",
    ewma_lambda: float = 0.94,
    min_periods: int = 60,
    annualize: bool = True,
    fill: Literal["drop", "mean", "none"] = "drop",
    psd: bool = True,
    as_frame: Literal[True] = True,
) -> pd.DataFrame: ...
@overload
def covariance(
    df_ret_wide: pl.DataFrame | pd.DataFrame,
    *,
    method: CovMethod = "oas",
    ewma_lambda: float = 0.94,
    min_periods: int = 60,
    annualize: bool = True,
    fill: Literal["drop", "mean", "none"] = "drop",
    psd: bool = True,
    as_frame: Literal[False],
) -> tuple[np.ndarray, list[str]]: ...


def covariance(
    df_ret_wide: pl.DataFrame | pd.DataFrame,
    *,
    method: CovMethod = "oas",
    ewma_lambda: float = 0.94,
    min_periods: int = 60,
    annualize: bool = True,
    fill: Literal["drop", "mean", "none"] = "drop",
    psd: bool = True,
    as_frame: bool = True,
) -> pd.DataFrame | tuple[np.ndarray, list[str]]:
    """
    Estimate covariance matrix Σ.
    - sample / lw / oas / ewma
    """
    X, names = _wide_to_matrix(df_ret_wide, fill=fill)
    T, N = X.shape if X.ndim == 2 else (0, 0)

    if max(2, min_periods) > T and method in ("lw", "oas", "ewma"):
        method = "sample"

    per = _infer_periodicity(df_ret_wide)
    ann = float(per.per_year) if annualize else 1.0

    if method == "sample":
        Sigma = np.cov(X, rowvar=False, ddof=1) if T > 1 else np.zeros((N, N), dtype=np.float64)
        Sigma = np.asarray(Sigma, dtype=np.float64)

    elif method == "lw":
        if np.isnan(X).any():
            raise ValueError("NaNs not allowed in lw; use fill='drop' or 'mean'.")
        Sigma = np.asarray(LedoitWolf().fit(X).covariance_, dtype=np.float64)

    elif method == "oas":
        if np.isnan(X).any():
            raise ValueError("NaNs not allowed in oas; use fill='drop' or 'mean'.")
        Sigma = np.asarray(OAS().fit(X).covariance_, dtype=np.float64)
    elif method == "ewma":
        lam = float(ewma_lambda)
        mu = np.nanmean(X, axis=0, keepdims=True)
        Xc = X - mu
        Sigma = np.zeros((N, N), dtype=np.float64)
        for t in range(T):
            xt = Xc[t : t + 1, :]
            if np.isnan(xt).any():
                continue
            Sigma = lam * Sigma + (1.0 - lam) * (xt.T @ xt)
    else:
        raise ValueError(f"Unknown covariance method: {method}")

    if psd:
        Sigma = _ensure_psd(np.asarray(Sigma, dtype=np.float64))

    Sigma = np.asarray(Sigma * ann, dtype=np.float64)
    if as_frame:
        return pd.DataFrame(Sigma, index=names, columns=names)
    return Sigma, names


# ──────────────────────────────────────────────────────────────────────────────
# Correlación
# ──────────────────────────────────────────────────────────────────────────────


def correlation_from_cov(Sigma: np.ndarray) -> np.ndarray:
    """
    Convierte Σ a ρ. Clipa divisiones por cero y simetriza.
    """
    S = np.asarray(Sigma, dtype=np.float64)
    if S.size == 0:
        return S
    d = np.sqrt(np.clip(np.diag(S), 0.0, None))
    d[d == 0.0] = 1e-16
    Corr = (S / d[:, None]) / d[None, :]
    Corr = np.clip(Corr, -1.0, 1.0)
    Corr_sym = 0.5 * (Corr + Corr.T)
    return np.asarray(Corr_sym, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Black-Litterman (posterior μ) — forma canónica
# ──────────────────────────────────────────────────────────────────────────────


def black_litterman(
    mu_prior: np.ndarray,  # (N,)
    Sigma: np.ndarray,  # (N,N)
    P: np.ndarray,  # (K,N)
    Q: np.ndarray,  # (K,)
    *,
    tau: float = 0.05,
    Omega: np.ndarray | None = None,  # (K,K)
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
    return np.asarray(mu_post, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Wrappers convenientes para la UI
# ──────────────────────────────────────────────────────────────────────────────


def compute_mu_sigma(
    df_ret_wide: pl.DataFrame | pd.DataFrame,
    *,
    mu_method: MuMethod = "ema",
    mu_span: int | None = 60,
    mu_shrink_to: np.ndarray | None = None,
    cov_method: CovMethod = "oas",
    ewma_lambda: float = 0.94,
    min_periods: int = 60,
    annualize: bool = True,
    fill: Literal["drop", "mean", "none"] = "drop",
    psd: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    s = expected_returns(
        df_ret_wide,
        method=mu_method,
        span=mu_span,
        shrink_to=mu_shrink_to,
        annualize=annualize,
        fill=fill,
    )
    mu = np.asarray(s.to_numpy(dtype=np.float64))
    names = list(s.index)

    Sigma, names2 = covariance(
        df_ret_wide,
        method=cov_method,
        ewma_lambda=ewma_lambda,
        min_periods=min_periods,
        annualize=annualize,
        fill=fill,
        psd=psd,
        as_frame=False,
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
    rf: float = 0.0,  # por periodo (misma frecuencia que df_ret_wide)
    fill: Literal["drop", "mean", "none"] = "drop",
    annualize: bool = True,
) -> np.ndarray:
    """
    Estima μ vía CAPM: μ_i = rf + β_i * (E[R_m] - rf).
    """
    tickers = [c for c in df_ret_wide.columns if c != "date"]
    if not tickers:
        return np.array([], dtype=np.float64)

    X, _ = _wide_to_matrix(df_ret_wide.select(["date"] + tickers), fill=fill)  # (T,N)

    if market in df_ret_wide.columns:
        Rm_mat, _ = _wide_to_matrix(
            df_ret_wide.select(["date", market]).rename({market: "MKT"}), fill=fill
        )
        Rm = Rm_mat[:, 0] if Rm_mat.size else np.array([], dtype=np.float64)
    else:
        Rm = np.nanmean(X, axis=1) if X.size else np.array([], dtype=np.float64)

    if X.size == 0 or Rm.size == 0:
        mu_hist = expected_returns(
            df_ret_wide, method="historical", annualize=annualize, fill=fill
        ).to_numpy(dtype=np.float64)
        return np.asarray(mu_hist, dtype=np.float64)

    Er_m = float(np.nanmean(Rm))
    Rm_c = Rm - np.nanmean(Rm)
    var_m = float(np.nanvar(Rm_c, ddof=1))
    if var_m <= 0:
        mu_hist = expected_returns(
            df_ret_wide, method="historical", annualize=annualize, fill=fill
        ).to_numpy(dtype=np.float64)
        return np.asarray(mu_hist, dtype=np.float64)

    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    cov_im = Xc * Rm_c[:, None]
    beta = np.nanmean(cov_im, axis=0) / var_m  # (N,)

    mu_period = rf + beta * (Er_m - rf)

    per = _infer_periodicity(df_ret_wide)
    ann = float(per.per_year) if annualize else 1.0
    return np.asarray(mu_period * ann, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Black–Litterman prior μ (equilibrium)
# ──────────────────────────────────────────────────────────────────────────────


def black_litterman_mu(
    data_or_names: pl.DataFrame | Sequence[str],
    *,
    Sigma: np.ndarray | None = None,
    market_weights: np.ndarray | None = None,
    delta: float = 2.5,
    annualize: bool = True,
    fill: Literal["drop", "mean", "none"] = "drop",
    cov_method: CovMethod = "oas",
) -> np.ndarray:
    """
    Devuelve el prior de BL (equilibrium returns) π = δ Σ w_mkt.
    """
    if not isinstance(data_or_names, pl.DataFrame):
        names = list(data_or_names)
        return np.zeros(len(names), dtype=np.float64)

    df_ret_wide = data_or_names
    tickers = [c for c in df_ret_wide.columns if c != "date"]
    N = len(tickers)
    if N == 0:
        return np.array([], dtype=np.float64)

    if Sigma is None:
        Sigma_est, _ = covariance(
            df_ret_wide,
            method=cov_method,
            annualize=False,
            fill=fill,
            psd=True,
            as_frame=False,
        )
        Sigma = np.asarray(Sigma_est, dtype=np.float64)

    if market_weights is None:
        w = np.full(N, 1.0 / N, dtype=np.float64)
    else:
        w = np.asarray(market_weights, dtype=np.float64)
        if w.shape != (N,):
            raise ValueError("market_weights must have shape (N,)")
        s = float(np.sum(w))
        if s <= 0:
            raise ValueError("market_weights sum must be > 0")
        w = w / s

    pi = delta * (Sigma @ w)  # por periodo

    per = _infer_periodicity(df_ret_wide)
    ann = float(per.per_year) if annualize else 1.0
    return np.asarray(pi * ann, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# PCA factor covariance (Σ ≈ V_k Λ_k V_k^T + diag(specific))
# ──────────────────────────────────────────────────────────────────────────────


def pca_factor_cov(
    df_ret_wide: pl.DataFrame,
    *,
    mu_method: MuMethod = "ema",
    mu_span: int | None = 60,
    n_factors: int = 5,
    annualize: bool = False,
    fill: Literal["drop", "mean", "none"] = "drop",
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, object]]:
    """
    Estima μ (método indicado) y Σ vía modelo de factores PCA:
        S = V Λ V^T
        Σ ≈ V_k Λ_k V_k^T + diag( diag(S - V_k Λ_k V_k^T) )
    """
    X, names = _wide_to_matrix(df_ret_wide, fill=fill)  # (T,N)
    T, N = X.shape if X.ndim == 2 else (0, 0)
    if T < 2 or N == 0:
        s = expected_returns(
            df_ret_wide, method=mu_method, span=mu_span, annualize=annualize, fill=fill
        )
        mu = np.asarray(s.to_numpy(dtype=np.float64))
        return (
            mu,
            np.zeros((N, N), dtype=np.float64),
            names,
            {"eigvals": np.asarray([], dtype=np.float64), "explained": 0.0, "k": 0},
        )

    mu = expected_returns(
        df_ret_wide, method=mu_method, span=mu_span, annualize=False, fill=fill
    ).to_numpy(dtype=np.float64)

    S = np.cov(X, rowvar=False, ddof=1) if T > 1 else np.zeros((N, N), dtype=np.float64)
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
    Sigma = S_k + specific  # PSD

    if annualize:
        per = _infer_periodicity(df_ret_wide)
        a = float(per.per_year)
        mu = np.asarray(mu * a, dtype=np.float64)
        Sigma = np.asarray(Sigma * a, dtype=np.float64)

    total_var = float(np.sum(eigvals)) if eigvals.size else 0.0
    explained = float(np.sum(eigvals[:k]) / total_var) if total_var > 0 else 0.0

    eigvals = np.asarray(eigvals, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    info: dict[str, object] = {"eigvals": eigvals, "explained": explained, "k": k}
    return mu, Sigma, names, info


# ──────────────────────────────────────────────────────────────────────────────
# Diagnóstico / Métricas auxiliares
# ──────────────────────────────────────────────────────────────────────────────


def nan_policy_stats(
    df_ret_wide: pl.DataFrame,
    *,
    fill: Literal["drop", "mean", "none"] = "drop",
) -> dict[str, object]:
    """
    Reporta impacto de la política de NaN.
    """
    value_cols = [c for c in df_ret_wide.columns if c != "date"]
    X = np.asarray(df_ret_wide.select(value_cols).to_numpy(), dtype=float)
    X[~np.isfinite(X)] = np.nan

    out: dict[str, object] = {"policy": fill}
    if fill == "drop":
        before = X.shape[0]
        after = (~np.isnan(X).any(axis=1)).sum()
        out["rows_dropped"] = int(before - after)
        out["imputed_pct_by_ticker"] = {c: 0.0 for c in value_cols}
    elif fill == "mean":
        total_cells = X.size
        n_na = int(np.isnan(X).sum())
        imputed: dict[str, float] = {}
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
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    mrc = Sigma @ w
    rc = w * mrc
    if normalize:
        tot = float(w.T @ mrc)
        rc = rc / (tot if abs(tot) > 1e-16 else 1.0)
    return np.asarray(rc, dtype=np.float64)


def rolling_metrics(
    df_ret_wide: pl.DataFrame,
    *,
    window: int = 26,
    pair: tuple[str, str] | None = None,
) -> dict[str, pl.DataFrame]:
    """
    Métricas rolling:
      - vol (por activo): std rolling
      - corr (par de activos): corr rolling de un par
    """
    if "date" not in df_ret_wide.columns:
        raise ValueError("'date' column required.")
    tickers = [c for c in df_ret_wide.columns if c != "date"]

    lf = df_ret_wide.lazy().sort("date")
    vol_exprs = [pl.col(t).rolling_std(window, ddof=1).alias(t) for t in tickers]
    vol_df = lf.with_columns(vol_exprs).select(["date"] + tickers).collect()

    out: dict[str, pl.DataFrame] = {"vol": vol_df}

    if pair is not None:
        a, b = pair
        if a in tickers and b in tickers:
            ra = pl.col(a) - pl.col(a).rolling_mean(window)
            rb = pl.col(b) - pl.col(b).rolling_mean(window)
            cov = (ra * rb).rolling_mean(window)
            std_a = pl.col(a).rolling_std(window, ddof=1)
            std_b = pl.col(b).rolling_std(window, ddof=1)
            corr = (cov / (std_a * std_b)).alias("corr")
            corr_df = (
                lf.select(["date", a, b]).with_columns(corr).select(["date", "corr"]).collect()
            )
            out["corr"] = corr_df
    return out
