# portfolio/optim/robust.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def apply_ridge(Sigma: FloatArray, eps: float) -> FloatArray:
    """
    Añade una ridge (eps * I) a la covarianza.
    Garantiza dtype float64 y salida simétrica.
    """
    S = np.asarray(Sigma, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("Sigma must be a square (n x n) array.")
    if eps <= 0:
        return S
    n = S.shape[0]
    # np.eye devuelve float64; aseguramos float64 final
    return (S + np.eye(n, dtype=np.float64) * float(eps)).astype(np.float64, copy=False)


def shrink_mu(mu: FloatArray, target: FloatArray, lam: float) -> FloatArray:
    """
    Encoge el vector de medias hacia `target` con factor lam in [0,1].
    """
    m = np.asarray(mu, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    if m.shape != t.shape:
        raise ValueError(f"`mu` and `target` must have same shape, got {m.shape} vs {t.shape}.")
    lam_f = float(np.clip(lam, 0.0, 1.0))
    return ((1.0 - lam_f) * m + lam_f * t).astype(np.float64, copy=False)


# -----------------------------------------------------------------------------
# Random Matrix Theory (Marcenko-Pastur) Denoising
# -----------------------------------------------------------------------------
def marcenko_pastur_limits(T: int, N: int, var_eps: float = 1.0) -> tuple[float, float]:
    """
    Theoretical bounds [lambda_min, lambda_max] for the eigenvalues of a random
    correlation matrix (Wishart) with ratio Q = T/N.
    """
    q = float(T) / float(N)
    lambda_min = var_eps * (1 - np.sqrt(1.0 / q)) ** 2
    lambda_max = var_eps * (1 + np.sqrt(1.0 / q)) ** 2
    return lambda_min, lambda_max


def clean_covariance_rmt(Sigma: FloatArray, T: int, N: int) -> FloatArray:
    """
    Denoises the covariance matrix using Constant Residual Eigenvalue Method.
    1. Transforms Sigma -> Correlation C.
    2. Computes eigenvalues.
    3. Identifies noise eigenvalues (those below Marcenko-Pastur max chreshold).
    4. Replaces noise eigenvalues with their average.
    5. Transforms back -> Cleaned Sigma.

    Ref: Lopez de Prado, "Machine Learning for Asset Managers".
    """
    S = np.asarray(Sigma, dtype=np.float64)
    # Extract variance and correlation
    v = np.diag(S)
    std = np.sqrt(v).reshape(-1, 1)
    # correlation = S / (std * std.T)
    # Safe division
    C = S / (std @ std.T)
    np.fill_diagonal(C, 1.0)

    # Eigendecomposition
    # eigh for symmetric matrices
    evals, evecs = np.linalg.eigh(C)

    # Sort descending
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Determine cutoff
    _, lambda_max = marcenko_pastur_limits(T, N, var_eps=1.0)

    # Find noise eigenvalues (evals <= lambda_max)
    # In practice, empirical eigenvalues often bleed slightly past theoretical max due to finite size.
    # We treat all k eigenvalues <= lambda_max as noise?
    # Or typically the last N-k.
    # Logic: Signal eigenvalues are theoretically > lambda_max.

    n_signal = np.sum(evals > lambda_max)

    # If no signal found (all noise), return shrinkage or identity?
    if n_signal == 0:
        # Extreme case: everything is noise. Return Identity or heavily shrunk C?
        # Let's return Identity-like structure (Pure independence)
        C_clean = np.eye(N)
    elif n_signal == N:
        # All signal, no cleaning needed
        C_clean = C
    else:
        # Replace noise evals with their average
        evals_clean = evals.copy()
        mean_noise = np.mean(evals[n_signal:])
        evals_clean[n_signal:] = mean_noise

        # Reconstruct C
        # C = V Lambda V^T
        C_clean = evecs @ np.diag(evals_clean) @ evecs.T

        # Rescale diagonal to 1.0 (denoising changes diagonal)
        d = np.diag(C_clean)
        norm_factor = 1.0 / np.sqrt(d).reshape(-1, 1)
        C_clean = C_clean * (norm_factor @ norm_factor.T)
        np.fill_diagonal(C_clean, 1.0)

    # Transform back to Covariance
    Sigma_clean = C_clean * (std @ std.T)
    return Sigma_clean.astype(np.float64)
