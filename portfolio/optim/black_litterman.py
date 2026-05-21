# portfolio/optim/black_litterman.py
from __future__ import annotations

import numpy as np

from .mean_variance import ensure_psd

# ──────────────────────────────────────────────────────────────────────────────
# Black-Litterman Core
# ──────────────────────────────────────────────────────────────────────────────


def market_implied_prior(
    Sigma: np.ndarray, w_market: np.ndarray, delta: float = 2.5, rf: float = 0.0
) -> np.ndarray:
    """
    Reverse Optimization: Calcular los retornos implícitos del mercado (Pi).

    Formula: Pi = delta * Sigma @ w_market

    Args:
        Sigma: Matriz de covarianza (N x N).
        w_market: Pesos del portafolio de mercado (N,).
        delta: Coeficiente de aversión al riesgo del mercado.
        rf: Tasa libre de riesgo (se suma al exceso de retorno).

    Returns:
        Pi: Vector de retornos esperados implícitos (N,).
    """
    Sigma = np.asarray(Sigma, dtype=float)
    w_market = np.asarray(w_market, dtype=float)

    # Validar dimensiones
    if Sigma.shape[0] != w_market.shape[0]:
        raise ValueError("Dimensions of Sigma and w_market do not match.")

    # Pi (exceso de retorno)
    Pi_excess = delta * (Sigma @ w_market)

    return np.asarray(Pi_excess + rf)


def black_litterman_posterior(
    Sigma: np.ndarray,
    Pi: np.ndarray,
    tau: float,
    P: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    Omega: np.ndarray | None = None,
    view_confidences: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula los retornos esperados y la covarianza posterior usando el modelo Black-Litterman.

    Soporta dos modos de especificación de incertidumbre en las vistas (Omega):
    1. Directo: Pasando 'Omega' explícitamente.
    2. Idzorek (Confianza): Pasando 'view_confidences' (0.0 a 1.0).
       Se usa el método de He & Litterman modificado por Idzorek para inferir Omega.

    Args:
        Sigma: Covarianza Prior (N x N).
        Pi: Retornos Implícitos Prior (N,).
        tau: Escalar de incertidumbre del prior (normalmente pequeño, e.g., 0.05).
        P: Matriz de Vistas (K x N). K vistas, N activos.
           Cada fila suma 0 (vista relativa) o 1 (vista absoluta).
        Q: Vector de Retornos de las Vistas (K,).
        Omega: Matriz de covarianza de las vistas (K x K). Si es None, se intenta calcular.
        view_confidences: Array (K,) con confianzas [0, 1] para cada vista. Usado si Omega is None.

    Returns:
        mu_bl: Retornos esperados posteriores (N,).
        sigma_bl: Covarianza posterior (N x N).
    """
    Sigma = ensure_psd(Sigma)
    Pi = np.asarray(Pi, dtype=float)
    N = Sigma.shape[0]

    # Caso base: Sin vistas -> Devuelve el Prior ajustado por tau
    if P is None or Q is None or len(P) == 0:
        return (
            Pi,
            Sigma,
        )  # En puridad BL retorna Sigma + tau*Sigma, pero para mean-variance solemos usar Sigma actualizado o el original.
        # Standard BL: mu = Pi, Cov = Sigma + tau*Sigma.
        # Pero a menudo solo queremos los alphas. Devolvemos Pi tal cual.

    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    K = P.shape[0]

    if P.shape[1] != N:
        raise ValueError(f"View matrix P columns ({P.shape[1]}) must match assets N ({N}).")

    tau_Sigma = tau * Sigma

    # Calcular Omega si no se da
    if Omega is None:
        if view_confidences is not None:
            # Metodo Idzorek: Inferir Omega desde confianzas
            # 1. Calcular Omega "proporcional" base (He-Litterman): P @ (tau*Sigma) @ P.T
            # 2. Ajustar diagonal basada en confianza alpha. alpha -> 0 (ruido infinito), alpha -> 1 (ruido cero).
            #    Formula Idzorek es compleja iterativa, usamos una aproximacion comun:
            #    omega_k = p_k @ (tau*Sigma) @ p_k' * ( (1-conf)/conf )

            Omega_diag = []
            for k in range(K):
                pk = P[k, :]
                conf = float(view_confidences[k])
                # Evitar div por cero if conf=0
                conf = np.clip(conf, 1e-4, 0.9999)

                # Varianza de la vista inducida por el prior
                var_pk = pk @ tau_Sigma @ pk.T

                # Escalar de incertidumbre alfa: alpha = (1-conf)/conf
                # Si conf=0.5 -> alpha=1 -> incertidumbre igual al prior
                # Si conf=0.9 -> alpha=0.1 -> muy cierto (poca varianza extra)
                # Si conf=0.1 -> alpha=9 -> muy incierto
                scaling = (1.0 - conf) / conf
                omega_k = var_pk * scaling
                Omega_diag.append(omega_k)

            Omega = np.diag(Omega_diag)

        else:
            # Metodo default He-Litterman: Diagonal de P @ (tau*Sigma) @ P.T
            # Asume que la incertidumbre de la vista es proporcional a la del prior.
            Omega = np.diag(np.diag(P @ tau_Sigma @ P.T))

    Omega = np.asarray(Omega, dtype=float)

    # ── Formula BL Maestra ──────────────────────────────────────────────────────
    # mu_bl = [(tau*Sigma)^-1 + P' Omega^-1 P]^-1 [ (tau*Sigma)^-1 Pi + P' Omega^-1 Q ]
    #
    # Para estabilidad numerica usamos inversion via Woodbury o solucion LS,
    # pero dado N usualmente < 1000, inversion directa es ok si tau*Sigma es estable.

    # Inversas
    try:
        ts_inv = np.linalg.inv(tau_Sigma)
        omega_inv = np.linalg.inv(Omega)
    except np.linalg.LinAlgError:
        # Fallback a pseudo-inversa si singular
        ts_inv = np.linalg.pinv(tau_Sigma)
        omega_inv = np.linalg.pinv(Omega)

    # M = [(tau*Sigma)^-1 + P' Omega^-1 P]^-1
    # Parte izq del termino de mu
    post_cov_inv = ts_inv + (P.T @ omega_inv @ P)

    try:
        M = np.linalg.inv(post_cov_inv)  # Esta es la varianza del estimado de la media
    except np.linalg.LinAlgError:
        M = np.linalg.pinv(post_cov_inv)

    # Parte Der del termino de mu: [ (tau*Sigma)^-1 Pi + P' Omega^-1 Q ]
    term_b = (ts_inv @ Pi) + (P.T @ omega_inv @ Q)

    mu_bl = M @ term_b

    # Covarianza Posterior Total = Sigma + Sigma_estimacion_media = Sigma + M
    # Aunque algunos practicos usan M directamente para optimizacion si asumen que 'Sigma' ya contiene todo el riesgo.
    # El standard BL dice: Sigma_posterior = Sigma + M.
    # Pero cuidado: M escala con tau (pequeño). Sigma es grande.
    # Si optimizamos, usamos Sigma_posterior.

    sigma_bl = Sigma + M

    return mu_bl, sigma_bl
