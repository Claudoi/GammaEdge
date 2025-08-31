from __future__ import annotations
import numpy as np

def portfolio_stats(
    w: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    rf: float = 0.0
) -> tuple[float, float, float]:
    """
    Devuelve (mu_p, sigma_p, sharpe).
    - mu_p = w' mu
    - sigma_p = sqrt(w' Σ w), con clip a 0 para robustez numérica
    - sharpe = (mu_p - rf) / sigma_p  (NaN si sigma ~ 0)
    """
    w = np.asarray(w, float).reshape(-1)
    mu = np.asarray(mu, float).reshape(-1)
    Sigma = np.asarray(Sigma, float)
    mu_p = float(w @ mu)
    var_p = float(max(w @ Sigma @ w, 0.0))
    sigma_p = float(np.sqrt(var_p))
    sharpe = (mu_p - rf) / sigma_p if sigma_p > 1e-12 else np.nan
    return mu_p, sigma_p, sharpe


def gini(x: np.ndarray) -> float:
    """
    Índice de Gini para medir concentración (p.ej. weights o RC).
    """
    x = np.asarray(x, float).reshape(-1)
    if x.size == 0:
        return np.nan
    x = np.abs(x)
    s = x.sum()
    if s <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    # fórmula estándar discreta
    g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(g)


def cvar_estimate(
    port_rets: np.ndarray,
    alpha: float = 0.95
) -> float:
    """
    Estimación in-sample del CVaR (cola izquierda).
    Devuelve la media de la cola de pérdidas <= VaR_(alpha).
    """
    r = np.asarray(port_rets, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan
    q = np.percentile(r, (1.0 - alpha) * 100.0)  # VaR (cuantil cola izqda)
    tail = r[r <= q]
    return float(tail.mean()) if tail.size > 0 else float(q)
