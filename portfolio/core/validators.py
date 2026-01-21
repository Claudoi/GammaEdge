"""
Validator module to prevent extreme portfolio metric values.

Guards against calculation errors like double annualization.
"""
from __future__ import annotations

import numpy as np


class MetricValidationError(ValueError):
    """Raised when portfolio metrics fall outside reasonable bounds."""
    pass


def validate_annual_metrics(
    mu_annual: float,
    sigma_annual: float,
    *,
    max_return: float = 3.0,
    max_volatility: float = 2.0,
    min_return: float = -0.99,
    min_volatility: float = 0.0,
) -> None:
    """
    Validate that annualized portfolio metrics are within reasonable bounds.
    
    This catches calculation errors like double annualization that produce
    extreme values (e.g., 5302% returns).
    
    Parameters
    ----------
    mu_annual : float
        Annual expected return (e.g., 0.10 for 10%)
    sigma_annual : float
        Annual volatility (e.g., 0.15 for 15%)
    max_return : float, default=3.0
        Maximum reasonable annual return (300%)
    max_volatility : float, default=2.0
        Maximum reasonable annual volatility (200%)
    min_return : float, default=-0.99
        Minimum reasonable annual return (-99%)
    min_volatility : float, default=0.0
        Minimum reasonable volatility (0%)
        
    Raises
    ------
    MetricValidationError
        If metrics are outside reasonable bounds
        
    Examples
    --------
    >>> validate_annual_metrics(0.10, 0.15)  # OK: 10% return, 15% vol
    >>> validate_annual_metrics(53.02, 3.49)  # Raises: extreme values
    MetricValidationError: Suspicious annual return: 5302.00%
    """
    # Validate inputs are numeric
    if not (np.isfinite(mu_annual) and np.isfinite(sigma_annual)):
        raise MetricValidationError(
            f"Non-finite metrics detected: return={mu_annual}, vol={sigma_annual}"
        )
    
    # Check return bounds
    if mu_annual > max_return:
        raise MetricValidationError(
            f"Suspicious annual return: {mu_annual:.2%}. "
            f"Expected < {max_return:.0%}. "
            "Possible double annualization bug."
        )
    
    if mu_annual < min_return:
        raise MetricValidationError(
            f"Suspicious annual return: {mu_annual:.2%}. "
            f"Expected > {min_return:.0%}."
        )
    
    # Check volatility bounds
    if sigma_annual > max_volatility:
        raise MetricValidationError(
            f"Suspicious annual volatility: {sigma_annual:.2%}. "
            f"Expected < {max_volatility:.0%}. "
            "Possible double annualization bug."
        )
    
    if sigma_annual < min_volatility:
        raise MetricValidationError(
            f"Suspicious annual volatility: {sigma_annual:.2%}. "
            f"Expected > {min_volatility:.0%}."
        )


def validate_correlation_matrix(Corr: np.ndarray, tol: float = 1e-6) -> None:
    """
    Validate that a correlation matrix has valid properties.
    
    Parameters
    ----------
    Corr : np.ndarray
        Correlation matrix (N x N)
    tol : float
        Tolerance for validation checks
        
    Raises
    ------
    MetricValidationError
        If correlation matrix is invalid
    """
    if Corr.ndim != 2 or Corr.shape[0] != Corr.shape[1]:
        raise MetricValidationError(
            f"Correlation matrix must be square, got shape {Corr.shape}"
        )
    
    # Diagonal should be 1
    diag = np.diag(Corr)
    if not np.allclose(diag, 1.0, atol=tol):
        raise MetricValidationError(
            f"Correlation matrix diagonal should be 1, got {diag}"
        )
    
    # Values should be in [-1, 1]
    if np.any(Corr < -1.0 - tol) or np.any(Corr > 1.0 + tol):
        raise MetricValidationError(
            f"Correlation values must be in [-1, 1], got range [{Corr.min():.3f}, {Corr.max():.3f}]"
        )
    
    # Should be symmetric
    if not np.allclose(Corr, Corr.T, atol=tol):
        raise MetricValidationError("Correlation matrix must be symmetric")


def validate_covariance_matrix(Sigma: np.ndarray, tol: float = 1e-10) -> None:
    """
    Validate that a covariance matrix is positive semi-definite.
    
    Parameters
    ----------
    Sigma : np.ndarray
        Covariance matrix (N x N)
    tol : float
        Tolerance for minimum eigenvalue
        
    Raises
    ------
    MetricValidationError
        If covariance matrix is invalid
    """
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise MetricValidationError(
            f"Covariance matrix must be square, got shape {Sigma.shape}"
        )
    
    # Should be symmetric
    if not np.allclose(Sigma, Sigma.T, atol=tol):
        raise MetricValidationError("Covariance matrix must be symmetric")
    
    # Should be PSD (all eigenvalues >= 0)
    eigvals = np.linalg.eigvalsh(Sigma)
    min_eigval = float(np.min(eigvals))
    
    if min_eigval < -tol:
        raise MetricValidationError(
            f"Covariance matrix not PSD: minimum eigenvalue = {min_eigval:.2e}"
        )
