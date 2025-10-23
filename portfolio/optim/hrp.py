# portfolio/optim/hrp.py
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from .mean_variance import ensure_psd, project_to_box_simplex


def _corr_from_cov(S: np.ndarray) -> np.ndarray:
    """Compute correlation matrix from covariance matrix."""
    d = np.sqrt(np.clip(np.diag(S), 1e-16, None))
    R = (S / d[:, None]) / d[None, :]
    R = np.clip(0.5 * (R + R.T), -1.0, 1.0)
    return np.asarray(R, dtype=np.float64)  # ✅ ensures ndarray type


def _seriation_order(
    Sigma: np.ndarray,
    method: Literal["single", "complete", "average", "ward"] = "ward",
    optimal: bool = True,
) -> np.ndarray:
    """Compute hierarchical ordering of assets."""
    Corr = _corr_from_cov(Sigma)
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - Corr)))
    Z = linkage(squareform(dist, checks=False), method=method)
    if optimal:
        Z = optimal_leaf_ordering(Z, squareform(dist, checks=False))
    order = leaves_list(Z)
    return np.asarray(order, dtype=int)  # ✅ explicit ndarray



def hrp_weights(
    Sigma: np.ndarray,
    *,
    method: Literal["single", "complete", "average", "ward"] = "ward",
    optimal: bool = True,
    w_min: float = 0.0,
    w_max: float = 1.0,
) -> np.ndarray:
    """
    Hierarchical Risk Parity (López de Prado) con proyección a caja+simplex.
    """
    S = ensure_psd(Sigma)
    order = _seriation_order(S, method=method, optimal=optimal)
    S = S[np.ix_(order, order)]

    def _cluster_var(Ss: np.ndarray) -> float:
        invdiag = 1.0 / np.clip(np.diag(Ss), 1e-16, None)
        w = invdiag / invdiag.sum()
        return float(w @ Ss @ w)

    def _split_allocation(Ss: np.ndarray, idx: np.ndarray) -> np.ndarray:
        if len(idx) == 1:
            return np.array([1.0], dtype=np.float64)
        k = len(idx) // 2
        left = idx[:k]
        right = idx[k:]
        w_left = _split_allocation(Ss[np.ix_(left, left)], left)
        w_right = _split_allocation(Ss[np.ix_(right, right)], right)
        v_left = _cluster_var(Ss[np.ix_(left, left)])
        v_right = _cluster_var(Ss[np.ix_(right, right)])
        alpha = 1.0 - v_left / (v_left + v_right)
        w = np.zeros(len(idx), dtype=np.float64)
        w[:k] = alpha * w_left
        w[k:] = (1.0 - alpha) * w_right
        return w

    base = _split_allocation(S, np.arange(S.shape[0]))
    w = np.zeros_like(base)
    w[order] = base
    w = project_to_box_simplex(w, w_min, w_max)
    return np.asarray(w, dtype=np.float64)
