# portfolio/bscktest/allocators.py

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import polars as pl

from portfolio.core.utils import ensure_psd, hrp_safe, project_to_box_simplex
from portfolio.optim.hrp import hrp_weights
from portfolio.optim.mean_variance import pgd_box_simplex_l2
from portfolio.optim.risk_parity import risk_parity

CovEstimator = Literal["Sample", "EWMA"]


def cov_ewma(R: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Compute EWMA covariance matrix (robust, PSD-enforced)."""
    if R.size == 0:
        return np.eye(0)
    T, N = R.shape
    S = np.zeros((N, N), dtype=float)
    w = 0.0
    mu = np.nanmean(R, axis=0)
    for t in range(T):
        x = (R[t] - mu).reshape(1, -1)
        S = lam * S + (1 - lam) * (x.T @ x)
        w = lam * w + (1 - lam)
    S = S / max(w, 1e-12)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    return ensure_psd(S, eps=1e-10, clip=True)


def enforce_turnover(
    prev_w: np.ndarray | None,
    new_w: np.ndarray,
    *,
    max_to: float = 0.10,
    band: float = 0.01,
    w_min: float = 0.0,
    w_max: float = 1.0,
) -> np.ndarray:
    """
    Enforce turnover and rebalancing rules.
    - Ignores small median weight changes (band threshold).
    - Limits portfolio turnover (L1/2) to 'max_to' budget.
    """
    if prev_w is None or prev_w.size == 0:
        return project_to_box_simplex(new_w, w_min, w_max)
    if np.median(np.abs(new_w - prev_w)) < band:
        return prev_w
    to = 0.5 * float(np.sum(np.abs(new_w - prev_w)))
    if to <= max_to:
        return project_to_box_simplex(new_w, w_min, w_max)
    lam = min(1.0, max_to / (to + 1e-12))
    w_lim = prev_w + lam * (new_w - prev_w)
    return project_to_box_simplex(w_lim, w_min, w_max)


def min_te_to_bench(
    Sigma: np.ndarray,
    w_bench: np.ndarray,
    *,
    w_min: float,
    w_max: float,
) -> np.ndarray:
    """
    Minimize Tracking Error vs benchmark:
        min (w - w_b)' Σ (w - w_b)
    Equivalent to min w'Σw - 2 w'(Σ w_b) + const
    -> solved via L2 PGD with effective μ = 2 Σ w_b.
    """
    if Sigma.size == 0:
        return project_to_box_simplex(w_bench.copy(), w_min, w_max)
    mu_eff = 2.0 * (Sigma @ w_bench)
    w = pgd_box_simplex_l2(mu_eff, Sigma, gamma=1.0, w_min=w_min, w_max=w_max, lam_turnover=0.0)
    return project_to_box_simplex(w, w_min, w_max)


def _get_cov(
    win: pl.DataFrame,
    cols: list[str],
    cov_estimator: CovEstimator,
    ewma_lambda: float,
) -> np.ndarray:
    if not cols:
        return np.eye(0)
    R = win.select(cols).to_numpy()
    if R.size == 0:
        return np.eye(len(cols)) * 1e-4

    S = cov_ewma(R, lam=float(ewma_lambda)) if cov_estimator == "EWMA" else np.cov(R, rowvar=False)

    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    return ensure_psd(S, eps=1e-10, clip=True)


def make_allocator(
    kind: str,
    *,
    w_min: float,
    w_max: float,
    cov_estimator: CovEstimator = "Sample",
    ewma_lambda: float = 0.97,
    use_to_budget: bool = True,
    max_turnover: float = 0.10,
    band_eps: float = 0.01,
) -> Callable[[pl.DataFrame], np.ndarray]:
    """
    Factory for allocators used by the Backtest UI.

    It returns a closure:
        window of returns (Polars DataFrame) -> weights (np.ndarray, shape (N,))
    """

    prev: dict[str, np.ndarray | None] = {"w": None}

    def base_alloc_equal(win: pl.DataFrame) -> np.ndarray:
        n = win.width - 1
        if n <= 0:
            return np.array([], dtype=float)
        w = np.ones(n, dtype=float) / n
        return project_to_box_simplex(w, w_min, w_max)

    def base_alloc_min_var(win: pl.DataFrame) -> np.ndarray:
        cols = [c for c in win.columns if c != "date"]
        R = win.select(cols).to_numpy() if cols else np.zeros((0, 0), dtype=float)
        mu_w = np.nanmean(R, axis=0) if R.size else np.zeros(len(cols))
        mu_w = np.nan_to_num(mu_w, nan=0.0, posinf=0.0, neginf=0.0)
        Sigma_w = _get_cov(win, cols, cov_estimator, ewma_lambda)
        w = pgd_box_simplex_l2(
            mu_w,
            Sigma_w,
            gamma=100.0,
            w_min=w_min,
            w_max=w_max,
            lam_turnover=0.0,
        )
        return project_to_box_simplex(w, w_min, w_max)

    def base_alloc_risk_parity(win: pl.DataFrame) -> np.ndarray:
        cols = [c for c in win.columns if c != "date"]
        Sigma_w = _get_cov(win, cols, cov_estimator, ewma_lambda)
        try:
            w = risk_parity(Sigma_w, w_min=w_min, w_max=w_max)
        except Exception:
            w = np.ones(len(cols), dtype=float) / max(len(cols), 1)
        return project_to_box_simplex(w, w_min, w_max)

    def base_alloc_hrp(win: pl.DataFrame) -> np.ndarray:
        cols = [c for c in win.columns if c != "date"]
        Sigma_w = _get_cov(win, cols, cov_estimator, ewma_lambda)
        w = hrp_safe(
            hrp_func=hrp_weights,
            cov=Sigma_w,
            method="ward",
            optimal=True,
            w_min=w_min,
            w_max=w_max,
        )
        return project_to_box_simplex(w, w_min, w_max)

    def base_alloc_min_te(win: pl.DataFrame) -> np.ndarray:
        cols = [c for c in win.columns if c != "date"]
        Sigma_w = _get_cov(win, cols, cov_estimator, ewma_lambda)
        w_bench = np.full(len(cols), 1.0 / max(len(cols), 1))
        w = min_te_to_bench(Sigma_w, w_bench, w_min=w_min, w_max=w_max)
        return project_to_box_simplex(w, w_min, w_max)

    if kind == "Equal-Weight":
        base_alloc = base_alloc_equal
    elif kind == "Min-Var (L2 PGD)":
        base_alloc = base_alloc_min_var
    elif kind == "Risk Parity":
        base_alloc = base_alloc_risk_parity
    elif kind == "HRP":
        base_alloc = base_alloc_hrp
    elif kind == "Min-TE (to Bench)":
        base_alloc = base_alloc_min_te
    else:
        base_alloc = base_alloc_equal

    def alloc(win: pl.DataFrame) -> np.ndarray:
        """Final allocator with optional turnover control."""
        w_new = base_alloc(win)
        if use_to_budget:
            w_final = enforce_turnover(
                prev["w"],
                w_new,
                max_to=max_turnover,
                band=band_eps,
                w_min=w_min,
                w_max=w_max,
            )
        else:
            w_final = w_new
        prev["w"] = w_final
        return w_final

    return alloc
