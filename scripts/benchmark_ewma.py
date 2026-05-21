"""Benchmark: EWMA covariance — vectorized vs naive loop.

Reproduces the perf claim in the TFG (~70x speedup on T=500, N=20).

Usage:
    poetry run python scripts/benchmark_ewma.py
    poetry run python scripts/benchmark_ewma.py --n-obs 1000 --n-assets 30 --reps 100

The naive loop implementation is included locally for the comparison; the
production code uses the vectorized form (``portfolio.features.risk_models``).
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def ewma_cov_loop(X: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Naive Python-loop EWMA covariance (the pre-vectorization implementation).

    Equivalent to the production vectorized version but iterates over T,
    accumulating ``Sigma = lam * Sigma + (1 - lam) * x_t x_t^T``.
    """
    T, N = X.shape
    mu = np.nanmean(X, axis=0, keepdims=True)
    Xc = X - mu
    Sigma = np.zeros((N, N), dtype=np.float64)
    for t in range(T):
        xt = Xc[t : t + 1, :]
        if np.isnan(xt).any():
            continue
        Sigma = lam * Sigma + (1.0 - lam) * (xt.T @ xt)
    return Sigma


def ewma_cov_vectorized(X: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Vectorized EWMA covariance — production form.

    Equivalent to the loop but expresses the recursion as a weighted matrix
    multiply: ``Sigma = X^T diag(w) X`` with
    ``w[t] = (1 - lam) * lam^(T_eff - 1 - t)``.
    """
    T, N = X.shape
    mu_center = np.nanmean(X, axis=0, keepdims=True)
    Xc = X - mu_center
    valid_mask = ~np.isnan(Xc).any(axis=1)
    Xc_valid = Xc[valid_mask]
    T_eff = Xc_valid.shape[0]
    if T_eff == 0:
        return np.eye(N) * 1e-6
    exponents = np.arange(T_eff - 1, -1, -1, dtype=np.float64)
    weights = (1.0 - lam) * (lam**exponents)
    Xc_weighted = Xc_valid * weights[:, None]
    return Xc_weighted.T @ Xc_valid


def time_fn(fn, *args, reps: int = 50) -> tuple[float, float]:
    """Return (mean_ms, std_ms) over ``reps`` runs."""
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(samples)), float(np.std(samples))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=500, help="Number of observations (T)")
    parser.add_argument("--n-assets", type=int, default=20, help="Number of assets (N)")
    parser.add_argument("--lam", type=float, default=0.94, help="EWMA decay factor")
    parser.add_argument("--reps", type=int, default=50, help="Repetitions per measurement")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X = rng.normal(0.001, 0.02, (args.n_obs, args.n_assets))

    # Warm up (avoid JIT or first-call overhead from numpy/blas)
    _ = ewma_cov_loop(X[:50], args.lam)
    _ = ewma_cov_vectorized(X[:50], args.lam)

    loop_mean, loop_std = time_fn(ewma_cov_loop, X, args.lam, reps=args.reps)
    vec_mean, vec_std = time_fn(ewma_cov_vectorized, X, args.lam, reps=args.reps)

    # Verify numerical equivalence (sanity)
    S_loop = ewma_cov_loop(X, args.lam)
    S_vec = ewma_cov_vectorized(X, args.lam)
    max_abs_diff = float(np.max(np.abs(S_loop - S_vec)))

    speedup = loop_mean / vec_mean if vec_mean > 0 else float("inf")

    print("=" * 60)
    print(f"EWMA covariance benchmark — T={args.n_obs}, N={args.n_assets}, lam={args.lam}")
    print(f"Repetitions: {args.reps}")
    print("=" * 60)
    print(f"Loop (naive):      {loop_mean:8.3f} ms  ± {loop_std:.3f} ms")
    print(f"Vectorized:        {vec_mean:8.3f} ms  ± {vec_std:.3f} ms")
    print(f"Speedup:           {speedup:8.1f}x")
    print(f"Max |Σ_loop - Σ_vec|: {max_abs_diff:.2e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
