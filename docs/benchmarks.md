# GammaEdge — Performance Benchmarks

## EWMA Covariance: Vectorized vs Naive Loop

The EWMA covariance estimator in `portfolio/features/risk_models.py` was rewritten
from a per-timestep Python loop to a vectorized weighted matrix multiply. The
new form is mathematically equivalent and produces results numerically
indistinguishable from the loop (max absolute difference < 1e-18).

### Reproducing the benchmark

```bash
poetry run python scripts/benchmark_ewma.py
poetry run python scripts/benchmark_ewma.py --n-obs 1000 --n-assets 30 --reps 100
```

### Reference result

Hardware: Apple Silicon (Darwin 25.3.0)
Python 3.11.14, numpy 1.26.4

| T (obs) | N (assets) | Loop (ms)     | Vectorized (ms) | Speedup |
|---------|-----------|---------------|-----------------|---------|
| 500     | 20        | 2.993 ± 0.466 | 0.114 ± 0.005   | 26.2x   |
| 1000    | 30        | 6.150 ± 0.707 | 0.478 ± 0.819   | 12.9x   |

Numerical equivalence verified: `max |Σ_loop − Σ_vec| ≈ 1.6e−19` (T=500, N=20)
and `≈ 2.2e−19` (T=1000, N=30), well below float64 epsilon.

### Mathematical equivalence

Both implementations compute:

Σ = (1 − λ) · Σ_{t=0}^{T−1} λ^(T−1−t) · x_t x_tᵀ

The loop accumulates one term per iteration; the vectorized form expresses
the same sum as `X^T diag(w) X` with `w[t] = (1 − λ) λ^(T_eff − 1 − t)`,
which lets BLAS handle the inner products in a single call.
