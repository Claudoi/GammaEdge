"""Regression tests for portfolio.optim.hrp.hrp_weights.

Covers the latent bug where ``_split_allocation`` passed the original
``idx`` array to recursive calls instead of re-indexing into the sliced
submatrix, producing an IndexError for N>=3 assets.
"""

from __future__ import annotations

import numpy as np
import pytest

from portfolio.optim.hrp import hrp_weights


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
def test_hrp_weights_sum_to_one(n: int) -> None:
    """HRP weights must sum to 1.0 and be non-negative for any N >= 2."""
    rng = np.random.default_rng(42)
    R = rng.normal(0.001, 0.02, (252, n))
    Sigma = np.cov(R, rowvar=False)
    w = hrp_weights(Sigma)
    assert w.shape == (n,)
    assert abs(w.sum() - 1.0) < 1e-6, f"weights sum {w.sum()} != 1.0"
    assert (w >= -1e-9).all(), f"negative weights: {w}"
    assert not np.isnan(w).any(), "HRP returned NaN weights"


def test_hrp_weights_single_asset_works_or_raises_cleanly() -> None:
    """HRP with N=1 should either return [1.0] or raise ValueError clearly."""
    Sigma = np.array([[0.04]])
    try:
        w = hrp_weights(Sigma)
        assert w.shape == (1,)
        assert abs(w[0] - 1.0) < 1e-6
    except ValueError:
        pass  # acceptable
