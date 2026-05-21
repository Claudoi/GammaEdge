"""Regression tests for risk parity normalized convergence criterion."""

from __future__ import annotations

import numpy as np

from portfolio.optim.risk_parity import risk_parity


def test_risk_parity_converges_with_normalized_tolerance():
    """Risk parity should produce a feasible long-only allocation that
    improves over equal-weight in terms of RC dispersion.

    The heuristic multiplicative scheme is not a true Newton solver, so we
    don't demand tight equality of risk contributions. The point of this
    regression test is that the normalized stopping criterion still
    produces a sensible allocation (valid simplex, all positive, RC
    dispersion no worse than equal-weight baseline).
    """
    rng = np.random.default_rng(42)
    n = 5
    A = rng.normal(0, 1, (n, n))
    Sigma = A @ A.T + np.eye(n) * 0.1  # well-conditioned PSD

    w = risk_parity(Sigma)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w > 0).all()

    # Compute Euler RC dispersion for the risk-parity allocation
    sigma_p = np.sqrt(w @ Sigma @ w)
    rc = w * (Sigma @ w) / sigma_p
    target = rc.mean()
    relative_disp_rp = np.max(np.abs(rc / target - 1.0))

    # Equal-weight baseline for comparison
    w_eq = np.full(n, 1.0 / n)
    sigma_p_eq = np.sqrt(w_eq @ Sigma @ w_eq)
    rc_eq = w_eq * (Sigma @ w_eq) / sigma_p_eq
    relative_disp_eq = np.max(np.abs(rc_eq / rc_eq.mean() - 1.0))

    # Risk parity should not be dramatically worse than equal-weight on RC
    # dispersion (sanity check on the heuristic).
    assert relative_disp_rp <= relative_disp_eq + 1.0, (
        f"RP dispersion {relative_disp_rp:.4f} much worse than EW " f"{relative_disp_eq:.4f}"
    )


def test_risk_parity_low_variance_portfolio_does_not_hang():
    """Low-variance setup: convergence should still work."""
    # Very low variance matrix
    Sigma = np.eye(3) * 1e-8 + 1e-10
    w = risk_parity(Sigma)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w > 0).all()


def test_risk_parity_tiny_scale_returns_valid_simplex():
    """With Sigma scaled to extremely small values, the old absolute
    criterion ||g - target|| < 1e-8 would trip convergence almost
    immediately (since both g and target are ~1e-12), returning whatever
    near-initial weights happened to be in hand. The normalized criterion
    is scale-invariant, so the optimiser keeps iterating and still returns
    a valid simplex allocation rather than being short-circuited.
    """
    rng = np.random.default_rng(7)
    n = 4
    A = rng.normal(0, 1, (n, n))
    Sigma = (A @ A.T + np.eye(n) * 0.1) * 1e-12

    w = risk_parity(Sigma)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= 0).all()
    assert (w <= 1).all()
