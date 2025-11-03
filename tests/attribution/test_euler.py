# tests/attribution/test_euler.py

import numpy as np
from numpy.testing import assert_allclose

from portfolio.attribution import euler_risk_contributions


def test_euler_risk_contributions_basic():
    w = np.array([0.5, 0.5])
    S = np.array([[0.04, 0.01], [0.01, 0.09]])

    rc = euler_risk_contributions(w, S)
    # Euler property: sum(rc) == portfolio variance
    port_var = float(w @ S @ w)
    assert_allclose(rc.sum(), port_var, rtol=1e-12, atol=1e-12)
    # Contributions must be non-negative for PSD covariance when weights >= 0
    assert np.all(rc >= 0.0)
