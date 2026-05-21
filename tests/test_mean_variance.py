"""Tests for portfolio.optim.mean_variance.

Covers EDGE-1: frontier_closed_form must guard against single-asset inputs.
"""

import numpy as np
import pytest

from portfolio.optim.mean_variance import frontier_closed_form, markowitz_closed_form


def test_frontier_single_asset_raises():
    """EDGE-1: frontier with 1 asset must raise ValueError."""
    mu = np.array([0.1])
    Sigma = np.array([[0.04]])
    with pytest.raises(ValueError, match="2 assets"):
        frontier_closed_form(mu, Sigma, r_min=0.05, r_max=0.15, npts=10)


def test_frontier_two_assets_valid():
    """frontier with 2 assets must return a valid curve without NaN."""
    mu = np.array([0.10, 0.06])
    Sigma = np.array([[0.04, 0.01], [0.01, 0.02]])
    risks, rets = frontier_closed_form(mu, Sigma, r_min=0.06, r_max=0.10, npts=20)
    assert len(risks) == len(rets) == 20
    assert not np.isnan(risks).any(), "risks contains NaN"
    assert not np.isnan(rets).any(), "rets contains NaN"
    assert (risks >= 0).all(), "risks contains negative values"


def test_markowitz_two_assets():
    """markowitz_closed_form with 2 assets produces weights summing to 1."""
    mu = np.array([0.10, 0.06])
    Sigma = np.array([[0.04, 0.01], [0.01, 0.02]])
    w_mvp, w_tan = markowitz_closed_form(mu, Sigma, rf=0.02)
    assert abs(w_mvp.sum() - 1.0) < 1e-6
    assert abs(w_tan.sum() - 1.0) < 1e-6


def test_frontier_no_nan_with_correlated_assets():
    """Frontier with highly (but not perfectly) correlated assets produces no NaN."""
    from datetime import date, timedelta

    import polars as pl

    from portfolio.features.risk_models import covariance as cov_fn

    rng = np.random.default_rng(42)
    base = rng.normal(0.001, 0.02, 300)
    noise = rng.normal(0, 0.001, 300)
    X = np.column_stack([base, base + noise, base * 0.5 + noise * 2])

    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(300)]
    cols = {"date": dates, "A": X[:, 0].tolist(), "B": X[:, 1].tolist(), "C": X[:, 2].tolist()}
    df = pl.DataFrame(cols)
    Sigma, _names = cov_fn(df, method="oas", psd=True, as_frame=False)
    mu = np.array([0.10, 0.08, 0.06])
    risks, rets = frontier_closed_form(mu, Sigma, r_min=0.06, r_max=0.10, npts=15)
    assert not np.isnan(risks).any()
    assert not np.isinf(risks).any()
