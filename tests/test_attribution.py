"""Tests for portfolio/attribution/ — Brinson-Fachler and Euler risk contributions.

Covers the public API exported by ``portfolio.attribution``:

- ``euler_risk_contributions`` (Euler decomposition of portfolio volatility)
- ``compute_brinson_attribution`` (Brinson-style timeseries aggregation)
- ``BrinsonAttribution`` (result container)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from portfolio.attribution import (
    BrinsonAttribution,
    compute_brinson_attribution,
    euler_risk_contributions,
)


# ────────────────────────────────────────────────────────────────────
# Euler risk contributions
# ────────────────────────────────────────────────────────────────────
def test_euler_risk_contributions_sum_to_portfolio_volatility():
    """Euler decomposition invariant: sum of RC_i equals portfolio sigma."""
    rng = np.random.default_rng(42)
    n = 5
    # Build a symmetric PSD covariance matrix
    A = rng.normal(0.0, 1.0, (n, n))
    sigma = A @ A.T + 1e-6 * np.eye(n)

    # Random long-only weights summing to 1
    w = rng.uniform(0.0, 1.0, n)
    w /= w.sum()

    rc = euler_risk_contributions(w, sigma)

    port_sigma = float(np.sqrt(w @ sigma @ w))
    assert isinstance(rc, pd.Series)
    assert rc.shape == (n,)
    assert np.isclose(
        rc.sum(), port_sigma, atol=1e-10
    ), f"Sum of risk contributions ({rc.sum()}) must equal portfolio sigma ({port_sigma})."


def test_euler_risk_contributions_preserves_pandas_index():
    """When weights is a pd.Series, the returned RC must share its index."""
    idx = pd.Index(["AAPL", "MSFT", "GOOG"], name="ticker")
    w = pd.Series([0.5, 0.3, 0.2], index=idx)
    cov = pd.DataFrame(
        np.array(
            [
                [0.04, 0.01, 0.00],
                [0.01, 0.09, 0.02],
                [0.00, 0.02, 0.16],
            ]
        ),
        index=idx,
        columns=idx,
    )

    rc = euler_risk_contributions(w, cov)
    assert list(rc.index) == ["AAPL", "MSFT", "GOOG"]
    assert rc.name == "risk_contribution"
    # Invariant still holds
    port_sigma = float(np.sqrt(w.to_numpy() @ cov.to_numpy() @ w.to_numpy()))
    assert np.isclose(rc.sum(), port_sigma, atol=1e-12)


def test_euler_risk_contributions_rejects_zero_variance_portfolio():
    """A zero-vol portfolio (all-zero weights) must raise ValueError."""
    w = np.zeros(3)
    cov = np.eye(3)
    with pytest.raises(ValueError, match="variance"):
        euler_risk_contributions(w, cov)


def test_euler_risk_contributions_rejects_mismatched_covariance_shape():
    """Mismatch between weights length and covariance dimensions must raise."""
    w = np.array([0.5, 0.5])
    cov = np.eye(3)  # 3x3 but only 2 weights
    with pytest.raises(ValueError, match="Covariance"):
        euler_risk_contributions(w, cov)


# ────────────────────────────────────────────────────────────────────
# Brinson-Fachler attribution
# ────────────────────────────────────────────────────────────────────
def test_compute_brinson_attribution_aggregates_long_format():
    """Long-format Brinson dataframe is aggregated by group and totalled."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)]
    rows = []
    for d in dates:
        for gid, (a, s, i, t) in enumerate(
            [
                (0.001, 0.002, 0.0001, 0.0031),
                (-0.0005, 0.001, 0.00005, 0.00055),
            ]
        ):
            rows.append(
                {
                    "date": d,
                    "group_id": gid,
                    "alloc": a,
                    "select": s,
                    "interact": i,
                    "total": t,
                }
            )
    df = pl.DataFrame(rows)

    result = compute_brinson_attribution(df, how="sum")

    assert isinstance(result, BrinsonAttribution)
    # by_group: one row per group_id, sums across dates
    assert set(result.by_group.columns) >= {"group_id", "alloc", "select", "interact", "total"}
    assert result.by_group.height == 2

    # Group 0: alloc = 3 * 0.001 = 0.003
    g0 = result.by_group.filter(pl.col("group_id") == 0).row(0, named=True)
    assert np.isclose(g0["alloc"], 0.003, atol=1e-12)
    assert np.isclose(g0["select"], 0.006, atol=1e-12)

    # Total = sum across all (date, group_id) rows
    total_row = result.total.row(0, named=True)
    expected_alloc = 3 * (0.001 + -0.0005)
    assert np.isclose(total_row["alloc"], expected_alloc, atol=1e-12)


def test_compute_brinson_attribution_rejects_unsupported_format():
    """A df missing both group_id/group and metric_* columns must raise."""
    df = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1)],
            "foo": [1.0],
            "bar": [2.0],
        }
    )
    with pytest.raises(ValueError, match="Unsupported Brinson timeseries format"):
        compute_brinson_attribution(df)
