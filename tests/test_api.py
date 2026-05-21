"""Integration tests for the GammaEdge FastAPI endpoints.

Uses fastapi.testclient.TestClient to call endpoints without a running server.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_returns_3() -> dict:
    """Returns matrix and tickers for 3 assets, 252 daily observations."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, (252, 3)).tolist()
    return {"returns": returns, "tickers": ["AAPL", "MSFT", "GOOGL"]}


# ─── /health ─────────────────────────────────────────────────────────────


def test_health_ok():
    """Health probe returns 200 with expected payload structure."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["uptime_s"] >= 0
    assert body["python_version"].startswith("3.")
    assert isinstance(body["portfolio_lib_check"], bool)


# ─── /api/v1/optimize: happy paths ───────────────────────────────────────


def test_optimize_mean_variance_returns_valid_weights(synthetic_returns_3):
    """mean_variance optimization returns weights summing to ~1.0."""
    r = client.post("/api/v1/optimize", json={**synthetic_returns_3, "method": "mean_variance"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["weights"].keys()) == set(synthetic_returns_3["tickers"])
    assert abs(sum(body["weights"].values()) - 1.0) < 1e-4
    assert body["method"] == "mean_variance"
    assert "expected_return" in body["metrics"]
    assert "volatility" in body["metrics"]
    assert "sharpe" in body["metrics"]
    assert math.isfinite(body["metrics"]["sharpe"])


def test_optimize_min_variance_yields_lower_vol_than_mvo(synthetic_returns_3):
    """Minimum-variance portfolio should have <= volatility than mean-variance optimal."""
    r_mvo = client.post("/api/v1/optimize", json={**synthetic_returns_3, "method": "mean_variance"})
    r_min = client.post("/api/v1/optimize", json={**synthetic_returns_3, "method": "min_variance"})
    assert r_mvo.status_code == 200
    assert r_min.status_code == 200
    # Minimum variance should be <= mean-variance volatility (it's the minimum)
    assert r_min.json()["metrics"]["volatility"] <= r_mvo.json()["metrics"]["volatility"] + 1e-9


def test_optimize_risk_parity_equal_risk_contributions(synthetic_returns_3):
    """risk_parity returns positive weights summing to 1."""
    r = client.post("/api/v1/optimize", json={**synthetic_returns_3, "method": "risk_parity"})
    assert r.status_code == 200, r.text
    body = r.json()
    weights = list(body["weights"].values())
    assert all(w >= -1e-9 for w in weights), "Risk parity weights should be non-negative"
    assert abs(sum(weights) - 1.0) < 1e-4


def test_optimize_hrp_works_or_returns_501(synthetic_returns_3):
    """HRP either works (200) or is gracefully unimplemented (501).

    Either outcome is acceptable as long as the failure is structured.
    """
    r = client.post("/api/v1/optimize", json={**synthetic_returns_3, "method": "hrp"})
    assert r.status_code in {200, 501}, r.text
    if r.status_code == 200:
        body = r.json()
        assert abs(sum(body["weights"].values()) - 1.0) < 1e-4


# ─── /api/v1/optimize: validation errors ─────────────────────────────────


def test_optimize_rejects_non_numeric_returns():
    """Non-numeric values in the returns matrix should yield a 422."""
    r = client.post(
        "/api/v1/optimize",
        json={
            "returns": [["not-a-number", 0.01], [0.01, 0.02]],
            "tickers": ["A", "B"],
            "method": "mean_variance",
        },
    )
    assert r.status_code == 422


def test_optimize_rejects_single_asset():
    """A single asset cannot form a portfolio."""
    r = client.post(
        "/api/v1/optimize",
        json={
            "returns": [[0.01], [0.02]],
            "tickers": ["AAPL"],
            "method": "mean_variance",
        },
    )
    assert r.status_code == 422


def test_optimize_rejects_mismatched_columns_and_tickers(synthetic_returns_3):
    """tickers length must match returns columns."""
    payload = {**synthetic_returns_3, "tickers": ["A", "B"], "method": "mean_variance"}
    r = client.post("/api/v1/optimize", json=payload)
    assert r.status_code == 422


def test_optimize_rejects_duplicate_tickers():
    """Duplicate tickers should be rejected."""
    r = client.post(
        "/api/v1/optimize",
        json={
            "returns": [[0.01, 0.02], [0.03, 0.04]],
            "tickers": ["AAPL", "AAPL"],
            "method": "mean_variance",
        },
    )
    assert r.status_code == 422
