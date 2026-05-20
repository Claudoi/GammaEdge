"""Pydantic schemas for POST /api/v1/optimize.

Request: matrix of returns + ticker labels + optimizer method + risk-free rate.
Response: weights per ticker + portfolio metrics + metadata.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator

OptimizerMethod = Literal[
    "mean_variance",
    "min_variance",
    "risk_parity",
    "hrp",
]


class OptimizeRequest(BaseModel):
    """Input payload for /optimize.

    Returns are passed as a T x N matrix (T time periods, N assets).
    Tickers are the column labels and define the asset order.
    """

    returns: list[list[float]] = Field(
        ...,
        description="Matrix of returns shape (T, N). T must be >= 2.",
        min_length=2,
    )
    tickers: list[str] = Field(
        ...,
        description="Ticker symbols, one per column of `returns`.",
        min_length=2,
    )
    method: OptimizerMethod = Field(
        default="mean_variance",
        description="Optimization method to apply.",
    )
    rf: float = Field(
        default=0.04,
        ge=0.0,
        le=1.0,
        description="Annualized risk-free rate used to compute the Sharpe ratio.",
    )

    @field_validator("returns")
    @classmethod
    def _validate_returns_finite(cls, v: list[list[float]]) -> list[list[float]]:
        if not v:
            raise ValueError("returns must not be empty")
        n_cols = len(v[0])
        if n_cols < 2:
            raise ValueError("returns must have at least 2 columns")
        for i, row in enumerate(v):
            if len(row) != n_cols:
                raise ValueError(f"returns row {i} has {len(row)} values, expected {n_cols}")
            for j, x in enumerate(row):
                if math.isnan(x) or math.isinf(x):
                    raise ValueError(f"returns[{i}][{j}] is not finite (NaN or Inf)")
        return v

    @field_validator("tickers")
    @classmethod
    def _validate_tickers_unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("tickers must be unique")
        if any(not t or not t.strip() for t in v):
            raise ValueError("tickers must be non-empty strings")
        return v


class OptimizeMetrics(BaseModel):
    """Annualized portfolio metrics computed from the resulting weights."""

    expected_return: float
    volatility: float
    sharpe: float


class OptimizeResponse(BaseModel):
    """Output of /optimize."""

    weights: dict[str, float] = Field(
        ..., description="Mapping ticker -> weight. Weights sum to ~1.0."
    )
    metrics: OptimizeMetrics
    method: OptimizerMethod
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp at which the optimization ran.")
