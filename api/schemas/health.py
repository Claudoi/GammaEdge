"""Pydantic schema for GET /health."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Liveness probe payload."""

    status: str  # "ok" | "degraded"
    uptime_s: int
    python_version: str
    portfolio_lib_check: bool
