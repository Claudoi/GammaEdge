"""GET /health — liveness probe."""

from __future__ import annotations

import sys
import time as _time

from fastapi import APIRouter

from api.schemas import HealthResponse

router = APIRouter(tags=["health"])
_START_TIME = _time.time()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        import portfolio  # noqa: F401

        lib_ok = True
    except Exception:
        lib_ok = False

    return HealthResponse(
        status="ok" if lib_ok else "degraded",
        uptime_s=int(_time.time() - _START_TIME),
        python_version=sys.version.split()[0],
        portfolio_lib_check=lib_ok,
    )
