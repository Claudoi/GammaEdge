"""GammaEdge REST API — FastAPI application entry point.

Run with: uvicorn api.main:app --reload
OpenAPI docs: http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI

from api.routes.health import router as health_router
from api.routes.optimize import router as optimize_router

app = FastAPI(
    title="GammaEdge API",
    description=(
        "REST interface for the GammaEdge portfolio optimization platform. "
        "Currently exposes portfolio optimization and a health probe; "
        "additional endpoints are future work."
    ),
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(optimize_router)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"message": "GammaEdge API. See /docs for OpenAPI documentation."}
