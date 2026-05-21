"""POST /api/v1/optimize — portfolio optimization endpoint."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, HTTPException, status

from api.schemas import (
    OptimizeMetrics,
    OptimizeRequest,
    OptimizeResponse,
    OptimizerMethod,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["optimize"])

TRADING_DAYS = 252


def _annualize(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute annualized mean returns and covariance from a TxN return matrix."""
    mu = R.mean(axis=0) * TRADING_DAYS
    Sigma = np.cov(R, rowvar=False) * TRADING_DAYS
    return mu, Sigma


def _solve(method: OptimizerMethod, mu: np.ndarray, Sigma: np.ndarray, rf: float) -> np.ndarray:
    """Dispatch to the appropriate portfolio/optim/ function."""
    if method == "mean_variance":
        from portfolio.optim.mean_variance import markowitz_closed_form

        _w_mvp, w_tan = markowitz_closed_form(mu, Sigma, rf=rf)
        return np.asarray(w_tan, dtype=np.float64)
    if method == "min_variance":
        from portfolio.optim.mean_variance import markowitz_closed_form

        w_mvp, _w_tan = markowitz_closed_form(mu, Sigma, rf=rf)
        return np.asarray(w_mvp, dtype=np.float64)
    if method == "risk_parity":
        from portfolio.optim.risk_parity import risk_parity

        return np.asarray(risk_parity(Sigma), dtype=np.float64)
    if method == "hrp":
        from portfolio.optim.hrp import hrp_weights

        return np.asarray(hrp_weights(Sigma), dtype=np.float64)
    raise ValueError(f"Unknown method: {method}")


@router.post(
    "/optimize",
    response_model=OptimizeResponse,
    summary="Optimize a portfolio given returns and a method",
    responses={
        400: {"description": "Infeasible optimization problem"},
        422: {"description": "Validation error in request body"},
        500: {"description": "Solver failure"},
    },
)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    R = np.asarray(req.returns, dtype=np.float64)
    if R.shape[1] != len(req.tickers):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(f"returns has {R.shape[1]} columns but tickers has {len(req.tickers)} entries"),
        )

    try:
        mu, Sigma = _annualize(R)
        w = _solve(req.method, mu, Sigma, req.rf)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Optimization failed: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Optimizer crashed for method=%s", req.method)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Solver error: {type(exc).__name__}",
        ) from exc

    if np.any(np.isnan(w)):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Solver returned NaN weights",
        )

    weights_dict = {t: float(wi) for t, wi in zip(req.tickers, w, strict=True)}
    er = float(w @ mu)
    vol = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
    sharpe = (er - req.rf) / vol if vol > 1e-12 else 0.0

    return OptimizeResponse(
        weights=weights_dict,
        metrics=OptimizeMetrics(
            expected_return=er,
            volatility=vol,
            sharpe=sharpe,
        ),
        method=req.method,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
