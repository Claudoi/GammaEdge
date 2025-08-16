# Risk model calculations
from __future__ import annotations
import warnings
from typing import Literal, Optional
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS

CovMethod = Literal["sample", "lw", "oas", "ewma"]
MuMethod  = Literal["historical", "ema", "shrunk"]

def _infer_ann_factor(index: pd.Index) -> int:
    if len(index) > 1:
        dt = (index[1] - index[0]).days or 1
        return 252 if dt <= 3 else (52 if dt <= 7 else 12)
    return 252

def expected_returns(
    returns: pd.DataFrame,
    method: MuMethod = "ema",
    span: Optional[int] = 60,
    shrink_to: Optional[pd.Series] = None,
) -> pd.Series:
    r = returns.dropna(how="all")
    if r.empty:
        return pd.Series(dtype=float)
    ann = _infer_ann_factor(r.index)

    if method == "historical":
        mu = r.mean() * ann

    elif method == "ema":
        if span is None:
            span = 60
        mu = r.ewm(span=span, adjust=False).mean().iloc[-1] * ann

    elif method == "shrunk":
        base = r.mean()
        target = (shrink_to.reindex(base.index).fillna(0.0)
                  if shrink_to is not None else pd.Series(0.0, index=base.index))
        # peso de shrinkage simple (convexo) basado en varianzas
        lam = 0.5 if base.var() == 0 else float(r.var().mean() / (r.var().mean() + base.var()))
        mu = ((1 - lam) * base + lam * target) * ann

    else:
        raise ValueError(f"Unknown expected return method: {method}")

    return mu.astype(float)

def covariance(
    returns: pd.DataFrame,
    method: CovMethod = "oas",
    ewma_lambda: float = 0.94,
    min_periods: int = 60,
) -> pd.DataFrame:
    r = returns.dropna(how="all")
    if r.shape[0] < max(2, min_periods):
        warnings.warn("Too few observations for stable covariance; falling back to 'sample'.")
        method = "sample"

    ann = _infer_ann_factor(r.index)

    X = r.values
    if method == "sample":
        Sigma = np.cov(X, rowvar=False)

    elif method == "lw":
        Sigma = LedoitWolf().fit(X).covariance_

    elif method == "oas":
        Sigma = OAS().fit(X).covariance_

    elif method == "ewma":
        lam = float(ewma_lambda)
        Xc = r - r.mean()
        Sigma = np.zeros((Xc.shape[1], Xc.shape[1]))
        for t in range(Xc.shape[0]):
            x = Xc.iloc[t].values.reshape(-1, 1)
            Sigma = lam * Sigma + (1 - lam) * (x @ x.T)
    else:
        raise ValueError(f"Unknown covariance method: {method}")

    return pd.DataFrame(Sigma, index=r.columns, columns=r.columns) * ann
