# tests/unit/test_kpis_edge_cases.py
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from portfolio.backtest.kpis import compute_kpis


def _kpis_to_dict(obj) -> dict[str, float]:
    """
    Normalise compute_kpis output to a flat dict[str, float].

    Supports:
    - dict-like (already key -> value)
    - Polars DataFrame (single row, metrics as columns)
    - Pandas DataFrame (single row, metrics as columns)
    """
    if isinstance(obj, dict):
        out: dict[str, float] = {}
        for k, v in obj.items():
            try:
                fv = float(v)
            except Exception:
                continue
            out[str(k).lower()] = fv
        return out

    if isinstance(obj, pl.DataFrame):
        if obj.height == 0:
            return {}
        row = obj.row(0, named=True)
        out = {}
        for k, v in row.items():
            try:
                fv = float(v)
            except Exception:
                continue
            out[str(k).lower()] = fv
        return out

    if isinstance(obj, pd.DataFrame):
        if len(obj) == 0:
            return {}
        row = obj.iloc[0].to_dict()
        out = {}
        for k, v in row.items():
            try:
                fv = float(v)
            except Exception:
                continue
            out[str(k).lower()] = fv
        return out

    # Fallback: nothing we recognise
    return {}


def test_compute_kpis_constant_equity_zero_risk():
    """
    If equity is perfectly flat, CAGR and MaxDD should be ~0 and
    risk-related ratios should not explode.
    """
    equity = np.full(252, 100.0, dtype=float)

    kpis = compute_kpis(equity, rf_daily=0.0, periods_per_year=252)
    metrics = _kpis_to_dict(kpis)

    # CAGR ~ 0
    if "cagr" in metrics:
        cagr = metrics["cagr"]
        assert abs(cagr) < 1e-6

    # MaxDD ~ 0
    if "maxdd" in metrics:
        maxdd = metrics["maxdd"]
        # Allow tiny numerical noise, sign can depend on implementation
        assert abs(maxdd) < 1e-6

    # Volatility very small if exists
    for name in ("vol", "volatility", "ann_vol", "annual_vol"):
        if name in metrics:
            vol = metrics[name]
            assert vol >= 0.0
            assert vol < 1e-6
            break

    # Sharpe / Sortino / Calmar must not explode to inf
    for ratio_name in ("sharpe", "sortino", "calmar"):
        if ratio_name in metrics:
            val = metrics[ratio_name]
            assert not np.isinf(val)
            # NaN is acceptable when vol == 0


def test_compute_kpis_monotone_up_equity_positive_cagr():
    """
    Monotonically increasing equity should give positive CAGR
    and small drawdown.
    """
    T = 252
    r_daily = 0.001  # ~0.1% per day
    equity = 100.0 * (1.0 + r_daily) ** np.arange(T, dtype=float)

    kpis = compute_kpis(equity, rf_daily=0.0, periods_per_year=252)
    metrics = _kpis_to_dict(kpis)

    if "cagr" in metrics:
        cagr = metrics["cagr"]
        assert cagr > 0.0
        approx_annual = (1.0 + r_daily) ** 252 - 1.0
        # 5% tolerance
        assert abs(cagr - approx_annual) < 5e-2

    if "maxdd" in metrics:
        maxdd = metrics["maxdd"]
        # monotone path → drawdown almost zero (allow sign conventions)
        assert abs(maxdd) < 1e-3


def test_compute_kpis_handles_nans_and_short_series():
    """
    Function should handle:
    - short series
    - some NaNs in the path
    without crashing.
    """
    equity = np.array([100.0, np.nan, 101.0, 100.5, np.nan, 102.0], dtype=float)

    kpis = compute_kpis(equity, rf_daily=0.0, periods_per_year=252)
    metrics = _kpis_to_dict(kpis)

    # Just check it returns something sensible-ish
    assert isinstance(metrics, dict)
    assert metrics != {}

    if "cagr" in metrics:
        cagr = metrics["cagr"]
        assert not np.isinf(cagr)

    if "maxdd" in metrics:
        maxdd = metrics["maxdd"]
        # Solo exigimos que no sea infinito y que esté en un rango razonable.
        # Puede ser ligeramente negativo según la convención de signo.
        assert not np.isinf(maxdd)
        assert abs(maxdd) <= 1.0
