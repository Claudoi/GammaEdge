# tests/quick/test_kpis_property.py
from __future__ import annotations

import numpy as np

from portfolio.backtest.kpis import (
    compute_kpis,
    equity_to_drawdown,
    equity_to_returns,
)


def test_equity_to_returns_empty_and_short():
    assert equity_to_returns([]).size == 0
    assert equity_to_returns([1.0]).size == 0


def test_equity_to_returns_constant_series_zero_returns():
    eq = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    rets = equity_to_returns(eq)
    assert rets.size == 3
    assert np.allclose(rets, 0.0)


def test_equity_to_drawdown_basic():
    eq = np.array([100.0, 110.0, 105.0, 120.0, 108.0], dtype=np.float64)
    dd = equity_to_drawdown(eq)
    assert dd.size == eq.size
    # drawdown nunca positivo; en máximos debe ser ~0
    assert np.nanmax(dd) <= 1e-12
    assert np.isclose(dd[1], 0.0, atol=1e-12)  # en 110 iguala máximo
    assert dd[-1] < 0  # 108 por debajo del máximo 120


def test_compute_kpis_handles_nan_inf_and_empty():
    eq_bad = np.array([np.nan, np.inf, -np.inf], dtype=np.float64)
    k = compute_kpis(eq_bad)
    # todas las keys presentes
    for key in [
        "Total Return",
        "CAGR",
        "Ann. Return (excess)",
        "Ann. Vol",
        "Sharpe",
        "Sortino",
        "MaxDD",
        "Calmar",
        "Hit Ratio",
    ]:
        assert key in k

    # vacío
    k2 = compute_kpis(np.array([], dtype=np.float64))
    assert np.isnan(k2["Total Return"])
    assert np.isnan(k2["CAGR"])


def test_compute_kpis_constant_equity_is_well_defined():
    eq = np.full(252, 100.0, dtype=np.float64)
    k = compute_kpis(eq)
    # rentabilidad total ~0 y métricas que dependen de volatilidad => NaN o 0 donde aplique
    assert abs(k["Total Return"]) < 1e-12
    # Vol anualizada debería ser NaN (no hay varianza muestral con ddof=1)
    assert np.isnan(k["Ann. Vol"]) or k["Ann. Vol"] == 0.0
