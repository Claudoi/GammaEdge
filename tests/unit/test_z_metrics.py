# tests/unit/test_z_metrics.py
import numpy as np

from portfolio.backtest.metrics import cvar_historic, omega_ratio, var_historic


def test_var_cvar_gaussian():
    np.random.seed(42)
    # 10000 dias de retornos normales N(0, 0.01)
    rets = np.random.normal(0, 0.01, 100000)

    # VaR 95% teorico para N(0, 1) es 1.645 * sigma
    sigma = 0.01
    var95_theo = 1.645 * sigma  # ~0.01645

    var_calc = var_historic(rets, 0.95)

    # Check close
    assert abs(var_calc - var95_theo) < 1e-3

    # CVaR 95% teorico para N(0, sigma) es (pdf(1.645)/(1-0.95)) * sigma
    # phi(1.645) ~ 0.103
    # 0.103 / 0.05 * 0.01 = 2.06 * 0.01 = 0.0206

    cvar_calc = cvar_historic(rets, 0.95)
    assert cvar_calc > var_calc  # CVaR siempre > VaR (en perdida absoluta)
    assert abs(cvar_calc - 0.0206) < 2e-3


def test_omega_ratio():
    # Caso simple: Ganamos siempre
    rets = np.array([0.01, 0.02, 0.01])
    om = omega_ratio(rets, 0.0)
    assert np.isinf(om) or om > 1000

    # Caso simetrico
    rets2 = np.array([0.01, -0.01])
    om2 = omega_ratio(rets2, 0.0)
    assert abs(om2 - 1.0) < 1e-6
