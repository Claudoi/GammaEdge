import numpy as np

from portfolio.backtest.kpis import compute_kpis


def test_kpis_smoke():
    # equity creciente con pequeña volatilidad
    np.random.seed(0)
    rets = 0.0005 + 0.01 * np.random.randn(300)
    eq = np.cumprod(1.0 + rets)
    k = compute_kpis(eq, rf_daily=0.0, periods_per_year=252)
    assert "Sharpe" in k and "MaxDD" in k and "CAGR" in k
