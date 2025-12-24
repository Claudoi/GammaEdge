# tests/unit/test_black_litterman.py
import numpy as np

from portfolio.optim.black_litterman import black_litterman_posterior, market_implied_prior


def test_market_implied_prior():
    # Mercado super simple 2 activos
    Sigma = np.array([[0.1, 0.0], [0.0, 0.1]])
    w_mkt = np.array([0.6, 0.4])
    delta = 2.5

    # Prior = 2.5 * Sigma @ w = 2.5 * [0.06, 0.04] = [0.15, 0.10]
    pi = market_implied_prior(Sigma, w_mkt, delta, rf=0.0)

    expected = np.array([0.15, 0.10])
    np.testing.assert_allclose(pi, expected, atol=1e-6)


def test_black_litterman_basic():
    # 2 Activos
    # Prior = [10%, 10%]
    # View: Activo 0 va a subir -> [15%]
    Sigma = np.eye(2) * 0.1  # alta varianza para magnificar
    Pi = np.array([0.10, 0.10])
    tau = 0.05

    # View: P=[1, 0], Q=[0.15] -> Absoluta sobre activo 0
    P = np.array([[1.0, 0.0]])
    Q = np.array([0.15])
    # Alta confianza en la vista (Omega pequeño)
    Omega = np.array([[0.0001]])

    mu_bl, _ = black_litterman_posterior(Sigma, Pi, tau, P, Q, Omega=Omega)

    # bl activo 0 debe estar cerca de 0.15 (view) porque Omega es muy bajo (strong view)
    assert mu_bl[0] > 0.14
    # bl activo 1 debe quedarse cerca del prior 0.10
    assert abs(mu_bl[1] - 0.10) < 0.01


def test_black_litterman_with_conf():
    # Idzorek method check
    Sigma = np.eye(2)
    Pi = np.array([0.05, 0.05])
    tau = 0.05
    P = np.array([[1.0, -1.0]])  # Relative View: A > B
    Q = np.array([0.02])
    conf = np.array([0.90])  # 90% confidence

    mu_bl, _ = black_litterman_posterior(Sigma, Pi, tau, P, Q, view_confidences=conf)

    # El spread posterior deberia ser positivo
    spread_post = mu_bl[0] - mu_bl[1]
    assert spread_post > 0.0
