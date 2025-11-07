# tests/attribution/test_euler.py

import numpy as np
import pandas as pd
import pytest

from portfolio.attribution.euler import euler_risk_contributions


def test_euler_risk_contributions_basic_2x2():
    # Pesos de la cartera
    w = pd.Series([0.6, 0.4], index=["A", "B"])

    # Matriz de covarianzas simple
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    rc = euler_risk_contributions(w, cov)

    # 1) mismo índice que los pesos
    assert list(rc.index) == ["A", "B"]

    # 2) la suma de contribuciones = volatilidad total de la cartera
    w_vec = w.values.reshape(-1, 1)
    sigma = cov.values
    port_var = float(w_vec.T @ sigma @ w_vec)
    port_sigma = float(np.sqrt(port_var))

    assert np.isclose(rc.sum(), port_sigma)

    # 3) ninguna contribución negativa en este ejemplo (todo positivo)
    assert (rc.values > 0).all()


def test_euler_raises_on_wrong_cov_shape():
    """Debe fallar si la matriz de covarianzas no es NxN compatible con w."""
    w = pd.Series([0.5, 0.5], index=["A", "B"])
    # Matriz 3x3 incompatible con 2 pesos
    cov = np.eye(3)

    with pytest.raises(ValueError):
        euler_risk_contributions(w, cov)


def test_euler_raises_on_non_positive_variance():
    """Debe fallar si la varianza de cartera no es estrictamente positiva."""
    w = pd.Series([0.5, 0.5], index=["A", "B"])
    # Covarianza cero ⇒ varianza 0
    cov = np.zeros((2, 2))

    with pytest.raises(ValueError):
        euler_risk_contributions(w, cov)
