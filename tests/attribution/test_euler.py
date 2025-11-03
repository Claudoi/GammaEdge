# tests/attribution/test_euler.py

import numpy as np
import pandas as pd

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

    # Propiedades básicas:
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
