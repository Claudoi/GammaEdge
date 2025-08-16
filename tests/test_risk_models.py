# Test for risk models
import numpy as np
import pandas as pd
from portfolio.features.risk_models import covariance, expected_returns

def test_oas_is_psd():
    rng = np.random.default_rng(0)
    R = pd.DataFrame(rng.normal(0, 0.01, size=(300, 5)), columns=list("ABCDE"))
    S = covariance(R, method="oas")
    eig = np.linalg.eigvalsh(S.values)
    assert (eig >= -1e-10).all()

def test_ema_mu_changes_with_span():
    rng = np.random.default_rng(1)
    R = pd.DataFrame(rng.normal(0.0005, 0.01, size=(200, 3)))
    mu_short = expected_returns(R, method="ema", span=20)
    mu_long  = expected_returns(R, method="ema", span=100)
    assert not np.allclose(mu_short.values, mu_long.values)
