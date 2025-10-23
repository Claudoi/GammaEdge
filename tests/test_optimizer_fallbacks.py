import numpy as np

from portfolio.core.utils import hrp_safe


def dummy_hrp_fail(cov: np.ndarray, **kwargs):
    raise RuntimeError("clustering failed")


def test_hrp_safe_fallback_equal_weight():
    S = np.array([[0.01, 0.0], [0.0, 0.02]])
    w = hrp_safe(dummy_hrp_fail, cov=S)
    assert w.shape == (2,)
    assert abs(w.sum() - 1.0) < 1e-8
    assert (w >= 0.0).all()
