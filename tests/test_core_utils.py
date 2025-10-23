import numpy as np

from portfolio.core.utils import cond_number, ensure_psd, project_to_box_simplex


def test_ensure_psd_and_cond():
    S = np.array([[1.0, 2.0], [2.0, 1.0]])  # no PSD
    S_psd = ensure_psd(S, eps=1e-10, clip=True)
    w = np.linalg.eigvalsh(S_psd)
    assert (w >= -1e-12).all()
    c = cond_number(S_psd)
    assert np.isfinite(c) and c > 0


def test_project_to_box_simplex_basic():
    v = np.array([0.5, 0.7, -0.2])
    x = project_to_box_simplex(v, w_min=0.0, w_max=0.8)
    assert np.isfinite(x).all()
    assert abs(x.sum() - 1.0) < 1e-8
    assert (x >= -1e-8).all() and (x <= 0.8 + 1e-8).all()
