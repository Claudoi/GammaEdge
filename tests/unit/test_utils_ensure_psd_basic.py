# tests/unit/test_utils_ensure_psd_basic.py
from __future__ import annotations

import numpy as np

from portfolio.core.utils import ensure_psd


def test_ensure_psd_basic_symmetrises_and_clips():
    # matriz casi-PSD con un pequeño negativo numérico
    A = np.array([[1.0, 0.99], [0.99, 0.98]], dtype=float)
    # fuerza una ligera asimetría y un autovalor negativo minúsculo
    A[0, 1] = 0.991
    A[1, 0] = 0.989
    A_psd = ensure_psd(A)

    # Debe ser simétrica y PSD (eigs >= 0 dentro de tolerancia)
    assert np.allclose(A_psd, A_psd.T, atol=1e-12)
    w = np.linalg.eigvalsh(A_psd)
    assert np.min(w) >= -1e-12

    # Mantiene escala razonable
    assert np.isfinite(A_psd).all()
