# tests/test_risk_models_edge.py
import numpy as np


def test_lw_covariance_is_psd():
    """HIGH-1: Ledoit-Wolf must always produce a PSD matrix."""
    from datetime import date, timedelta

    import polars as pl

    from portfolio.features.risk_models import covariance

    rng = np.random.default_rng(0)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(300)]
    data = {"date": dates}
    for i in range(10):
        data[f"A{i}"] = rng.normal(0.001, 0.02, 300).tolist()
    df = pl.DataFrame(data)
    Sigma, names = covariance(df, method="lw", psd=True, as_frame=False)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert eigvals.min() >= -1e-9, f"LW covariance has negative eigenvalue: {eigvals.min()}"


def test_ensure_psd_removes_negative_eigenvalues():
    """_ensure_psd in risk_models must fix negative eigenvalues."""
    from portfolio.features.risk_models import _ensure_psd as rm_psd

    rng = np.random.default_rng(1)
    A = rng.normal(0, 1, (5, 5))
    S = A @ A.T
    # Introduce a negative eigenvalue
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals[0] = -0.001
    S_bad = eigvecs @ np.diag(eigvals) @ eigvecs.T

    S_fixed = rm_psd(S_bad)
    eigvals_fixed = np.linalg.eigvalsh(S_fixed)
    assert (
        eigvals_fixed.min() >= -1e-9
    ), f"_ensure_psd left negative eigenvalue: {eigvals_fixed.min()}"


def test_oas_covariance_is_psd():
    """OAS covariance (default) must produce PSD matrix."""
    from datetime import date, timedelta

    import polars as pl

    from portfolio.features.risk_models import covariance

    rng = np.random.default_rng(2)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(300)]
    data = {"date": dates}
    for i in range(10):
        data[f"A{i}"] = rng.normal(0.001, 0.02, 300).tolist()
    df = pl.DataFrame(data)
    Sigma, names = covariance(df, method="oas", psd=True, as_frame=False)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert eigvals.min() >= -1e-9, f"OAS covariance has negative eigenvalue: {eigvals.min()}"
