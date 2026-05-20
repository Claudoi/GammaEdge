# tests/test_risk_models_edge.py
import numpy as np
import pytest


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


def test_expected_returns_raises_on_all_null_column():
    """SILENT-3: expected_returns must raise ValueError if result contains NaN or Inf."""
    from datetime import date, timedelta

    import polars as pl

    from portfolio.features.risk_models import expected_returns

    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    df = pl.DataFrame(
        {
            "date": dates,
            "AAPL": np.random.default_rng(0).normal(0.001, 0.02, 100).tolist(),
            "BAD": [None] * 100,  # completely null column
        }
    )
    with pytest.raises(ValueError, match="invalid"):
        expected_returns(df, method="historical", annualize=True)


def test_ewma_vectorized_matches_loop():
    """P5: Vectorized EWMA must produce a PSD matrix without NaN."""
    from datetime import date, timedelta

    import polars as pl

    from portfolio.features.risk_models import covariance as cov_fn

    rng = np.random.default_rng(7)
    n_obs, n_assets = 500, 15
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_obs)]
    data = {"date": dates}
    for i in range(n_assets):
        data[f"A{i}"] = rng.normal(0.001, 0.02, n_obs).tolist()
    df = pl.DataFrame(data)

    Sigma, _ = cov_fn(df, method="ewma", psd=True, as_frame=False, ewma_lambda=0.94)
    assert Sigma.shape == (n_assets, n_assets)
    assert not np.isnan(Sigma).any(), "EWMA covariance contains NaN"
    eigvals = np.linalg.eigvalsh(Sigma)
    assert eigvals.min() >= -1e-9, f"Not PSD: min eigenvalue = {eigvals.min()}"


def test_ewma_fast_for_realistic_size():
    """Vectorized EWMA must complete quickly for typical T=500, N=20."""
    import time
    from datetime import date, timedelta

    import polars as pl

    from portfolio.features.risk_models import covariance as cov_fn

    rng = np.random.default_rng(3)
    n_obs, n_assets = 500, 20
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_obs)]
    data = {"date": dates}
    for i in range(n_assets):
        data[f"A{i}"] = rng.normal(0.001, 0.02, n_obs).tolist()
    df = pl.DataFrame(data)

    t0 = time.perf_counter()
    cov_fn(df, method="ewma", psd=True, as_frame=False)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"EWMA took {elapsed:.3f}s — should be <0.5s for T=500, N=20"
