# tests/test_attribution_euler.py
# tests/test_attribution_euler.py
import numpy as np
import polars as pl

from portfolio.backtest.attribution import DailyAlignment, euler_rc_by_asset
from tests.utils_dates import make_dates


def _aln_eqw(T=8, N=4, seed=7):
    rng = np.random.default_rng(seed)
    R = rng.normal(0.0, 0.01, size=(T, N))
    W = np.full((T, N), 1.0 / N, dtype=float)
    dates = make_dates(T, tz="UTC")  # or tz=None
    tickers = [f"A{i}" for i in range(N)]
    return DailyAlignment(dates=dates, tickers=tickers, returns=R, weights=W)


def test_euler_rc_sums_to_sigma_diagonal():
    aln = _aln_eqw(T=8, N=4)
    # Fixed diagonal covariance (per-date identical)
    variances = np.array([0.04, 0.01, 0.09, 0.16])  # std = [0.2, 0.1, 0.3, 0.4]
    C = np.diag(variances)
    df = euler_rc_by_asset(aln, cov=C)

    # For each date: sum rc ≈ sigma
    chk = df.group_by("date").agg(
        [
            pl.col("rc").sum().alias("rc_sum"),
            pl.col("sigma").first().alias("sigma"),
        ]
    )
    assert np.allclose(chk["rc_sum"].to_numpy(), chk["sigma"].to_numpy(), atol=1e-10)


def test_euler_rc_dense_cov_timevarying():
    aln = _aln_eqw(T=10, N=3)
    # Time-varying cov list: mix of diagonal + dense
    covs = []
    base = np.array([[0.03, 0.01, 0.00], [0.01, 0.02, 0.00], [0.00, 0.00, 0.05]], dtype=float)
    for t in range(10):
        C = base + 0.001 * t * np.array([[1, 0.2, 0.1], [0.2, 1, 0.1], [0.1, 0.1, 1]], dtype=float)
        covs.append(C)
    df = euler_rc_by_asset(aln, cov=covs)

    chk = df.group_by("date").agg(
        [
            pl.col("rc").sum().alias("rc_sum"),
            pl.col("sigma").first().alias("sigma"),
        ]
    )
    assert np.allclose(chk["rc_sum"].to_numpy(), chk["sigma"].to_numpy(), atol=1e-10)
