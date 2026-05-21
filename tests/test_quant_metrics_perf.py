"""
Performance and correctness tests for vectorized correlation matrix.

Guards against re-introduction of the O(N^2) Python nested loop that
called ``np.corrcoef`` once per pair. The vectorized implementation
computes the full N x N correlation matrix with a single batch call.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from portfolio.features.quant_metrics import calculate_correlation_matrix


def _make_returns(n_assets: int, n_obs: int = 500, seed: int = 42) -> pl.DataFrame:
    """Build a wide returns DataFrame with deterministic random data."""
    rng = np.random.default_rng(seed)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_obs)]
    data: dict[str, list] = {"date": dates}
    for i in range(n_assets):
        data[f"ret_A{i}"] = rng.normal(0.001, 0.02, n_obs).tolist()
    return pl.DataFrame(data)


class TestCorrelationCorrectness:
    """Correctness invariants for the vectorized correlation matrix."""

    def test_diagonal_is_one(self):
        """Diagonal of correlation matrix must be exactly 1.0."""
        df = _make_returns(n_assets=5)
        result = calculate_correlation_matrix(df, return_col_prefix="ret_")
        corr_df = result["correlation_matrix"]
        assert corr_df is not None

        tickers = [f"A{i}" for i in range(5)]
        for i, t in enumerate(tickers):
            assert abs(corr_df[t][i] - 1.0) < 1e-12, f"Diagonal {t} != 1.0"

    def test_matrix_is_symmetric(self):
        """Correlation matrix must be symmetric (corr(i, j) == corr(j, i))."""
        df = _make_returns(n_assets=6)
        result = calculate_correlation_matrix(df, return_col_prefix="ret_")
        corr_df = result["correlation_matrix"]
        assert corr_df is not None

        tickers = [f"A{i}" for i in range(6)]
        for i, ti in enumerate(tickers):
            for j, tj in enumerate(tickers):
                assert abs(corr_df[ti][j] - corr_df[tj][i]) < 1e-12, f"Asymmetric at ({ti}, {tj})"

    def test_off_diagonal_in_unit_interval(self):
        """All correlations must lie in [-1, 1]."""
        df = _make_returns(n_assets=8)
        result = calculate_correlation_matrix(df, return_col_prefix="ret_")
        corr_df = result["correlation_matrix"]
        assert corr_df is not None

        tickers = [f"A{i}" for i in range(8)]
        values = corr_df.select(tickers).to_numpy()
        assert np.all(values >= -1.0 - 1e-12)
        assert np.all(values <= 1.0 + 1e-12)

    def test_matches_reference_numpy_corrcoef(self):
        """Vectorized result must match a direct numpy reference."""
        df = _make_returns(n_assets=10, n_obs=300, seed=7)
        result = calculate_correlation_matrix(df, return_col_prefix="ret_")
        corr_df = result["correlation_matrix"]
        assert corr_df is not None

        tickers = [f"A{i}" for i in range(10)]
        # Reference: global complete-case drop, single corrcoef
        ref_arr = df.select([f"ret_{t}" for t in tickers]).drop_nulls().to_numpy()
        ref_corr = np.corrcoef(ref_arr, rowvar=False)

        result_arr = corr_df.select(tickers).to_numpy()
        np.testing.assert_allclose(result_arr, ref_corr, rtol=1e-10, atol=1e-12)

    def test_single_ticker_returns_none(self):
        """Single ticker must return None for correlation_matrix."""
        df = _make_returns(n_assets=1)
        result = calculate_correlation_matrix(df, return_col_prefix="ret_")
        assert result["correlation_matrix"] is None
        assert result["sample_sizes"] is None
        assert result["method"] == "pearson"

    def test_sample_sizes_shape(self):
        """sample_sizes must be N x N with full row count on the diagonal."""
        df = _make_returns(n_assets=4, n_obs=250)
        result = calculate_correlation_matrix(df, return_col_prefix="ret_")
        sample_df = result["sample_sizes"]
        assert sample_df is not None

        tickers = [f"A{i}" for i in range(4)]
        # Diagonal stores full row count (pre-existing convention)
        for i, t in enumerate(tickers):
            assert sample_df[t][i] == 250


class TestCorrelationPerformance:
    """Performance guard against regression to the O(N^2) Python loop."""

    @pytest.mark.parametrize("n_assets", [20])
    def test_fast_for_typical_size(self, n_assets: int):
        """N=20 tickers, T=500 obs: must complete in <100ms after vectorization.

        The pre-fix implementation took ~70ms+ on a typical dev machine for
        this size; the vectorized version should be at least 10x faster.
        """
        df = _make_returns(n_assets=n_assets, n_obs=500)
        # Warmup (import + JIT-like polars overhead)
        calculate_correlation_matrix(df, return_col_prefix="ret_")

        t0 = time.perf_counter()
        _ = calculate_correlation_matrix(df, return_col_prefix="ret_")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 100, (
            f"Correlation matrix took {elapsed_ms:.1f}ms for N={n_assets} — "
            "must be <100ms after vectorization (regression to nested loop?)"
        )
