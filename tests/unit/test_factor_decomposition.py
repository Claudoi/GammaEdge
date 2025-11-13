# tests/unit/test_factor_decomposition.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.attribution.factor_decomposition import (
    euler_factor_contributions,
    factor_attribution_matrix,
)


def _toy_inputs():
    assets = ["A", "B", "C"]
    factors = ["MKT", "SIZE"]

    w = pd.Series([0.5, 0.3, 0.2], index=assets, name="w")

    B = pd.DataFrame(
        [[1.0, 0.2], [0.8, -0.1], [1.2, 0.4]],
        index=assets,
        columns=factors,
    )

    Sigma_f = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=factors,
        columns=factors,
    )
    return w, B, Sigma_f


def test_euler_factor_contributions_shapes_and_sum():
    w, B, Sigma_f = _toy_inputs()
    out = euler_factor_contributions(w, B, Sigma_f)

    assert "sigma_p" in out and "factor_rc" in out and "asset_factor_rc" in out
    sigma_p = out["sigma_p"]
    factor_rc = out["factor_rc"]
    asset_factor_rc = out["asset_factor_rc"]

    assert isinstance(sigma_p, float) and sigma_p >= 0
    assert list(factor_rc.index) == list(B.columns)
    assert list(asset_factor_rc.index) == list(B.index)
    assert list(asset_factor_rc.columns) == list(B.columns)

    assert np.isclose(factor_rc.sum(), sigma_p, atol=1e-10)
    assert np.isclose(asset_factor_rc.values.sum(), sigma_p, atol=1e-10)


def test_factor_attribution_matrix_long_format_and_filtering():
    w, B, Sigma_f = _toy_inputs()
    df_long = factor_attribution_matrix(w, B, Sigma_f, top_factors=1)

    assert set(df_long.columns) == {"asset", "factor", "rc", "abs_rc"}
    assert df_long["factor"].nunique() == 1
    assert df_long["abs_rc"].is_monotonic_decreasing


def test_zero_variance_returns_zero_everywhere():
    w, B, _ = _toy_inputs()
    Sigma_zero = pd.DataFrame(
        np.zeros((B.shape[1], B.shape[1])), index=B.columns, columns=B.columns
    )
    out = euler_factor_contributions(w, B, Sigma_zero)
    assert out["sigma_p"] == 0.0
    assert float(out["factor_rc"].sum()) == 0.0
    assert float(out["asset_factor_rc"].values.sum()) == 0.0


def test_accepts_numpy_inputs_and_recovers_shapes():
    w, B, Sigma_f = _toy_inputs()
    out = euler_factor_contributions(w.to_numpy(), B.to_numpy(), Sigma_f.to_numpy())

    sigma_p = out["sigma_p"]
    factor_rc = out["factor_rc"]
    asset_factor_rc = out["asset_factor_rc"]

    assert isinstance(sigma_p, float)
    assert factor_rc.shape[0] == B.shape[1]
    assert asset_factor_rc.shape == (B.shape[0], B.shape[1])

    assert np.isclose(factor_rc.sum(), sigma_p, atol=1e-10)
    assert np.isclose(asset_factor_rc.sum().sum(), sigma_p, atol=1e-10)


def test_top_factors_none_returns_all_factors_in_long_format():
    w, B, Sigma_f = _toy_inputs()
    df_long_all = factor_attribution_matrix(w, B, Sigma_f, top_factors=None)
    assert set(df_long_all["factor"].unique()) == set(B.columns)
    assert set(df_long_all["asset"].unique()) == set(B.index)
    assert (df_long_all["abs_rc"] >= 0).all()


def test_top_factors_bigger_than_available_caps_to_n_factors():
    w, B, Sigma_f = _toy_inputs()
    df_long = factor_attribution_matrix(w, B, Sigma_f, top_factors=10)
    assert set(df_long["factor"].unique()) == set(B.columns)


def test_mismatched_index_raises():
    w, B, Sigma_f = _toy_inputs()
    w_bad = pd.Series(w.values, index=[f"X{i}" for i in range(len(w))], name="w")
    with pytest.raises(ValueError):
        euler_factor_contributions(w_bad, B, Sigma_f)
