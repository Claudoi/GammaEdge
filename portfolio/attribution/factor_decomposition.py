# portfolio/attribution/factor_decomposition.py
from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd

from portfolio.core.utils import ensure_psd


class FactorDecompOut(TypedDict):
    sigma_p: float
    factor_rc: pd.Series
    asset_factor_rc: pd.DataFrame


def _to_series(
    x: pd.Series | list[float] | tuple[float, ...] | np.ndarray,
    name: str,
) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x.copy()
        if s.name is None:
            s.name = name
        return s
    if isinstance(x, (list, tuple, np.ndarray)):
        return pd.Series(x, name=name)
    raise TypeError(f"Unsupported type for {name}. Expected pandas.Series, list, tuple or ndarray.")


def _to_dataframe(
    x: pd.DataFrame | np.ndarray,
    *,
    index: list[str] | pd.Index | None = None,
    columns: list[str] | pd.Index | None = None,
    name: str = "df",
) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, np.ndarray):
        return pd.DataFrame(x, index=index, columns=columns)
    raise TypeError(f"Unsupported type for {name}. Expected pandas.DataFrame or numpy.ndarray.")


def _validate_shapes(
    w: pd.Series,
    B: pd.DataFrame,
    Sigma_f: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    if not isinstance(w, pd.Series):
        raise TypeError("w must be a pandas Series with asset index.")
    if not isinstance(B, pd.DataFrame):
        raise TypeError("B must be a pandas DataFrame with assets as index and factors as columns.")
    if not isinstance(Sigma_f, pd.DataFrame):
        raise TypeError("Sigma_f must be a pandas DataFrame with factors as index and columns.")

    if not w.index.equals(B.index):
        raise ValueError("w.index must match B.index (assets).")
    if list(Sigma_f.index) != list(Sigma_f.columns):
        raise ValueError("Sigma_f must be square with matching factor index and columns.")
    if not set(B.columns).issubset(set(Sigma_f.columns)):
        raise ValueError("All B columns (factors) must be present in Sigma_f.")

    # Align Sigma_f to factor order in B
    Sigma_f = Sigma_f.reindex(index=B.columns, columns=B.columns)
    assert Sigma_f is not None

    return w, B, Sigma_f


def euler_factor_contributions(
    w: pd.Series | np.ndarray,
    B: pd.DataFrame | np.ndarray,
    Sigma_f: pd.DataFrame | np.ndarray,
    *,
    lambda_reg: float = 1e-8,
) -> FactorDecompOut:
    """
    Euler factor risk decomposition under a linear factor model:

      R = B F
      Var(R) = B Sigma_f B^T
      sigma_p = sqrt(w^T B Sigma_f B^T w)

    Returns
    -------
    sigma_p : float
        Portfolio risk (standard deviation).
    factor_rc : pd.Series
        Risk contributions by factor (sum ≈ sigma_p).
    asset_factor_rc : pd.DataFrame
        Asset × factor risk contributions (rows=assets, cols=factors).
    """
    # Coerce to pandas
    w_s: pd.Series = _to_series(w, name="w")
    B_df: pd.DataFrame = _to_dataframe(B, index=w_s.index, name="B")
    if not isinstance(Sigma_f, pd.DataFrame):
        Sigma_f_df: pd.DataFrame = _to_dataframe(
            Sigma_f,
            index=B_df.columns,
            columns=B_df.columns,
            name="Sigma_f",
        )
    else:
        Sigma_f_df = Sigma_f.copy()

    # Validate shapes and align factor order
    w_s, B_df, Sigma_f_df = _validate_shapes(w_s, B_df, Sigma_f_df)

    # ------------------------------------------------------------------
    # 1) Comprobación de varianza "real" SIN tocar Sigma (ni PSD ni ridge)
    # ------------------------------------------------------------------
    Sigma_input = Sigma_f_df.to_numpy(dtype=float)

    # Portfolio factor exposure g = B^T w
    g_ser: pd.Series = B_df.T @ w_s
    g_np = g_ser.to_numpy(dtype=float)

    var_raw = float(g_np.T @ (Sigma_input @ g_np))

    # Si la varianza verdadera es cero (o numéricamente ~0) → todo cero
    if var_raw <= 0.0 or np.isclose(var_raw, 0.0, atol=1e-12):
        factors = B_df.columns
        assets = B_df.index
        return {
            "sigma_p": 0.0,
            "factor_rc": pd.Series(0.0, index=factors, name="factor_rc"),
            "asset_factor_rc": pd.DataFrame(0.0, index=assets, columns=factors),
        }

    # ------------------------------------------------------------------
    # 2) Ahora sí: limpieza PSD + ridge para estabilidad numérica
    # ------------------------------------------------------------------
    Sigma_base = ensure_psd(Sigma_input)
    Sigma_np = Sigma_base
    if lambda_reg > 0.0:
        n_factors = Sigma_base.shape[0]
        Sigma_np = Sigma_base + float(lambda_reg) * np.eye(n_factors)

    Sigma_f_df = pd.DataFrame(Sigma_np, index=Sigma_f_df.index, columns=Sigma_f_df.columns)

    # Portfolio variance with stabilized covariance
    var_p = float(g_np.T @ (Sigma_np @ g_np))
    if var_p < 0.0:
        var_p = 0.0
    sigma_p = float(np.sqrt(var_p))

    if sigma_p == 0.0:
        factors = B_df.columns
        assets = B_df.index
        return {
            "sigma_p": 0.0,
            "factor_rc": pd.Series(0.0, index=factors, name="factor_rc"),
            "asset_factor_rc": pd.DataFrame(0.0, index=assets, columns=factors),
        }

    # Marginal risk per factor: d sigma / d g = (Sigma g) / sigma
    mrc_np = (Sigma_np @ g_np) / sigma_p
    mrc_factor = pd.Series(mrc_np, index=Sigma_f_df.index, name="mrc_factor")

    # Euler by factor: RC_f = g_f * MRC_f (sum_f RC_f = sigma_p)
    factor_rc = pd.Series(g_np * mrc_np, index=Sigma_f_df.index, name="factor_rc")

    # Per-asset per-factor: RC_{i,f} = w_i * B_if * MRC_f
    asset_factor_rc = B_df.mul(w_s, axis=0).mul(mrc_factor, axis=1)

    return {
        "sigma_p": sigma_p,
        "factor_rc": factor_rc,
        "asset_factor_rc": asset_factor_rc,
    }


def factor_attribution_matrix(
    w: pd.Series | np.ndarray,
    B: pd.DataFrame | np.ndarray,
    Sigma_f: pd.DataFrame | np.ndarray,
    top_factors: int | None = None,
    *,
    lambda_reg: float = 1e-8,
) -> pd.DataFrame:
    """
    Long-format per-asset per-factor risk contributions with |rc| for sorting.

    Columns: ["asset", "factor", "rc", "abs_rc"].
    """
    out = euler_factor_contributions(w, B, Sigma_f, lambda_reg=lambda_reg)
    A: pd.DataFrame = out["asset_factor_rc"]

    stacked = A.stack()  # MultiIndex (asset, factor)
    stacked.name = "rc"
    df = stacked.reset_index()
    df.columns = ["asset", "factor", "rc"]

    if top_factors is not None and top_factors > 0:
        order = out["factor_rc"].abs().sort_values(ascending=False).head(top_factors).index
        df = df[df["factor"].isin(order)].copy()

    df["abs_rc"] = df["rc"].abs()
    df = df.sort_values(["factor", "abs_rc"], ascending=[True, False]).reset_index(drop=True)
    return df


__all__ = [
    "euler_factor_contributions",
    "factor_attribution_matrix",
    "FactorDecompOut",
]
