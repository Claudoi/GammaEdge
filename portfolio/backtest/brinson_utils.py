# portfolio/backtest/brinson_utils.py
"""
Backtest-side Brinson helpers.

``ensure_datetime`` and ``coerce_brinson_timeseries_to_long`` were moved to
``portfolio.attribution.brinson_utils`` to break the import cycle between
``portfolio.backtest`` and ``portfolio.attribution``. They are re-exported
here for backward compatibility with existing tests and pages.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable

import numpy as np
import pandas as pd
import polars as pl

# Backward-compatible re-exports — their canonical home is the attribution package.
from portfolio.attribution.brinson_utils import (
    coerce_brinson_timeseries_to_long,
    ensure_datetime,
)

__all__ = [
    "ensure_datetime",
    "coerce_brinson_timeseries_to_long",
    "to_pandas_2d_returns",
    "align_columns",
]


# ────────────────────────────────────────────────────────────────────
# Helpers: coerción de retornos 2D y alineación de columnas
# ────────────────────────────────────────────────────────────────────
def to_pandas_2d_returns(obj: object) -> pd.DataFrame:
    """
    Convierte diferentes entradas a un DataFrame de pandas 2D con floats (retornos).
    Acepta:
      - pandas.DataFrame (con o sin 'date' como columna)
      - polars.DataFrame (con o sin 'date')
      - np.ndarray / secuencias 2D
    - Elimina/ignora 'date' como columna y la usa como índice si es parseable.
    - Reemplaza NaN/±inf por 0.0 para evitar explosiones.
    """
    # Pandas
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if "date" in df.columns:
            if not isinstance(df.index, pd.DatetimeIndex):
                with contextlib.suppress(Exception):
                    df.index = pd.to_datetime(df["date"])
            df = df.drop(columns=["date"])
        df = df.select_dtypes(include=[np.number]).astype(float)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df

    # Polars
    if isinstance(obj, pl.DataFrame):
        df_pl = obj
        if "date" in df_pl.columns:
            pdf = df_pl.to_pandas()
            with contextlib.suppress(Exception):
                pdf.index = pd.to_datetime(pdf["date"])
            if "date" in pdf.columns:
                pdf = pdf.drop(columns=["date"])
        else:
            pdf = df_pl.to_pandas()
        pdf = pdf.select_dtypes(include=[np.number]).astype(float)
        pdf = pdf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pdf

    # Numpy / secuencias
    try:
        arr = np.asarray(obj, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = [f"c{i}" for i in range(arr.shape[1])]
        pdf = pd.DataFrame(arr, columns=cols)
        pdf = pdf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pdf
    except Exception as _:
        raise AttributeError("Unsupported input for to_pandas_2d_returns(...)")  # noqa: B904


def align_columns(cols_a: Iterable[str], cols_b: Iterable[str]) -> list[str]:
    """
    Devuelve la intersección de columnas preservando el orden de `cols_a`.
    Si la intersección es vacía, intenta el orden de `cols_b`.
    """
    set_b = set(cols_b)
    out = [c for c in cols_a if c in set_b]
    if not out:
        set_a = set(cols_a)
        out = [c for c in cols_b if c in set_a]
    return out
