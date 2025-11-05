# portfolio/attribution/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

Method = Literal["generic", "brinson", "euler"]


@dataclass
class AttributionResult:
    """
    Contenedor ligero para salidas de atribución.

    Por ahora solo guarda:
    - method: etiqueta del método ('generic', 'brinson', 'euler', ...)
    - contributions: DataFrame de Polars con contribuciones por fecha/asset
    - meta: diccionario opcional con metadatos (nombre de columnas, etc.)
    """

    method: Method
    contributions: pl.DataFrame
    meta: dict[str, Any] | None = None


def _ensure_lazy(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Normaliza a LazyFrame para poder encadenar joins y selects."""
    if isinstance(df, pl.LazyFrame):
        return df
    return df.lazy()


def compute_portfolio_contributions(
    *,
    weights: pl.DataFrame | pl.LazyFrame,
    returns: pl.DataFrame | pl.LazyFrame,
    date_col: str = "date",
    method: Method = "generic",
) -> AttributionResult:
    """
    Calcula contribuciones simples de cartera: contrib = peso * retorno.

    Parameters
    ----------
    weights:
        DataFrame/LazyFrame de Polars con la columna `date_col` y una columna
        por asset/bucket con los pesos de cartera.
    returns:
        DataFrame/LazyFrame de Polars con la columna `date_col` y una columna
        por asset/bucket con los retornos de cada fecha, en la misma malla
        temporal que `weights`.
    date_col:
        Nombre de la columna fecha para alinear ambos DataFrames.
    method:
        Etiqueta del método de atribución. Actualmente es solo informativa,
        pero deja el API preparado para enganchar 'brinson' / 'euler' después.

    Returns
    -------
    AttributionResult
        `contributions` tendrá el esquema:
        [date_col] + mismas columnas de assets que `weights`/`returns`,
        con valores igual a peso * retorno.
    """
    w_lazy = _ensure_lazy(weights)
    r_lazy = _ensure_lazy(returns)

    # Join interno por fecha; las columnas de retornos se sufijan con '_ret'
    joined = w_lazy.join(r_lazy, on=date_col, how="inner", suffix="_ret")

    # Columnas de assets: todas menos la fecha y las sufijadas con '_ret'
    asset_cols = [c for c in joined.schema if c != date_col and not c.endswith("_ret")]

    # Construimos expresiones: fecha + contribuciones por asset
    exprs: list[pl.Expr] = [pl.col(date_col)]
    for c in asset_cols:
        exprs.append((pl.col(c) * pl.col(f"{c}_ret")).alias(c))

    contributions = joined.select(exprs).collect()

    return AttributionResult(
        method=method,
        contributions=contributions,
        meta={"date_col": date_col, "asset_cols": asset_cols},
    )
