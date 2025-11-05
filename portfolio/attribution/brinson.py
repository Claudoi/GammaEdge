# portfolio/attribution/brinson.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from portfolio.attribution.engine import compute_portfolio_contributions
from portfolio.backtest.brinson_utils import coerce_brinson_timeseries_to_long

BRINSON_METRICS = ["alloc", "select", "interact", "total"]


@dataclass
class BrinsonAttribution:
    """
    Contenedor sencillo para resultados de Brinson.

    - timeseries: dataframe largo (date, group_id, métricas).
    - by_group: agregado por group_id.
    - total: fila única con el total del periodo.
    """

    timeseries: pl.DataFrame
    by_group: pl.DataFrame
    total: pl.DataFrame


def _build_agg_exprs(how: Literal["sum", "mean"]) -> list[pl.Expr]:
    if how == "sum":
        return [pl.col(m).sum().alias(m) for m in BRINSON_METRICS]
    if how == "mean":
        return [pl.col(m).mean().alias(m) for m in BRINSON_METRICS]
    msg = f"Unsupported aggregation '{how}'. Use 'sum' or 'mean'."
    raise ValueError(msg)


def compute_brinson_attribution(
    df: pl.DataFrame,
    how: Literal["sum", "mean"] = "sum",
) -> BrinsonAttribution:
    """
    Normaliza cualquier timeseries "tipo Brinson" y devuelve:

    - timeseries: formato largo estándar vía `coerce_brinson_timeseries_to_long`.
    - by_group: agregado por group_id (sum o mean).
    - total: métricas agregadas sobre todo el periodo.
    """
    ts = coerce_brinson_timeseries_to_long(df)

    agg_exprs = _build_agg_exprs(how)

    by_group = ts.group_by("group_id").agg(agg_exprs).sort("group_id")
    total = ts.select([pl.col(m).sum().alias(m) for m in BRINSON_METRICS])

    return BrinsonAttribution(timeseries=ts, by_group=by_group, total=total)


def _extract_contrib_frame(result: object) -> pl.DataFrame:
    """
    Normaliza la salida de `compute_portfolio_contributions` a un DataFrame.

    Soporta:
      - Devolver directamente un `pl.DataFrame`
      - Devolver un contenedor tipo AttributionResult con atributos:
        `contributions`, `frame`, `df` o `data` que sean DataFrame.
    """
    if isinstance(result, pl.DataFrame):
        return result

    for attr in ("contributions", "frame", "df", "data"):
        if hasattr(result, attr):
            candidate = getattr(result, attr)
            if isinstance(candidate, pl.DataFrame):
                return candidate

    msg = (
        "compute_portfolio_contributions(...) must return either a Polars "
        "DataFrame, or an object with a 'contributions'/'frame'/'df'/'data' "
        "attribute holding a Polars DataFrame."
    )
    raise TypeError(msg)


def _call_portfolio_contributions_from_long(
    df: pl.DataFrame,
    weights_col: str,
    returns_col: str,
    date_col: str,
) -> pl.DataFrame:
    """
    A partir de un df *largo* (date, asset, w, r) construye los dataframes
    anchos que espera `compute_portfolio_contributions` y devuelve un df
    largo con contribuciones por (date, asset).

    Salida: columnas ['date', 'asset', 'contribution'].
    """
    if "asset" not in df.columns:
        msg = "Expected a long dataframe with an 'asset' column."
        raise ValueError(msg)

    # 1) Pivot a formato ancho para pesos y retornos
    weights_wide = (
        df.select([date_col, "asset", weights_col])
        .pivot(index=date_col, columns="asset", values=weights_col)
        .sort(date_col)
    )

    returns_wide = (
        df.select([date_col, "asset", returns_col])
        .pivot(index=date_col, columns="asset", values=returns_col)
        .sort(date_col)
    )

    # 2) Llamada al engine en formato ancho
    result = compute_portfolio_contributions(
        weights=weights_wide,
        returns=returns_wide,
        date_col=date_col,
    )
    contrib_wide = _extract_contrib_frame(result)

    # 3) Volver a largo: (date, asset, contribution)
    contrib_long = contrib_wide.melt(
        id_vars=date_col,
        variable_name="asset",
        value_name="contribution",
    )

    return contrib_long


def run_brinson_engine(
    df: pl.DataFrame,
    weights_col: str = "w",
    returns_col: str = "r",
    date_col: str = "date",
) -> pl.DataFrame:
    """
    Integra un dataframe largo estilo Brinson con el engine de atribución.

    Entrada esperada (mínimo):
        ['date', 'asset', weights_col, returns_col]

    Devuelve el dataframe original enriquecido con la columna 'contribution'
    por (date, asset), calculada vía `compute_portfolio_contributions`.
    """
    contributions = _call_portfolio_contributions_from_long(
        df=df,
        weights_col=weights_col,
        returns_col=returns_col,
        date_col=date_col,
    )

    # Unimos por (date, asset) para no duplicar columnas.
    return df.join(contributions, on=[date_col, "asset"], how="left")
