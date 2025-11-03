# portfolio/attribution/brinson.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

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

    Parameters
    ----------
    df:
        DataFrame de entrada, en cualquiera de los formatos soportados
        por `coerce_brinson_timeseries_to_long` (global-only, long, wide).
    how:
        Agregación para `by_group`: "sum" (por defecto) o "mean".
    """
    ts = coerce_brinson_timeseries_to_long(df)

    agg_exprs = _build_agg_exprs(how)

    by_group = ts.group_by("group_id").agg(agg_exprs).sort("group_id")

    total = ts.select([pl.col(m).sum().alias(m) for m in BRINSON_METRICS])

    return BrinsonAttribution(timeseries=ts, by_group=by_group, total=total)
