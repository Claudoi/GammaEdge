# portfolio/backtest/attribution_reporting.py

from __future__ import annotations

from typing import Literal, TypedDict, Union

import pandas as pd
import polars as pl

from portfolio.attribution.brinson import (
    BRINSON_METRICS,
    BrinsonAttribution,
    compute_brinson_attribution,
)

# Alias útil para documentación / tipos de alto nivel
BrinsonInputFrame = Union[pl.DataFrame, pd.DataFrame]


class BrinsonReport(TypedDict):
    """
    Estructura de alto nivel para reporting de Brinson:

    - timeseries: dataframe largo (date, group_id, métricas).
    - by_group: agregado por group_id.
    - total: fila única con el total del periodo.
    """

    timeseries: pl.DataFrame
    by_group: pl.DataFrame
    total: pl.DataFrame


def _to_polars_timeseries(df: object) -> pl.DataFrame:
    """
    Normaliza un dataframe de entrada (pandas o polars) a polars.DataFrame.

    Reglas:
    - Si es polars.DataFrame, se devuelve tal cual.
    - Si es pandas.DataFrame:
        - Si el índice es DatetimeIndex y no hay columna 'date',
          se resetea el índice y se renombra a 'date'.
        - En caso contrario, se usa tal cual y se convierte a polars.
    - Para cualquier otro tipo se lanza TypeError en tiempo de ejecución.
    """
    if isinstance(df, pl.DataFrame):
        return df

    if isinstance(df, pd.DataFrame):
        pdf = df.copy()

        if "date" not in pdf.columns and isinstance(pdf.index, pd.DatetimeIndex):
            pdf = pdf.reset_index().rename(columns={"index": "date"})

        return pl.from_pandas(pdf)

    msg = f"Unsupported input type {type(df)!r} for Brinson attribution timeseries."
    raise TypeError(msg)


def build_brinson_attribution_report(
    timeseries: BrinsonInputFrame,
    how: Literal["sum", "mean"] = "sum",
) -> BrinsonReport:
    """
    Helper de alto nivel para reporting de atribución tipo Brinson.

    Acepta un dataframe (pandas o polars) en cualquiera de los formatos
    soportados por `coerce_brinson_timeseries_to_long`:

      - Global-only (alloc/select/interact/total sin group_id).
      - Long (con 'group_id' o 'group').
      - Wide (columnas alloc_0, select_0, ...).

    Devuelve un dict con:

      - 'timeseries': dataframe largo normalizado.
      - 'by_group': métricas agregadas por group_id.
      - 'total': métricas agregadas en todo el periodo.

    Parameters
    ----------
    timeseries:
        DataFrame de entrada (pandas o polars).
    how:
        Tipo de agregación para by_group: "sum" (por defecto) o "mean".
    """
    ts_pl = _to_polars_timeseries(timeseries)

    result: BrinsonAttribution = compute_brinson_attribution(ts_pl, how=how)

    # Pequeña sanidad: aseguramos que las métricas clave estén presentes.
    for m in BRINSON_METRICS:
        if m not in result.timeseries.columns:
            msg = f"Metric '{m}' not found in Brinson timeseries."
            raise ValueError(msg)

    return BrinsonReport(
        timeseries=result.timeseries,
        by_group=result.by_group,
        total=result.total,
    )
