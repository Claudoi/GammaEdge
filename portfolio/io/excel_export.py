# portfolio/io/excel_export.py
"""
Módulo para exportación de métricas cuantitativas a Excel.

Proporciona funciones para descargar datos OHLCV, aplicar transformaciones
cuantitativas y exportar a formato Excel (.xlsx).
"""

from __future__ import annotations

import io
import logging
from datetime import date
from typing import Literal

import polars as pl

from portfolio.features.quant_transforms import (
    VolatilityMethod,
    compute_all_quant_metrics,
)
from portfolio.io.data_loader import get_ohlcv_long

logger = logging.getLogger(__name__)


def export_quant_metrics_to_excel(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    *,
    volume_lookback: int = 20,
    volatility_method: VolatilityMethod = "parkinson",
    output_format: Literal["bytes", "path"] = "bytes",
    output_path: str | None = None,
) -> bytes | str:
    """
    Descarga datos OHLCV, aplica transformaciones cuantitativas y exporta a Excel.

    Las métricas exportadas son:
    - **log_return**: Retorno logarítmico del cierre ajustado
    - **rel_volume**: Volumen relativo (vol / SMA(vol, lookback))
    - **intraday_vol**: Volatilidad intradía (estimador Parkinson o Garman-Klass)

    El archivo Excel tendrá una hoja por ticker con datos ordenados por fecha.

    Args:
        tickers: Lista de símbolos de acciones
        start: Fecha de inicio (YYYY-MM-DD). Default: 2010-01-01
        end: Fecha de fin (YYYY-MM-DD). Default: hoy
        volume_lookback: Ventana para SMA del volumen (default: 20 días)
        volatility_method: 'parkinson' o 'garman_klass'
        output_format: 'bytes' para devolver bytes, 'path' para guardar a archivo
        output_path: Ruta del archivo (requerido si output_format='path')

    Returns:
        bytes del archivo Excel si output_format='bytes'
        str con la ruta del archivo si output_format='path'

    Example:
        >>> excel_bytes = export_quant_metrics_to_excel(
        ...     tickers=["AAPL", "MSFT"],
        ...     start="2015-01-01",
        ...     end="2024-01-01",
        ... )
        >>> # Usar en Streamlit:
        >>> st.download_button("Download", excel_bytes, "quant_metrics.xlsx")
    """
    if end is None:
        end = str(date.today())

    logger.info(
        "Exporting quant metrics for %d tickers: %s to %s",
        len(tickers),
        start,
        end,
    )

    # 1. Descargar OHLCV
    df_ohlcv = get_ohlcv_long(tickers, start=start, end=end)

    if df_ohlcv.height == 0:
        raise ValueError(f"No data available for tickers: {tickers}")

    # 2. Calcular métricas cuantitativas
    df_metrics = compute_all_quant_metrics(
        df_ohlcv,
        volume_lookback=volume_lookback,
        volatility_method=volatility_method,
    )

    # 3. Exportar a Excel (una hoja por ticker)
    buffer = io.BytesIO()

    with pl.Config(set_fmt_float="full"):
        # Convertir a pandas para usar ExcelWriter con openpyxl
        unique_tickers = df_metrics.select(pl.col("ticker").unique()).to_series().to_list()

        # Usar pandas ExcelWriter con openpyxl
        import pandas as pd

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for ticker in sorted(unique_tickers):
                ticker_df = (
                    df_metrics.filter(pl.col("ticker") == ticker)
                    .sort("date")
                    .select(["date", "log_ret", "rel_volume", "intraday_vol"])
                    .to_pandas()
                )

                # Formatear fecha como date (sin hora)
                ticker_df["date"] = pd.to_datetime(ticker_df["date"]).dt.date

                # Nombre de hoja: max 31 caracteres (límite de Excel)
                sheet_name = ticker[:31]
                ticker_df.to_excel(writer, sheet_name=sheet_name, index=False)

                logger.info("Wrote %d rows for ticker %s", len(ticker_df), ticker)

    buffer.seek(0)
    excel_bytes = buffer.getvalue()

    if output_format == "path":
        if output_path is None:
            raise ValueError("output_path is required when output_format='path'")
        with open(output_path, "wb") as f:
            f.write(excel_bytes)
        logger.info("Excel file saved to: %s", output_path)
        return output_path

    return excel_bytes


def get_quant_metrics_summary(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
) -> pl.DataFrame:
    """
    Obtiene un resumen de las métricas cuantitativas para cada ticker.

    Útil para preview antes de la exportación completa.

    Returns:
        DataFrame con estadísticas resumidas por ticker:
        - n_obs: número de observaciones
        - first_date, last_date: rango de fechas
        - mean_log_ret, std_log_ret: media y desv. std. de log-returns
        - mean_rel_volume: media del volumen relativo
        - mean_intraday_vol: media de la volatilidad intradía
    """
    if end is None:
        end = str(date.today())

    df_ohlcv = get_ohlcv_long(tickers, start=start, end=end)

    if df_ohlcv.height == 0:
        return pl.DataFrame()

    df_metrics = compute_all_quant_metrics(df_ohlcv)

    summary = (
        df_metrics.group_by("ticker")
        .agg(
            [
                pl.len().alias("n_obs"),
                pl.col("date").min().alias("first_date"),
                pl.col("date").max().alias("last_date"),
                pl.col("log_ret").mean().alias("mean_log_ret"),
                pl.col("log_ret").std().alias("std_log_ret"),
                pl.col("rel_volume").mean().alias("mean_rel_volume"),
                pl.col("intraday_vol").mean().alias("mean_intraday_vol"),
            ]
        )
        .sort("ticker")
    )

    return summary
