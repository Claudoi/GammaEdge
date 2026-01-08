# portfolio/features/quant_transforms.py
"""
Transformaciones cuantitativas para datos OHLCV.

Este módulo proporciona funciones para convertir datos de precios en métricas
cuantitativas utilizadas en análisis financiero:
- Log-returns (retornos logarítmicos)
- Volumen relativo (volumen normalizado por su media móvil)
- Volatilidad intradía (estimadores Parkinson y Garman-Klass)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

VolatilityMethod = Literal["parkinson", "garman_klass"]


def compute_log_returns(df_ohlcv_long: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula log-returns a partir de precios de cierre ajustados.

    Log-return: r_t = ln(close_t / close_{t-1})

    Args:
        df_ohlcv_long: DataFrame con columnas [date, ticker, close, ...]

    Returns:
        DataFrame con columnas [date, ticker, log_ret]
        El primer valor por ticker será null (drop_first=True por defecto)
    """
    df = df_ohlcv_long.sort(["ticker", "date"])

    result = (
        df.with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ticker")).log().alias("log_ret")
        )
        .select(["date", "ticker", "log_ret"])
        .drop_nulls()
    )

    return result


def compute_relative_volume(df_ohlcv_long: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """
    Calcula el volumen relativo respecto a la media móvil.

    Relative Volume: rel_vol_t = volume_t / SMA(volume, lookback)

    Un valor > 1 indica volumen por encima del promedio,
    < 1 indica volumen por debajo del promedio.

    Args:
        df_ohlcv_long: DataFrame con columnas [date, ticker, volume, ...]
        lookback: Ventana para la media móvil (por defecto 20 días)

    Returns:
        DataFrame con columnas [date, ticker, rel_volume]
    """
    df = df_ohlcv_long.sort(["ticker", "date"])

    result = df.with_columns(
        (
            pl.col("volume") / pl.col("volume").rolling_mean(window_size=lookback).over("ticker")
        ).alias("rel_volume")
    ).select(["date", "ticker", "rel_volume"])

    return result


def compute_intraday_volatility(
    df_ohlcv_long: pl.DataFrame, method: VolatilityMethod = "parkinson"
) -> pl.DataFrame:
    """
    Calcula la volatilidad intradía usando estimadores de rango.

    Métodos disponibles:

    1. **Parkinson** (1980): Usa solo high-low
       σ² = (ln(H/L))² / (4 * ln(2))

    2. **Garman-Klass** (1980): Usa OHLC completo
       σ² = 0.5 * (ln(H/L))² - (2*ln(2)-1) * (ln(C/O))²

    Ambos estimadores son más eficientes que el tradicional close-to-close
    porque utilizan información intradía.

    Args:
        df_ohlcv_long: DataFrame con columnas [date, ticker, open, high, low, close]
        method: 'parkinson' o 'garman_klass'

    Returns:
        DataFrame con columnas [date, ticker, intraday_vol]
    """
    df = df_ohlcv_long.sort(["ticker", "date"])

    if method == "parkinson":
        # Parkinson estimator: σ = sqrt((ln(H/L))² / (4*ln(2)))
        result = df.with_columns(
            (((pl.col("high") / pl.col("low")).log().pow(2) / (4 * np.log(2))).sqrt()).alias(
                "intraday_vol"
            )
        )
    else:  # garman_klass
        # Garman-Klass estimator
        # σ² = 0.5 * (ln(H/L))² - (2*ln(2)-1) * (ln(C/O))²
        ln_hl_sq = (pl.col("high") / pl.col("low")).log().pow(2)
        ln_co_sq = (pl.col("close") / pl.col("open")).log().pow(2)
        gk_var = 0.5 * ln_hl_sq - (2 * np.log(2) - 1) * ln_co_sq

        result = df.with_columns(
            # Clip to avoid sqrt of negative (can happen with bad data)
            gk_var.clip(lower_bound=0).sqrt().alias("intraday_vol")
        )

    return result.select(["date", "ticker", "intraday_vol"])


def compute_all_quant_metrics(
    df_ohlcv_long: pl.DataFrame,
    *,
    volume_lookback: int = 20,
    volatility_method: VolatilityMethod = "parkinson",
) -> pl.DataFrame:
    """
    Calcula todas las métricas cuantitativas en una sola pasada.

    Combina:
    - Log-returns (de close ajustado)
    - Volumen relativo (vol / SMA(vol, lookback))
    - Volatilidad intradía (Parkinson o Garman-Klass)

    Args:
        df_ohlcv_long: DataFrame con columnas [date, ticker, open, high, low, close, volume]
        volume_lookback: Ventana para SMA del volumen (default 20)
        volatility_method: 'parkinson' o 'garman_klass'

    Returns:
        DataFrame con columnas [date, ticker, log_ret, rel_volume, intraday_vol]
    """
    log_rets = compute_log_returns(df_ohlcv_long)
    rel_vol = compute_relative_volume(df_ohlcv_long, lookback=volume_lookback)
    intra_vol = compute_intraday_volatility(df_ohlcv_long, method=volatility_method)

    # Join all metrics
    result = (
        log_rets.join(rel_vol, on=["date", "ticker"], how="left")
        .join(intra_vol, on=["date", "ticker"], how="left")
        .sort(["ticker", "date"])
    )

    return result
