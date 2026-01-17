# Advanced Quantitative Features
# ==============================
"""
Features cuantitativos avanzados para el Feature Store.

Este módulo extiende las transformaciones básicas con métricas adicionales
usadas en estrategias cuantitativas profesionales:

- Multi-horizon returns
- Momentum y mean reversion
- Volatility avanzada (ATR, realized vol, vol-of-vol)
- Liquidez (Amihud, dollar volume)
- Higher moments (skew, kurtosis)
- Cross-sectional ranks
"""

from __future__ import annotations

import numpy as np
import polars as pl

# =============================================================================
# Returns - Multiple Horizons
# =============================================================================


def compute_multi_horizon_returns(
    df: pl.DataFrame,
    horizons: list[int] | None = None,
) -> pl.DataFrame:
    """
    Calcula retornos a múltiples horizontes.

    Args:
        df: DataFrame con columnas [date, ticker, close]
        horizons: Lista de horizontes en días (default: [1, 5, 20, 60])

    Returns:
        DataFrame con columnas adicionales ret_{n}d y log_ret_{n}d
    """
    if horizons is None:
        horizons = [1, 5, 20, 60]

    df = df.sort(["ticker", "date"])

    expressions = []
    for n in horizons:
        # Simple returns
        expressions.append(
            (pl.col("close") / pl.col("close").shift(n).over("ticker") - 1).alias(f"ret_{n}d")
        )
        # Log returns
        expressions.append(
            (pl.col("close") / pl.col("close").shift(n).over("ticker")).log().alias(f"log_ret_{n}d")
        )

    return df.with_columns(expressions)


# =============================================================================
# Momentum and Mean Reversion
# =============================================================================


def compute_momentum_12_1(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula momentum clásico: retorno 12 meses menos retorno 1 mes.

    Este es el factor de momentum más estudiado en la literatura académica.
    Evita el "short-term reversal" restando el último mes.
    """
    df = df.sort(["ticker", "date"])

    return df.with_columns(
        [
            (
                (pl.col("close") / pl.col("close").shift(252).over("ticker") - 1)
                - (pl.col("close") / pl.col("close").shift(21).over("ticker") - 1)
            ).alias("momentum_12_1")
        ]
    )


def compute_momentum_zscore(
    df: pl.DataFrame,
    return_col: str = "ret_1d",
    window: int = 20,
) -> pl.DataFrame:
    """
    Calcula Z-score de retornos para detectar mean reversion.

    Z-score alto = precio subió mucho vs. su volatilidad reciente
    Z-score bajo = precio cayó mucho vs. su volatilidad reciente

    Útil para estrategias de mean reversion.
    """
    df = df.sort(["ticker", "date"])

    if return_col not in df.columns:
        # Calcular retorno simple si no existe
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias(return_col)
        )

    return df.with_columns(
        [
            (
                (pl.col(return_col) - pl.col(return_col).rolling_mean(window).over("ticker"))
                / pl.col(return_col).rolling_std(window).over("ticker")
            ).alias(f"momentum_zscore_{window}d")
        ]
    )


# =============================================================================
# Volatility - Advanced Estimators
# =============================================================================


def compute_realized_volatility(
    df: pl.DataFrame,
    windows: list[int] | None = None,
) -> pl.DataFrame:
    """
    Calcula volatilidad realizada (rolling std de log-returns).

    Args:
        df: DataFrame con columna close
        windows: Lista de ventanas (default: [5, 20, 60])
    """
    if windows is None:
        windows = [5, 20, 60]

    df = df.sort(["ticker", "date"])

    # Calcular log-return si no existe
    if "log_ret_1d" not in df.columns:
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ticker")).log().alias("log_ret_1d")
        )

    expressions = []
    for w in windows:
        # Volatilidad anualizada (asumiendo 252 días de trading)
        expressions.append(
            (pl.col("log_ret_1d").rolling_std(w).over("ticker") * np.sqrt(252)).alias(
                f"realized_vol_{w}d"
            )
        )

    return df.with_columns(expressions)


def compute_atr(
    df: pl.DataFrame,
    window: int = 14,
) -> pl.DataFrame:
    """
    Calcula Average True Range (ATR).

    True Range = max(H-L, |H-Prev_C|, |L-Prev_C|)
    ATR = SMA(True Range, window)

    Útil para sizing de posiciones y stops dinámicos.
    """
    df = df.sort(["ticker", "date"])

    # Primero calcular prev_close y True Range
    df = df.with_columns([pl.col("close").shift(1).over("ticker").alias("_prev_close")])

    # True Range components
    df = df.with_columns(
        [
            (pl.col("high") - pl.col("low")).alias("_tr1"),
            (pl.col("high") - pl.col("_prev_close")).abs().alias("_tr2"),
            (pl.col("low") - pl.col("_prev_close")).abs().alias("_tr3"),
        ]
    )

    # True Range = max de los tres componentes
    df = df.with_columns([pl.max_horizontal("_tr1", "_tr2", "_tr3").alias("_true_range")])

    # ATR = rolling mean de True Range
    df = df.with_columns(
        [pl.col("_true_range").rolling_mean(window).over("ticker").alias(f"atr_{window}")]
    )

    # Limpiar columnas temporales
    return df.drop(["_prev_close", "_tr1", "_tr2", "_tr3", "_true_range"])


def compute_volatility_of_volatility(
    df: pl.DataFrame,
    vol_window: int = 20,
    vov_window: int = 60,
) -> pl.DataFrame:
    """
    Calcula volatilidad de la volatilidad (vol-of-vol).

    Mide la estabilidad/inestabilidad de la volatilidad.
    Alta vol-of-vol = régimen cambiante, difícil de modelar.
    """
    df = df.sort(["ticker", "date"])

    # Primero computar volatilidad rolling
    vol_col = f"realized_vol_{vol_window}d"
    if vol_col not in df.columns:
        df = compute_realized_volatility(df, windows=[vol_window])

    return df.with_columns(
        [pl.col(vol_col).rolling_std(vov_window).over("ticker").alias(f"vol_of_vol_{vov_window}d")]
    )


# =============================================================================
# Liquidity Metrics
# =============================================================================


def compute_amihud_illiquidity(
    df: pl.DataFrame,
    window: int = 20,
) -> pl.DataFrame:
    """
    Calcula ratio de iliquidez de Amihud (2002).

    ILLIQ = |return| / dollar_volume

    Mayor valor = menos líquido (más impacto de mercado).
    Esta es una proxy de market impact sin necesitar datos L2.

    Reference:
        Amihud, Y. (2002). "Illiquidity and stock returns: cross-section
        and time-series effects." Journal of Financial Markets 5(1):31-56.
    """
    df = df.sort(["ticker", "date"])

    # Dollar volume
    df = df.with_columns([(pl.col("close") * pl.col("volume")).alias("dollar_volume")])

    # Retorno si no existe
    if "ret_1d" not in df.columns:
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias("ret_1d")
        )

    # Amihud ratio (evitar división por cero)
    return df.with_columns(
        [
            (pl.col("ret_1d").abs() / pl.col("dollar_volume").clip(lower_bound=1e-10))
            .rolling_mean(window)
            .over("ticker")
            .alias(f"amihud_illiq_{window}d")
        ]
    )


def compute_relative_volume(
    df: pl.DataFrame,
    window: int = 20,
) -> pl.DataFrame:
    """
    Calcula volumen relativo vs. promedio.

    rel_volume > 1 = volumen por encima del normal
    rel_volume < 1 = volumen por debajo del normal
    """
    df = df.sort(["ticker", "date"])

    return df.with_columns(
        [
            (pl.col("volume") / pl.col("volume").rolling_mean(window).over("ticker")).alias(
                f"rel_volume_{window}d"
            )
        ]
    )


# =============================================================================
# Higher Moments
# =============================================================================


def compute_rolling_skewness(
    df: pl.DataFrame,
    window: int = 20,
) -> pl.DataFrame:
    """
    Calcula skewness rolling de retornos.

    Skewness positivo = cola derecha más larga (posibles gains grandes)
    Skewness negativo = cola izquierda más larga (posibles crashes)
    """
    df = df.sort(["ticker", "date"])

    if "ret_1d" not in df.columns:
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias("ret_1d")
        )

    return df.with_columns(
        [pl.col("ret_1d").rolling_skew(window).over("ticker").alias(f"skew_{window}d")]
    )


def compute_rolling_kurtosis(
    df: pl.DataFrame,
    window: int = 20,
) -> pl.DataFrame:
    """
    Calcula kurtosis rolling de retornos.

    Kurtosis > 3 = colas más gruesas que normal (más eventos extremos)
    Kurtosis < 3 = colas más delgadas que normal (menos extremos)
    """
    df = df.sort(["ticker", "date"])

    if "ret_1d" not in df.columns:
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias("ret_1d")
        )

    # Polars no tiene rolling_kurt nativo, calculamos manualmente
    # Kurtosis = E[(X-μ)^4] / σ^4
    return df.with_columns(
        [
            (
                pl.col("ret_1d").map_batches(
                    lambda s: _rolling_kurtosis_series(s, window),
                    return_dtype=pl.Float64,
                )
            ).alias(f"kurtosis_{window}d")
        ]
    )


def _rolling_kurtosis_series(series: pl.Series, window: int) -> pl.Series:
    """Helper para calcular kurtosis rolling."""
    import numpy as np
    from scipy.stats import kurtosis

    arr = series.to_numpy()
    result = np.full(len(arr), np.nan)

    for i in range(window - 1, len(arr)):
        window_data = arr[i - window + 1 : i + 1]
        if np.all(np.isfinite(window_data)):
            result[i] = kurtosis(window_data, fisher=True)  # Excess kurtosis

    return pl.Series(result)


# =============================================================================
# Risk Metrics
# =============================================================================


def compute_drawdown(
    df: pl.DataFrame,
    window: int = 20,
) -> pl.DataFrame:
    """
    Calcula drawdown local (distancia al máximo reciente).

    Drawdown = (precio actual / máximo en ventana) - 1

    Siempre es <= 0. Más negativo = más lejos del máximo.
    """
    df = df.sort(["ticker", "date"])

    return df.with_columns(
        [
            (pl.col("close") / pl.col("close").rolling_max(window).over("ticker") - 1).alias(
                f"drawdown_{window}d"
            )
        ]
    )


def compute_rolling_beta(
    df: pl.DataFrame,
    benchmark_df: pl.DataFrame,
    benchmark_ticker: str = "SPY",
    window: int = 60,
) -> pl.DataFrame:
    """
    Calcula beta rolling vs benchmark.

    Beta = Cov(asset, benchmark) / Var(benchmark)

    Args:
        df: DataFrame de activos
        benchmark_df: DataFrame del benchmark con misma estructura
        benchmark_ticker: Nombre del ticker benchmark
        window: Ventana para cálculo rolling
    """
    df = df.sort(["ticker", "date"])

    # Calcular retornos si no existen
    if "ret_1d" not in df.columns:
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias("ret_1d")
        )

    # Obtener retornos del benchmark
    bench = benchmark_df.filter(pl.col("ticker") == benchmark_ticker)
    if "ret_1d" not in bench.columns:
        bench = bench.with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("ret_1d"))

    bench_returns = bench.select(["date", pl.col("ret_1d").alias("bench_ret")])

    # Unir con datos del activo
    df = df.join(bench_returns, on="date", how="left")

    # Calcular beta rolling (simplificado - usa correlación y ratio de vols)
    return df.with_columns(
        [
            (
                pl.col("ret_1d").rolling_cov(pl.col("bench_ret"), window).over("ticker")
                / pl.col("bench_ret").rolling_var(window)
            ).alias(f"beta_{window}d")
        ]
    ).drop("bench_ret")


# =============================================================================
# Cross-Sectional Ranks
# =============================================================================


def compute_cross_sectional_ranks(
    df: pl.DataFrame,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """
    Calcula ranks percentiles dentro del universo (por fecha).

    Normalizado a [0, 1] donde 1 = mayor valor en el universo ese día.

    Args:
        df: DataFrame con columnas a rankear
        columns: Lista de columnas a rankear (default: ret_1d, momentum_12_1)
    """
    if columns is None:
        columns = ["ret_1d", "momentum_12_1"]

    expressions = []
    for col in columns:
        if col in df.columns:
            # Rank dentro de cada fecha
            expressions.append(
                (
                    pl.col(col).rank(method="average").over("date")
                    / pl.col(col).count().over("date")
                ).alias(f"{col}_rank")
            )

    return df.with_columns(expressions) if expressions else df


# =============================================================================
# All-in-One Feature Builder
# =============================================================================


def compute_all_features(
    df: pl.DataFrame,
    benchmark_df: pl.DataFrame | None = None,
    *,
    return_horizons: list[int] | None = None,
    vol_windows: list[int] | None = None,
    include_momentum: bool = True,
    include_liquidity: bool = True,
    include_higher_moments: bool = True,
    include_risk: bool = True,
    include_ranks: bool = True,
) -> pl.DataFrame:
    """
    Calcula todas las features de una sola vez.

    Args:
        df: DataFrame con OHLCV
        benchmark_df: DataFrame del benchmark (para beta)
        return_horizons: Horizontes de retorno
        vol_windows: Ventanas de volatilidad
        include_*: Flags para incluir/excluir grupos de features

    Returns:
        DataFrame con todas las features calculadas
    """
    if return_horizons is None:
        return_horizons = [1, 5, 20]
    if vol_windows is None:
        vol_windows = [5, 20]

    result = df.clone()

    # Returns
    result = compute_multi_horizon_returns(result, return_horizons)

    # Volatility
    result = compute_realized_volatility(result, vol_windows)
    result = compute_atr(result, window=14)

    # Momentum
    if include_momentum:
        result = compute_momentum_12_1(result)
        result = compute_momentum_zscore(result, window=20)

    # Liquidity
    if include_liquidity:
        result = compute_amihud_illiquidity(result, window=20)
        result = compute_relative_volume(result, window=20)

    # Higher moments
    if include_higher_moments:
        result = compute_rolling_skewness(result, window=20)
        # Kurtosis es costoso, lo hacemos opcional
        # result = compute_rolling_kurtosis(result, window=20)

    # Risk
    if include_risk:
        result = compute_drawdown(result, window=20)
        if benchmark_df is not None:
            result = compute_rolling_beta(result, benchmark_df, window=60)

    # Cross-sectional ranks
    if include_ranks:
        rank_cols = [c for c in ["ret_1d", "momentum_12_1"] if c in result.columns]
        result = compute_cross_sectional_ranks(result, rank_cols)

    return result
