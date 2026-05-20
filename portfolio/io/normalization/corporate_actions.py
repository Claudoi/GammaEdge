# Corporate Actions Processor v2
# ==============================
"""
Procesador de corporate actions para ajustar precios de forma determinística.

PRINCIPIOS:
1. Nunca confiar en ajustes del provider
2. Calcular factores de ajuste de forma auditable
3. Volumen se ajusta INVERSO a precio en splits
4. Factores separados: split_factor vs dividend_factor
5. Threshold dinámico basado en volatilidad
6. Idempotencia garantizada

VERSIÓN: 2.0.0 — Corrige volumen, threshold dinámico, factores separados
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import date, datetime

import polars as pl

from portfolio.core.compat import UTC
from portfolio.io.schemas import (
    create_empty_bars_adjusted_df,
)

logger = logging.getLogger(__name__)

# Versión del algoritmo de ajuste — CAMBIAR si cambia la lógica
ADJUSTMENT_VERSION = "2.0.0"


@dataclass
class AdjustmentFactors:
    """Factores de ajuste separados para auditoría."""

    date: date
    ticker: str
    split_factor: float = 1.0  # Factor acumulado de splits
    dividend_factor: float = 1.0  # Factor acumulado de dividendos
    total_factor: float = 1.0  # split_factor * dividend_factor
    is_event_day: bool = False  # True si hay corp action en este día


@dataclass
class AdjustmentResult:
    """Resultado del proceso de ajuste."""

    df_adjusted: pl.DataFrame
    df_factors: pl.DataFrame
    anomalies: pl.DataFrame
    stats: dict
    content_hash: str = ""  # Hash para verificar idempotencia

    @property
    def has_anomalies(self) -> bool:
        return self.anomalies.height > 0


class CorporateActionsProcessor:
    """
    Aplica ajustes de corporate actions de forma determinística e idempotente.

    CORRECCIONES v2.0:
    - Volumen se ajusta INVERSO al split (split 2:1 → vol * 2)
    - Factores split vs dividend separados
    - Threshold dinámico: max(base_threshold, k * rolling_vol)
    - Hash de output para verificar idempotencia
    - Event days marcados para features/labels

    Example:
        >>> processor = CorporateActionsProcessor()
        >>> result = processor.process(df_raw, df_actions)
        >>>
        >>> # Verificar idempotencia
        >>> result2 = processor.process(df_raw, df_actions)
        >>> assert result.content_hash == result2.content_hash
    """

    def __init__(
        self,
        base_threshold: float = 0.15,
        volatility_multiplier: float = 5.0,
        rolling_vol_window: int = 20,
        min_dividend_yield: float = 0.001,
    ):
        """
        Args:
            base_threshold: Umbral base para anomalías (15%)
            volatility_multiplier: k en max(base, k * vol)
            rolling_vol_window: Ventana para calcular volatilidad
            min_dividend_yield: Mínimo yield para considerar dividendo
        """
        self.base_threshold = base_threshold
        self.volatility_multiplier = volatility_multiplier
        self.rolling_vol_window = rolling_vol_window
        self.min_dividend_yield = min_dividend_yield

    def process(
        self,
        df_raw: pl.DataFrame,
        df_actions: pl.DataFrame,
        *,
        source_provider: str = "unknown",
        source_run_id: str = "unknown",
    ) -> AdjustmentResult:
        """
        Procesa raw prices y aplica corporate actions.

        GARANTÍA: Output determinístico e idempotente.
        """
        logger.info(
            "Processing corporate actions v%s: %d rows, %d actions",
            ADJUSTMENT_VERSION,
            df_raw.height,
            df_actions.height,
        )

        if df_raw.height == 0:
            return AdjustmentResult(
                df_adjusted=create_empty_bars_adjusted_df(),
                df_factors=pl.DataFrame(),
                anomalies=pl.DataFrame(),
                stats={"rows_processed": 0},
                content_hash="empty",
            )

        # 1. Calcular factores de ajuste SEPARADOS
        df_factors = self._compute_adjustment_factors(df_raw, df_actions)

        # 2. Detectar anomalías con threshold DINÁMICO
        anomalies = self._detect_anomalies_dynamic(df_raw, df_actions)

        # 3. Aplicar ajustes (precio Y volumen)
        df_adjusted = self._apply_adjustments(
            df_raw,
            df_factors,
            source_provider=source_provider,
            source_run_id=source_run_id,
        )

        # 4. Marcar quality flags y event days
        df_adjusted = self._mark_quality_flags(df_adjusted, anomalies, df_factors)

        # 5. Calcular hash para idempotencia
        content_hash = self._compute_content_hash(df_adjusted)

        stats = {
            "rows_processed": df_raw.height,
            "rows_adjusted": df_adjusted.height,
            "splits_applied": len(df_actions.filter(pl.col("action_type") == "split")),
            "dividends_applied": len(df_actions.filter(pl.col("action_type") == "dividend")),
            "anomalies_detected": anomalies.height,
            "content_hash": content_hash,
            "adjustment_version": ADJUSTMENT_VERSION,
        }

        logger.info(
            "Corporate actions processed: %d rows, %d splits, %d dividends, %d anomalies, hash=%s",
            stats["rows_adjusted"],
            stats["splits_applied"],
            stats["dividends_applied"],
            stats["anomalies_detected"],
            content_hash[:8],
        )

        return AdjustmentResult(
            df_adjusted=df_adjusted,
            df_factors=df_factors,
            anomalies=anomalies,
            stats=stats,
            content_hash=content_hash,
        )

    def _compute_adjustment_factors(
        self,
        df_raw: pl.DataFrame,
        df_actions: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Calcula factores de ajuste SEPARADOS por ticker y fecha.

        Retorna: date, ticker, split_factor, dividend_factor, total_factor, is_event_day
        """
        dates_tickers = df_raw.select(["date", "ticker"]).unique()

        if df_actions.height == 0:
            return dates_tickers.with_columns(
                [
                    pl.lit(1.0).alias("split_factor"),
                    pl.lit(1.0).alias("dividend_factor"),
                    pl.lit(1.0).alias("total_factor"),
                    pl.lit(False).alias("is_event_day"),
                ]
            )

        result_dfs = []

        for ticker in df_raw.select(pl.col("ticker").unique()).to_series():
            ticker_raw = df_raw.filter(pl.col("ticker") == ticker).sort("date")
            ticker_actions = df_actions.filter(pl.col("ticker") == ticker).sort("ex_date")

            if ticker_actions.height == 0:
                factor_df = ticker_raw.select(["date", "ticker"]).with_columns(
                    [
                        pl.lit(1.0).alias("split_factor"),
                        pl.lit(1.0).alias("dividend_factor"),
                        pl.lit(1.0).alias("total_factor"),
                        pl.lit(False).alias("is_event_day"),
                    ]
                )
            else:
                factor_df = self._compute_ticker_factors_v2(ticker_raw, ticker_actions)

            result_dfs.append(factor_df)

        return pl.concat(result_dfs)

    def _compute_ticker_factors_v2(
        self,
        df_raw: pl.DataFrame,
        df_actions: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Calcula factores separados para un ticker.

        IMPORTANTE: Los factores se aplican RETROACTIVAMENTE.
        """
        ticker = df_raw["ticker"][0]
        dates = df_raw["date"].to_list()

        # Factores separados por fecha
        split_factors = {d: 1.0 for d in dates}
        dividend_factors = {d: 1.0 for d in dates}
        event_days = {d: False for d in dates}

        actions = df_actions.sort("ex_date", descending=True)

        cumulative_split = 1.0
        cumulative_div = 1.0

        for row in actions.iter_rows(named=True):
            ex_date = row["ex_date"]
            action_type = row["action_type"]

            # Marcar event day
            if ex_date in event_days:
                event_days[ex_date] = True

            if action_type == "split":
                split_ratio = row.get("split_ratio") or 1.0
                if split_ratio > 0 and split_ratio != 1.0:
                    cumulative_split *= split_ratio

            elif action_type == "dividend":
                dividend = row.get("dividend_amount") or 0.0
                close_before = self._get_price_before(df_raw, ex_date)
                if close_before and close_before > dividend and dividend > 0:
                    div_factor = (close_before - dividend) / close_before
                    cumulative_div *= div_factor

            # Aplicar a fechas anteriores
            for d in dates:
                if d < ex_date:
                    split_factors[d] = cumulative_split
                    dividend_factors[d] = cumulative_div

        return pl.DataFrame(
            {
                "date": list(dates),
                "ticker": [ticker] * len(dates),
                "split_factor": [split_factors[d] for d in dates],
                "dividend_factor": [dividend_factors[d] for d in dates],
                "total_factor": [split_factors[d] * dividend_factors[d] for d in dates],
                "is_event_day": [event_days[d] for d in dates],
            }
        )

    def _get_price_before(self, df_raw: pl.DataFrame, ex_date: date) -> float | None:
        """Obtiene el precio de cierre justo antes de una fecha."""
        close_col = "close_raw" if "close_raw" in df_raw.columns else "close"
        before = df_raw.filter(pl.col("date") < ex_date).sort("date", descending=True)
        if before.height > 0:
            return before[close_col][0]
        return None

    def _apply_adjustments(
        self,
        df_raw: pl.DataFrame,
        df_factors: pl.DataFrame,
        source_provider: str,
        source_run_id: str,
    ) -> pl.DataFrame:
        """
        Aplica factores de ajuste a precios Y volumen.

        IMPORTANTE:
        - Precios: se multiplican por factor
        - Volumen: se divide por split_factor (ajuste INVERSO)
        """
        df = df_raw.join(df_factors, on=["date", "ticker"], how="left")

        # Fill nulls
        df = df.with_columns(
            [
                pl.col("split_factor").fill_null(1.0),
                pl.col("dividend_factor").fill_null(1.0),
                pl.col("total_factor").fill_null(1.0),
                pl.col("is_event_day").fill_null(False),
            ]
        )

        now = datetime.now(UTC)

        # CORRECCIÓN: Volumen se ajusta INVERSO al split
        # Si split 2:1 (ratio=2), precios/2, volumen*2
        df_adjusted = df.select(
            [
                pl.col("date"),
                pl.col("ticker"),
                (pl.col("open_raw") * pl.col("total_factor")).alias("open"),
                (pl.col("high_raw") * pl.col("total_factor")).alias("high"),
                (pl.col("low_raw") * pl.col("total_factor")).alias("low"),
                (pl.col("close_raw") * pl.col("total_factor")).alias("close"),
                # Volumen: ajuste INVERSO al split (no afectado por dividendos)
                (pl.col("volume_raw") / pl.col("split_factor")).alias("volume"),
                pl.col("total_factor").alias("adj_factor"),
                pl.col("split_factor"),
                pl.col("dividend_factor"),
                pl.col("is_event_day"),
                pl.lit(ADJUSTMENT_VERSION).alias("adjustment_version"),
                pl.lit("pending").alias("quality_flag"),
                pl.lit(source_provider).alias("source_provider"),
                pl.lit(source_run_id).alias("source_run_id"),
                pl.lit(now).alias("adjusted_at"),
            ]
        )

        return df_adjusted

    def _detect_anomalies_dynamic(
        self,
        df_raw: pl.DataFrame,
        df_actions: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Detecta anomalías con THRESHOLD DINÁMICO.

        Threshold = max(base_threshold, k * rolling_vol_20d)
        """
        anomalies = []

        for ticker in df_raw.select(pl.col("ticker").unique()).to_series():
            ticker_raw = df_raw.filter(pl.col("ticker") == ticker).sort("date")
            ticker_actions = df_actions.filter(pl.col("ticker") == ticker)

            if ticker_raw.height < self.rolling_vol_window + 2:
                continue

            close_col = "close_raw" if "close_raw" in ticker_raw.columns else "close"

            # Calcular retornos y volatilidad rolling
            ticker_df = ticker_raw.with_columns(
                [
                    (pl.col(close_col) / pl.col(close_col).shift(1) - 1).alias("ret"),
                ]
            ).with_columns(
                [
                    pl.col("ret").rolling_std(self.rolling_vol_window).alias("rolling_vol"),
                ]
            )

            for row in ticker_df.iter_rows(named=True):
                ret = row.get("ret")
                rolling_vol = row.get("rolling_vol")
                date_val = row.get("date")

                if ret is None or rolling_vol is None:
                    continue

                # Threshold dinámico
                dynamic_threshold = max(
                    self.base_threshold,
                    (
                        self.volatility_multiplier * rolling_vol
                        if rolling_vol
                        else self.base_threshold
                    ),
                )

                if abs(ret) > dynamic_threshold:
                    # Verificar si hay action
                    has_action = ticker_actions.filter(pl.col("ex_date") == date_val).height > 0

                    if not has_action:
                        anomalies.append(
                            {
                                "date": date_val,
                                "ticker": ticker,
                                "return_pct": ret,
                                "threshold_used": dynamic_threshold,
                                "rolling_vol": rolling_vol,
                                "close_prev": row.get(close_col),
                                "reason": "dynamic_threshold_exceeded",
                            }
                        )

        if not anomalies:
            return pl.DataFrame(
                schema={
                    "date": pl.Date,
                    "ticker": pl.Utf8,
                    "return_pct": pl.Float64,
                    "threshold_used": pl.Float64,
                    "rolling_vol": pl.Float64,
                    "close_prev": pl.Float64,
                    "reason": pl.Utf8,
                }
            )

        return pl.DataFrame(anomalies)

    def _mark_quality_flags(
        self,
        df_adjusted: pl.DataFrame,
        anomalies: pl.DataFrame,
        df_factors: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Marca quality_flag y añade info de event days.

        Flags:
        - clean: Sin problemas
        - suspect: Anomalía detectada
        - event_day: Corporate action en este día
        """
        if anomalies.height == 0:
            return df_adjusted.with_columns(
                pl.when(pl.col("is_event_day"))
                .then(pl.lit("event_day"))
                .otherwise(pl.lit("clean"))
                .alias("quality_flag")
            )

        anomaly_keys = anomalies.select(["date", "ticker"])

        df_with_flag = (
            df_adjusted.join(
                anomaly_keys.with_columns(pl.lit(True).alias("_is_anomaly")),
                on=["date", "ticker"],
                how="left",
            )
            .with_columns(
                pl.when(pl.col("_is_anomaly"))
                .then(pl.lit("suspect"))
                .when(pl.col("is_event_day"))
                .then(pl.lit("event_day"))
                .otherwise(pl.lit("clean"))
                .alias("quality_flag")
            )
            .drop("_is_anomaly")
        )

        return df_with_flag

    def _compute_content_hash(self, df: pl.DataFrame) -> str:
        """
        Calcula hash del contenido para verificar idempotencia.

        El mismo input debe producir el mismo hash SIEMPRE.
        """
        # Excluir columnas con timestamps para que sea determinístico
        cols_for_hash = [c for c in df.columns if c not in ["adjusted_at", "ingested_at"]]
        df_for_hash = df.select(cols_for_hash).sort(["ticker", "date"])

        try:
            content = df_for_hash.write_ipc(None).getvalue()
            return hashlib.sha256(content).hexdigest()[:16]
        except Exception:
            return "hash_error"

    def verify_dollar_volume_consistency(
        self,
        df_raw: pl.DataFrame,
        df_adjusted: pl.DataFrame,
        tolerance: float = 0.01,
    ) -> pl.DataFrame:
        """
        Verifica que dollar_volume sea consistente antes/después del split.

        dollar_volume = close * volume debe ser ~igual (salvo noise).
        """
        close_col = "close_raw" if "close_raw" in df_raw.columns else "close"

        raw_dv = df_raw.with_columns(
            (pl.col(close_col) * pl.col("volume_raw")).alias("dollar_volume_raw")
        ).select(["date", "ticker", "dollar_volume_raw"])

        adj_dv = df_adjusted.with_columns(
            (pl.col("close") * pl.col("volume")).alias("dollar_volume_adj")
        ).select(["date", "ticker", "dollar_volume_adj"])

        comparison = raw_dv.join(adj_dv, on=["date", "ticker"]).with_columns(
            [
                (
                    (pl.col("dollar_volume_adj") - pl.col("dollar_volume_raw")).abs()
                    / pl.col("dollar_volume_raw")
                ).alias("pct_diff")
            ]
        )

        # Retornar filas donde la diferencia es > tolerancia
        return comparison.filter(pl.col("pct_diff") > tolerance)


# =============================================================================
# Helper Functions
# =============================================================================


def adjust_for_split(
    df: pl.DataFrame,
    split_date: date,
    split_ratio: float,
    price_cols: list[str] | None = None,
) -> pl.DataFrame:
    """
    Helper: Aplica un split a un DataFrame.

    CORRECCIÓN: También ajusta volumen en sentido inverso.
    """
    if price_cols is None:
        price_cols = ["open", "high", "low", "close"]

    # Ajustar precios (dividir por ratio)
    df = df.with_columns(
        [
            pl.when(pl.col("date") < split_date)
            .then(pl.col(col) / split_ratio)
            .otherwise(pl.col(col))
            .alias(col)
            for col in price_cols
        ]
    )

    # Ajustar volumen (multiplicar por ratio — inverso)
    if "volume" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("date") < split_date)
            .then(pl.col("volume") * split_ratio)
            .otherwise(pl.col("volume"))
            .alias("volume")
        )

    return df


def adjust_for_dividend(
    df: pl.DataFrame,
    ex_date: date,
    dividend_amount: float,
) -> pl.DataFrame:
    """
    Helper: Aplica ajuste por dividendo a un DataFrame.

    Ajuste: P_adj = P_raw * (1 - D/P_ex)
    Volumen NO se ajusta por dividendos.
    """
    close_before = df.filter(pl.col("date") < ex_date).sort("date", descending=True)

    if close_before.height == 0:
        return df

    price_before_ex = close_before["close"][0]

    if price_before_ex <= dividend_amount:
        return df

    factor = (price_before_ex - dividend_amount) / price_before_ex

    price_cols = ["open", "high", "low", "close"]

    return df.with_columns(
        [
            pl.when(pl.col("date") < ex_date)
            .then(pl.col(col) * factor)
            .otherwise(pl.col(col))
            .alias(col)
            for col in price_cols
        ]
    )
