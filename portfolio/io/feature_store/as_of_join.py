# As-Of Join Framework
# ====================
"""
Framework para joins point-in-time que eliminan leakage estructural.

PROBLEMA:
Un join normal entre features y labels puede introducir leakage:
- Features calculados con datos que no existían en decision_time
- Universo que incluye instrumentos que no existían/eran tradeables

SOLUCIÓN:
As-of joins que respetan timestamps de disponibilidad.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import polars as pl

from portfolio.core.compat import UTC

logger = logging.getLogger(__name__)


DecisionTimePolicy = Literal[
    "market_close",  # Decidir al cierre del día t
    "next_open",  # Decidir a la apertura del día t+1
    "next_close",  # Decidir al cierre del día t+1
]


@dataclass
class AsOfConfig:
    """Configuración para as-of joins."""

    # Cuándo se toma la decisión
    decision_time_policy: DecisionTimePolicy = "market_close"

    # Offset adicional (para ser conservador)
    decision_delay_minutes: int = 0

    # Exchange para horarios
    exchange: str = "NYSE"

    # Timezone
    timezone: str = "America/New_York"


class AsOfJoiner:
    """
    Ejecuta joins point-in-time para evitar leakage.

    Características:
    - Respeta timestamps de disponibilidad de features
    - Construye universo as-of
    - Valida que no haya leakage

    Example:
        >>> joiner = AsOfJoiner(config=AsOfConfig(
        ...     decision_time_policy="next_open",
        ... ))
        >>>
        >>> # Join features con labels sin leakage
        >>> df = joiner.join_features_labels(
        ...     df_features,
        ...     df_labels,
        ...     feature_date_col="date",
        ...     label_date_col="date",
        ... )
    """

    def __init__(self, config: AsOfConfig | None = None):
        self.config = config or AsOfConfig()
        self._tz = ZoneInfo(self.config.timezone)

    def get_decision_time(self, observation_date: date) -> datetime:
        """
        Calcula el decision_time para una fecha de observación.

        Args:
            observation_date: Fecha de las features (t)

        Returns:
            Datetime exacto de la decisión
        """
        policy = self.config.decision_time_policy

        if policy == "market_close":
            # Cierre del día t
            dt = datetime(
                observation_date.year,
                observation_date.month,
                observation_date.day,
                16,
                0,
                0,
                tzinfo=self._tz,
            )

        elif policy == "next_open":
            # Apertura del día t+1
            next_day = observation_date + timedelta(days=1)
            # TODO: Skip weekends/holidays
            dt = datetime(
                next_day.year,
                next_day.month,
                next_day.day,
                9,
                30,
                0,
                tzinfo=self._tz,
            )

        elif policy == "next_close":
            # Cierre del día t+1
            next_day = observation_date + timedelta(days=1)
            dt = datetime(
                next_day.year,
                next_day.month,
                next_day.day,
                16,
                0,
                0,
                tzinfo=self._tz,
            )

        else:
            raise ValueError(f"Unknown policy: {policy}")

        # Aplicar delay adicional
        if self.config.decision_delay_minutes:
            dt = dt + timedelta(minutes=self.config.decision_delay_minutes)

        return dt.astimezone(UTC)

    def add_decision_time_column(
        self,
        df: pl.DataFrame,
        date_col: str = "date",
        output_col: str = "decision_time",
    ) -> pl.DataFrame:
        """
        Añade columna de decision_time a un DataFrame.

        Esto permite verificar leakage row by row.
        """
        # Necesitamos calcular decision_time para cada fecha
        dates = df.select(pl.col(date_col).unique()).to_series().to_list()

        decision_times = {d: self.get_decision_time(d) for d in dates}

        return df.with_columns(
            pl.col(date_col)
            .map_elements(lambda d: decision_times.get(d), return_dtype=pl.Datetime)
            .alias(output_col)
        )

    def add_available_at_column(
        self,
        df: pl.DataFrame,
        date_col: str = "date",
        available_at_policy: str = "market_close",
        output_col: str = "available_at",
    ) -> pl.DataFrame:
        """
        Añade columna de available_at para features.

        available_at indica cuándo el feature está disponible.
        """
        from portfolio.io.feature_store.registry import AvailabilityTime, FeatureDefinition

        # Crear definición temporal para calcular
        temp_def = FeatureDefinition(
            name="temp",
            available_at=AvailabilityTime(available_at_policy),
        )

        dates = df.select(pl.col(date_col).unique()).to_series().to_list()
        available_times = {
            d: temp_def.get_availability_timestamp(d, self.config.exchange) for d in dates
        }

        return df.with_columns(
            pl.col(date_col)
            .map_elements(lambda d: available_times.get(d), return_dtype=pl.Datetime)
            .alias(output_col)
        )

    def join_features_labels(
        self,
        df_features: pl.DataFrame,
        df_labels: pl.DataFrame,
        feature_date_col: str = "date",
        label_date_col: str = "date",
        ticker_col: str = "ticker",
        label_horizon: int = 1,
    ) -> pl.DataFrame:
        """
        Join de features con labels respetando point-in-time.

        Args:
            df_features: Features calculados en fecha t
            df_labels: Labels (ej: forward returns) en fecha t+horizon
            feature_date_col: Columna de fecha en features
            label_date_col: Columna de fecha en labels
            ticker_col: Columna de ticker
            label_horizon: Horizonte del label (días)

        Returns:
            DataFrame con features de t y label de t+horizon

        IMPORTANTE:
        - Features de fecha t se joinean con label de t+horizon
        - Se añaden columnas de validación de timing
        """
        # Añadir decision_time a features
        df_f = self.add_decision_time_column(
            df_features,
            date_col=feature_date_col,
            output_col="decision_time",
        )

        # Calcular fecha del label (t + horizon)
        df_f = df_f.with_columns(
            (pl.col(feature_date_col) + pl.duration(days=label_horizon)).alias("label_date")
        )

        # Renombrar fecha del label en df_labels
        df_l = df_labels.rename({label_date_col: "label_date"})

        # Join
        df_joined = df_f.join(
            df_l,
            on=[ticker_col, "label_date"],
            how="left",
        )

        logger.info(
            "Joined features with labels: %d rows, horizon=%d days",
            df_joined.height,
            label_horizon,
        )

        return df_joined

    def validate_no_leakage(
        self,
        df: pl.DataFrame,
        feature_cols: list[str],
        decision_time_col: str = "decision_time",
        available_at_col: str = "available_at",
    ) -> tuple[bool, pl.DataFrame]:
        """
        Valida que no haya leakage en un dataset.

        Leakage = feature usado antes de que esté disponible
        (available_at > decision_time)

        Returns:
            (is_valid, df_with_leakage_rows)
        """
        if available_at_col not in df.columns or decision_time_col not in df.columns:
            logger.warning(
                "Missing timestamp columns for leakage validation: %s, %s",
                available_at_col,
                decision_time_col,
            )
            return True, pl.DataFrame()

        # Encontrar filas con leakage
        df_leakage = df.filter(pl.col(available_at_col) > pl.col(decision_time_col))

        if df_leakage.height > 0:
            logger.warning(
                "LEAKAGE DETECTED: %d rows have available_at > decision_time",
                df_leakage.height,
            )

        return df_leakage.height == 0, df_leakage


def build_as_of_dataset(
    df_features: pl.DataFrame,
    df_labels: pl.DataFrame,
    universe_as_of: Callable[[date], list[str]] | None = None,
    decision_policy: DecisionTimePolicy = "next_open",
    label_horizon: int = 1,
) -> pl.DataFrame:
    """
    Construye un dataset point-in-time para training.

    GARANTÍAS:
    1. Features de t solo usan datos disponibles hasta decision_time(t)
    2. Labels son de t+horizon (futuro)
    3. Universo es el que existía en t (no survivorship bias)

    Args:
        df_features: DataFrame con features
        df_labels: DataFrame con labels
        universe_as_of: Función que retorna tickers válidos en una fecha
        decision_policy: Cuándo se toma la decisión
        label_horizon: Horizonte del label

    Example:
        >>> from portfolio.io.metadata import UniverseHistory
        >>> universe = UniverseHistory(...)
        >>>
        >>> df = build_as_of_dataset(
        ...     df_features,
        ...     df_labels,
        ...     universe_as_of=universe.get_universe_as_of,
        ...     decision_policy="next_open",
        ...     label_horizon=5,
        ... )
    """
    config = AsOfConfig(decision_time_policy=decision_policy)
    joiner = AsOfJoiner(config)

    # Join features con labels
    df = joiner.join_features_labels(
        df_features,
        df_labels,
        label_horizon=label_horizon,
    )

    # Filtrar por universo as-of
    if universe_as_of is not None:
        dates = df.select(pl.col("date").unique()).to_series().to_list()

        # Para cada fecha, filtrar solo instrumentos que existían
        filtered_dfs = []
        for d in dates:
            valid_tickers = universe_as_of(d)
            df_date = df.filter((pl.col("date") == d) & (pl.col("ticker").is_in(valid_tickers)))
            filtered_dfs.append(df_date)

        df = pl.concat(filtered_dfs)
        logger.info("Filtered by universe as-of: %d rows remaining", df.height)

    return df
