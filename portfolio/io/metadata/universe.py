# Universe History Manager
# ========================
"""
Gestión de histórico de universo para evitar survivorship bias.

PROBLEMA:
Sin universe history, el modelo solo ve los "ganadores" que sobrevivieron.
Esto infla artificialmente los retornos esperados.

SOLUCIÓN:
Mantener registro de qué tickers existían en cada fecha, incluyendo
los que fueron delisted, merged, o cambiaron de ticker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import polars as pl

from portfolio.io.schemas import create_empty_universe_history_df

logger = logging.getLogger(__name__)


RemovalReason = Literal[
    "delisted",
    "merged",
    "acquired",
    "bankruptcy",
    "ticker_change",
    "dropped_from_index",
    "manual_exclusion",
]


@dataclass
class UniverseMembership:
    """Representa la membership de un instrumento en un universo."""

    instrument_id: str
    ticker: str
    universe: str
    start_date: date
    end_date: date | None
    reason_added: str
    reason_removed: str | None
    merged_into_id: str | None


class UniverseHistory:
    """
    Gestiona el histórico de membership de universos.

    Permite:
    - Construir universo as-of cualquier fecha histórica
    - Evitar survivorship bias
    - Trackear cambios de ticker y mergers

    Example:
        >>> universe = UniverseHistory(path="data_lake/metadata/universe_history")
        >>>
        >>> # Obtener universo al 1 de enero 2020
        >>> tickers = universe.get_universe_as_of(date(2020, 1, 1))
        >>>
        >>> # Registrar un delisting
        >>> universe.record_removal(
        ...     ticker="ENRON",
        ...     date=date(2001, 12, 2),
        ...     reason="bankruptcy",
        ... )
    """

    def __init__(self, path: Path | str | None = None):
        """
        Args:
            path: Ruta al archivo parquet con el histórico
        """
        self.path = Path(path) if path else None
        self._df: pl.DataFrame | None = None

    @property
    def df(self) -> pl.DataFrame:
        """Carga o crea el DataFrame de histórico."""
        if self._df is None:
            if self.path and self.path.exists():
                self._df = pl.read_parquet(self.path)
            else:
                self._df = create_empty_universe_history_df()
        return self._df

    def save(self) -> None:
        """Guarda el histórico a disco."""
        if self.path and self._df is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._df.write_parquet(self.path)
            logger.info("Saved universe history to %s", self.path)

    def get_universe_as_of(
        self,
        as_of_date: date,
        universe: str = "all",
    ) -> list[str]:
        """
        Obtiene la lista de tickers que existían en una fecha.

        Args:
            as_of_date: Fecha de referencia
            universe: Nombre del universo (all, sp500, nasdaq100, etc.)

        Returns:
            Lista de tickers activos en esa fecha
        """
        df = self.df

        if df.height == 0:
            return []

        # Filtrar por universo
        if universe != "all":
            df = df.filter(pl.col("universe") == universe)

        # Filtrar por fecha:
        # start_date <= as_of_date AND (end_date IS NULL OR end_date > as_of_date)
        active = df.filter(
            (pl.col("start_date") <= as_of_date)
            & (pl.col("end_date").is_null() | (pl.col("end_date") > as_of_date))
        )

        return active["ticker"].unique().to_list()

    def get_delistings(
        self,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """
        Obtiene tickers que fueron removidos en un período.

        Útil para:
        - Verificar si el backtest está considerando delistings
        - Analizar por qué desapareció un ticker
        """
        return self.df.filter(
            (pl.col("end_date").is_not_null())
            & (pl.col("end_date") >= start)
            & (pl.col("end_date") <= end)
        )

    def add_ticker(
        self,
        ticker: str,
        start_date: date,
        universe: str = "all",
        instrument_id: str | None = None,
        reason: str = "ipo",
    ) -> None:
        """
        Registra un nuevo ticker en el universo.

        Args:
            ticker: Símbolo
            start_date: Fecha de inicio
            universe: Universo al que pertenece
            instrument_id: ID interno (se genera si no se provee)
            reason: Razón de adición (ipo, added_to_index, etc.)
        """
        import uuid

        new_row = pl.DataFrame(
            [
                {
                    "instrument_id": instrument_id or str(uuid.uuid4()),
                    "ticker": ticker,
                    "universe": universe,
                    "start_date": start_date,
                    "end_date": None,
                    "reason_added": reason,
                    "reason_removed": None,
                    "merged_into_id": None,
                }
            ]
        )

        self._df = pl.concat([self.df, new_row])
        logger.info("Added %s to universe %s starting %s", ticker, universe, start_date)

    def record_removal(
        self,
        ticker: str,
        removal_date: date,
        reason: RemovalReason,
        universe: str = "all",
        merged_into_id: str | None = None,
    ) -> None:
        """
        Registra la remoción de un ticker.

        Args:
            ticker: Símbolo
            removal_date: Fecha de remoción
            reason: Razón (delisted, merged, etc.)
            universe: Universo del que se remueve
            merged_into_id: Si fue merger, el instrument_id del adquirente
        """
        # Encontrar el registro activo
        mask = (
            (pl.col("ticker") == ticker)
            & (pl.col("universe") == universe)
            & pl.col("end_date").is_null()
        )

        self._df = self.df.with_columns(
            [
                pl.when(mask)
                .then(pl.lit(removal_date))
                .otherwise(pl.col("end_date"))
                .alias("end_date"),
                pl.when(mask)
                .then(pl.lit(reason))
                .otherwise(pl.col("reason_removed"))
                .alias("reason_removed"),
                pl.when(mask)
                .then(pl.lit(merged_into_id))
                .otherwise(pl.col("merged_into_id"))
                .alias("merged_into_id"),
            ]
        )

        logger.info("Recorded removal of %s on %s: %s", ticker, removal_date, reason)

    def record_ticker_change(
        self,
        old_ticker: str,
        new_ticker: str,
        change_date: date,
        universe: str = "all",
    ) -> None:
        """
        Registra un cambio de ticker manteniendo continuidad del instrument.

        Args:
            old_ticker: Ticker anterior
            new_ticker: Nuevo ticker
            change_date: Fecha del cambio
            universe: Universo
        """
        # Obtener instrument_id del ticker viejo
        old_record = self.df.filter(
            (pl.col("ticker") == old_ticker)
            & (pl.col("universe") == universe)
            & pl.col("end_date").is_null()
        )

        if old_record.height == 0:
            logger.warning("Ticker %s not found in universe %s", old_ticker, universe)
            return

        instrument_id = old_record["instrument_id"][0]

        # Cerrar el registro viejo
        self.record_removal(old_ticker, change_date, "ticker_change", universe)

        # Crear nuevo registro manteniendo el instrument_id
        self.add_ticker(
            ticker=new_ticker,
            start_date=change_date,
            universe=universe,
            instrument_id=instrument_id,
            reason="ticker_change",
        )

        logger.info("Recorded ticker change: %s -> %s", old_ticker, new_ticker)

    def bulk_import(
        self,
        df: pl.DataFrame,
        universe: str = "all",
    ) -> None:
        """
        Importa memberships en bulk.

        El DataFrame debe tener columnas: ticker, start_date, end_date (opcional)
        """
        import uuid

        required_cols = {"ticker", "start_date"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

        # Añadir columnas faltantes
        if "instrument_id" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("instrument_id")).with_columns(
                pl.col("instrument_id")
                .map_elements(lambda _: str(uuid.uuid4()), return_dtype=pl.Utf8)
                .alias("instrument_id")
            )

        if "end_date" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Date).alias("end_date"))

        df = df.with_columns(
            [
                pl.lit(universe).alias("universe"),
                pl.lit("bulk_import").alias("reason_added"),
                pl.lit(None).cast(pl.Utf8).alias("reason_removed"),
                pl.lit(None).cast(pl.Utf8).alias("merged_into_id"),
            ]
        )

        self._df = pl.concat([self.df, df])
        logger.info("Bulk imported %d tickers to universe %s", df.height, universe)

    def get_stats(self) -> dict:
        """Obtiene estadísticas del universo."""
        df = self.df

        return {
            "total_instruments": df.select(pl.col("instrument_id").n_unique()).item(),
            "total_tickers": df.select(pl.col("ticker").n_unique()).item(),
            "active_tickers": df.filter(pl.col("end_date").is_null()).height,
            "delisted_count": df.filter(pl.col("reason_removed") == "delisted").height,
            "merged_count": df.filter(pl.col("reason_removed") == "merged").height,
            "earliest_date": df["start_date"].min(),
            "latest_removal": df["end_date"].max(),
        }


def build_universe_from_reference(
    df_reference: pl.DataFrame,
    universe_name: str = "all",
) -> UniverseHistory:
    """
    Construye un UniverseHistory desde datos de referencia.

    Útil para inicializar el universo desde datos existentes.
    """
    universe = UniverseHistory()

    df = df_reference.select(
        [
            pl.col("ticker"),
            pl.col("ipo_date").alias("start_date"),
            pl.col("delisted_date").alias("end_date"),
        ]
    ).filter(pl.col("start_date").is_not_null())

    universe.bulk_import(df, universe=universe_name)

    return universe
