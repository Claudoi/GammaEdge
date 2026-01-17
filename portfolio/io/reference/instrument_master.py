# Instrument Master — Stable Internal IDs
# ========================================
"""
Gestión de identidades de instrumentos con IDs internos estables.

PROBLEMA: Ticker no es identidad:
- Cambia ticker (FB → META)
- Cambia share class
- Merge/split/spin-off

SOLUCIÓN:
- instrument_id (UUID) es la identidad estable
- Mapeos: instrument_id ↔ ticker ↔ FIGI ↔ exchange ↔ provider_symbol
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import polars as pl

from portfolio.core.compat import UTC

logger = logging.getLogger(__name__)


@dataclass
class Instrument:
    """Representa un instrumento financiero con ID estable."""

    instrument_id: str
    current_ticker: str
    name: str | None = None
    asset_class: str = "equity"
    primary_exchange: str | None = None

    # External identifiers
    figi: str | None = None
    composite_figi: str | None = None
    cik: str | None = None
    isin: str | None = None
    cusip: str | None = None

    # Provider-specific symbols
    provider_symbols: dict[str, str] | None = None

    # Lifecycle
    ipo_date: date | None = None
    delisted_date: date | None = None
    is_active: bool = True

    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class TickerMapping:
    """Mapeo de ticker a instrument_id en un período."""

    instrument_id: str
    ticker: str
    exchange: str
    valid_from: date
    valid_to: date | None  # None = current
    reason: str  # ipo, ticker_change, merger, listing_change


class InstrumentMaster:
    """
    Registro maestro de instrumentos con IDs estables.

    Características:
    - instrument_id NUNCA cambia (UUID)
    - Ticker puede cambiar, se trackea en ticker_history
    - Mapeos bidireccionales para lookups rápidos
    - Provider-specific symbols para cada data source

    Example:
        >>> master = InstrumentMaster(path="data_lake/reference")
        >>>
        >>> # Crear instrumento
        >>> instr_id = master.create_instrument(
        ...     ticker="AAPL",
        ...     name="Apple Inc",
        ...     exchange="NASDAQ",
        ... )
        >>>
        >>> # Lookup por ticker (returns instrument_id)
        >>> master.get_instrument_id("AAPL")

        >>> # Lookup por instrument_id
        >>> master.get_current_ticker(instr_id)

        >>> # Registrar cambio de ticker
        >>> master.record_ticker_change("FB", "META", date(2022, 6, 9))
    """

    def __init__(self, path: Path | str | None = None):
        """
        Args:
            path: Directorio para persistir el master data
        """
        self.path = Path(path) if path else None
        self._instruments: dict[str, Instrument] = {}
        self._ticker_to_id: dict[str, str] = {}  # current ticker → instrument_id
        self._ticker_history: list[TickerMapping] = []

        if self.path and self.path.exists():
            self._load()

    def create_instrument(
        self,
        ticker: str,
        name: str | None = None,
        exchange: str | None = None,
        asset_class: str = "equity",
        figi: str | None = None,
        ipo_date: date | None = None,
        **kwargs,
    ) -> str:
        """
        Crea un nuevo instrumento y retorna su instrument_id.

        Returns:
            instrument_id (UUID)
        """
        # Check si ticker ya existe
        if ticker in self._ticker_to_id:
            logger.warning("Ticker %s already exists, returning existing ID", ticker)
            return self._ticker_to_id[ticker]

        instrument_id = str(uuid4())
        now = datetime.now(UTC)

        instrument = Instrument(
            instrument_id=instrument_id,
            current_ticker=ticker,
            name=name,
            asset_class=asset_class,
            primary_exchange=exchange,
            figi=figi,
            ipo_date=ipo_date,
            created_at=now,
            updated_at=now,
            **kwargs,
        )

        self._instruments[instrument_id] = instrument
        self._ticker_to_id[ticker] = instrument_id

        # Registrar en ticker history
        self._ticker_history.append(
            TickerMapping(
                instrument_id=instrument_id,
                ticker=ticker,
                exchange=exchange or "UNKNOWN",
                valid_from=ipo_date or date.today(),
                valid_to=None,
                reason="ipo",
            )
        )

        logger.info("Created instrument %s: %s (%s)", instrument_id[:8], ticker, name)
        return instrument_id

    def get_instrument_id(
        self,
        ticker: str,
        as_of: date | None = None,
    ) -> str | None:
        """
        Obtiene instrument_id para un ticker.

        Args:
            ticker: Símbolo a buscar
            as_of: Fecha para lookup histórico (None = current)

        Returns:
            instrument_id o None si no existe
        """
        if as_of is None:
            return self._ticker_to_id.get(ticker.upper())

        # Lookup histórico
        for mapping in self._ticker_history:
            if mapping.ticker.upper() == ticker.upper():
                if mapping.valid_from <= as_of:
                    if mapping.valid_to is None or mapping.valid_to > as_of:
                        return mapping.instrument_id

        return None

    def get_current_ticker(self, instrument_id: str) -> str | None:
        """Obtiene el ticker actual para un instrument_id."""
        instrument = self._instruments.get(instrument_id)
        if instrument:
            return instrument.current_ticker
        return None

    def get_ticker_as_of(self, instrument_id: str, as_of: date) -> str | None:
        """Obtiene el ticker que tenía un instrumento en una fecha."""
        for mapping in self._ticker_history:
            if mapping.instrument_id == instrument_id:
                if mapping.valid_from <= as_of:
                    if mapping.valid_to is None or mapping.valid_to > as_of:
                        return mapping.ticker
        return None

    def get_instrument(self, instrument_id: str) -> Instrument | None:
        """Obtiene el instrumento completo."""
        return self._instruments.get(instrument_id)

    def record_ticker_change(
        self,
        old_ticker: str,
        new_ticker: str,
        change_date: date,
        reason: str = "ticker_change",
    ) -> bool:
        """
        Registra un cambio de ticker manteniendo el mismo instrument_id.

        Example:
            >>> master.record_ticker_change("FB", "META", date(2022, 6, 9))
        """
        instrument_id = self._ticker_to_id.get(old_ticker.upper())
        if not instrument_id:
            logger.warning("Ticker %s not found for change", old_ticker)
            return False

        instrument = self._instruments.get(instrument_id)
        if not instrument:
            return False

        # Cerrar el mapping anterior
        for mapping in self._ticker_history:
            if (
                mapping.instrument_id == instrument_id
                and mapping.ticker.upper() == old_ticker.upper()
                and mapping.valid_to is None
            ):
                mapping.valid_to = change_date
                break

        # Crear nuevo mapping
        self._ticker_history.append(
            TickerMapping(
                instrument_id=instrument_id,
                ticker=new_ticker.upper(),
                exchange=instrument.primary_exchange or "UNKNOWN",
                valid_from=change_date,
                valid_to=None,
                reason=reason,
            )
        )

        # Actualizar current ticker
        del self._ticker_to_id[old_ticker.upper()]
        self._ticker_to_id[new_ticker.upper()] = instrument_id
        instrument.current_ticker = new_ticker.upper()
        instrument.updated_at = datetime.now(UTC)

        logger.info(
            "Recorded ticker change: %s → %s (instrument %s)",
            old_ticker,
            new_ticker,
            instrument_id[:8],
        )
        return True

    def record_delisting(
        self,
        ticker: str,
        delisted_date: date,
        reason: str = "delisted",
    ) -> bool:
        """Registra que un instrumento fue delisted."""
        instrument_id = self._ticker_to_id.get(ticker.upper())
        if not instrument_id:
            return False

        instrument = self._instruments.get(instrument_id)
        if not instrument:
            return False

        instrument.delisted_date = delisted_date
        instrument.is_active = False
        instrument.updated_at = datetime.now(UTC)

        # Cerrar mapping
        for mapping in self._ticker_history:
            if mapping.instrument_id == instrument_id and mapping.valid_to is None:
                mapping.valid_to = delisted_date
                mapping.reason = reason
                break

        logger.info("Recorded delisting: %s on %s", ticker, delisted_date)
        return True

    def set_provider_symbol(
        self,
        instrument_id: str,
        provider: str,
        symbol: str,
    ) -> None:
        """
        Establece el símbolo específico de un provider.

        Útil cuando providers usan símbolos diferentes:
        - Yahoo: BRK-B
        - Polygon: BRK.B
        - Alpaca: BRKB
        """
        instrument = self._instruments.get(instrument_id)
        if not instrument:
            return

        if instrument.provider_symbols is None:
            instrument.provider_symbols = {}

        instrument.provider_symbols[provider] = symbol
        instrument.updated_at = datetime.now(UTC)

    def get_provider_symbol(
        self,
        instrument_id: str,
        provider: str,
    ) -> str | None:
        """Obtiene el símbolo específico para un provider."""
        instrument = self._instruments.get(instrument_id)
        if not instrument:
            return None

        if instrument.provider_symbols:
            return instrument.provider_symbols.get(provider)

        # Fallback: usar current ticker
        return instrument.current_ticker

    def bulk_import(self, df: pl.DataFrame) -> int:
        """
        Importa instrumentos en bulk desde un DataFrame.

        Requiere columnas: ticker
        Opcionales: name, exchange, asset_class, figi, ipo_date
        """
        count = 0
        for row in df.iter_rows(named=True):
            ticker = row.get("ticker")
            if not ticker:
                continue

            if ticker in self._ticker_to_id:
                continue  # Skip existing

            self.create_instrument(
                ticker=ticker,
                name=row.get("name"),
                exchange=row.get("exchange"),
                asset_class=row.get("asset_class", "equity"),
                figi=row.get("figi"),
                ipo_date=row.get("ipo_date"),
            )
            count += 1

        logger.info("Bulk imported %d instruments", count)
        return count

    def to_dataframe(self) -> pl.DataFrame:
        """Exporta el master a DataFrame."""
        if not self._instruments:
            return pl.DataFrame(
                schema={
                    "instrument_id": pl.Utf8,
                    "current_ticker": pl.Utf8,
                    "name": pl.Utf8,
                    "asset_class": pl.Utf8,
                    "primary_exchange": pl.Utf8,
                    "figi": pl.Utf8,
                    "is_active": pl.Boolean,
                }
            )

        rows = []
        for instr in self._instruments.values():
            rows.append(
                {
                    "instrument_id": instr.instrument_id,
                    "current_ticker": instr.current_ticker,
                    "name": instr.name,
                    "asset_class": instr.asset_class,
                    "primary_exchange": instr.primary_exchange,
                    "figi": instr.figi,
                    "composite_figi": instr.composite_figi,
                    "cik": instr.cik,
                    "isin": instr.isin,
                    "ipo_date": instr.ipo_date,
                    "delisted_date": instr.delisted_date,
                    "is_active": instr.is_active,
                }
            )

        return pl.DataFrame(rows)

    def ticker_history_to_dataframe(self) -> pl.DataFrame:
        """Exporta el histórico de tickers a DataFrame."""
        if not self._ticker_history:
            return pl.DataFrame(
                schema={
                    "instrument_id": pl.Utf8,
                    "ticker": pl.Utf8,
                    "exchange": pl.Utf8,
                    "valid_from": pl.Date,
                    "valid_to": pl.Date,
                    "reason": pl.Utf8,
                }
            )

        rows = [
            {
                "instrument_id": m.instrument_id,
                "ticker": m.ticker,
                "exchange": m.exchange,
                "valid_from": m.valid_from,
                "valid_to": m.valid_to,
                "reason": m.reason,
            }
            for m in self._ticker_history
        ]

        return pl.DataFrame(rows)

    def save(self) -> None:
        """Persiste el master a disco."""
        if not self.path:
            return

        self.path.mkdir(parents=True, exist_ok=True)

        # Guardar instruments
        self.to_dataframe().write_parquet(self.path / "instruments.parquet")

        # Guardar ticker history
        self.ticker_history_to_dataframe().write_parquet(self.path / "ticker_history.parquet")

        logger.info("Saved instrument master to %s", self.path)

    def _load(self) -> None:
        """Carga el master desde disco."""
        instruments_path = self.path / "instruments.parquet"
        history_path = self.path / "ticker_history.parquet"

        if instruments_path.exists():
            df = pl.read_parquet(instruments_path)
            for row in df.iter_rows(named=True):
                instr = Instrument(
                    instrument_id=row["instrument_id"],
                    current_ticker=row["current_ticker"],
                    name=row.get("name"),
                    asset_class=row.get("asset_class", "equity"),
                    primary_exchange=row.get("primary_exchange"),
                    figi=row.get("figi"),
                    is_active=row.get("is_active", True),
                    ipo_date=row.get("ipo_date"),
                    delisted_date=row.get("delisted_date"),
                )
                self._instruments[instr.instrument_id] = instr
                self._ticker_to_id[instr.current_ticker] = instr.instrument_id

        if history_path.exists():
            df = pl.read_parquet(history_path)
            for row in df.iter_rows(named=True):
                self._ticker_history.append(
                    TickerMapping(
                        instrument_id=row["instrument_id"],
                        ticker=row["ticker"],
                        exchange=row.get("exchange", "UNKNOWN"),
                        valid_from=row["valid_from"],
                        valid_to=row.get("valid_to"),
                        reason=row.get("reason", "unknown"),
                    )
                )

        logger.info(
            "Loaded instrument master: %d instruments, %d ticker mappings",
            len(self._instruments),
            len(self._ticker_history),
        )

    def get_stats(self) -> dict:
        """Estadísticas del master."""
        return {
            "total_instruments": len(self._instruments),
            "active_instruments": sum(1 for i in self._instruments.values() if i.is_active),
            "delisted_instruments": sum(1 for i in self._instruments.values() if not i.is_active),
            "ticker_changes": sum(1 for m in self._ticker_history if m.reason == "ticker_change"),
            "total_ticker_mappings": len(self._ticker_history),
        }
