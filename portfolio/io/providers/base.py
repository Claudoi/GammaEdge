# Base Data Provider Interface
# ============================
"""
Interfaz abstracta para proveedores de datos de mercado.

Todos los proveedores (Polygon, Alpaca, Tiingo, Yahoo) deben implementar
esta interfaz para garantizar consistencia en el pipeline de ingesta.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

from portfolio.core.compat import UTC
from portfolio.io.schemas import (
    ProviderName,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


class BaseDataProvider(ABC):
    """
    Interfaz abstracta para proveedores de datos de mercado.

    Todos los proveedores deben implementar los métodos abstractos.
    La clase base proporciona funcionalidad común como logging,
    rate limiting y manejo de errores.

    Attributes:
        name: Identificador único del proveedor
        api_key: API key (si aplica)
        base_url: URL base de la API
    """

    name: ProviderName

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Inicializa el proveedor.

        Args:
            api_key: API key para autenticación (puede venir de env var)
            base_url: URL base de la API (override del default)
        """
        self.api_key = api_key
        self.base_url = base_url
        self._request_count = 0
        self._last_request_time: datetime | None = None

    # =========================================================================
    # Abstract Methods - Deben ser implementados por cada proveedor
    # =========================================================================

    @abstractmethod
    def get_bars_daily(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """
        Descarga barras diarias OHLCV.

        Args:
            tickers: Lista de símbolos de acciones
            start: Fecha de inicio (YYYY-MM-DD)
            end: Fecha de fin (YYYY-MM-DD)

        Returns:
            DataFrame con schema SCHEMA_BARS_1D:
            - date, ticker, open, high, low, close, volume
            - adj_factor, source, ingested_at

        Raises:
            ProviderError: Si hay un error en la descarga
            RateLimitError: Si se excede el rate limit
        """
        ...

    @abstractmethod
    def get_corporate_actions(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """
        Descarga corporate actions (splits, dividendos).

        Args:
            tickers: Lista de símbolos
            start: Fecha de inicio (YYYY-MM-DD)
            end: Fecha de fin (YYYY-MM-DD)

        Returns:
            DataFrame con schema SCHEMA_CORPORATE_ACTIONS:
            - ticker, ex_date, action_type, factor, amount
            - currency, source, ingested_at
        """
        ...

    @abstractmethod
    def get_reference_data(
        self,
        tickers: Iterable[str],
    ) -> pl.DataFrame:
        """
        Descarga metadata de referencia de los tickers.

        Args:
            tickers: Lista de símbolos

        Returns:
            DataFrame con schema SCHEMA_REFERENCE:
            - ticker, name, exchange, asset_class, sector, industry
            - figi, cik, ipo_date, delisted_date, is_active, last_updated
        """
        ...

    @property
    @abstractmethod
    def rate_limit(self) -> tuple[int, int]:
        """
        Rate limit del proveedor.

        Returns:
            (max_requests, period_seconds) - ej: (5, 60) = 5 req/minuto
        """
        ...

    @property
    @abstractmethod
    def supports_intraday(self) -> bool:
        """Indica si el proveedor soporta datos intraday."""
        ...

    @property
    @abstractmethod
    def supports_corporate_actions(self) -> bool:
        """Indica si el proveedor proporciona corporate actions."""
        ...

    # =========================================================================
    # Common Methods - Funcionalidad compartida
    # =========================================================================

    def _normalize_tickers(self, tickers: Iterable[str]) -> list[str]:
        """Normaliza tickers a mayúsculas y sin espacios."""
        return sorted(set(t.strip().upper() for t in tickers if t.strip()))

    def _add_metadata_columns(
        self,
        df: pl.DataFrame,
        source: str | None = None,
    ) -> pl.DataFrame:
        """Añade columnas de metadata (source, ingested_at)."""
        return df.with_columns(
            [
                pl.lit(source or self.name).alias("source"),
                pl.lit(datetime.now(UTC)).alias("ingested_at"),
            ]
        )

    def _validate_date_range(self, start: str, end: str) -> None:
        """Valida que el rango de fechas sea válido."""
        from datetime import date as date_cls

        start_dt = date_cls.fromisoformat(start)
        end_dt = date_cls.fromisoformat(end)

        if start_dt > end_dt:
            raise ValueError(f"start ({start}) must be <= end ({end})")

        if end_dt > date_cls.today():
            logger.warning("end date %s is in the future, using today", end)

    def _log_request(
        self,
        operation: str,
        tickers: list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        """Log de la request para auditoría."""
        self._request_count += 1
        self._last_request_time = datetime.now(UTC)

        ticker_str = ",".join(tickers[:5])
        if len(tickers) > 5:
            ticker_str += f"...+{len(tickers) - 5} more"

        logger.info(
            "[%s] %s: tickers=%s, range=%s to %s (request #%d)",
            self.name,
            operation,
            ticker_str,
            start or "N/A",
            end or "N/A",
            self._request_count,
        )

    def health_check(self) -> dict:
        """
        Verifica conectividad con el proveedor.

        Returns:
            Dict con status, latency_ms, error (si aplica)
        """
        import time

        t0 = time.perf_counter()
        try:
            # Intenta obtener un dato mínimo
            df = self.get_bars_daily(["AAPL"], "2024-01-02", "2024-01-02")
            latency = (time.perf_counter() - t0) * 1000

            return {
                "provider": self.name,
                "status": "ok" if df.height > 0 else "empty",
                "latency_ms": round(latency, 2),
                "error": None,
            }
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            return {
                "provider": self.name,
                "status": "error",
                "latency_ms": round(latency, 2),
                "error": str(e),
            }


# =============================================================================
# Custom Exceptions
# =============================================================================


class ProviderError(Exception):
    """Error genérico del proveedor de datos."""

    def __init__(self, provider: str, message: str, original_error: Exception | None = None):
        self.provider = provider
        self.message = message
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")


class RateLimitError(ProviderError):
    """Error de rate limit excedido."""

    def __init__(self, provider: str, retry_after: int | None = None):
        self.retry_after = retry_after
        msg = "Rate limit exceeded"
        if retry_after:
            msg += f", retry after {retry_after}s"
        super().__init__(provider, msg)


class AuthenticationError(ProviderError):
    """Error de autenticación (API key inválida)."""

    def __init__(self, provider: str):
        super().__init__(provider, "Authentication failed - check API key")


class DataNotFoundError(ProviderError):
    """No se encontraron datos para el request."""

    def __init__(self, provider: str, tickers: list[str], start: str, end: str):
        msg = f"No data found for tickers={tickers} from {start} to {end}"
        super().__init__(provider, msg)
