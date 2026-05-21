# Data Ingestion Orchestrator
# ===========================
"""
Orquestador multi-proveedor para ingesta de datos.

Gestiona la descarga desde múltiples proveedores con:
- Fallback automático si un proveedor falla
- Cross-validation entre proveedores
- Rate limiting y retry logic
- Logging de auditoría
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import polars as pl

from portfolio.core.compat import UTC
from portfolio.io.ingestion.quality_gates import QualityGates, QualityReport
from portfolio.io.providers.base import (
    BaseDataProvider,
    ProviderError,
    RateLimitError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Resultado de una operación de ingesta."""

    ingestion_id: str
    provider: str
    data_type: str
    tickers: list[str]
    start_date: str
    end_date: str
    rows_fetched: int
    quality_passed: bool
    quality_report: QualityReport | None
    duration_seconds: float
    error_message: str | None
    timestamp: datetime

    @property
    def success(self) -> bool:
        return self.error_message is None and self.quality_passed

    def to_dict(self) -> dict:
        return {
            "ingestion_id": self.ingestion_id,
            "provider": self.provider,
            "data_type": self.data_type,
            "tickers": self.tickers,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "rows_fetched": self.rows_fetched,
            "quality_passed": self.quality_passed,
            "duration_seconds": round(self.duration_seconds, 3),
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }


class DataOrchestrator:
    """
    Orquestador multi-proveedor para ingesta de datos.

    Gestiona la descarga desde múltiples proveedores con fallback automático,
    validación de calidad, y almacenamiento en el data lake.

    Example:
        >>> from portfolio.io.providers.yahoo import YahooProvider
        >>>
        >>> orchestrator = DataOrchestrator(
        ...     primary=YahooProvider(),
        ...     fallbacks=[],
        ...     data_lake_path="/path/to/data_lake",
        ... )
        >>>
        >>> result = orchestrator.ingest_bars_daily(
        ...     tickers=["AAPL", "MSFT"],
        ...     start="2020-01-01",
        ...     end="2024-01-01",
        ... )
    """

    def __init__(
        self,
        primary: BaseDataProvider,
        fallbacks: list[BaseDataProvider] | None = None,
        data_lake_path: str | Path | None = None,
        quality_gates: QualityGates | None = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 5.0,
    ):
        """
        Inicializa el orquestador.

        Args:
            primary: Proveedor principal de datos
            fallbacks: Lista de proveedores de fallback (en orden de preferencia)
            data_lake_path: Ruta al data lake (para guardar raw/)
            quality_gates: Instancia de QualityGates (usa default si None)
            max_retries: Número máximo de reintentos por proveedor
            retry_delay_seconds: Delay entre reintentos
        """
        self.primary = primary
        self.fallbacks = fallbacks or []
        self.data_lake_path = Path(data_lake_path) if data_lake_path else None
        self.quality_gates = quality_gates or QualityGates()
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        # Todos los proveedores en orden de preferencia
        self._providers = [self.primary] + self.fallbacks

        logger.info(
            "DataOrchestrator initialized: primary=%s, fallbacks=%s",
            self.primary.name,
            [p.name for p in self.fallbacks],
        )

    def ingest_bars_daily(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
        *,
        run_quality_gates: bool = True,
        save_to_raw: bool = True,
        cross_validate: bool = False,
    ) -> tuple[pl.DataFrame, IngestionResult]:
        """
        Ingesta barras diarias con fallback automático.

        Args:
            tickers: Lista de símbolos
            start: Fecha inicio (YYYY-MM-DD)
            end: Fecha fin (YYYY-MM-DD)
            run_quality_gates: Ejecutar validación de calidad
            save_to_raw: Guardar en data_lake/raw/
            cross_validate: Validar contra proveedor secundario

        Returns:
            (DataFrame, IngestionResult) - datos y metadata de la ingesta
        """
        ingestion_id = str(uuid4())
        tickers_list = sorted(set(t.strip().upper() for t in tickers if t.strip()))
        t0 = time.perf_counter()

        logger.info(
            "[%s] Starting bars_daily ingestion: %d tickers, %s to %s",
            ingestion_id[:8],
            len(tickers_list),
            start,
            end,
        )

        df = None
        used_provider = None
        error_message = None

        # Intentar con cada proveedor
        for provider in self._providers:
            try:
                df = self._fetch_with_retry(
                    provider,
                    "get_bars_daily",
                    tickers_list,
                    start,
                    end,
                )
                used_provider = provider.name
                logger.info(
                    "[%s] Fetched %d rows from %s",
                    ingestion_id[:8],
                    df.height,
                    provider.name,
                )
                break

            except RateLimitError as e:
                logger.warning("[%s] Rate limit on %s: %s", ingestion_id[:8], provider.name, e)
                continue

            except ProviderError as e:
                logger.warning("[%s] Provider error on %s: %s", ingestion_id[:8], provider.name, e)
                error_message = str(e)
                continue

            except Exception as e:
                logger.exception("[%s] Unexpected error on %s", ingestion_id[:8], provider.name)
                error_message = str(e)
                continue

        # Si ningún proveedor funcionó
        if df is None:
            duration = time.perf_counter() - t0
            result = IngestionResult(
                ingestion_id=ingestion_id,
                provider="none",
                data_type="bars_1d",
                tickers=tickers_list,
                start_date=start,
                end_date=end,
                rows_fetched=0,
                quality_passed=False,
                quality_report=None,
                duration_seconds=duration,
                error_message=error_message or "All providers failed",
                timestamp=datetime.now(UTC),
            )
            logger.error("[%s] All providers failed", ingestion_id[:8])
            return pl.DataFrame(), result

        # Quality gates
        quality_report = None
        quality_passed = True
        if run_quality_gates:
            quality_report = self.quality_gates.validate_bars(df)
            quality_passed = quality_report.passed

            if not quality_passed:
                logger.warning(
                    "[%s] Quality gates failed: %s",
                    ingestion_id[:8],
                    quality_report.summary,
                )

        # Guardar en raw/
        if save_to_raw and self.data_lake_path is not None:
            self._save_to_raw(df, "bars_1d", tickers_list, start, end, ingestion_id)

        # Cross-validation (opcional)
        if cross_validate and len(self._providers) > 1 and used_provider is not None:
            self._cross_validate(df, tickers_list, start, end, used_provider)

        duration = time.perf_counter() - t0

        result = IngestionResult(
            ingestion_id=ingestion_id,
            provider=used_provider or "unknown",
            data_type="bars_1d",
            tickers=tickers_list,
            start_date=start,
            end_date=end,
            rows_fetched=df.height,
            quality_passed=quality_passed,
            quality_report=quality_report,
            duration_seconds=duration,
            error_message=None,
            timestamp=datetime.now(UTC),
        )

        logger.info(
            "[%s] Ingestion complete: %d rows, quality=%s, %.2fs",
            ingestion_id[:8],
            df.height,
            "PASSED" if quality_passed else "FAILED",
            duration,
        )

        return df, result

    def _fetch_with_retry(
        self,
        provider: BaseDataProvider,
        method: str,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """Ejecuta un fetch con reintentos."""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                func = getattr(provider, method)
                result: pl.DataFrame = func(tickers, start, end)
                return result

            except RateLimitError:
                raise  # No reintentar rate limits

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        "[%s] Attempt %d/%d failed: %s. Retrying in %.1fs...",
                        provider.name,
                        attempt,
                        self.max_retries,
                        e,
                        self.retry_delay_seconds,
                    )
                    time.sleep(self.retry_delay_seconds)

        raise ProviderError(provider.name, f"All {self.max_retries} attempts failed", last_error)

    def _save_to_raw(
        self,
        df: pl.DataFrame,
        data_type: str,
        tickers: list[str],
        start: str,
        end: str,
        ingestion_id: str,
    ) -> None:
        """Guarda datos en data_lake/raw/."""
        if self.data_lake_path is None:
            return

        raw_path = self.data_lake_path / "raw" / data_type
        raw_path.mkdir(parents=True, exist_ok=True)

        # Guardar por ticker para mejor particionamiento
        for ticker in df.select(pl.col("ticker").unique()).to_series():
            ticker_path = raw_path / ticker
            ticker_path.mkdir(parents=True, exist_ok=True)

            ticker_df = df.filter(pl.col("ticker") == ticker)

            # Nombre de archivo incluye rango de fechas
            filename = f"{start}_{end}.parquet"
            filepath = ticker_path / filename

            ticker_df.write_parquet(filepath)
            logger.debug("Saved %d rows to %s", ticker_df.height, filepath)

    def _cross_validate(
        self,
        df: pl.DataFrame,
        tickers: list[str],
        start: str,
        end: str,
        used_provider: str,
    ) -> None:
        """Valida datos contra un proveedor secundario."""
        # Encontrar un proveedor diferente para validar
        for provider in self._providers:
            if provider.name != used_provider:
                try:
                    df_check = provider.get_bars_daily(
                        tickers[:1], start, end
                    )  # Solo primer ticker
                    if df_check.height > 0:
                        # Comparar precios de cierre
                        primary_closes = df.filter(pl.col("ticker") == tickers[0])[
                            "close"
                        ].to_list()
                        check_closes = df_check["close"].to_list()

                        if len(primary_closes) > 0 and len(check_closes) > 0:
                            # Comparar últimos 5 precios
                            diff = abs(primary_closes[-1] - check_closes[-1]) / check_closes[-1]
                            if diff > 0.01:  # >1% diferencia
                                logger.warning(
                                    "Cross-validation warning: %s vs %s differ by %.2f%%",
                                    used_provider,
                                    provider.name,
                                    diff * 100,
                                )
                            else:
                                logger.info("Cross-validation passed against %s", provider.name)
                        break
                except Exception as e:
                    logger.debug("Cross-validation skipped: %s", e)

    def health_check(self) -> dict[str, dict]:
        """Verifica el estado de todos los proveedores."""
        results: dict[str, dict] = {}
        for provider in self._providers:
            results[provider.name] = provider.health_check()
        return results
