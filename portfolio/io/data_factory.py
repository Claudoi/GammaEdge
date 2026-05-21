# Data Factory Configuration
# ==========================
"""
Factory functions para crear el stack de ingesta de datos.

Proporciona configuración automática basada en las API keys disponibles
y funciones helper para uso común.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from portfolio.io.ingestion.orchestrator import DataOrchestrator, IngestionResult
from portfolio.io.ingestion.quality_gates import QualityGates
from portfolio.io.providers.base import BaseDataProvider
from portfolio.io.providers.yahoo import YahooProvider

logger = logging.getLogger(__name__)

# Default paths - data_lake está en la raíz del proyecto GammaEdge
# __file__ = portfolio/io/data_factory.py
# parent.parent.parent = GammaEdge/
DEFAULT_DATA_LAKE_PATH = Path(__file__).parent.parent.parent / "data_lake"


def get_available_providers() -> dict[str, bool]:
    """
    Detecta qué proveedores están disponibles según API keys.

    Returns:
        Dict con nombre del proveedor y si está disponible
    """
    return {
        "yahoo": True,
    }


def create_provider(name: str) -> BaseDataProvider:
    """
    Crea una instancia de un proveedor por nombre.

    Args:
        name: Nombre del proveedor (yahoo, polygon, alpaca, tiingo)

    Returns:
        Instancia del proveedor

    Raises:
        ValueError: Si el proveedor no existe
        AuthenticationError: Si falta la API key
    """
    if name == "yahoo":
        return YahooProvider()

    else:
        # For simplicity in this Yahoo-only refactor, we raise error for others
        raise ValueError(f"Unknown provider: {name}. Only 'yahoo' is supported.")


def create_orchestrator(
    primary: str | None = None,
    fallbacks: list[str] | None = None,
    data_lake_path: str | Path | None = None,
    auto_detect: bool = True,
) -> DataOrchestrator:
    """
    Crea un DataOrchestrator con la configuración óptima.

    Si auto_detect=True, detecta automáticamente qué proveedores están
    disponibles y los configura en orden de preferencia:
    1. Massive (primary) - si tiene API key
    2. Alpaca (fallback 1) - si tiene API keys
    3. Tiingo (fallback 2) - si tiene API key
    4. Yahoo (último fallback) - siempre disponible

    Args:
        primary: Nombre del proveedor primario (override)
        fallbacks: Lista de fallbacks (override)
        data_lake_path: Ruta al data lake
        auto_detect: Si True, detecta providers disponibles

    Returns:
        DataOrchestrator configurado

    Example:
        >>> # Auto-detect based on environment variables
        >>> orch = create_orchestrator()
        >>>
        >>> # Manual configuration
        >>> orch = create_orchestrator(
        ...     primary="yahoo",
        ...     fallbacks=[],
        ...     data_lake_path="./my_data_lake",
        ... )
    """
    """
    Crea un DataOrchestrator con la configuración óptima (Yahoo Only).
    """
    primary = "yahoo"
    fallbacks = []

    logger.info("Auto-detect: forcing primary='yahoo' (sole provider).")

    # Crear providers
    primary_provider = create_provider(primary)
    fallback_providers = [create_provider(name) for name in fallbacks]

    if data_lake_path is None:
        data_lake_path = DEFAULT_DATA_LAKE_PATH

    logger.info(
        "Creating orchestrator: primary=%s, fallbacks=%s",
        primary,
        fallbacks,
    )

    return DataOrchestrator(
        primary=primary_provider,
        fallbacks=fallback_providers,
        data_lake_path=data_lake_path,
        quality_gates=QualityGates(),
    )


def quick_ingest(
    tickers: list[str],
    start: str,
    end: str,
    **kwargs: Any,
) -> tuple[Any, IngestionResult]:
    """
    Función helper para ingesta rápida.

    Crea un orchestrator con auto-detect y ejecuta la ingesta.

    Args:
        tickers: Lista de símbolos
        start: Fecha inicio (YYYY-MM-DD)
        end: Fecha fin (YYYY-MM-DD)
        **kwargs: Argumentos adicionales para ingest_bars_daily

    Returns:
        (DataFrame, IngestionResult)

    Example:
        >>> df, result = quick_ingest(
        ...     ["AAPL", "MSFT"],
        ...     "2024-01-01",
        ...     "2024-12-31",
        ... )
    """
    orch = create_orchestrator()
    return orch.ingest_bars_daily(tickers, start, end, **kwargs)


def provider_health_check() -> dict[str, dict]:
    """
    Verifica el estado de todos los proveedores disponibles.

    Returns:
        Dict con resultados de health check por proveedor
    """
    results = {}
    available = get_available_providers()

    for name, is_available in available.items():
        if is_available:
            try:
                provider = create_provider(name)
                results[name] = provider.health_check()
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                }
        else:
            results[name] = {
                "status": "not_configured",
                "error": "API key not set",
            }

    return results
