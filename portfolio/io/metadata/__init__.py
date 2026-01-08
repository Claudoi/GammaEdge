# Metadata Package Exports
"""
Metadata management for reproducibility and auditing.

- manifest: Ingestion manifests for reproducibility
- universe: Universe history for survivorship bias prevention
"""

from portfolio.io.metadata.manifest import (
    DataConfig,
    IngestionManifest,
    ManifestRegistry,
)
from portfolio.io.metadata.universe import (
    UniverseHistory,
    UniverseMembership,
    build_universe_from_reference,
)

__all__ = [
    # Manifests
    "DataConfig",
    "IngestionManifest",
    "ManifestRegistry",
    # Universe
    "UniverseHistory",
    "UniverseMembership",
    "build_universe_from_reference",
]
