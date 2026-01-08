# Ingestion Manifest for Reproducibility
# =======================================
"""
Manifests para garantizar reproducibilidad de ingestas.

Cada ejecución de ingesta genera un manifest que contiene:
- Qué se descargó (tickers, fechas)
- Cómo se descargó (provider, params)
- Cuándo (timestamps)
- Integridad (hashes)
- Reproducibilidad (versiones de código y dependencias)
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from portfolio.core.compat import UTC

logger = logging.getLogger(__name__)


def get_polars_version() -> str:
    """Obtiene la versión de polars instalada."""
    return pl.__version__


def get_git_commit() -> str | None:
    """Obtiene el commit hash actual del repositorio."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def get_gammaedge_version() -> str:
    """Obtiene la versión de GammaEdge."""
    # Intentar leer de __version__ o pyproject.toml
    try:
        from portfolio import __version__

        return __version__
    except ImportError:
        pass
    return "0.0.0-dev"


@dataclass
class IngestionManifest:
    """
    Metadata completa de una ejecución de ingesta.

    Esta clase captura TODA la información necesaria para reproducir
    exactamente la misma ingesta en el futuro.
    """

    # Identifiers
    manifest_id: str = field(default_factory=lambda: str(uuid4()))
    ingestion_run_id: str = field(default_factory=lambda: str(uuid4()))

    # What was ingested
    data_type: str = "bars_raw"
    tickers: list[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""

    # How it was ingested
    provider: str = ""
    provider_params: dict[str, Any] = field(default_factory=dict)
    fallback_providers: list[str] = field(default_factory=list)

    # When
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    duration_seconds: float = 0.0

    # Results
    row_count: int = 0
    ticker_count: int = 0
    quality_passed: bool = True
    quality_summary: str = ""

    # Integrity
    schema_hash: str = ""
    data_hash: str = ""

    # Reproducibility
    gammaedge_version: str = field(default_factory=get_gammaedge_version)
    gammaedge_commit: str | None = field(default_factory=get_git_commit)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    polars_version: str = field(default_factory=get_polars_version)

    # Output
    output_files: list[str] = field(default_factory=list)
    manifest_path: str | None = None

    def complete(
        self,
        df: pl.DataFrame | None = None,
        quality_passed: bool = True,
        quality_summary: str = "",
    ) -> IngestionManifest:
        """Marca la ingesta como completada y calcula stats."""
        self.completed_at = datetime.now(UTC)
        self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        self.quality_passed = quality_passed
        self.quality_summary = quality_summary

        if df is not None:
            self.row_count = df.height
            self.ticker_count = (
                df.select(pl.col("ticker").n_unique()).item() if "ticker" in df.columns else 0
            )

            # Compute hashes
            self.data_hash = self._compute_data_hash(df)
            self.schema_hash = self._compute_schema_hash(df)

        return self

    def _compute_data_hash(self, df: pl.DataFrame) -> str:
        """Hash del contenido del DataFrame."""
        try:
            content = df.write_ipc(None).getvalue()
            return hashlib.sha256(content).hexdigest()[:16]
        except Exception:
            return "hash_error"

    def _compute_schema_hash(self, df: pl.DataFrame) -> str:
        """Hash del schema del DataFrame."""
        schema_str = str(sorted(df.schema.items()))
        return hashlib.sha256(schema_str.encode()).hexdigest()[:8]

    def to_dict(self) -> dict[str, Any]:
        """Serializa a diccionario."""
        d = asdict(self)
        # Convertir datetimes a ISO strings
        if d["started_at"]:
            d["started_at"] = d["started_at"].isoformat()
        if d["completed_at"]:
            d["completed_at"] = d["completed_at"].isoformat()
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serializa a JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> IngestionManifest:
        """Deserializa desde JSON."""
        d = json.loads(json_str)

        # Convertir ISO strings a datetimes
        if d.get("started_at"):
            d["started_at"] = datetime.fromisoformat(d["started_at"])
        if d.get("completed_at"):
            d["completed_at"] = datetime.fromisoformat(d["completed_at"])

        return cls(**d)

    @classmethod
    def from_file(cls, path: Path | str) -> IngestionManifest:
        """Lee manifest desde archivo."""
        with open(path) as f:
            return cls.from_json(f.read())

    def save(self, path: Path | str) -> Path:
        """Guarda manifest a archivo."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(self.to_json())

        self.manifest_path = str(path)
        logger.info("Saved manifest to %s", path)
        return path


class ManifestRegistry:
    """
    Registro de manifests para búsqueda y auditoría.

    Permite:
    - Buscar ingestas por fecha, ticker, provider
    - Verificar reproducibilidad
    - Detectar cambios en el pipeline
    """

    def __init__(self, manifests_dir: Path | str):
        """
        Args:
            manifests_dir: Directorio donde se guardan los manifests
        """
        self.manifests_dir = Path(manifests_dir)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def save(self, manifest: IngestionManifest) -> Path:
        """Guarda un manifest en el registro."""
        filename = f"{manifest.ingestion_run_id}.json"
        path = self.manifests_dir / filename
        return manifest.save(path)

    def get(self, run_id: str) -> IngestionManifest | None:
        """Obtiene un manifest por run_id."""
        path = self.manifests_dir / f"{run_id}.json"
        if path.exists():
            return IngestionManifest.from_file(path)
        return None

    def list_all(self) -> list[IngestionManifest]:
        """Lista todos los manifests."""
        manifests = []
        for path in self.manifests_dir.glob("*.json"):
            try:
                manifests.append(IngestionManifest.from_file(path))
            except Exception as e:
                logger.warning("Failed to load manifest %s: %s", path, e)
        return sorted(manifests, key=lambda m: m.started_at or datetime.min, reverse=True)

    def find_by_ticker(self, ticker: str) -> list[IngestionManifest]:
        """Encuentra manifests que incluyen un ticker."""
        return [m for m in self.list_all() if ticker in m.tickers]

    def find_by_provider(self, provider: str) -> list[IngestionManifest]:
        """Encuentra manifests por provider."""
        return [m for m in self.list_all() if m.provider == provider]

    def find_by_date_range(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[IngestionManifest]:
        """Encuentra manifests en un rango de fechas de ingesta."""
        manifests = self.list_all()

        if start:
            manifests = [m for m in manifests if m.started_at and m.started_at >= start]
        if end:
            manifests = [m for m in manifests if m.started_at and m.started_at <= end]

        return manifests

    def verify_reproducibility(
        self,
        manifest: IngestionManifest,
    ) -> dict[str, bool]:
        """
        Verifica si una ingesta podría reproducirse.

        Compara versiones actuales con las del manifest.
        """
        current_commit = get_git_commit()
        current_polars = get_polars_version()
        current_python = sys.version.split()[0]

        return {
            "same_commit": manifest.gammaedge_commit == current_commit,
            "same_polars": manifest.polars_version == current_polars,
            "same_python": manifest.python_version == current_python,
            "can_reproduce": (
                manifest.gammaedge_commit == current_commit
                and manifest.polars_version == current_polars
            ),
        }


@dataclass
class DataConfig:
    """
    Configuración explícita para ingesta (anti auto-detect).

    En research serio, NUNCA usar auto-detect. Usar un DataConfig
    explícito y versionado.
    """

    config_id: str = field(default_factory=lambda: str(uuid4()))
    config_version: str = "1.0.0"

    # Providers (explícitos, no auto-detect)
    primary_provider: str = "yahoo"
    fallback_providers: list[str] = field(default_factory=list)

    # Provider params
    provider_params: dict[str, dict] = field(default_factory=dict)

    # Quality gates
    max_missing_pct: float = 0.05
    max_daily_return: float = 0.50
    require_calendar_check: bool = True

    # Universe
    universe: str = "all"  # all, sp500, nasdaq100, custom
    custom_tickers: list[str] = field(default_factory=list)

    # Adjustments
    adjustment_version: str = "1.0.0"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> DataConfig:
        return cls(**json.loads(json_str))

    @classmethod
    def from_file(cls, path: Path | str) -> DataConfig:
        with open(path) as f:
            return cls.from_json(f.read())

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())
