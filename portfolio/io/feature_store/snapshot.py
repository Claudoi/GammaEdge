# Dataset Snapshots for Reproducible Training
# ============================================
"""
Snapshots inmutables de datasets para training reproducible.

PROBLEMA:
Sin snapshots, no puedes:
- Reproducir un resultado 6 meses después
- Comparar modelos sin que cambie el suelo
- Debuggear por qué un modelo dejó de funcionar

SOLUCIÓN:
Cada training run apunta a un snapshot inmutable con:
- snapshot_id
- feature_set_version
- adjustment_version
- hash del contenido
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from portfolio.core.compat import UTC

logger = logging.getLogger(__name__)


@dataclass
class DatasetSnapshot:
    """
    Snapshot inmutable de un dataset para training.

    Contiene toda la información necesaria para reproducir
    exactamente el mismo dataset en el futuro.
    """

    # Identificación
    snapshot_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""

    # Alcance
    tickers: list[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""

    # Versiones (CRÍTICO para reproducibilidad)
    feature_set_version: str = ""  # Hash del feature registry
    adjustment_version: str = ""  # Versión del corporate actions processor
    reconciliation_policy: str = ""  # Qué policy se usó
    universe_version: str = ""  # Versión del universo

    # Configuración de timing
    decision_time_policy: str = "next_open"
    label_horizon: int = 1

    # Features incluidos
    feature_names: list[str] = field(default_factory=list)
    label_name: str = ""

    # Integridad
    row_count: int = 0
    ticker_count: int = 0
    feature_count: int = 0
    content_hash: str = ""  # SHA256 del contenido
    schema_hash: str = ""  # Hash del schema

    # Código
    gammaedge_commit: str | None = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Storage
    data_path: str = ""  # Ruta al parquet

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = d["created_at"].isoformat()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> DatasetSnapshot:
        d = d.copy()
        if d.get("created_at"):
            d["created_at"] = datetime.fromisoformat(d["created_at"])
        return cls(**d)

    @classmethod
    def from_json(cls, json_str: str) -> DatasetSnapshot:
        return cls.from_dict(json.loads(json_str))


class SnapshotManager:
    """
    Gestiona snapshots de datasets para training.

    Características:
    - Crea snapshots inmutables
    - Verifica integridad via hashes
    - Permite reproducir datasets exactos

    Example:
        >>> manager = SnapshotManager(path="data_lake/snapshots")
        >>>
        >>> # Crear snapshot
        >>> snapshot = manager.create_snapshot(
        ...     df_training,
        ...     name="model_v1_training",
        ...     feature_names=["ret_1d", "vol_20d", ...],
        ...     label_name="ret_5d_forward",
        ... )
        >>>
        >>> # Más tarde, cargar exactamente el mismo dataset
        >>> df = manager.load_snapshot(snapshot.snapshot_id)
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._catalog: dict[str, DatasetSnapshot] = {}
        self._load_catalog()

    def create_snapshot(
        self,
        df: pl.DataFrame,
        name: str,
        feature_names: list[str],
        label_name: str = "",
        description: str = "",
        feature_set_version: str = "",
        adjustment_version: str = "",
        decision_time_policy: str = "next_open",
        label_horizon: int = 1,
        **metadata: Any,
    ) -> DatasetSnapshot:
        """
        Crea un snapshot inmutable de un dataset.

        Args:
            df: DataFrame a guardar
            name: Nombre descriptivo
            feature_names: Features incluidos
            label_name: Nombre del label
            ...

        Returns:
            DatasetSnapshot con toda la metadata
        """
        snapshot_id = str(uuid4())

        # Calcular hashes
        content_hash = self._compute_content_hash(df)
        schema_hash = self._compute_schema_hash(df)

        # Extraer metadata del DataFrame
        tickers = []
        if "ticker" in df.columns:
            tickers = df.select(pl.col("ticker").unique()).to_series().to_list()

        start_date = ""
        end_date = ""
        if "date" in df.columns:
            start_date = str(df["date"].min())
            end_date = str(df["date"].max())

        # Guardar datos
        data_path = self.path / "data" / f"{snapshot_id}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(data_path)

        # Crear snapshot
        snapshot = DatasetSnapshot(
            snapshot_id=snapshot_id,
            name=name,
            description=description,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            feature_set_version=feature_set_version,
            adjustment_version=adjustment_version,
            decision_time_policy=decision_time_policy,
            label_horizon=label_horizon,
            feature_names=feature_names,
            label_name=label_name,
            row_count=df.height,
            ticker_count=len(tickers),
            feature_count=len(feature_names),
            content_hash=content_hash,
            schema_hash=schema_hash,
            gammaedge_commit=self._get_git_commit(),
            data_path=str(data_path),
        )

        # Guardar metadata
        self._save_snapshot_metadata(snapshot)
        self._catalog[snapshot_id] = snapshot

        logger.info(
            "Created snapshot %s: %s (%d rows, %d features, hash=%s)",
            snapshot_id[:8],
            name,
            df.height,
            len(feature_names),
            content_hash[:8],
        )

        return snapshot

    def load_snapshot(
        self,
        snapshot_id: str,
        verify_hash: bool = True,
    ) -> pl.DataFrame:
        """
        Carga un snapshot y opcionalmente verifica integridad.

        Args:
            snapshot_id: ID del snapshot
            verify_hash: Si True, verifica que el hash sea correcto

        Returns:
            DataFrame exactamente como fue guardado

        Raises:
            ValueError: Si el hash no coincide (data corruption)
        """
        snapshot = self.get_snapshot(snapshot_id)
        if snapshot is None:
            raise ValueError(f"Snapshot not found: {snapshot_id}")

        df = pl.read_parquet(snapshot.data_path)

        if verify_hash:
            current_hash = self._compute_content_hash(df)
            if current_hash != snapshot.content_hash:
                raise ValueError(
                    f"Hash mismatch! Expected {snapshot.content_hash}, "
                    f"got {current_hash}. Data may be corrupted."
                )

        logger.info(
            "Loaded snapshot %s: %d rows, verified=%s",
            snapshot_id[:8],
            df.height,
            verify_hash,
        )

        return df

    def get_snapshot(self, snapshot_id: str) -> DatasetSnapshot | None:
        """Obtiene metadata de un snapshot."""
        return self._catalog.get(snapshot_id)

    def list_snapshots(
        self,
        name_contains: str | None = None,
    ) -> list[DatasetSnapshot]:
        """Lista snapshots, opcionalmente filtrados."""
        snapshots = list(self._catalog.values())

        if name_contains:
            snapshots = [s for s in snapshots if name_contains in s.name]

        return sorted(snapshots, key=lambda s: s.created_at, reverse=True)

    def compare_snapshots(
        self,
        snapshot_id_1: str,
        snapshot_id_2: str,
    ) -> dict:
        """Compara dos snapshots."""
        s1 = self.get_snapshot(snapshot_id_1)
        s2 = self.get_snapshot(snapshot_id_2)

        if s1 is None or s2 is None:
            raise ValueError("One or both snapshots not found")

        return {
            "same_content": s1.content_hash == s2.content_hash,
            "same_schema": s1.schema_hash == s2.schema_hash,
            "same_features": set(s1.feature_names) == set(s2.feature_names),
            "same_tickers": set(s1.tickers) == set(s2.tickers),
            "row_diff": s1.row_count - s2.row_count,
            "ticker_diff": s1.ticker_count - s2.ticker_count,
            "s1_feature_set_version": s1.feature_set_version,
            "s2_feature_set_version": s2.feature_set_version,
        }

    def _compute_content_hash(self, df: pl.DataFrame) -> str:
        """Hash del contenido para integridad."""
        try:
            content = df.write_ipc(None).getvalue()
            return hashlib.sha256(content).hexdigest()[:16]
        except Exception:
            return "hash_error"

    def _compute_schema_hash(self, df: pl.DataFrame) -> str:
        """Hash del schema."""
        schema_str = str(sorted(df.schema.items()))
        return hashlib.sha256(schema_str.encode()).hexdigest()[:8]

    def _get_git_commit(self) -> str | None:
        """Obtiene commit hash actual."""
        import subprocess

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

    def _save_snapshot_metadata(self, snapshot: DatasetSnapshot) -> None:
        """Guarda metadata del snapshot."""
        meta_path = self.path / "metadata" / f"{snapshot.snapshot_id}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        with open(meta_path, "w") as f:
            f.write(snapshot.to_json())

    def _load_catalog(self) -> None:
        """Carga el catálogo de snapshots."""
        meta_dir = self.path / "metadata"
        if not meta_dir.exists():
            return

        for meta_file in meta_dir.glob("*.json"):
            try:
                with open(meta_file) as f:
                    snapshot = DatasetSnapshot.from_json(f.read())
                    self._catalog[snapshot.snapshot_id] = snapshot
            except Exception as e:
                logger.warning("Failed to load snapshot %s: %s", meta_file, e)

        logger.info("Loaded %d snapshots from catalog", len(self._catalog))


@dataclass
class TrainingRun:
    """
    Registro de un training run para reproducibilidad completa.

    Conecta:
    - Dataset snapshot
    - Configuración del modelo
    - Resultados
    - Artifacts
    """

    run_id: str = field(default_factory=lambda: str(uuid4()))

    # Dataset
    snapshot_id: str = ""

    # Modelo
    model_type: str = ""
    model_params: dict = field(default_factory=dict)

    # Resultados
    metrics: dict = field(default_factory=dict)

    # Artifacts
    model_path: str = ""
    backtest_path: str = ""

    # Meta
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    gammaedge_commit: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["started_at"] = d["started_at"].isoformat()
        if d["completed_at"]:
            d["completed_at"] = d["completed_at"].isoformat()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
