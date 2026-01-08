# Feature Registry — Anti-Leakage Feature Definitions
# ====================================================
"""
Registro de features con timestamps de disponibilidad.

PROBLEMA CRÍTICO:
Sin `available_at`, es trivial introducir leakage:
- Usar close(t) para decidir en t (pero close no está hasta market_close)
- Ranks cross-sectional con instrumentos que no existían
- Corporate actions con info que se publicó después

SOLUCIÓN:
Cada feature declara EXPLÍCITAMENTE cuándo está disponible.
El pipeline enforza que solo uses features disponibles antes de decision_time.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from portfolio.core.compat import UTC

logger = logging.getLogger(__name__)


class AvailabilityTime(Enum):
    """
    Cuándo está disponible un feature para decisión.

    IMPORTANTE: Esto define cuándo PUEDES usar el feature,
    no cuándo se CALCULA.
    """

    # Disponible al cierre del mercado del día t
    MARKET_CLOSE = "market_close"

    # Disponible al apertura del día t+1 (para decisiones overnight)
    NEXT_OPEN = "t+1_open"

    # Disponible al cierre del día t+1 (para features que requieren settle)
    NEXT_CLOSE = "t+1_close"

    # Disponible inmediatamente (datos de referencia, no cambian intraday)
    IMMEDIATE = "immediate"

    # Disponible con delay específico (ej: earnings se publican post-close)
    DELAYED = "delayed"


@dataclass
class FeatureDefinition:
    """
    Definición completa de un feature para el registry.

    TODOS los features deben tener:
    - available_at: Cuándo está disponible para decisión
    - lookback: Cuántos días de historia necesita
    - dependencies: Otros features que requiere
    """

    # Identificación
    name: str
    version: str = "1.0.0"

    # Timing (CRÍTICO para evitar leakage)
    available_at: AvailabilityTime = AvailabilityTime.MARKET_CLOSE
    delay_hours: float = 0.0  # Para DELAYED, cuántas horas después del close

    # Requisitos de datos
    lookback_days: int = 0  # Días de historia que necesita
    min_periods: int = 1  # Mínimo de observaciones válidas
    requires_columns: list[str] = field(default_factory=lambda: ["close"])

    # Dependencias
    requires_features: list[str] = field(default_factory=list)

    # Cálculo
    params: dict[str, Any] = field(default_factory=dict)

    # Metadata
    description: str = ""
    category: str = "general"  # momentum, volatility, liquidity, etc.
    frequency: str = "daily"  # daily, weekly, monthly

    # Cross-sectional
    is_cross_sectional: bool = False  # True si usa info del universo
    universe_required: bool = False  # True si necesita universo as-of

    def get_availability_timestamp(
        self,
        observation_date: date,
        exchange: str = "NYSE",
    ) -> datetime:
        """
        Calcula el timestamp exacto en que este feature está disponible.

        Args:
            observation_date: Fecha de la observación (t)
            exchange: Exchange para determinar horarios

        Returns:
            Datetime (UTC) cuando el feature está disponible
        """
        et = ZoneInfo("America/New_York")

        if self.available_at == AvailabilityTime.MARKET_CLOSE:
            # NYSE cierra 16:00 ET
            dt = datetime(
                observation_date.year,
                observation_date.month,
                observation_date.day,
                16,
                0,
                0,
                tzinfo=et,
            )

        elif self.available_at == AvailabilityTime.NEXT_OPEN:
            # Día siguiente 9:30 ET
            next_day = observation_date + timedelta(days=1)
            dt = datetime(
                next_day.year,
                next_day.month,
                next_day.day,
                9,
                30,
                0,
                tzinfo=et,
            )

        elif self.available_at == AvailabilityTime.NEXT_CLOSE:
            next_day = observation_date + timedelta(days=1)
            dt = datetime(
                next_day.year,
                next_day.month,
                next_day.day,
                16,
                0,
                0,
                tzinfo=et,
            )

        elif self.available_at == AvailabilityTime.IMMEDIATE:
            # Disponible a las 00:00 del día
            dt = datetime(
                observation_date.year,
                observation_date.month,
                observation_date.day,
                0,
                0,
                0,
                tzinfo=et,
            )

        elif self.available_at == AvailabilityTime.DELAYED:
            base = datetime(
                observation_date.year,
                observation_date.month,
                observation_date.day,
                16,
                0,
                0,
                tzinfo=et,
            )
            dt = base + timedelta(hours=self.delay_hours)

        else:
            raise ValueError(f"Unknown availability: {self.available_at}")

        return dt.astimezone(UTC)

    def validate_no_leakage(
        self,
        decision_time: datetime,
        observation_date: date,
    ) -> bool:
        """
        Verifica que el feature esté disponible antes de decision_time.

        Returns:
            True si es safe usar este feature para decidir en decision_time
        """
        available = self.get_availability_timestamp(observation_date)
        return available <= decision_time

    def to_dict(self) -> dict:
        d = asdict(self)
        d["available_at"] = self.available_at.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FeatureDefinition:
        d = d.copy()
        d["available_at"] = AvailabilityTime(d["available_at"])
        return cls(**d)


class FeatureRegistry:
    """
    Catálogo versionado de features con enforcement de availability.

    GARANTÍAS:
    1. Cada feature tiene definición explícita
    2. available_at es obligatorio
    3. El pipeline puede verificar leakage
    4. Versionado semántico para reproducibilidad

    Example:
        >>> registry = FeatureRegistry()
        >>>
        >>> # Registrar features
        >>> registry.register(FeatureDefinition(
        ...     name="ret_1d",
        ...     available_at=AvailabilityTime.MARKET_CLOSE,
        ...     lookback_days=1,
        ... ))
        >>>
        >>> # Verificar si feature está disponible para decisión
        >>> registry.is_available("ret_1d", decision_time, observation_date)
        >>>
        >>> # Obtener todos los features disponibles para una decisión
        >>> available = registry.get_available_features(decision_time)
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else None
        self._features: dict[str, dict[str, FeatureDefinition]] = {}  # name → {version → def}
        self._current_versions: dict[str, str] = {}  # name → latest version

        if self.path and self.path.exists():
            self._load()

    def register(
        self,
        definition: FeatureDefinition,
        set_current: bool = True,
    ) -> None:
        """
        Registra una definición de feature.

        Args:
            definition: Definición del feature
            set_current: Si True, marca esta versión como la actual
        """
        name = definition.name
        version = definition.version

        if name not in self._features:
            self._features[name] = {}

        if version in self._features[name]:
            logger.warning("Overwriting feature %s version %s", name, version)

        self._features[name][version] = definition

        if set_current:
            self._current_versions[name] = version

        logger.info("Registered feature %s v%s", name, version)

    def get(
        self,
        name: str,
        version: str | None = None,
    ) -> FeatureDefinition | None:
        """
        Obtiene definición de un feature.

        Args:
            name: Nombre del feature
            version: Versión específica (None = current)
        """
        if name not in self._features:
            return None

        if version is None:
            version = self._current_versions.get(name)
            if version is None:
                return None

        return self._features[name].get(version)

    def is_available(
        self,
        feature_name: str,
        decision_time: datetime,
        observation_date: date,
    ) -> bool:
        """
        Verifica si un feature está disponible para una decisión.

        Args:
            feature_name: Nombre del feature
            decision_time: Cuándo se toma la decisión
            observation_date: Fecha de la observación del feature
        """
        definition = self.get(feature_name)
        if definition is None:
            logger.warning("Feature %s not in registry", feature_name)
            return False

        return definition.validate_no_leakage(decision_time, observation_date)

    def get_available_features(
        self,
        decision_time: datetime,
        observation_date: date,
        category: str | None = None,
    ) -> list[str]:
        """
        Obtiene todos los features disponibles para una decisión.

        Args:
            decision_time: Cuándo se toma la decisión
            observation_date: Fecha de observación
            category: Filtrar por categoría
        """
        available = []

        for name, version in self._current_versions.items():
            definition = self._features[name][version]

            if category and definition.category != category:
                continue

            if definition.validate_no_leakage(decision_time, observation_date):
                available.append(name)

        return available

    def validate_feature_set(
        self,
        feature_names: list[str],
        decision_time: datetime,
        observation_date: date,
    ) -> tuple[bool, list[str]]:
        """
        Valida que un conjunto de features no tenga leakage.

        Returns:
            (all_valid, list of features with leakage)
        """
        leaking = []

        for name in feature_names:
            if not self.is_available(name, decision_time, observation_date):
                leaking.append(name)

        return len(leaking) == 0, leaking

    def get_all_current(self) -> list[FeatureDefinition]:
        """Retorna todas las definiciones actuales."""
        return [self._features[name][version] for name, version in self._current_versions.items()]

    def get_dependency_graph(self, feature_name: str) -> dict[str, list[str]]:
        """Construye grafo de dependencias de un feature."""
        graph = {}

        def _add_deps(name: str):
            if name in graph:
                return

            definition = self.get(name)
            if definition is None:
                graph[name] = []
                return

            deps = definition.requires_features
            graph[name] = deps

            for dep in deps:
                _add_deps(dep)

        _add_deps(feature_name)
        return graph

    def compute_hash(self) -> str:
        """Hash del registry para versionado."""
        content = json.dumps(
            {name: self._current_versions[name] for name in sorted(self._current_versions)},
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def save(self) -> None:
        """Persiste el registry."""
        if not self.path:
            return

        self.path.mkdir(parents=True, exist_ok=True)

        data = {
            "features": {
                name: {version: defn.to_dict() for version, defn in versions.items()}
                for name, versions in self._features.items()
            },
            "current_versions": self._current_versions,
        }

        with open(self.path / "registry.json", "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Saved feature registry to %s", self.path)

    def _load(self) -> None:
        """Carga el registry desde disco."""
        registry_file = self.path / "registry.json"
        if not registry_file.exists():
            return

        with open(registry_file) as f:
            data = json.load(f)

        for name, versions in data.get("features", {}).items():
            self._features[name] = {}
            for version, defn_dict in versions.items():
                self._features[name][version] = FeatureDefinition.from_dict(defn_dict)

        self._current_versions = data.get("current_versions", {})

        logger.info("Loaded feature registry: %d features", len(self._current_versions))


# =============================================================================
# Standard Feature Definitions
# =============================================================================


def get_standard_features() -> list[FeatureDefinition]:
    """
    Definiciones estándar de features con availability correcta.

    NOTA: Todos los features basados en close están disponibles
    en MARKET_CLOSE, no antes.
    """
    return [
        # Returns
        FeatureDefinition(
            name="ret_1d",
            version="1.0.0",
            available_at=AvailabilityTime.MARKET_CLOSE,
            lookback_days=1,
            requires_columns=["close"],
            description="1-day simple return",
            category="returns",
        ),
        FeatureDefinition(
            name="ret_5d",
            version="1.0.0",
            available_at=AvailabilityTime.MARKET_CLOSE,
            lookback_days=5,
            requires_columns=["close"],
            description="5-day simple return",
            category="returns",
        ),
        FeatureDefinition(
            name="ret_20d",
            version="1.0.0",
            available_at=AvailabilityTime.MARKET_CLOSE,
            lookback_days=20,
            min_periods=15,
            requires_columns=["close"],
            description="20-day simple return",
            category="returns",
        ),
        # Volatility
        FeatureDefinition(
            name="realized_vol_20d",
            version="1.0.0",
            available_at=AvailabilityTime.MARKET_CLOSE,
            lookback_days=20,
            min_periods=15,
            requires_columns=["close"],
            requires_features=["ret_1d"],
            description="20-day realized volatility",
            category="volatility",
        ),
        FeatureDefinition(
            name="atr_14",
            version="1.0.0",
            available_at=AvailabilityTime.MARKET_CLOSE,
            lookback_days=14,
            min_periods=10,
            requires_columns=["high", "low", "close"],
            description="14-day Average True Range",
            category="volatility",
        ),
        # Momentum
        FeatureDefinition(
            name="momentum_12_1",
            version="1.0.0",
            available_at=AvailabilityTime.MARKET_CLOSE,
            lookback_days=252,
            min_periods=230,
            requires_columns=["close"],
            description="12-month return minus 1-month return",
            category="momentum",
        ),
        FeatureDefinition(
            name="momentum_zscore_20d",
            version="1.0.0",
            available_at=AvailabilityTime.MARKET_CLOSE,
            lookback_days=20,
            requires_features=["ret_1d"],
            description="Z-score of cumulative return over 20 days",
            category="momentum",
        ),
        # Liquidity
        FeatureDefinition(
            name="rel_volume_20d",
            version="1.0.0",
            available_at=AvailabilityTime.MARKET_CLOSE,
            lookback_days=20,
            requires_columns=["volume"],
            description="Today's volume relative to 20-day average",
            category="liquidity",
        ),
        FeatureDefinition(
            name="amihud_illiq_20d",
            version="1.0.0",
            available_at=AvailabilityTime.MARKET_CLOSE,
            lookback_days=20,
            requires_columns=["close", "volume"],
            requires_features=["ret_1d"],
            description="Amihud illiquidity ratio",
            category="liquidity",
        ),
        # Cross-sectional (requieren universo)
        FeatureDefinition(
            name="ret_1d_rank",
            version="1.0.0",
            available_at=AvailabilityTime.MARKET_CLOSE,
            lookback_days=1,
            requires_features=["ret_1d"],
            is_cross_sectional=True,
            universe_required=True,
            description="Percentile rank of 1-day return in universe",
            category="cross_sectional",
        ),
        FeatureDefinition(
            name="momentum_rank",
            version="1.0.0",
            available_at=AvailabilityTime.MARKET_CLOSE,
            lookback_days=252,
            requires_features=["momentum_12_1"],
            is_cross_sectional=True,
            universe_required=True,
            description="Percentile rank of 12-1 momentum in universe",
            category="cross_sectional",
        ),
        # Risk
        FeatureDefinition(
            name="drawdown_20d",
            version="1.0.0",
            available_at=AvailabilityTime.MARKET_CLOSE,
            lookback_days=20,
            requires_columns=["close"],
            description="Drawdown from 20-day high",
            category="risk",
        ),
        FeatureDefinition(
            name="skew_20d",
            version="1.0.0",
            available_at=AvailabilityTime.MARKET_CLOSE,
            lookback_days=20,
            min_periods=15,
            requires_features=["ret_1d"],
            description="20-day rolling skewness",
            category="risk",
        ),
    ]


def create_standard_registry() -> FeatureRegistry:
    """Crea un registry con todos los features estándar."""
    registry = FeatureRegistry()

    for feature_def in get_standard_features():
        registry.register(feature_def)

    return registry
