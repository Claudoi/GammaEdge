# Feature Store Package Exports
"""
Feature store components for GammaEdge.

Components:
- registry: Feature definitions with availability timestamps
- as_of_join: Point-in-time joins to prevent leakage
- snapshot: Immutable dataset snapshots for reproducibility
- advanced_features: Quantitative feature computation
"""

from portfolio.features.advanced_features import (
    compute_all_features,
    compute_amihud_illiquidity,
    compute_atr,
    compute_cross_sectional_ranks,
    compute_drawdown,
    compute_momentum_12_1,
    compute_momentum_zscore,
    compute_multi_horizon_returns,
    compute_realized_volatility,
    compute_relative_volume,
    compute_rolling_beta,
    compute_rolling_skewness,
    compute_volatility_of_volatility,
)
from portfolio.io.feature_store.as_of_join import (
    AsOfConfig,
    AsOfJoiner,
    build_as_of_dataset,
)
from portfolio.io.feature_store.registry import (
    AvailabilityTime,
    FeatureDefinition,
    FeatureRegistry,
    create_standard_registry,
    get_standard_features,
)
from portfolio.io.feature_store.snapshot import (
    DatasetSnapshot,
    SnapshotManager,
    TrainingRun,
)

__all__ = [
    # Feature Registry
    "AvailabilityTime",
    "FeatureDefinition",
    "FeatureRegistry",
    "create_standard_registry",
    "get_standard_features",
    # As-Of Joins
    "AsOfConfig",
    "AsOfJoiner",
    "build_as_of_dataset",
    # Snapshots
    "DatasetSnapshot",
    "SnapshotManager",
    "TrainingRun",
    # Advanced Features
    "compute_atr",
    "compute_all_features",
    "compute_amihud_illiquidity",
    "compute_cross_sectional_ranks",
    "compute_drawdown",
    "compute_momentum_12_1",
    "compute_momentum_zscore",
    "compute_multi_horizon_returns",
    "compute_realized_volatility",
    "compute_relative_volume",
    "compute_rolling_beta",
    "compute_rolling_skewness",
    "compute_volatility_of_volatility",
]
