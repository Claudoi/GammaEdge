# Trading Package Exports
"""
Trading configuration and execution components.

- config: Trading policies, cost models, tradability filters
- calendar: Enhanced calendar with early closes and DST
- v1_allocation: QQQ/VOO/BIL allocation system
- v1_features: Feature builder for V1
"""

from portfolio.trading.calendar import (
    EnhancedCalendar,
    TradingSession,
    get_availability_timestamp,
    get_enhanced_calendar,
)
from portfolio.trading.config import (
    ExecutionChecks,
    ExecutionPolicy,
    LabelBuilder,
    TradabilityFilter,
    TradingConfig,
    get_v1_etf_config,
    get_v2_top_liquid_config,
    get_v3_growth_config,
)
from portfolio.trading.v1_allocation import (
    AllocationBacktest,
    AllocationLabelBuilder,
    BacktestResult,
    MeanVarianceOptimizer,
    SampleValidityChecker,
    V1AllocationConfig,
    get_v1_allocation_config,
    run_v1_backtest,
)
from portfolio.trading.v1_features import (
    V1FeatureBuilder,
    V1FeatureConfig,
    build_v1_features,
)

__all__ = [
    # Config
    "ExecutionPolicy",
    "TradingConfig",
    "LabelBuilder",
    "TradabilityFilter",
    "ExecutionChecks",
    "get_v1_etf_config",
    "get_v2_top_liquid_config",
    "get_v3_growth_config",
    # Calendar
    "EnhancedCalendar",
    "TradingSession",
    "get_enhanced_calendar",
    "get_availability_timestamp",
    # V1 Allocation
    "V1AllocationConfig",
    "AllocationLabelBuilder",
    "MeanVarianceOptimizer",
    "AllocationBacktest",
    "BacktestResult",
    "SampleValidityChecker",
    "get_v1_allocation_config",
    "run_v1_backtest",
    # V1 Features
    "V1FeatureConfig",
    "V1FeatureBuilder",
    "build_v1_features",
]
