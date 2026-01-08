# Normalization Package Exports
"""
Data normalization pipeline components.

- corporate_actions: Processor for splits/dividends adjustments
- reconciliation: Cross-provider data validation
"""

from portfolio.io.normalization.corporate_actions import (
    ADJUSTMENT_VERSION,
    AdjustmentResult,
    CorporateActionsProcessor,
    adjust_for_dividend,
    adjust_for_split,
)
from portfolio.io.normalization.reconciliation import (
    CrossProviderReconciler,
    ReconciliationConfig,
    ReconciliationResult,
    reconcile_providers,
)

__all__ = [
    # Corporate Actions
    "ADJUSTMENT_VERSION",
    "AdjustmentResult",
    "CorporateActionsProcessor",
    "adjust_for_dividend",
    "adjust_for_split",
    # Reconciliation
    "CrossProviderReconciler",
    "ReconciliationConfig",
    "ReconciliationResult",
    "reconcile_providers",
]
