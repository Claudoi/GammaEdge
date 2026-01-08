# Ingestion Package Exports
"""
Data ingestion pipeline components.

- orchestrator: Multi-provider data orchestrator with fallback
- quality_gates: Automated data quality validation
- corporate_actions: Split/dividend adjustment logic
"""

from portfolio.io.ingestion.orchestrator import (
    DataOrchestrator,
    IngestionResult,
)
from portfolio.io.ingestion.quality_gates import (
    CheckResult,
    QualityGates,
    QualityReport,
    run_quality_gates,
)

__all__ = [
    # Orchestrator
    "DataOrchestrator",
    "IngestionResult",
    # Quality Gates
    "CheckResult",
    "QualityGates",
    "QualityReport",
    "run_quality_gates",
]
