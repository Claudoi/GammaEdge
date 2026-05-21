# Data Lake Schemas and Data Contracts — Production Grade
# =========================================================
"""
Definiciones canónicas de schemas para el Data Lake de GammaEdge.

PRINCIPIOS:
1. Separación raw/adjusted: Nunca confiar en ajustes del provider
2. Canonical naming: Campos consistentes entre todos los providers
3. Audit trail: Trazabilidad completa de origen y transformaciones
4. Temporal consistency: Timestamps explícitos para evitar leakage
"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

import polars as pl

# =============================================================================
# Type Aliases
# =============================================================================

ProviderName = Literal["polygon", "alpaca", "tiingo", "yahoo"]
AssetClass = Literal["equity", "etf", "adr", "index", "crypto"]
ActionType = Literal["split", "dividend", "spinoff", "rights_issue", "stock_dividend"]
DataQualityFlag = Literal["clean", "suspect", "failed"]

# =============================================================================
# OHLCV RAW Schema (sin ajustar — del provider tal cual)
# =============================================================================

SCHEMA_BARS_RAW: dict[str, type[pl.DataType]] = {
    # Identifiers
    "date": pl.Date,
    "ticker": pl.Utf8,
    # Price data (RAW — sin modificar del provider)
    "open_raw": pl.Float64,
    "high_raw": pl.Float64,
    "low_raw": pl.Float64,
    "close_raw": pl.Float64,
    "volume_raw": pl.Float64,
    # Audit fields
    "provider": pl.Utf8,
    "ingestion_run_id": pl.Utf8,  # UUID de la ejecución
    "ingested_at": pl.Datetime,  # UTC
}

SCHEMA_BARS_RAW_ORDERED = [
    ("date", pl.Date),
    ("ticker", pl.Utf8),
    ("open_raw", pl.Float64),
    ("high_raw", pl.Float64),
    ("low_raw", pl.Float64),
    ("close_raw", pl.Float64),
    ("volume_raw", pl.Float64),
    ("provider", pl.Utf8),
    ("ingestion_run_id", pl.Utf8),
    ("ingested_at", pl.Datetime),
]

# =============================================================================
# OHLCV Adjusted Schema (generado por NUESTRO pipeline, no provider)
# =============================================================================

SCHEMA_BARS_ADJUSTED: dict[str, type[pl.DataType]] = {
    # Identifiers
    "date": pl.Date,
    "ticker": pl.Utf8,
    # Adjusted prices (calculados por nosotros)
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    # Adjustment info
    "adj_factor": pl.Float64,  # Factor acumulado (calculado por nosotros)
    "adjustment_version": pl.Utf8,  # Versión del algoritmo: "v1.0.0"
    # Data quality
    "quality_flag": pl.Utf8,  # clean, suspect, failed
    # Audit fields
    "source_provider": pl.Utf8,  # Provider original del raw
    "source_run_id": pl.Utf8,  # ingestion_run_id del raw
    "adjusted_at": pl.Datetime,  # Cuándo se aplicó el ajuste
}

SCHEMA_BARS_ADJUSTED_ORDERED = [
    ("date", pl.Date),
    ("ticker", pl.Utf8),
    ("open", pl.Float64),
    ("high", pl.Float64),
    ("low", pl.Float64),
    ("close", pl.Float64),
    ("volume", pl.Float64),
    ("adj_factor", pl.Float64),
    ("adjustment_version", pl.Utf8),
    ("quality_flag", pl.Utf8),
    ("source_provider", pl.Utf8),
    ("source_run_id", pl.Utf8),
    ("adjusted_at", pl.Datetime),
]

# =============================================================================
# Corporate Actions Schema (explícitas, verificables)
# =============================================================================

SCHEMA_CORPORATE_ACTIONS: dict[str, type[pl.DataType]] = {
    "ticker": pl.Utf8,
    "ex_date": pl.Date,  # Fecha ex-dividendo o efectiva del split
    "record_date": pl.Date,  # Record date (puede ser null)
    "payment_date": pl.Date,  # Payment date para dividendos
    "action_type": pl.Utf8,  # split, dividend, stock_dividend, spinoff
    # Split info
    "split_from": pl.Float64,  # Ej: 1 (de 1-for-2)
    "split_to": pl.Float64,  # Ej: 2 (de 1-for-2)
    "split_ratio": pl.Float64,  # split_to / split_from = 2.0
    # Dividend info
    "dividend_amount": pl.Float64,  # Monto en cash
    "dividend_type": pl.Utf8,  # regular, special, return_of_capital
    "currency": pl.Utf8,
    # Verification
    "provider": pl.Utf8,
    "verified": pl.Boolean,  # True si cross-checked con otro provider
    "verification_provider": pl.Utf8,  # Qué provider usamos para verificar
    # Audit
    "ingestion_run_id": pl.Utf8,
    "ingested_at": pl.Datetime,
}

SCHEMA_CORPORATE_ACTIONS_ORDERED = [
    ("ticker", pl.Utf8),
    ("ex_date", pl.Date),
    ("record_date", pl.Date),
    ("payment_date", pl.Date),
    ("action_type", pl.Utf8),
    ("split_from", pl.Float64),
    ("split_to", pl.Float64),
    ("split_ratio", pl.Float64),
    ("dividend_amount", pl.Float64),
    ("dividend_type", pl.Utf8),
    ("currency", pl.Utf8),
    ("provider", pl.Utf8),
    ("verified", pl.Boolean),
    ("verification_provider", pl.Utf8),
    ("ingestion_run_id", pl.Utf8),
    ("ingested_at", pl.Datetime),
]

# =============================================================================
# Instrument Master Schema (ID estable, ticker cambia)
# =============================================================================

SCHEMA_INSTRUMENT_MASTER: dict[str, type[pl.DataType]] = {
    "instrument_id": pl.Utf8,  # UUID interno - NUNCA cambia
    "current_ticker": pl.Utf8,  # Ticker actual
    "name": pl.Utf8,
    "asset_class": pl.Utf8,  # equity, etf, adr, index, crypto
    "primary_exchange": pl.Utf8,
    # External identifiers
    "figi": pl.Utf8,  # OpenFIGI
    "composite_figi": pl.Utf8,
    "cik": pl.Utf8,  # SEC
    "isin": pl.Utf8,
    "cusip": pl.Utf8,
    # Lifecycle
    "ipo_date": pl.Date,
    "delisted_date": pl.Date,  # null si activo
    "is_active": pl.Boolean,
    # Audit
    "created_at": pl.Datetime,
    "updated_at": pl.Datetime,
}

# =============================================================================
# Ticker History Schema (mapeo ticker → instrument a través del tiempo)
# =============================================================================

SCHEMA_TICKER_HISTORY: dict[str, type[pl.DataType]] = {
    "instrument_id": pl.Utf8,  # FK a instrument_master
    "ticker": pl.Utf8,
    "valid_from": pl.Date,
    "valid_to": pl.Date,  # null si actual
    "exchange": pl.Utf8,
    "reason": pl.Utf8,  # ipo, ticker_change, merger, delisting
}

# =============================================================================
# Universe History Schema (membership para evitar survivorship bias)
# =============================================================================

SCHEMA_UNIVERSE_HISTORY: dict[str, type[pl.DataType]] = {
    "instrument_id": pl.Utf8,
    "ticker": pl.Utf8,  # Ticker en esa fecha
    "universe": pl.Utf8,  # sp500, nasdaq100, all_us, etc.
    "start_date": pl.Date,
    "end_date": pl.Date,  # null si todavía en universo
    "reason_added": pl.Utf8,
    "reason_removed": pl.Utf8,  # delisted, dropped, merged
    "merged_into_id": pl.Utf8,  # instrument_id si fue merger
}

SCHEMA_UNIVERSE_HISTORY_ORDERED = [
    ("instrument_id", pl.Utf8),
    ("ticker", pl.Utf8),
    ("universe", pl.Utf8),
    ("start_date", pl.Date),
    ("end_date", pl.Date),
    ("reason_added", pl.Utf8),
    ("reason_removed", pl.Utf8),
    ("merged_into_id", pl.Utf8),
]

# =============================================================================
# Provider Reconciliation Schema
# =============================================================================

SCHEMA_RECONCILIATION: dict[str, type[pl.DataType]] = {
    "date": pl.Date,
    "ticker": pl.Utf8,
    # Primary provider values
    "primary_provider": pl.Utf8,
    "primary_close": pl.Float64,
    "primary_volume": pl.Float64,
    # Secondary provider values
    "secondary_provider": pl.Utf8,
    "secondary_close": pl.Float64,
    "secondary_volume": pl.Float64,
    # Divergences
    "price_diff_pct": pl.Float64,  # (p1 - p2) / p2
    "volume_ratio": pl.Float64,  # v1 / v2
    # Flags
    "is_suspect": pl.Boolean,  # True si divergencia > umbral
    "suspect_reason": pl.Utf8,
    "reconciled_at": pl.Datetime,
}

# =============================================================================
# Ingestion Manifest Schema (para reproducibilidad)
# =============================================================================

SCHEMA_INGESTION_MANIFEST: dict[str, type[pl.DataType] | pl.DataType] = {
    "manifest_id": pl.Utf8,  # UUID
    "ingestion_run_id": pl.Utf8,  # UUID de esta ejecución
    # What
    "data_type": pl.Utf8,  # bars_raw, corporate_actions, etc.
    "tickers": pl.List(pl.Utf8),
    "start_date": pl.Date,
    "end_date": pl.Date,
    # How
    "provider": pl.Utf8,
    "provider_params": pl.Utf8,  # JSON con parámetros usados
    # When
    "started_at": pl.Datetime,
    "completed_at": pl.Datetime,
    # Integrity
    "row_count": pl.Int64,
    "schema_hash": pl.Utf8,
    "data_hash": pl.Utf8,  # Hash del contenido
    # Reproducibility
    "gammaedge_version": pl.Utf8,
    "gammaedge_commit": pl.Utf8,
    "python_version": pl.Utf8,
    "polars_version": pl.Utf8,
    # Output
    "output_path": pl.Utf8,
}

# =============================================================================
# Feature Store Schema (con availability timestamps)
# =============================================================================

SCHEMA_FEATURES_DAILY: dict[str, type[pl.DataType]] = {
    # Identifiers
    "date": pl.Date,  # Fecha de la observación
    "ticker": pl.Utf8,
    # Timing (CRÍTICO para evitar leakage)
    "feature_time": pl.Datetime,  # Cuándo se calculó el feature
    "available_at": pl.Datetime,  # Cuándo está disponible para decisión
    # Returns (multiple horizons)
    "ret_1d": pl.Float64,
    "ret_5d": pl.Float64,
    "ret_20d": pl.Float64,
    "log_ret_1d": pl.Float64,
    # Momentum
    "momentum_12_1": pl.Float64,
    "momentum_zscore": pl.Float64,
    # Volatility
    "realized_vol_5d": pl.Float64,
    "realized_vol_20d": pl.Float64,
    "parkinson_vol": pl.Float64,
    "garman_klass_vol": pl.Float64,
    "atr_14": pl.Float64,
    # Liquidity
    "rel_volume_20d": pl.Float64,
    "amihud_illiq": pl.Float64,
    "dollar_volume": pl.Float64,
    # Cross-sectional ranks
    "ret_1d_rank": pl.Float64,
    "momentum_rank": pl.Float64,
    # Risk
    "beta_60d": pl.Float64,
    "drawdown_20d": pl.Float64,
    # Higher moments
    "skew_20d": pl.Float64,
    "kurtosis_20d": pl.Float64,
    # Metadata
    "feature_version": pl.Utf8,  # Semántico: "1.2.0"
    "universe_as_of": pl.Utf8,  # Qué universo se usó para ranks
}

# =============================================================================
# Quality Report Schema (mejorado)
# =============================================================================

SCHEMA_QUALITY_REPORT: dict[str, type[pl.DataType]] = {
    "report_id": pl.Utf8,
    "ingestion_run_id": pl.Utf8,
    "check_name": pl.Utf8,
    "severity": pl.Utf8,  # error, warning, info
    "passed": pl.Boolean,
    "message": pl.Utf8,
    "details": pl.Utf8,  # JSON
    "affected_rows": pl.Int64,
    "checked_at": pl.Datetime,
}

# =============================================================================
# Helper Functions
# =============================================================================


def generate_run_id() -> str:
    """Genera un UUID para identificar una ejecución de ingesta."""
    return str(uuid4())


def create_empty_bars_raw_df() -> pl.DataFrame:
    """Crea un DataFrame vacío con el schema de bars_raw."""
    return pl.DataFrame(schema=SCHEMA_BARS_RAW_ORDERED)


def create_empty_bars_adjusted_df() -> pl.DataFrame:
    """Crea un DataFrame vacío con el schema de bars_adjusted."""
    return pl.DataFrame(schema=SCHEMA_BARS_ADJUSTED_ORDERED)


def create_empty_corporate_actions_df() -> pl.DataFrame:
    """Crea un DataFrame vacío con el schema de corporate_actions."""
    return pl.DataFrame(schema=SCHEMA_CORPORATE_ACTIONS_ORDERED)


def create_empty_universe_history_df() -> pl.DataFrame:
    """Crea un DataFrame vacío con el schema de universe_history."""
    return pl.DataFrame(schema=SCHEMA_UNIVERSE_HISTORY_ORDERED)


# Backward compatibility aliases
SCHEMA_BARS_1D = SCHEMA_BARS_ADJUSTED  # Legacy
create_empty_bars_df = create_empty_bars_adjusted_df  # Legacy


def create_empty_reference_df() -> pl.DataFrame:
    """Crea un DataFrame vacío con el schema de reference."""
    schema = [
        ("ticker", pl.Utf8),
        ("name", pl.Utf8),
        ("exchange", pl.Utf8),
        ("asset_class", pl.Utf8),
        ("sector", pl.Utf8),
        ("industry", pl.Utf8),
        ("figi", pl.Utf8),
        ("cik", pl.Utf8),
        ("ipo_date", pl.Date),
        ("delisted_date", pl.Date),
        ("is_active", pl.Boolean),
        ("last_updated", pl.Datetime),
    ]
    return pl.DataFrame(schema=schema)


def validate_schema(
    df: pl.DataFrame,
    expected_schema: dict[str, type[pl.DataType]],
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """
    Valida que un DataFrame tenga el schema esperado.

    Args:
        df: DataFrame a validar
        expected_schema: Schema esperado
        strict: Si True, falla por columnas extra

    Returns:
        (passed, errors)
    """
    errors = []

    # Required columns
    for col, dtype in expected_schema.items():
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
        elif df[col].dtype != dtype:
            errors.append(f"Column {col}: expected {dtype}, got {df[col].dtype}")

    # Extra columns
    extra_cols = set(df.columns) - set(expected_schema.keys())
    if extra_cols:
        msg = f"Extra columns: {extra_cols}"
        if strict:
            errors.append(msg)
        else:
            errors.append(f"(warning) {msg}")

    passed = len(errors) == 0 or all("warning" in e.lower() for e in errors)
    return passed, errors


def compute_data_hash(df: pl.DataFrame) -> str:
    """Calcula hash del contenido de un DataFrame para integridad."""
    import hashlib

    # Serializar a bytes y hashear
    content = df.write_ipc(None).getvalue()
    return hashlib.sha256(content).hexdigest()[:16]


def compute_schema_hash(schema: dict[str, type[pl.DataType]]) -> str:
    """Calcula hash del schema para versionado."""
    import hashlib

    schema_str = str(sorted(schema.items()))
    return hashlib.sha256(schema_str.encode()).hexdigest()[:8]
