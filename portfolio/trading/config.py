# Live Trading Configuration
# ==========================
"""
Configuración específica para trading real.

V1: EOD → Next Open con ETFs ultra líquidos (VOO, QQQ)
V2: Top liquids US equities
V3: Growth / IPOs

GARANTÍAS:
- Labels con costos de ejecución realistas
- Tradability filters enforced
- Calendar-aware execution windows
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

import polars as pl

logger = logging.getLogger(__name__)


# =============================================================================
# Decision/Execution Policies
# =============================================================================


class ExecutionPolicy(Enum):
    """Políticas de decisión/ejecución soportadas."""

    # V1: Para ETFs ultra líquidos
    EOD_NEXT_OPEN = "eod_next_open"

    # Variantes
    EOD_NEXT_VWAP = "eod_next_vwap"
    PREMARKET_OPEN = "premarket_open"


@dataclass
class TradingConfig:
    """
    Configuración completa para un setup de trading.

    V1 Default: EOD → Next Open para VOO/QQQ
    """

    # =========================================================================
    # Timing
    # =========================================================================

    policy: ExecutionPolicy = ExecutionPolicy.EOD_NEXT_OPEN

    # Decision: close(t) + offset
    decision_offset_minutes: int = 5  # 5 min después del close

    # Execution: open(t+1) + offset
    execution_offset_minutes: int = 1  # 1 min después del open

    # Holding period
    holding_days: int = 1  # 1 día = open-to-open

    # =========================================================================
    # Costs (en basis points)
    # =========================================================================

    # Modelo simple (para ETFs ultra líquidos)
    cost_per_side_bps: float = 3.0  # Conservador: 3 bps
    round_trip_cost_bps: float = 6.0  # 2 × 3 bps

    # Modelo liquidity-aware (para acciones)
    use_liquidity_cost: bool = False
    base_cost_bps: float = 5.0  # Base para acciones
    liquidity_factor: float = 50.0  # 50 / sqrt(dollar_vol_in_millions)
    min_cost_bps: float = 2.0
    max_cost_bps: float = 25.0  # Cap: si supera → no tradable

    # =========================================================================
    # Tradability Filters
    # =========================================================================

    min_price: float = 1.0  # ETFs: $1, Acciones: $5
    min_dollar_volume_20d: float = 100_000_000  # $100M para ETFs
    require_clean_quality: bool = True
    exclude_disputed: bool = True
    exclude_event_days: bool = True

    # =========================================================================
    # Labels
    # =========================================================================

    # Forward return calculado como:
    # open(t+1+holding) / open(t+1) - 1 - round_trip_cost
    include_costs_in_label: bool = True

    # =========================================================================
    # Universe
    # =========================================================================

    universe: str = "etf_liquid"  # etf_liquid, top_liquid, growth
    tickers: list[str] = field(default_factory=lambda: ["VOO", "QQQ"])

    # IPO warmup (solo aplica a growth)
    ipo_warmup_days: int = 60

    def get_cost_bps(self, dollar_volume_20d: float | None = None) -> float:
        """
        Calcula el costo por lado en bps.

        Args:
            dollar_volume_20d: Dollar volume promedio 20 días (en USD)
        """
        if not self.use_liquidity_cost or dollar_volume_20d is None:
            return self.cost_per_side_bps

        # Modelo: base + factor / sqrt(dv_millions)
        dv_millions = dollar_volume_20d / 1_000_000
        if dv_millions <= 0:
            return self.max_cost_bps

        import math

        cost = self.base_cost_bps + self.liquidity_factor / math.sqrt(dv_millions)
        return max(self.min_cost_bps, min(self.max_cost_bps, cost))

    def is_tradable(self, dollar_volume_20d: float | None = None) -> bool:
        """Verifica si el costo está dentro del cap."""
        cost = self.get_cost_bps(dollar_volume_20d)
        return cost < self.max_cost_bps


# =============================================================================
# Pre-configured Setups
# =============================================================================


def get_v1_etf_config() -> TradingConfig:
    """V1: ETFs ultra líquidos (VOO, QQQ)."""
    return TradingConfig(
        policy=ExecutionPolicy.EOD_NEXT_OPEN,
        decision_offset_minutes=5,
        execution_offset_minutes=1,
        holding_days=1,
        cost_per_side_bps=3.0,
        round_trip_cost_bps=6.0,
        use_liquidity_cost=False,
        min_price=1.0,
        min_dollar_volume_20d=100_000_000,
        universe="etf_liquid",
        tickers=["VOO", "QQQ"],
    )


def get_v2_top_liquid_config() -> TradingConfig:
    """V2: Top líquidos US equities."""
    return TradingConfig(
        policy=ExecutionPolicy.EOD_NEXT_OPEN,
        decision_offset_minutes=5,
        execution_offset_minutes=1,
        holding_days=5,  # 5 días para reducir turnover
        cost_per_side_bps=5.0,
        round_trip_cost_bps=10.0,
        use_liquidity_cost=True,
        base_cost_bps=5.0,
        liquidity_factor=50.0,
        min_cost_bps=2.0,
        max_cost_bps=25.0,
        min_price=5.0,
        min_dollar_volume_20d=20_000_000,
        universe="top_liquid",
        tickers=[],  # Se define dinámicamente
    )


def get_v3_growth_config() -> TradingConfig:
    """V3: Growth / IPOs (más restrictivo)."""
    return TradingConfig(
        policy=ExecutionPolicy.EOD_NEXT_OPEN,
        decision_offset_minutes=5,
        execution_offset_minutes=1,
        holding_days=5,
        cost_per_side_bps=15.0,  # Más alto
        round_trip_cost_bps=30.0,
        use_liquidity_cost=True,
        base_cost_bps=10.0,
        liquidity_factor=100.0,  # Más sensible a liquidez
        min_cost_bps=5.0,
        max_cost_bps=50.0,
        min_price=10.0,
        min_dollar_volume_20d=10_000_000,
        universe="growth",
        tickers=[],
        ipo_warmup_days=60,  # No tocar hasta 60 días post-IPO
    )


# =============================================================================
# Label Builder (con costos realistas)
# =============================================================================


class LabelBuilder:
    """
    Construye labels de trading con costos de ejecución.

    IMPORTANTE:
    - Label = forward return DESPUÉS de costos
    - Horizonte = open(t+1) → open(t+1+holding)
    - NO usar close-to-close si ejecutas en open

    Example:
        >>> builder = LabelBuilder(get_v1_etf_config())
        >>> df = builder.add_labels(df_ohlcv)
        >>> # Ahora df tiene columna 'label' = open-to-open return - costs
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    def add_labels(
        self,
        df: pl.DataFrame,
        open_col: str = "open",
        date_col: str = "date",
        ticker_col: str = "ticker",
        dollar_volume_col: str | None = "dollar_volume_20d",
    ) -> pl.DataFrame:
        """
        Añade columna 'label' con forward returns netos de costos.

        Label para holding=1:
            y_t = open(t+2) / open(t+1) - 1 - round_trip_cost

        Label para holding=5:
            y_t = open(t+6) / open(t+1) - 1 - round_trip_cost
        """
        holding = self.config.holding_days

        # Shift opens para calcular forward return
        # entry = open(t+1), exit = open(t+1+holding)
        df = df.sort([ticker_col, date_col])

        df = df.with_columns(
            [
                pl.col(open_col).shift(-1).over(ticker_col).alias("open_entry"),
                pl.col(open_col).shift(-(1 + holding)).over(ticker_col).alias("open_exit"),
            ]
        )

        # Calcular return bruto
        df = df.with_columns(
            ((pl.col("open_exit") / pl.col("open_entry")) - 1).alias("gross_return")
        )

        # Calcular costos
        if self.config.use_liquidity_cost and dollar_volume_col and dollar_volume_col in df.columns:
            # Modelo liquidity-aware
            df = df.with_columns(self._compute_cost_expr(dollar_volume_col).alias("cost_bps"))
        else:
            # Costo fijo
            df = df.with_columns(pl.lit(self.config.round_trip_cost_bps).alias("cost_bps"))

        # Label = gross return - round trip cost
        if self.config.include_costs_in_label:
            df = df.with_columns(
                (pl.col("gross_return") - pl.col("cost_bps") / 10000).alias("label")
            )
        else:
            df = df.with_columns(pl.col("gross_return").alias("label"))

        return df

    def _compute_cost_expr(self, dollar_volume_col: str) -> pl.Expr:
        """Expresión para calcular costo liquidity-aware."""
        base = self.config.base_cost_bps
        factor = self.config.liquidity_factor
        min_cost = self.config.min_cost_bps
        max_cost = self.config.max_cost_bps

        # cost = base + factor / sqrt(dv_millions)
        dv_millions = pl.col(dollar_volume_col) / 1_000_000

        cost = base + factor / dv_millions.sqrt()

        # Clip y multiplicar por 2 para round-trip
        return cost.clip(min_cost, max_cost) * 2


# =============================================================================
# Tradability Filters
# =============================================================================


class TradabilityFilter:
    """
    Aplica filtros de tradability a un dataset.

    Filtra:
    - quality_flag != clean
    - recon_flag == disputed
    - event_day
    - price < min
    - dollar_volume < min
    - corporate action proximity
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    def apply(
        self,
        df: pl.DataFrame,
        date_col: str = "date",
        ticker_col: str = "ticker",
    ) -> pl.DataFrame:
        """
        Aplica todos los filtros de tradability.

        Retorna DataFrame con solo filas tradables.
        """
        original_count = df.height

        # Price filter
        if "close" in df.columns:
            df = df.filter(pl.col("close") >= self.config.min_price)

        # Dollar volume filter
        if "dollar_volume_20d" in df.columns:
            df = df.filter(pl.col("dollar_volume_20d") >= self.config.min_dollar_volume_20d)

        # Quality flag
        if self.config.require_clean_quality and "quality_flag" in df.columns:
            df = df.filter(pl.col("quality_flag") == "clean")

        # Disputed reconciliation
        if self.config.exclude_disputed and "recon_flag" in df.columns:
            df = df.filter(pl.col("recon_flag") != "disputed")

        # Event days
        if self.config.exclude_event_days and "is_event_day" in df.columns:
            df = df.filter(~pl.col("is_event_day"))

        filtered_count = df.height
        logger.info(
            "Tradability filter: %d → %d rows (%.1f%% passed)",
            original_count,
            filtered_count,
            100 * filtered_count / max(original_count, 1),
        )

        return df

    def add_tradability_flag(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Añade columna 'is_tradable' sin filtrar.

        Útil para análisis.
        """
        conditions = []

        if "close" in df.columns:
            conditions.append(pl.col("close") >= self.config.min_price)

        if "dollar_volume_20d" in df.columns:
            conditions.append(pl.col("dollar_volume_20d") >= self.config.min_dollar_volume_20d)

        if self.config.require_clean_quality and "quality_flag" in df.columns:
            conditions.append(pl.col("quality_flag") == "clean")

        if self.config.exclude_disputed and "recon_flag" in df.columns:
            conditions.append(pl.col("recon_flag") != "disputed")

        if self.config.exclude_event_days and "is_event_day" in df.columns:
            conditions.append(~pl.col("is_event_day"))

        if not conditions:
            return df.with_columns(pl.lit(True).alias("is_tradable"))

        combined = conditions[0]
        for cond in conditions[1:]:
            combined = combined & cond

        return df.with_columns(combined.alias("is_tradable"))


# =============================================================================
# Execution Checks
# =============================================================================


class ExecutionChecks:
    """
    Validaciones específicas para el pipeline de ejecución.
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    def check_calendar_chain(
        self,
        df: pl.DataFrame,
        trading_days: list[date],
        date_col: str = "date",
    ) -> pl.DataFrame:
        """
        Verifica que existan los trading days necesarios.

        Para cada t, deben existir t+1 y t+1+holding como trading days.
        """
        trading_set = set(trading_days)
        holding = self.config.holding_days

        def has_chain(d: date) -> bool:
            for offset in range(1, 2 + holding):
                check_date = d + timedelta(days=offset)
                # Buscar el siguiente trading day
                while check_date not in trading_set:
                    check_date += timedelta(days=1)
                    if (check_date - d).days > offset + 10:  # Safety
                        return False
            return True

        dates = df.select(pl.col(date_col).unique()).to_series().to_list()
        valid_dates = [d for d in dates if has_chain(d)]

        df_filtered = df.filter(pl.col(date_col).is_in(valid_dates))

        logger.info("Calendar chain check: %d → %d dates valid", len(dates), len(valid_dates))

        return df_filtered

    def check_execution_availability(
        self,
        df: pl.DataFrame,
        open_col: str = "open",
        ticker_col: str = "ticker",
        date_col: str = "date",
    ) -> pl.DataFrame:
        """
        Verifica que open(t+1) y open(t+2) no sean null.
        """
        holding = self.config.holding_days

        df = df.sort([ticker_col, date_col])

        df = df.with_columns(
            [
                pl.col(open_col).shift(-1).over(ticker_col).alias("_open_t1"),
                pl.col(open_col).shift(-(1 + holding)).over(ticker_col).alias("_open_exit"),
            ]
        )

        df_valid = df.filter(
            pl.col("_open_t1").is_not_null() & pl.col("_open_exit").is_not_null()
        ).drop(["_open_t1", "_open_exit"])

        logger.info(
            "Execution availability: %d → %d rows with valid opens", df.height, df_valid.height
        )

        return df_valid

    def check_corporate_action_proximity(
        self,
        df: pl.DataFrame,
        df_actions: pl.DataFrame,
        date_col: str = "date",
        ticker_col: str = "ticker",
    ) -> pl.DataFrame:
        """
        Marca samples que tienen corporate action en t+1 a t+1+holding.

        Añade columna 'event_proximity' = True si hay action cerca.
        """
        holding = self.config.holding_days

        # Crear set de (ticker, date) con actions
        if df_actions.height == 0:
            return df.with_columns(pl.lit(False).alias("event_proximity"))

        action_dates = set(
            (row["ticker"], row["ex_date"]) for row in df_actions.iter_rows(named=True)
        )

        def has_action_in_window(ticker: str, d: date) -> bool:
            for offset in range(1, 2 + holding):
                check_date = d + timedelta(days=offset)
                if (ticker, check_date) in action_dates:
                    return True
            return False

        # Esto es lento para datasets grandes; en producción usar join
        # Por ahora: approach funcional
        df = df.with_columns(
            pl.struct([ticker_col, date_col])
            .map_elements(
                lambda row: has_action_in_window(row[ticker_col], row[date_col]),
                return_dtype=pl.Boolean,
            )
            .alias("event_proximity")
        )

        proximity_count = df.filter(pl.col("event_proximity")).height
        logger.info("Corporate action proximity: %d samples flagged", proximity_count)

        return df
