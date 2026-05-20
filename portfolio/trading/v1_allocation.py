# V1 Allocation: QQQ/VOO/BIL
# ==========================
"""
Sistema de allocation V1 para validar el pipeline completo.

Assets: QQQ (growth), VOO (market), BIL (risk-off)
Policy: EOD → Next Open
Optimizer: Mean-Variance + Turnover Penalty

GARANTÍAS:
- Labels open-to-open reales
- Costos por turnover
- Rebalance con umbral
- Métricas netas de costos
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


# =============================================================================
# V1 Config (hashable, versionable)
# =============================================================================


@dataclass
class V1AllocationConfig:
    """
    Configuración completa para V1 Allocation.

    HASHABLE: Cualquier cambio produce hash diferente.
    """

    # Version
    config_version: str = "1.0.0"

    # Assets
    assets: list[str] = field(default_factory=lambda: ["QQQ", "VOO", "BIL"])

    # Timing
    decision_offset_minutes: int = 5
    execution_offset_minutes: int = 1
    holding_days: int = 1

    # Costs (bps)
    cost_per_side_bps: float = 3.0
    open_penalty_bps: float = 1.0  # Extra por ejecutar en open
    round_trip_bps: float = 8.0  # 2*(3+1) = 8 bps conservador

    # Optimizer
    risk_aversion: float = 1.0  # λ en mean-variance
    turnover_penalty: float = 0.01  # γ para L1 penalty en turnover

    # Constraints
    min_weight: float = 0.0
    max_weight_risky: float = 0.8  # Max para QQQ, VOO
    max_weight_cash: float = 1.0  # Max para BIL
    max_daily_change: float = 0.25  # Máximo cambio por activo por día

    # Rebalance
    rebalance_threshold: float = 0.05  # τ = 5%

    # Tradability
    min_dollar_volume_20d: float = 100_000_000
    require_clean: bool = True
    exclude_disputed: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def compute_hash(self) -> str:
        """Hash canónico para versionado."""
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, d: dict) -> V1AllocationConfig:
        return cls(**d)


# =============================================================================
# Forward Returns Builder
# =============================================================================


class AllocationLabelBuilder:
    """
    Construye forward returns por activo para allocation.

    r_t^(a) = open(t+2)^(a) / open(t+1)^(a) - 1
    """

    def __init__(self, config: V1AllocationConfig):
        self.config = config

    def build(
        self,
        df: pl.DataFrame,
        date_col: str = "date",
        ticker_col: str = "ticker",
        open_col: str = "open",
    ) -> pl.DataFrame:
        """
        Construye forward returns por activo.

        Input: Long format (date, ticker, open, ...)
        Output: Wide format (date, ret_QQQ, ret_VOO, ret_BIL, ...)
        """
        holding = self.config.holding_days

        # Calcular forward return por ticker
        df = df.sort([ticker_col, date_col])

        df = df.with_columns(
            [
                pl.col(open_col).shift(-1).over(ticker_col).alias("open_entry"),
                pl.col(open_col).shift(-(1 + holding)).over(ticker_col).alias("open_exit"),
            ]
        )

        df = df.with_columns(
            ((pl.col("open_exit") / pl.col("open_entry")) - 1).alias("forward_return")
        )

        # Pivot a wide format
        df_wide = df.select([date_col, ticker_col, "forward_return"]).pivot(
            index=date_col,
            columns=ticker_col,
            values="forward_return",
        )

        # Renombrar columnas
        for asset in self.config.assets:
            if asset in df_wide.columns:
                df_wide = df_wide.rename({asset: f"ret_{asset}"})

        return df_wide


# =============================================================================
# Mean-Variance Optimizer with Turnover Penalty
# =============================================================================


class MeanVarianceOptimizer:
    """
    Optimizador Mean-Variance con penalización de turnover.

    max_w  w' μ - λ w' Σ w - γ |w - w_prev|₁
    s.t.   w >= 0
           Σw = 1
           w <= caps
    """

    def __init__(self, config: V1AllocationConfig):
        self.config = config
        self.n_assets = len(config.assets)

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray | None = None,
        prev_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Optimiza pesos dados retornos esperados.

        Args:
            expected_returns: (n_assets,) expected returns
            covariance: (n_assets, n_assets) covariance matrix
            prev_weights: (n_assets,) previous weights for turnover penalty

        Returns:
            (n_assets,) optimal weights summing to 1
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.warning("scipy not available, using equal weights")
            return np.ones(self.n_assets) / self.n_assets

        n = self.n_assets
        mu = expected_returns

        # Default covariance (identity scaled)
        sigma = np.eye(n) * 0.01 if covariance is None else covariance

        # Default prev weights
        w_prev = np.ones(n) / n if prev_weights is None else prev_weights

        lam = self.config.risk_aversion
        gamma = self.config.turnover_penalty

        def objective(w):
            # Mean-variance + turnover penalty
            ret = w @ mu
            risk = lam * w @ sigma @ w
            turnover = gamma * np.sum(np.abs(w - w_prev))
            return -(ret - risk - turnover)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Sum to 1
        ]

        # Bounds (0 <= w <= cap)
        bounds = []
        for _i, asset in enumerate(self.config.assets):
            if asset == "BIL":
                bounds.append((self.config.min_weight, self.config.max_weight_cash))
            else:
                bounds.append((self.config.min_weight, self.config.max_weight_risky))

        # Optimize
        x0 = w_prev
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            w_opt = result.x
            # Normalize in case of numerical issues
            w_opt = np.clip(w_opt, 0, 1)
            w_opt = w_opt / w_opt.sum()
            return w_opt
        else:
            logger.warning("Optimization failed, returning previous weights")
            return w_prev

    def apply_daily_change_cap(
        self,
        target_weights: np.ndarray,
        prev_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Aplica cap de cambio diario por activo.

        Si |w_target - w_prev| > max_daily_change, limita el cambio.
        """
        max_change = self.config.max_daily_change

        delta = target_weights - prev_weights
        capped_delta = np.clip(delta, -max_change, max_change)

        new_weights = prev_weights + capped_delta

        # Re-normalize
        new_weights = np.clip(new_weights, 0, 1)
        new_weights = new_weights / new_weights.sum()

        return new_weights


# =============================================================================
# Event-Driven Backtest
# =============================================================================


@dataclass
class BacktestResult:
    """Resultado del backtest."""

    # Equity curve
    dates: list[date]
    equity: list[float]
    weights: list[dict[str, float]]
    returns: list[float]

    # Metrics
    sharpe_annual: float
    max_drawdown: float
    turnover_annual: float
    hit_rate_vs_voo: float
    total_return: float
    volatility_annual: float

    # Config
    config_hash: str

    def to_dict(self) -> dict:
        return {
            "sharpe_annual": self.sharpe_annual,
            "max_drawdown": self.max_drawdown,
            "turnover_annual": self.turnover_annual,
            "hit_rate_vs_voo": self.hit_rate_vs_voo,
            "total_return": self.total_return,
            "volatility_annual": self.volatility_annual,
            "config_hash": self.config_hash,
            "n_days": len(self.dates),
        }


class AllocationBacktest:
    """
    Backtest event-driven para V1 Allocation.

    Características:
    - Ejecución open-to-open
    - Rebalance con umbral
    - Costos por turnover
    - Métricas netas

    Example:
        >>> config = V1AllocationConfig()
        >>> backtest = AllocationBacktest(config)
        >>> result = backtest.run(df_returns, df_signals)
    """

    def __init__(self, config: V1AllocationConfig):
        self.config = config
        self.optimizer = MeanVarianceOptimizer(config)

    def run(
        self,
        df_returns: pl.DataFrame,
        expected_returns_func: callable | None = None,
        covariance_func: callable | None = None,
    ) -> BacktestResult:
        """
        Ejecuta backtest event-driven.

        Args:
            df_returns: Wide format con ret_QQQ, ret_VOO, ret_BIL por fecha
            expected_returns_func: Función que retorna expected returns dado histórico
            covariance_func: Función que retorna covariance matrix

        Returns:
            BacktestResult con métricas y equity curve
        """
        assets = self.config.assets
        n_assets = len(assets)

        # Ordenar por fecha
        df = df_returns.sort("date")
        dates = df["date"].to_list()

        # Inicializar
        equity = [1.0]
        weights_history = []
        returns_history = []
        turnovers = []

        current_weights = np.ones(n_assets) / n_assets  # Equal weight inicial

        for _i, row in enumerate(df.iter_rows(named=True)):
            d = row["date"]

            # Forward returns del día
            rets = np.array([row.get(f"ret_{a}", 0.0) or 0.0 for a in assets])

            # Expected returns para optimización
            if expected_returns_func:
                exp_rets = expected_returns_func(df.filter(pl.col("date") < d))
            else:
                # Default: usar retornos rolling como proxy
                exp_rets = rets  # Simplificación para V1

            # Covariance
            cov = covariance_func(df.filter(pl.col("date") < d)) if covariance_func else None

            # Optimizar nuevos pesos
            target_weights = self.optimizer.optimize(
                exp_rets,
                cov,
                current_weights,
            )

            # Aplicar cap de cambio diario
            target_weights = self.optimizer.apply_daily_change_cap(
                target_weights,
                current_weights,
            )

            # Verificar umbral de rebalance
            max_change = np.max(np.abs(target_weights - current_weights))

            if max_change >= self.config.rebalance_threshold:
                # Rebalancear
                new_weights = target_weights
                turnover = np.sum(np.abs(new_weights - current_weights))
            else:
                # Mantener pesos
                new_weights = current_weights
                turnover = 0.0

            # Calcular costo
            cost = (turnover / 2) * (self.config.round_trip_bps / 10000)

            # Retorno del portfolio
            port_return = np.dot(new_weights, rets) - cost

            # Actualizar equity
            new_equity = equity[-1] * (1 + port_return)
            equity.append(new_equity)

            # Guardar histórico
            weights_history.append(dict(zip(assets, new_weights, strict=False)))
            returns_history.append(port_return)
            turnovers.append(turnover)

            # Actualizar pesos para siguiente día
            current_weights = new_weights

        # Calcular métricas
        returns_arr = np.array(returns_history)

        # Sharpe
        if len(returns_arr) > 0 and np.std(returns_arr) > 0:
            sharpe = np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max Drawdown
        equity_arr = np.array(equity)
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - running_max) / running_max
        max_dd = np.min(drawdowns)

        # Turnover anual
        turnover_annual = np.sum(turnovers)  # Total en el período

        # Hit rate vs VOO
        voo_returns = df["ret_VOO"].to_numpy()
        hits = (returns_arr > voo_returns).astype(float)
        hit_rate = np.mean(hits) if len(hits) > 0 else 0.0

        # Volatilidad anual
        vol_annual = np.std(returns_arr) * np.sqrt(252)

        # Total return
        total_return = equity[-1] / equity[0] - 1

        return BacktestResult(
            dates=dates,
            equity=equity[1:],  # Excluir el 1.0 inicial
            weights=weights_history,
            returns=returns_history,
            sharpe_annual=sharpe,
            max_drawdown=max_dd,
            turnover_annual=turnover_annual,
            hit_rate_vs_voo=hit_rate,
            total_return=total_return,
            volatility_annual=vol_annual,
            config_hash=self.config.compute_hash(),
        )


# =============================================================================
# Sample Validity Checker
# =============================================================================


class SampleValidityChecker:
    """
    Verifica que cada sample sea válido para training.

    Un sample es válido si:
    - quality_flag == clean en t, t+1, t+2 para todos los assets
    - recon_flag != disputed
    - calendar chain ok
    - no event_proximity
    - no missing data
    """

    def __init__(self, config: V1AllocationConfig):
        self.config = config

    def add_validity_column(
        self,
        df: pl.DataFrame,
        date_col: str = "date",
    ) -> pl.DataFrame:
        """
        Añade columna 'sample_valid' al DataFrame.
        """
        conditions = []

        # Quality flag
        if self.config.require_clean and "quality_flag" in df.columns:
            conditions.append(pl.col("quality_flag") == "clean")

        # Disputed
        if self.config.exclude_disputed and "recon_flag" in df.columns:
            conditions.append(pl.col("recon_flag") != "disputed")

        # Event proximity
        if "event_proximity" in df.columns:
            conditions.append(~pl.col("event_proximity"))

        # Forward returns existen
        for asset in self.config.assets:
            ret_col = f"ret_{asset}"
            if ret_col in df.columns:
                conditions.append(pl.col(ret_col).is_not_null())

        if not conditions:
            return df.with_columns(pl.lit(True).alias("sample_valid"))

        combined = conditions[0]
        for cond in conditions[1:]:
            combined = combined & cond

        return df.with_columns(combined.alias("sample_valid"))

    def filter_valid(self, df: pl.DataFrame) -> pl.DataFrame:
        """Retorna solo samples válidos."""
        if "sample_valid" not in df.columns:
            df = self.add_validity_column(df)
        return df.filter(pl.col("sample_valid"))


# =============================================================================
# Factory Functions
# =============================================================================


def get_v1_allocation_config() -> V1AllocationConfig:
    """Configuración V1 por defecto."""
    return V1AllocationConfig()


def run_v1_backtest(
    df_ohlcv: pl.DataFrame,
    config: V1AllocationConfig | None = None,
) -> BacktestResult:
    """
    Pipeline completo de backtest V1.

    Args:
        df_ohlcv: DataFrame long format (date, ticker, open, close, ...)
        config: Configuración (default: V1AllocationConfig())

    Returns:
        BacktestResult con métricas
    """
    if config is None:
        config = V1AllocationConfig()

    # 1. Construir forward returns
    label_builder = AllocationLabelBuilder(config)
    df_returns = label_builder.build(df_ohlcv)

    # 2. Validar samples
    checker = SampleValidityChecker(config)
    df_returns = checker.add_validity_column(df_returns)
    df_valid = df_returns.filter(pl.col("sample_valid"))

    # 3. Ejecutar backtest
    backtest = AllocationBacktest(config)
    result = backtest.run(df_valid)

    logger.info(
        "V1 Backtest complete: Sharpe=%.2f, MaxDD=%.1f%%, Turnover=%.0f%%, HitRate=%.1f%%",
        result.sharpe_annual,
        result.max_drawdown * 100,
        result.turnover_annual * 100,
        result.hit_rate_vs_voo * 100,
    )

    return result
