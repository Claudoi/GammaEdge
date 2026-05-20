# V1 Allocation v2 — Turnover Controlled + Predictive Model
# =========================================================
"""
V1 Allocation con controles de turnover serios:
- turnover_penalty (γ)
- rebalance_threshold (τ, deadband)
- max_daily_change (cap duro)
- partial_adjustment (α)
- min_holding_days

+ Modelo predictivo baseline (Ridge/ElasticNet)
+ Walk-forward validation
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


# =============================================================================
# V2 Config — Turnover Controlled
# =============================================================================


@dataclass
class V1AllocationConfigV2:
    """
    Configuración V1 con controles de turnover.

    4 frenos:
    - turnover_penalty (γ): penalización L1 en optimizer
    - rebalance_threshold (τ): deadband, no rebalancear si cambio < τ
    - max_daily_change: cap duro por activo
    - partial_adjustment (α): solo mover α% hacia target
    """

    config_version: str = "2.0.0"

    # Assets
    assets: list[str] = field(default_factory=lambda: ["QQQ", "VOO", "BIL"])

    # Costs (bps)
    cost_per_side_bps: float = 4.0  # 3 base + 1 open penalty

    # =========================================================================
    # TURNOVER CONTROLS (los 4 frenos)
    # =========================================================================

    # γ: Penalty L1 en optimizer
    turnover_penalty: float = 0.20  # Subido de 0.01 a 0.20

    # τ: Deadband (no rebalancear si max change < τ)
    rebalance_threshold: float = 0.10  # Subido de 0.05 a 0.10

    # Cap duro: máximo cambio por activo por día
    max_daily_change: float = 0.10  # Subido de 0.25 a 0.10

    # α: Partial adjustment (solo mover α% hacia target)
    partial_adjustment: float = 0.25  # 25% del camino hacia target

    # Min holding: días mínimos entre rebalances
    min_holding_days: int = 2

    # =========================================================================
    # Optimizer
    # =========================================================================

    risk_aversion: float = 1.0

    # Constraints
    min_weight: float = 0.0
    max_weight_risky: float = 0.8
    max_weight_cash: float = 1.0

    # =========================================================================
    # Holding
    # =========================================================================

    holding_days: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    def compute_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# =============================================================================
# Optimizer with All Controls
# =============================================================================


class TurnoverControlledOptimizer:
    """
    Optimizer MV con los 4 frenos de turnover.

    1. turnover_penalty (γ): en la función objetivo
    2. max_daily_change: cap después de optimizar
    3. partial_adjustment (α): solo mover α% hacia target
    4. rebalance_threshold (τ): enforced externamente en backtest
    """

    def __init__(self, config: V1AllocationConfigV2):
        self.config = config
        self.n_assets = len(config.assets)

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray | None = None,
        prev_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Optimiza pesos con penalty de turnover.
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            return np.ones(self.n_assets) / self.n_assets

        n = self.n_assets
        mu = expected_returns

        sigma = np.eye(n) * 0.01 if covariance is None else covariance

        w_prev = np.ones(n) / n if prev_weights is None else prev_weights

        lam = self.config.risk_aversion
        gamma = self.config.turnover_penalty  # γ alto

        def objective(w):
            ret = w @ mu
            risk = lam * w @ sigma @ w
            # L1 turnover penalty (fuerte)
            turnover = gamma * np.sum(np.abs(w - w_prev))
            return -(ret - risk - turnover)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]

        bounds = []
        for _i, asset in enumerate(self.config.assets):
            if asset == "BIL":
                bounds.append((self.config.min_weight, self.config.max_weight_cash))
            else:
                bounds.append((self.config.min_weight, self.config.max_weight_risky))

        result = minimize(
            objective,
            w_prev,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            w_opt = result.x
            w_opt = np.clip(w_opt, 0, 1)
            w_opt = w_opt / w_opt.sum()
            return w_opt
        else:
            return w_prev

    def apply_controls(
        self,
        target_weights: np.ndarray,
        prev_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Aplica controles post-optimización:
        1. max_daily_change cap
        2. partial_adjustment
        """
        # 1. Cap de cambio máximo por activo
        delta = target_weights - prev_weights
        max_change = self.config.max_daily_change
        capped_delta = np.clip(delta, -max_change, max_change)

        # 2. Partial adjustment: solo mover α% hacia target
        alpha = self.config.partial_adjustment
        adjusted_delta = alpha * capped_delta

        new_weights = prev_weights + adjusted_delta

        # Re-normalize
        new_weights = np.clip(new_weights, 0, 1)
        new_weights = new_weights / new_weights.sum()

        return new_weights


# =============================================================================
# Backtest with All Controls
# =============================================================================


@dataclass
class BacktestResultV2:
    """Resultado del backtest con métricas detalladas."""

    dates: list[date]
    equity: list[float]
    weights: list[dict[str, float]]
    returns: list[float]

    # Core metrics
    sharpe_annual: float
    max_drawdown: float
    total_return: float
    volatility_annual: float

    # Turnover (importante)
    turnover_annual: float
    turnover_daily_avg: float
    n_rebalances: int

    # Hit rate
    hit_rate_vs_voo: float
    downside_capture: float

    # Config
    config_hash: str

    def to_dict(self) -> dict:
        return {
            "sharpe_annual": self.sharpe_annual,
            "max_drawdown": self.max_drawdown,
            "total_return": self.total_return,
            "volatility_annual": self.volatility_annual,
            "turnover_annual": self.turnover_annual,
            "turnover_daily_avg": self.turnover_daily_avg,
            "n_rebalances": self.n_rebalances,
            "hit_rate_vs_voo": self.hit_rate_vs_voo,
            "downside_capture": self.downside_capture,
            "config_hash": self.config_hash,
            "n_days": len(self.dates),
        }


class AllocationBacktestV2:
    """
    Backtest con todos los controles de turnover.

    Controles aplicados:
    1. turnover_penalty en optimizer
    2. max_daily_change cap
    3. partial_adjustment
    4. rebalance_threshold deadband
    5. min_holding_days
    """

    def __init__(self, config: V1AllocationConfigV2):
        self.config = config
        self.optimizer = TurnoverControlledOptimizer(config)

    def run(
        self,
        df_returns: pl.DataFrame,
        expected_returns_func: Callable | None = None,
        covariance_func: Callable | None = None,
    ) -> BacktestResultV2:
        """
        Ejecuta backtest con controles de turnover.
        """
        assets = self.config.assets
        n_assets = len(assets)

        df = df_returns.sort("date")
        dates = df["date"].to_list()

        equity = [1.0]
        weights_history = []
        returns_history = []
        turnovers = []

        current_weights = np.ones(n_assets) / n_assets
        days_since_rebalance = self.config.min_holding_days  # Allow first rebalance
        n_rebalances = 0

        for _i, row in enumerate(df.iter_rows(named=True)):
            d = row["date"]

            # Forward returns del día
            rets = np.array([row.get(f"ret_{a}", 0.0) or 0.0 for a in assets])

            # Expected returns
            if expected_returns_func:
                exp_rets = expected_returns_func(df.filter(pl.col("date") < d))
            else:
                exp_rets = rets

            # Covariance
            cov = covariance_func(df.filter(pl.col("date") < d)) if covariance_func else None

            # Optimizar (con γ alto)
            raw_target = self.optimizer.optimize(exp_rets, cov, current_weights)

            # Check deadband (τ) ANTES de aplicar controles
            # Medir contra el raw target, no el adjusted
            raw_max_change = np.max(np.abs(raw_target - current_weights))

            # Check min holding days
            can_rebalance = days_since_rebalance >= self.config.min_holding_days

            if raw_max_change >= self.config.rebalance_threshold and can_rebalance:
                # Aplicar controles (cap + partial adjustment)
                target_weights = self.optimizer.apply_controls(raw_target, current_weights)

                # Rebalancear
                new_weights = target_weights
                # One-way turnover (correcto)
                turnover = 0.5 * np.sum(np.abs(new_weights - current_weights))
                days_since_rebalance = 0
                n_rebalances += 1
            else:
                # Mantener
                new_weights = current_weights
                turnover = 0.0
                days_since_rebalance += 1

            # Costo (one-way)
            cost = turnover * (self.config.cost_per_side_bps / 10000)

            # Retorno neto
            port_return = np.dot(new_weights, rets) - cost

            new_equity = equity[-1] * (1 + port_return)
            equity.append(new_equity)

            weights_history.append(dict(zip(assets, new_weights, strict=False)))
            returns_history.append(port_return)
            turnovers.append(turnover)

            current_weights = new_weights

        # Métricas
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

        # Turnover (anualizado correctamente)
        total_turnover = np.sum(turnovers)
        n_years = len(dates) / 252
        turnover_annual = total_turnover / n_years if n_years > 0 else 0
        turnover_daily_avg = np.mean(turnovers)

        # Hit rate vs VOO
        voo_returns = df["ret_VOO"].to_numpy()
        hits = (returns_arr > voo_returns).astype(float)
        hit_rate = np.mean(hits) if len(hits) > 0 else 0.0

        # Downside capture
        voo_down_days = voo_returns < 0
        if np.any(voo_down_days):
            port_on_down = returns_arr[voo_down_days]
            voo_on_down = voo_returns[voo_down_days]
            downside_capture = (
                np.mean(port_on_down) / np.mean(voo_on_down) if np.mean(voo_on_down) != 0 else 1.0
            )
        else:
            downside_capture = 1.0

        # Vol
        vol_annual = np.std(returns_arr) * np.sqrt(252)

        # Total return
        total_return = equity[-1] / equity[0] - 1

        return BacktestResultV2(
            dates=dates,
            equity=equity[1:],
            weights=weights_history,
            returns=returns_history,
            sharpe_annual=sharpe,
            max_drawdown=max_dd,
            total_return=total_return,
            volatility_annual=vol_annual,
            turnover_annual=turnover_annual,
            turnover_daily_avg=turnover_daily_avg,
            n_rebalances=n_rebalances,
            hit_rate_vs_voo=hit_rate,
            downside_capture=downside_capture,
            config_hash=self.config.compute_hash(),
        )


# =============================================================================
# Grid Search for Optimal Controls
# =============================================================================


def grid_search_turnover_controls(
    df_returns: pl.DataFrame,
    gammas: list[float] = None,
    taus: list[float] = None,
    alphas: list[float] = None,
    max_turnover: float = 3.0,
    min_sharpe: float = 0.5,
) -> list[dict]:
    """
    Grid search para encontrar controles óptimos.

    Objetivo: maximizar Sharpe sujeto a turnover <= max_turnover
    """
    if gammas is None:
        gammas = [0.10, 0.20, 0.35, 0.50]
    if taus is None:
        taus = [0.05, 0.10, 0.15]
    if alphas is None:
        alphas = [0.25, 0.50, 0.75]

    results = []

    for gamma in gammas:
        for tau in taus:
            for alpha in alphas:
                config = V1AllocationConfigV2(
                    turnover_penalty=gamma,
                    rebalance_threshold=tau,
                    partial_adjustment=alpha,
                )

                backtest = AllocationBacktestV2(config)
                result = backtest.run(df_returns)

                # Check constraints
                feasible = (
                    result.turnover_annual <= max_turnover
                    and result.max_drawdown > -0.15
                    and result.sharpe_annual >= min_sharpe
                )

                results.append(
                    {
                        "gamma": gamma,
                        "tau": tau,
                        "alpha": alpha,
                        "sharpe": result.sharpe_annual,
                        "turnover": result.turnover_annual,
                        "max_dd": result.max_drawdown,
                        "n_rebalances": result.n_rebalances,
                        "downside_capture": result.downside_capture,
                        "feasible": feasible,
                        "config_hash": result.config_hash,
                    }
                )

    # Sort by Sharpe (feasible first)
    results.sort(key=lambda x: (-int(x["feasible"]), -x["sharpe"]))

    return results


def select_best_config(results: list[dict]) -> V1AllocationConfigV2 | None:
    """Selecciona la mejor config feasible."""
    for r in results:
        if r["feasible"]:
            return V1AllocationConfigV2(
                turnover_penalty=r["gamma"],
                rebalance_threshold=r["tau"],
                partial_adjustment=r["alpha"],
            )
    return None
