# Predictive Model + Walk-Forward Validation
# ==========================================
"""
Modelo predictivo baseline para V1 Allocation:
- Ridge/ElasticNet para predecir expected returns
- Walk-forward validation con train/test splits
- Integración con SnapshotManager

BASELINE SERIO (hard to beat):
- Features: momentum, vol regime, drawdown, spreads
- Modelo: Ridge (rápido, estable)
- Validation: Walk-forward 252 train / 21 test
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


# =============================================================================
# Predictive Model Baseline
# =============================================================================


@dataclass
class PredictorConfig:
    """Configuración del predictor."""

    model_type: Literal["ridge", "elasticnet", "lasso"] = "ridge"
    alpha: float = 1.0  # Regularización
    l1_ratio: float = 0.5  # Solo para elasticnet

    # Features a usar
    feature_cols: list[str] = field(
        default_factory=lambda: [
            "ret_1d",
            "ret_5d",
            "ret_20d",
            "vol_20d",
            "mom_60d",
            "dd_20d",
        ]
    )

    # Assets
    assets: list[str] = field(default_factory=lambda: ["QQQ", "VOO", "BIL"])

    def compute_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


class ReturnPredictor:
    """
    Predictor de returns baseline usando Ridge/ElasticNet.

    Predice expected return para cada activo basado en features.
    """

    def __init__(self, config: PredictorConfig | None = None):
        self.config = config or PredictorConfig()
        self.models = {}  # asset → trained model
        self._is_fitted = False

    def fit(
        self,
        df_features: pl.DataFrame,
        df_returns: pl.DataFrame,
        date_col: str = "date",
    ) -> ReturnPredictor:
        """
        Entrena modelo para cada activo.

        Args:
            df_features: Features wide format (date, feat1_QQQ, feat2_QQQ, ...)
            df_returns: Returns wide format (date, ret_QQQ, ret_VOO, ret_BIL)
        """
        try:
            from sklearn.linear_model import ElasticNet, Lasso, Ridge
        except ImportError:
            logger.warning("sklearn not available, using dummy predictor")
            self._is_fitted = True
            return self

        # Merge features con returns
        df = df_features.join(df_returns, on=date_col, how="inner")

        for asset in self.config.assets:
            # Feature columns para este asset
            X_cols = [
                f"{f}_{asset}" for f in self.config.feature_cols if f"{f}_{asset}" in df.columns
            ]

            if not X_cols:
                X_cols = [
                    c
                    for c in df.columns
                    if asset in c and c.startswith(("ret_", "vol_", "mom_", "dd_"))
                ]

            y_col = f"ret_{asset}"

            if y_col not in df.columns:
                continue

            # Extraer X, y
            df_clean = df.select([date_col] + X_cols + [y_col]).drop_nulls()

            if df_clean.height < 30:
                logger.warning(f"Not enough data for {asset}: {df_clean.height} rows")
                continue

            X = df_clean.select(X_cols).to_numpy()
            y = df_clean[y_col].to_numpy()

            # Entrenar modelo
            if self.config.model_type == "ridge":
                model = Ridge(alpha=self.config.alpha)
            elif self.config.model_type == "elasticnet":
                model = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio)
            else:
                model = Lasso(alpha=self.config.alpha)

            model.fit(X, y)
            self.models[asset] = {
                "model": model,
                "feature_cols": X_cols,
            }

        self._is_fitted = True
        logger.info("Fitted predictor for %d assets", len(self.models))
        return self

    def predict(
        self,
        df_features: pl.DataFrame,
    ) -> np.ndarray:
        """
        Predice expected returns para la última fila.

        Returns:
            (n_assets,) array de expected returns
        """
        if not self._is_fitted:
            return np.zeros(len(self.config.assets))

        if df_features.height == 0:
            return np.zeros(len(self.config.assets))

        # Tomar última fila
        last_row = df_features.tail(1)

        predictions = []
        for asset in self.config.assets:
            if asset not in self.models:
                predictions.append(0.0)
                continue

            model_info = self.models[asset]
            model = model_info["model"]
            feature_cols = model_info["feature_cols"]

            # Extraer features
            X = last_row.select(feature_cols).to_numpy()

            # Handle NaNs
            if np.any(np.isnan(X)):
                predictions.append(0.0)
            else:
                pred = model.predict(X)[0]
                predictions.append(pred)

        return np.array(predictions)


# =============================================================================
# Walk-Forward Validation
# =============================================================================


@dataclass
class WalkForwardConfig:
    """Configuración de walk-forward."""

    train_days: int = 252  # 1 año de training
    test_days: int = 21  # 1 mes de test
    step_days: int = 21  # Avanzar 1 mes

    min_train_samples: int = 200


@dataclass
class WalkForwardResult:
    """Resultado de una ventana de walk-forward."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date

    # Métricas en test
    sharpe: float
    total_return: float
    turnover: float
    max_dd: float
    n_days: int

    # Hashes para reproducibilidad
    model_hash: str
    config_hash: str


class WalkForwardValidator:
    """
    Validación walk-forward para V1 Allocation.

    Train 252d → Test 21d → Step 21d → Repeat
    """

    def __init__(
        self,
        predictor_config: PredictorConfig | None = None,
        allocation_config=None,  # V1AllocationConfigV2
        wf_config: WalkForwardConfig | None = None,
    ):
        self.predictor_config = predictor_config or PredictorConfig()
        self.allocation_config = allocation_config
        self.wf_config = wf_config or WalkForwardConfig()

    def run(
        self,
        df_features: pl.DataFrame,
        df_returns: pl.DataFrame,
        date_col: str = "date",
    ) -> list[WalkForwardResult]:
        """
        Ejecuta walk-forward validation completa.
        """
        from portfolio.trading.v1_allocation_v2 import (
            AllocationBacktestV2,
            V1AllocationConfigV2,
        )

        if self.allocation_config is None:
            self.allocation_config = V1AllocationConfigV2()

        # Ordenar por fecha
        df_features = df_features.sort(date_col)
        df_returns = df_returns.sort(date_col)

        dates = df_features[date_col].to_list()
        n_dates = len(dates)

        results = []

        # Walk-forward loop
        train_start_idx = 0

        while True:
            train_end_idx = train_start_idx + self.wf_config.train_days
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + self.wf_config.test_days

            if test_end_idx > n_dates:
                break

            train_start = dates[train_start_idx]
            train_end = dates[train_end_idx - 1]
            test_start = dates[test_start_idx]
            test_end = dates[test_end_idx - 1]

            logger.info(
                "Walk-forward: train %s to %s, test %s to %s",
                train_start,
                train_end,
                test_start,
                test_end,
            )

            # Split data
            df_train_feat = df_features.filter(
                (pl.col(date_col) >= train_start) & (pl.col(date_col) <= train_end)
            )
            df_train_ret = df_returns.filter(
                (pl.col(date_col) >= train_start) & (pl.col(date_col) <= train_end)
            )

            df_test_feat = df_features.filter(
                (pl.col(date_col) >= test_start) & (pl.col(date_col) <= test_end)
            )
            df_test_ret = df_returns.filter(
                (pl.col(date_col) >= test_start) & (pl.col(date_col) <= test_end)
            )

            if df_train_feat.height < self.wf_config.min_train_samples:
                train_start_idx += self.wf_config.step_days
                continue

            # Entrenar predictor
            predictor = ReturnPredictor(self.predictor_config)
            predictor.fit(df_train_feat, df_train_ret)

            # Función para expected returns
            def expected_returns_func(df_history):
                # Merge con features
                if df_history.height == 0:
                    return np.zeros(len(self.allocation_config.assets))

                last_date = df_history["date"].max()
                feat_row = df_test_feat.filter(pl.col(date_col) <= last_date).tail(1)
                return predictor.predict(feat_row)

            # Rolling covariance
            def covariance_func(df_history):
                if df_history.height < 60:
                    return None

                ret_cols = [f"ret_{a}" for a in self.allocation_config.assets]
                ret_matrix = df_history.tail(60).select(ret_cols).to_numpy()

                # Handle NaNs
                ret_matrix = np.nan_to_num(ret_matrix, nan=0.0)

                if ret_matrix.shape[0] < 30:
                    return None

                return np.cov(ret_matrix.T)

            # Backtest en test period
            backtest = AllocationBacktestV2(self.allocation_config)
            bt_result = backtest.run(
                df_test_ret,
                expected_returns_func=expected_returns_func,
                covariance_func=covariance_func,
            )

            # Guardar resultado
            result = WalkForwardResult(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                sharpe=bt_result.sharpe_annual,
                total_return=bt_result.total_return,
                turnover=bt_result.turnover_annual,
                max_dd=bt_result.max_drawdown,
                n_days=len(bt_result.dates),
                model_hash=self.predictor_config.compute_hash(),
                config_hash=self.allocation_config.compute_hash(),
            )
            results.append(result)

            # Avanzar
            train_start_idx += self.wf_config.step_days

        return results

    def summarize(self, results: list[WalkForwardResult]) -> dict:
        """Resume resultados de walk-forward."""
        if not results:
            return {}

        sharpes = [r.sharpe for r in results]
        returns = [r.total_return for r in results]
        turnovers = [r.turnover for r in results]
        drawdowns = [r.max_dd for r in results]

        return {
            "n_windows": len(results),
            "sharpe_mean": np.mean(sharpes),
            "sharpe_std": np.std(sharpes),
            "sharpe_min": np.min(sharpes),
            "sharpe_max": np.max(sharpes),
            "return_mean": np.mean(returns),
            "turnover_mean": np.mean(turnovers),
            "max_dd_worst": np.min(drawdowns),
            "pct_positive_sharpe": np.mean([s > 0 for s in sharpes]),
        }


# =============================================================================
# Integration with Snapshot/Training Run
# =============================================================================


def run_walk_forward_with_snapshot(
    df_features: pl.DataFrame,
    df_returns: pl.DataFrame,
    snapshot_path: str = "data_lake/snapshots",
) -> dict:
    """
    Ejecuta walk-forward y guarda resultados con snapshot.
    """

    # Config
    predictor_config = PredictorConfig()

    from portfolio.trading.v1_allocation_v2 import V1AllocationConfigV2

    allocation_config = V1AllocationConfigV2()

    # Walk-forward
    validator = WalkForwardValidator(
        predictor_config=predictor_config,
        allocation_config=allocation_config,
    )

    results = validator.run(df_features, df_returns)
    summary = validator.summarize(results)

    # Log summary
    logger.info("Walk-forward summary: %s", summary)

    return {
        "results": results,
        "summary": summary,
        "predictor_config": predictor_config.compute_hash(),
        "allocation_config": allocation_config.compute_hash(),
    }
