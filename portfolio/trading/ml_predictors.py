"""
XGBoost Predictor Extension with Purged K-Fold Cross-Validation

Extends the baseline predictor with gradient boosting and advanced CV techniques
to avoid data leakage in time series prediction.

Based on:
- López de Prado (2018): "Advances in Financial Machine Learning" (Chapter 7-12)
- Chen & Guestrin (2016): "XGBoost: A Scalable Tree Boosting System"

Author: GammaEdge TIER 1 Enhancement - Phase 2
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost predictor."""

    # XGBoost hyperparameters
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.0
    min_child_weight: int = 1
    reg_alpha: float = 0.0  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization

    # Training params
    early_stopping_rounds: int = 10
    eval_metric: str = "rmse"

    # Features
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

    # Cross-validation
    cv_folds: int = 5
    purge_days: int = 5  # Days to purge before/after test fold
    embargo_days: int = 3  # Additional embargo after test fold

    def compute_hash(self) -> str:
        """Compute deterministic hash of config."""
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    def to_xgb_params(self) -> dict:
        """Convert to XGBoost parameters dict.

        Note:
            ``early_stopping_rounds`` is included here (constructor-level)
            because XGBoost >=2.0 removed support for passing it as a kwarg
            to ``XGBRegressor.fit()``. It must now be configured on the
            estimator itself.
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "objective": "reg:squarederror",
            "tree_method": "hist",  # Fast histogram-based
            "random_state": 42,
            "early_stopping_rounds": self.early_stopping_rounds,
            "eval_metric": self.eval_metric,
        }


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation for time series.

    Prevents data leakage by:
    1. Splitting data into K folds chronologically
    2. Purging samples near the test fold (before and after)
    3. Adding embargo period after test fold

    Based on López de Prado (2018) Chapter 7.

    Example:
        >>> cv = PurgedKFold(n_splits=5, purge_days=5, embargo_days=3)
        >>> for train_idx, test_idx in cv.split(dates):
        >>>     X_train, X_test = X[train_idx], X[test_idx]
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 5,
        embargo_days: int = 3,
    ):
        """
        Initialize Purged K-Fold.

        Args:
            n_splits: Number of folds
            purge_days: Days to remove before/after test fold
            embargo_days: Additional days to remove after test fold
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, dates: np.ndarray | list) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices with purging and embargo.

        Args:
            dates: Array of dates (must be sorted chronologically)

        Returns:
            List of (train_indices, test_indices) tuples
        """
        dates = np.array(dates)
        n = len(dates)

        # Use standard KFold for base splits
        kf = KFold(n_splits=self.n_splits, shuffle=False)

        splits = []

        for train_idx, test_idx in kf.split(dates):
            # Purge samples near test fold
            test_start = test_idx[0]
            test_end = test_idx[-1]

            # Remove samples within purge_days of test fold
            purge_start = max(0, test_start - self.purge_days)
            purge_end = min(n - 1, test_end + self.purge_days + self.embargo_days)

            # Filter train indices
            train_purged = train_idx[(train_idx < purge_start) | (train_idx > purge_end)]

            # Embargo: remove samples immediately after test fold
            embargo_end = min(n - 1, test_end + self.embargo_days)
            train_purged = train_purged[train_purged <= test_start - self.purge_days - 1]
            train_purged = np.append(train_purged, train_idx[train_idx > embargo_end])

            splits.append((train_purged, test_idx))

        return splits


class GradientBoostingPredictor:
    """
    XGBoost predictor for expected returns with purged cross-validation.

    Features:
    - Gradient boosting (more flexible than linear models)
    - Purged K-Fold CV (prevents leakage)
    - Feature importance via SHAP values
    - Early stopping to prevent overfitting

    Example:
        >>> config = XGBoostConfig(n_estimators=100, max_depth=3)
        >>> predictor = GradientBoostingPredictor(config)
        >>> predictor.fit(df_features, df_returns)
        >>> predictions = predictor.predict(df_features)
    """

    def __init__(self, config: XGBoostConfig | None = None):
        """
        Initialize XGBoost predictor.

        Args:
            config: XGBoost configuration
        """
        self.config = config or XGBoostConfig()
        self.models: dict[str, Any] = {}  # asset → trained XGBoost model
        self.feature_importances: dict[str, np.ndarray] = {}
        self._is_fitted = False

    def fit(
        self,
        df_features: pl.DataFrame,
        df_returns: pl.DataFrame,
        date_col: str = "date",
        use_cv: bool = True,
    ) -> GradientBoostingPredictor:
        """
        Train XGBoost model for each asset with purged cross-validation.

        Args:
            df_features: Features wide format (date, feat1_QQQ, feat2_QQQ, ...)
            df_returns: Returns wide format (date, ret_QQQ, ret_VOO, ret_BIL)
            date_col: Date column name
            use_cv: Whether to use purged K-Fold CV for validation

        Returns:
            self (fitted)
        """
        # Merge features with returns
        df = df_features.join(df_returns, on=date_col, how="inner")

        for asset in self.config.assets:
            # Feature columns for this asset
            X_cols = [
                f"{f}_{asset}" for f in self.config.feature_cols if f"{f}_{asset}" in df.columns
            ]

            y_col = f"ret_{asset}"

            if not X_cols:
                # Fallback: find any feature columns for this asset.
                # Exclude the target return column (``ret_{asset}``) so the
                # label never leaks into the feature matrix and to avoid
                # polars duplicate-column errors downstream.
                X_cols = [
                    c
                    for c in df.columns
                    if asset in c and c != y_col and c.startswith(("ret_", "vol_", "mom_", "dd_"))
                ]

            if y_col not in df.columns:
                logger.warning(f"Return column {y_col} not found, skipping {asset}")
                continue

            if not X_cols:
                logger.warning(f"No feature columns found for {asset}, skipping")
                continue

            # Extract X, y
            df_clean = df.select([date_col] + X_cols + [y_col]).drop_nulls()

            if df_clean.height < 100:
                logger.warning(f"Not enough data for {asset}: {df_clean.height} rows")
                continue

            X = df_clean.select(X_cols).to_numpy()
            y = df_clean[y_col].to_numpy()
            clean_dates = df_clean[date_col].to_numpy()

            # Train with optional cross-validation
            if use_cv and df_clean.height >= 200:
                # Purged K-Fold CV
                cv = PurgedKFold(
                    n_splits=self.config.cv_folds,
                    purge_days=self.config.purge_days,
                    embargo_days=self.config.embargo_days,
                )

                cv_scores = []

                for train_idx, val_idx in cv.split(clean_dates):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    # Train XGBoost
                    # NOTE: early_stopping_rounds is set on the estimator
                    # (see XGBoostConfig.to_xgb_params) because XGBoost >=2.0
                    # removed the fit() kwarg.
                    model = xgb.XGBRegressor(**self.config.to_xgb_params())
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                    )

                    # Evaluate
                    val_pred = model.predict(X_val)
                    mse = np.mean((val_pred - y_val) ** 2)
                    cv_scores.append(mse)

                logger.info(f"{asset} CV MSE: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")

            # Train final model on all data
            model = xgb.XGBRegressor(**self.config.to_xgb_params())

            # Use 20% validation split for early stopping
            val_size = max(20, int(0.2 * len(X)))
            X_train, X_val = X[:-val_size], X[-val_size:]
            y_train, y_val = y[:-val_size], y[-val_size:]

            # NOTE: early_stopping_rounds is configured on the estimator
            # via to_xgb_params() (XGBoost >=2.0 removed the fit() kwarg).
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # Store model and feature importance
            self.models[asset] = {
                "model": model,
                "feature_cols": X_cols,
                "best_iteration": (
                    model.best_iteration if hasattr(model, "best_iteration") else model.n_estimators
                ),
            }

            self.feature_importances[asset] = model.feature_importances_

        self._is_fitted = True
        logger.info(f"Fitted XGBoost predictor for {len(self.models)} assets")
        return self

    def predict(
        self,
        df_features: pl.DataFrame,
    ) -> np.ndarray:
        """
        Predict expected returns for the last row.

        Args:
            df_features: Features DataFrame

        Returns:
            (n_assets,) array of expected returns
        """
        if not self._is_fitted:
            return np.zeros(len(self.config.assets))

        if df_features.height == 0:
            return np.zeros(len(self.config.assets))

        # Take last row
        last_row = df_features.tail(1)

        predictions = []
        for asset in self.config.assets:
            if asset not in self.models:
                predictions.append(0.0)
                continue

            model_info = self.models[asset]
            model = model_info["model"]
            feature_cols = model_info["feature_cols"]

            # Extract features
            X = last_row.select(feature_cols).to_numpy()

            # Handle NaNs
            if np.any(np.isnan(X)):
                predictions.append(0.0)
            else:
                pred = model.predict(X)[0]
                predictions.append(pred)

        return np.array(predictions)

    def get_feature_importance(self, asset: str) -> dict[str, float] | None:
        """
        Get feature importance for an asset.

        Args:
            asset: Asset ticker

        Returns:
            Dict mapping feature names to importance scores
        """
        if asset not in self.models:
            return None

        feature_cols = self.models[asset]["feature_cols"]
        importances = self.feature_importances[asset]

        return dict(zip(feature_cols, importances, strict=False))

    def get_shap_values(self, df_features: pl.DataFrame, asset: str) -> np.ndarray | None:
        """
        Compute SHAP values for interpretability.

        Args:
            df_features: Features DataFrame
            asset: Asset ticker

        Returns:
            SHAP values array (n_samples, n_features)

        Note:
            Requires shap library. Install with: pip install shap
        """
        if asset not in self.models:
            return None

        try:
            import shap
        except ImportError:
            logger.warning("SHAP library not available. Install with: pip install shap")
            return None

        model_info = self.models[asset]
        model = model_info["model"]
        feature_cols = model_info["feature_cols"]

        # Extract features
        df_clean = df_features.select(feature_cols).drop_nulls()
        X = df_clean.to_numpy()

        if len(X) == 0:
            return None

        # Compute SHAP values (TreeExplainer for XGBoost)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        return np.asarray(shap_values) if shap_values is not None else None
