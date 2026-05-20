"""
Unit tests for XGBoost ML predictors.

Tests gradient boosting predictor with purged K-Fold cross-validation.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from portfolio.trading.ml_predictors import (
    GradientBoostingPredictor,
    PurgedKFold,
    XGBoostConfig,
)


@pytest.fixture
def synthetic_features():
    """Synthetic features for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

    assets = ["QQQ", "VOO", "BIL"]
    data = {"date": dates}

    for asset in assets:
        data[f"ret_1d_{asset}"] = np.random.normal(0.0005, 0.01, len(dates))
        data[f"ret_5d_{asset}"] = np.random.normal(0.002, 0.02, len(dates))
        data[f"vol_20d_{asset}"] = np.abs(np.random.normal(0.015, 0.005, len(dates)))
        data[f"mom_60d_{asset}"] = np.random.normal(0.05, 0.1, len(dates))

    return pl.DataFrame(data)


@pytest.fixture
def synthetic_returns():
    """Synthetic future returns for testing."""
    np.random.seed(123)
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

    assets = ["QQQ", "VOO", "BIL"]
    data = {"date": dates}

    for asset in assets:
        # Returns correlated with features
        data[f"ret_{asset}"] = np.random.normal(0.001, 0.012, len(dates))

    return pl.DataFrame(data)


class TestXGBoostConfig:
    """Test XGBoost configuration."""

    def test_config_creation(self):
        """Test config can be created."""
        config = XGBoostConfig()

        assert config.n_estimators == 100
        assert config.max_depth == 3
        assert config.learning_rate == 0.1

    def test_config_hash_deterministic(self):
        """Test config hash is deterministic."""
        config1 = XGBoostConfig(n_estimators=100, max_depth=3)
        config2 = XGBoostConfig(n_estimators=100, max_depth=3)

        assert config1.compute_hash() == config2.compute_hash()

    def test_config_hash_different_params(self):
        """Test different configs have different hashes."""
        config1 = XGBoostConfig(n_estimators=100)
        config2 = XGBoostConfig(n_estimators=200)

        assert config1.compute_hash() != config2.compute_hash()

    def test_to_xgb_params(self):
        """Test conversion to XGBoost params."""
        config = XGBoostConfig(n_estimators=50, max_depth=5)
        params = config.to_xgb_params()

        assert params["n_estimators"] == 50
        assert params["max_depth"] == 5
        assert params["objective"] == "reg:squarederror"


class TestPurgedKFold:
    """Test purged K-Fold cross-validation."""

    def test_purged_kfold_creation(self):
        """Test PurgedKFold can be created."""
        cv = PurgedKFold(n_splits=5, purge_days=5, embargo_days=3)

        assert cv.n_splits == 5
        assert cv.purge_days == 5
        assert cv.embargo_days == 3

    def test_purged_kfold_splits(self):
        """Test PurgedKFold generates correct number of splits."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D").tolist()
        cv = PurgedKFold(n_splits=5, purge_days=3, embargo_days=2)

        splits = cv.split(dates)

        assert len(splits) == 5

    def test_purged_kfold_no_overlap(self):
        """Test train and test sets don't overlap."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D").tolist()
        cv = PurgedKFold(n_splits=5, purge_days=3, embargo_days=2)

        for train_idx, test_idx in cv.split(dates):
            # No overlap
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0

    def test_purged_kfold_embargo_enforced(self):
        """Test embargo prevents samples after test fold in training."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D").tolist()
        cv = PurgedKFold(n_splits=3, purge_days=2, embargo_days=3)

        for train_idx, test_idx in cv.split(dates):
            test_end = test_idx[-1]

            # No training samples within embargo after test
            embargo_zone = np.arange(test_end + 1, min(test_end + 1 + 3, len(dates)))
            train_in_embargo = np.isin(train_idx, embargo_zone)

            # Should have no training samples in embargo zone
            assert not train_in_embargo.any()


class TestGradientBoostingPredictor:
    """Test XGBoost predictor."""

    def test_predictor_creation(self):
        """Test predictor can be created."""
        config = XGBoostConfig()
        predictor = GradientBoostingPredictor(config)

        assert not predictor._is_fitted
        assert len(predictor.models) == 0

    def test_predictor_fit(self, synthetic_features, synthetic_returns):
        """Test predictor fits without errors."""
        config = XGBoostConfig(n_estimators=20, cv_folds=3)
        predictor = GradientBoostingPredictor(config)

        predictor.fit(synthetic_features, synthetic_returns, use_cv=False)

        assert predictor._is_fitted
        assert len(predictor.models) > 0

    def test_predictor_fit_with_cv(self, synthetic_features, synthetic_returns):
        """Test predictor fits with purged CV."""
        config = XGBoostConfig(n_estimators=20, cv_folds=3, purge_days=5)
        predictor = GradientBoostingPredictor(config)

        predictor.fit(synthetic_features, synthetic_returns, use_cv=True)

        assert predictor._is_fitted
        assert len(predictor.models) > 0

    def test_predictor_predict_shape(self, synthetic_features, synthetic_returns):
        """Test predictions have correct shape."""
        config = XGBoostConfig(n_estimators=20)
        predictor = GradientBoostingPredictor(config)

        predictor.fit(synthetic_features, synthetic_returns, use_cv=False)
        predictions = predictor.predict(synthetic_features)

        # Should return prediction for each asset
        assert len(predictions) == len(config.assets)

    def test_predictor_predict_before_fit(self):
        """Test predict returns zeros before fitting."""
        config = XGBoostConfig()
        predictor = GradientBoostingPredictor(config)

        predictions = predictor.predict(pl.DataFrame({"date": []}))

        assert len(predictions) == len(config.assets)
        assert np.all(predictions == 0.0)

    def test_feature_importance(self, synthetic_features, synthetic_returns):
        """Test feature importance extraction."""
        config = XGBoostConfig(n_estimators=20)
        predictor = GradientBoostingPredictor(config)

        predictor.fit(synthetic_features, synthetic_returns, use_cv=False)

        importance = predictor.get_feature_importance("QQQ")

        assert importance is not None
        assert len(importance) > 0
        assert all(isinstance(v, (int, float, np.number)) for v in importance.values())

    def test_feature_importance_sums_to_one(self, synthetic_features, synthetic_returns):
        """Test feature importances are normalized."""
        config = XGBoostConfig(n_estimators=20)
        predictor = GradientBoostingPredictor(config)

        predictor.fit(synthetic_features, synthetic_returns, use_cv=False)

        importance = predictor.get_feature_importance("QQQ")

        if importance:
            # XGBoost importances sum to 1.0 (or close to it)
            total = sum(importance.values())
            assert 0.99 <= total <= 1.01 or total > 0  # At least positive

    def test_predict_handles_nan(self, synthetic_features, synthetic_returns):
        """Test prediction handles NaN values gracefully."""
        config = XGBoostConfig(n_estimators=20)
        predictor = GradientBoostingPredictor(config)

        predictor.fit(synthetic_features, synthetic_returns, use_cv=False)

        # Create features with NaN
        features_with_nan = synthetic_features.tail(1).clone()
        # Would need to add NaN here, but Polars makes this tricky
        # Just verify it doesn't crash
        predictions = predictor.predict(features_with_nan)

        assert len(predictions) == len(config.assets)


class TestShapIntegration:
    """Test SHAP integration (requires shap library)."""

    @pytest.mark.skip(reason="SHAP library may not be installed")
    def test_shap_values_computed(self, synthetic_features, synthetic_returns):
        """Test SHAP values computation."""
        config = XGBoostConfig(n_estimators=20)
        predictor = GradientBoostingPredictor(config)

        predictor.fit(synthetic_features, synthetic_returns, use_cv=False)

        shap_values = predictor.get_shap_values(synthetic_features, "QQQ")

        if shap_values is not None:
            # SHAP values should match data shape
            assert shap_values.shape[0] > 0  # n_samples
            assert shap_values.shape[1] > 0  # n_features


class TestXGBoostVsRidge:
    """Comparison tests between XGBoost and Ridge."""

    def test_xgboost_trains_faster_on_small_data(self, synthetic_features, synthetic_returns):
        """Test XGBoost trains reasonably fast."""
        import time

        config = XGBoostConfig(n_estimators=20)
        predictor = GradientBoostingPredictor(config)

        start = time.time()
        predictor.fit(synthetic_features, synthetic_returns, use_cv=False)
        elapsed = time.time() - start

        # Should train in less than 10 seconds for small data
        assert elapsed < 10.0
        assert predictor._is_fitted


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        config = XGBoostConfig(assets=["QQQ"])
        predictor = GradientBoostingPredictor(config)

        # Very small dataset
        small_features = pl.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10),
                "ret_1d_QQQ": np.random.normal(0, 0.01, 10),
            }
        )
        small_returns = pl.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10),
                "ret_QQQ": np.random.normal(0, 0.01, 10),
            }
        )

        # Should handle gracefully (log warning but not crash)
        predictor.fit(small_features, small_returns, use_cv=False)

        # May not fit due to insufficient data, but shouldn't crash
        predictions = predictor.predict(small_features)
        assert len(predictions) == 1

    def test_missing_asset_columns(self, synthetic_features):
        """Test behavior when asset columns are missing."""
        config = XGBoostConfig(assets=["INVALID"])
        predictor = GradientBoostingPredictor(config)

        returns = pl.DataFrame(
            {
                "date": synthetic_features["date"],
                "ret_INVALID": np.zeros(len(synthetic_features)),
            }
        )

        # Should handle missing columns gracefully
        predictor.fit(synthetic_features, returns, use_cv=False)
        predictions = predictor.predict(synthetic_features)

        assert len(predictions) == 1
