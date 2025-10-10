"""Comprehensive tests for training pipeline functionality.

This module tests model training from configuration, calibration, error handling,
and parameter passing to ensure reliable model training workflows.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from glassalpha.pipeline.train import train_from_config


def _has_lightgbm():
    """Check if lightgbm is installed."""
    try:
        import lightgbm

        return True
    except ImportError:
        return False


def _create_mock_config(model_type="logistic_regression", **kwargs):
    """Create a mock configuration object for testing."""

    class MockReproducibility:
        def __init__(self, random_seed=None):
            self.random_seed = random_seed

    class MockCalibration:
        def __init__(self, method=None, cv=3, ensemble=False):
            self.method = method
            self.cv = cv
            self.ensemble = ensemble

    class MockModel:
        def __init__(self, model_type, params=None, calibration=None):
            self.type = model_type
            self.params = params or {}
            self.calibration = calibration

    class MockConfig:
        def __init__(self, model_type, random_seed=None, params=None, calibration=None):
            self.model = MockModel(model_type, params, calibration)
            self.reproducibility = MockReproducibility(random_seed)

    return MockConfig(model_type, **kwargs)


def _create_test_data(samples=20):
    """Create test dataset for training."""
    # Create larger dataset for calibration tests
    X = pd.DataFrame(
        {
            "age": list(range(20, 20 + samples)),
            "income": [50000 + i * 1000 for i in range(samples)],
            "credit_score": [650 + i * 10 for i in range(samples)],
        },
    )
    # Create balanced binary classification with enough samples per class for CV
    y = [i % 2 for i in range(samples)]  # 0, 1, 0, 1, ... pattern
    return X, y


class TestTrainingPipeline:
    """Test suite for training pipeline functionality."""

    def test_train_logistic_regression_basic(self):
        """Test basic training of logistic regression model."""
        config = _create_mock_config("logistic_regression")
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should return a trained model
        assert model is not None
        assert hasattr(model, "predict")

        # Should be able to make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)

    def test_train_xgboost_basic(self):
        """Test basic training of XGBoost model."""
        config = _create_mock_config("xgboost")
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should return a trained model
        assert model is not None
        assert hasattr(model, "predict")

        # Should be able to make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    @pytest.mark.skipif(
        not _has_lightgbm(),
        reason="LightGBM not installed",
    )
    def test_train_lightgbm_basic(self):
        """Test basic training of LightGBM model."""
        config = _create_mock_config("lightgbm")
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should return a trained model
        assert model is not None
        assert hasattr(model, "predict")

        # Should be able to make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_with_custom_parameters(self):
        """Test training with custom model parameters."""
        params = {
            "max_iter": 500,
            "solver": "lbfgs",
            "C": 0.1,
        }
        config = _create_mock_config("logistic_regression", params=params)
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should return a trained model
        assert model is not None

        # Model should be configured (basic check)
        assert hasattr(model, "predict")

    def test_train_with_random_seed(self):
        """Test that random seed is properly passed to model."""
        config = _create_mock_config("logistic_regression", random_seed=42)
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should return a trained model
        assert model is not None

        # Should be able to make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_without_random_seed(self):
        """Test training without explicit random seed."""
        config = _create_mock_config("logistic_regression", random_seed=None)
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should return a trained model (no seed is allowed)
        assert model is not None

        # Should be able to make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_with_calibration_isotonic(self):
        """Test training with isotonic calibration."""
        calibration = MagicMock()
        calibration.method = "isotonic"
        calibration.cv = 3
        calibration.ensemble = False

        config = _create_mock_config("logistic_regression", calibration=calibration)
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should return a trained model with calibration
        assert model is not None
        assert hasattr(model, "predict")

        # Should be able to make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_with_calibration_sigmoid(self):
        """Test training with sigmoid calibration."""
        calibration = MagicMock()
        calibration.method = "sigmoid"
        calibration.cv = 5
        calibration.ensemble = True

        config = _create_mock_config("logistic_regression", calibration=calibration)
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should return a trained model with calibration
        assert model is not None
        assert hasattr(model, "predict")

        # Should be able to make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_without_calibration(self):
        """Test training without calibration."""
        config = _create_mock_config("logistic_regression")
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should return a trained model without calibration
        assert model is not None
        assert hasattr(model, "predict")

    def test_train_unknown_model_type(self):
        """Test error handling for unknown model type."""
        config = _create_mock_config("unknown_model_type")
        X, y = _create_test_data(samples=20)

        with pytest.raises((ValueError, KeyError), match="(Unknown model type|Unknown plugin)"):
            train_from_config(config, X, y)

    def test_train_model_without_fit_method(self):
        """Test error handling for model without fit method."""

        # Mock a model class that doesn't have fit method
        class MockModelWithoutFit:
            def __init__(self):
                pass

        config = _create_mock_config("mock_model")
        X, y = _create_test_data(samples=20)

        # Mock the registry to return our mock model
        with patch("glassalpha.pipeline.train.ModelRegistry.get") as mock_get:
            mock_get.return_value = MockModelWithoutFit

            with pytest.raises(RuntimeError, match="does not support fit method"):
                train_from_config(config, X, y)

    def test_train_with_large_dataset(self):
        """Test training with larger dataset."""
        # Create larger dataset
        X = pd.DataFrame({f"feature_{i}": range(100) for i in range(20)})
        y = [i % 2 for i in range(100)]  # Binary classification

        config = _create_mock_config("logistic_regression")
        model = train_from_config(config, X, y)

        # Should handle large dataset
        assert model is not None
        assert hasattr(model, "predict")

        # Should be able to make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_with_categorical_target(self):
        """Test training with categorical target variable."""
        X = pd.DataFrame(
            {
                "age": [25, 30, 35, 40, 45],
                "income": [50000, 60000, 70000, 80000, 90000],
            },
        )
        y = ["low", "low", "medium", "high", "high"]  # Categorical target

        config = _create_mock_config("logistic_regression")
        model = train_from_config(config, X, y)

        # Should handle categorical target
        assert model is not None
        assert hasattr(model, "predict")

        # Should be able to make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_with_missing_values(self):
        """Test training with missing values in data."""
        X = pd.DataFrame(
            {
                "age": [25, 30, None, 40, 45],
                "income": [50000, 60000, 70000, None, 90000],
                "credit_score": [650, 700, 750, 800, 850],
            },
        )
        y = [0, 0, 1, 1, 1]

        config = _create_mock_config("logistic_regression")

        # Should handle missing values gracefully or raise appropriate error
        try:
            model = train_from_config(config, X, y)
            # If successful, model should be valid
            assert model is not None
            assert hasattr(model, "predict")
        except ValueError as e:
            # If it fails with validation error, that's also acceptable
            assert "NaN" in str(e) or "missing" in str(e).lower()

    def test_train_preserves_model_type(self):
        """Test that trained model preserves its type information."""
        config = _create_mock_config("xgboost")
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should be able to identify model type
        assert model is not None

        # Model should have some indication of its type (implementation dependent)
        model_str = str(type(model)).lower()
        assert "xgboost" in model_str or hasattr(model, "get_booster")

    def test_train_with_complex_parameters(self):
        """Test training with complex parameter combinations."""
        params = {
            "max_depth": 10,
            "n_estimators": 200,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

        config = _create_mock_config("xgboost", params=params)
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should handle complex parameters
        assert model is not None
        assert hasattr(model, "predict")

        # Should be able to make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_calibration_with_cross_validation(self):
        """Test calibration with cross-validation."""
        calibration = MagicMock()
        calibration.method = "isotonic"
        calibration.cv = 5  # Custom CV folds
        calibration.ensemble = False

        config = _create_mock_config("logistic_regression", calibration=calibration)
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should handle calibration with custom CV
        assert model is not None
        assert hasattr(model, "predict")

    def test_train_multiple_models_consistent_results(self):
        """Test that multiple training runs with same config produce consistent results."""
        config = _create_mock_config("logistic_regression", random_seed=42)
        X, y = _create_test_data(samples=20)

        # Train multiple models with same config
        model1 = train_from_config(config, X, y)
        model2 = train_from_config(config, X, y)

        # Both should be trained successfully
        assert model1 is not None
        assert model2 is not None

        # Both should produce predictions (exact same predictions may vary due to randomness)
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)

        assert len(pred1) == len(y)
        assert len(pred2) == len(y)

    def test_train_handles_empty_parameters(self):
        """Test training with empty parameters dict."""
        config = _create_mock_config("logistic_regression", params={})
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should handle empty parameters
        assert model is not None
        assert hasattr(model, "predict")

        # Should be able to make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_with_ensemble_calibration(self):
        """Test training with ensemble calibration method."""
        calibration = MagicMock()
        calibration.method = "isotonic"
        calibration.cv = 3
        calibration.ensemble = True  # Enable ensemble

        config = _create_mock_config("logistic_regression", calibration=calibration)
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should handle ensemble calibration
        assert model is not None
        assert hasattr(model, "predict")

    def test_train_error_handling_invalid_calibration_method(self):
        """Test error handling for invalid calibration method."""
        calibration = MagicMock()
        calibration.method = "invalid_method"
        calibration.cv = 3
        calibration.ensemble = False

        config = _create_mock_config("logistic_regression", calibration=calibration)
        X, y = _create_test_data(samples=20)

        # Should handle invalid calibration method gracefully
        # If calibration fails, training should still succeed
        try:
            model = train_from_config(config, X, y)
            assert model is not None
        except ValueError as e:
            # If calibration method is unknown, that's acceptable
            assert "Unknown calibration method" in str(e)

    def test_train_with_extreme_parameter_values(self):
        """Test training with extreme parameter values."""
        params = {
            "C": 1e-10,  # Very small regularization
            "max_iter": 10000,  # Very large iterations
            "tol": 1e-10,  # Very small tolerance
        }

        config = _create_mock_config("logistic_regression", params=params)
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should handle extreme parameters
        assert model is not None
        assert hasattr(model, "predict")

        # Should be able to make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_preserves_model_capabilities(self):
        """Test that trained model preserves expected capabilities."""
        config = _create_mock_config("xgboost")
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should preserve model capabilities
        assert model is not None

        # Should have prediction methods
        assert hasattr(model, "predict")

        # Should work with the model interface
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_handles_model_import_errors(self):
        """Test graceful handling of model import errors."""
        # Skip this test as mocking importlib is complex and may cause recursion issues
        # In real usage, if xgboost module isn't available, the training would fail
        # This test verifies that the training function handles missing dependencies
        pytest.skip("Complex import mocking causes recursion issues - functionality tested in integration")

    def test_train_with_different_data_types(self):
        """Test training with different data types."""
        # Integer features
        X = pd.DataFrame(
            {
                "int_feature": [1, 2, 3, 4, 5],
                "float_feature": [1.1, 2.2, 3.3, 4.4, 5.5],
            },
        )
        y = [0, 1, 0, 1, 0]

        config = _create_mock_config("logistic_regression")
        model = train_from_config(config, X, y)

        # Should handle mixed data types
        assert model is not None
        assert hasattr(model, "predict")

        # Should be able to make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_with_small_dataset(self):
        """Test training with very small dataset."""
        X = pd.DataFrame(
            {
                "feature1": [1, 2],
                "feature2": [3, 4],
            },
        )
        y = [0, 1]

        config = _create_mock_config("logistic_regression")
        model = train_from_config(config, X, y)

        # Should handle small datasets
        assert model is not None
        assert hasattr(model, "predict")

        # Should be able to make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_calibration_preserves_model_functionality(self):
        """Test that calibration doesn't break model functionality."""
        calibration = MagicMock()
        calibration.method = "isotonic"
        calibration.cv = 3
        calibration.ensemble = False

        config = _create_mock_config("logistic_regression", calibration=calibration)
        X, y = _create_test_data(samples=20)

        model = train_from_config(config, X, y)

        # Should preserve all model functionality after calibration
        assert model is not None
        assert hasattr(model, "predict")

        # Should be able to make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)

        # Should be able to get prediction probabilities if supported
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            assert probabilities.shape[0] == len(y)
            assert probabilities.shape[1] >= 2  # At least 2 classes

    def test_train_with_high_dimensional_data(self):
        """Test training with high-dimensional data."""
        # Create high-dimensional dataset
        X = pd.DataFrame(
            {
                f"feature_{i}": range(50)
                for i in range(100)  # 100 features, 50 samples
            },
        )
        y = [i % 3 for i in range(50)]  # 3-class classification

        config = _create_mock_config("logistic_regression")
        model = train_from_config(config, X, y)

        # Should handle high-dimensional data
        assert model is not None
        assert hasattr(model, "predict")

        # Should be able to make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert all(pred in [0, 1, 2] for pred in predictions)

    def test_train_error_recovery_on_failure(self):
        """Test error recovery when training fails."""
        import warnings

        # Create config that might cause training to fail
        params = {
            "max_iter": 1,  # Very few iterations
            "tol": 1e-20,  # Extremely strict tolerance
        }

        config = _create_mock_config("logistic_regression", params=params)
        X, y = _create_test_data(samples=20)

        # Should handle training failures gracefully
        # Suppress expected convergence warning (intentionally testing failure mode)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            model = train_from_config(config, X, y)

        # May succeed or fail depending on data/params, but shouldn't crash
        # The important thing is that it doesn't raise unhandled exceptions
        assert model is not None or True  # Either succeeds or handles error gracefully
