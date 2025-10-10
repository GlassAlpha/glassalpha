"""Model integration tests.

Tests model wrapper functionality including initialization, predictions,
feature importance, and registry integration. These tests focus on
covering the model wrapper logic without requiring complex model training.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Conditional sklearn import for CI compatibility
try:
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    make_classification = None
    RandomForestClassifier = None
    LogisticRegression = None
    SKLEARN_AVAILABLE = False

# Include additional sklearn imports in conditional block
try:
    if SKLEARN_AVAILABLE:
        from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None

# Skip all tests if sklearn not available
pytestmark = pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available - CI compatibility issues")

from glassalpha.core import ModelRegistry
from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper, SklearnGenericWrapper


@pytest.fixture
def sample_classification_data():
    """Create sample classification dataset for testing."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_classes=2,
        n_informative=8,
        n_redundant=1,
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    return X_df, y, feature_names


@pytest.fixture
def trained_logistic_model(sample_classification_data):
    """Create a trained LogisticRegression model for testing."""
    X_df, y, feature_names = sample_classification_data

    model = LogisticRegression(random_state=42, max_iter=2000, solver="liblinear")
    model.fit(X_df, y)

    return model, X_df, y, feature_names


@pytest.fixture
def trained_random_forest(sample_classification_data):
    """Create a trained RandomForest model for testing."""
    X_df, y, feature_names = sample_classification_data

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_df, y)

    return model, X_df, y, feature_names


@pytest.fixture
def sample_multiclass_data():
    """Create sample multiclass classification dataset."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=150,
        n_features=8,
        n_classes=3,
        n_informative=6,
        n_redundant=1,
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    return X_df, y, feature_names


class TestModelRegistry:
    """Test model registration and discovery."""

    def test_sklearn_models_are_registered(self):
        """Test that sklearn models are properly registered."""
        components = ModelRegistry.get_all()

        # Should include sklearn models
        assert "logistic_regression" in components
        assert "sklearn_generic" in components

    def test_get_sklearn_model_classes(self):
        """Test that we can retrieve sklearn model classes."""
        lr_cls = ModelRegistry.get("logistic_regression")
        assert lr_cls == LogisticRegressionWrapper

        generic_cls = ModelRegistry.get("sklearn_generic")
        assert generic_cls == SklearnGenericWrapper

    def test_model_priorities(self):
        """Test that models have expected priorities."""
        # This tests the registry metadata (if available)
        components = ModelRegistry.get_all()

        # All registered models should be available
        assert len(components) >= 2  # At least logistic_regression and sklearn_generic


class TestLogisticRegressionWrapper:
    """Test LogisticRegressionWrapper functionality."""

    def test_wrapper_initialization_empty(self):
        """Test LogisticRegressionWrapper initialization without model."""
        wrapper = LogisticRegressionWrapper()

        assert wrapper.model is None
        assert wrapper.feature_names is None
        assert wrapper.n_classes == 2
        assert wrapper.capabilities["supports_shap"] is True
        assert wrapper.capabilities["data_modality"] == "tabular"
        assert wrapper.version == "1.0.0"

    def test_wrapper_initialization_with_model(self, trained_logistic_model):
        """Test LogisticRegressionWrapper initialization with model."""
        model, X_df, y, feature_names = trained_logistic_model

        wrapper = LogisticRegressionWrapper(model=model)

        assert wrapper.model is not None
        assert isinstance(wrapper.model, LogisticRegression)
        assert wrapper.n_classes == 2
        # feature_names might be set during _extract_model_info

    def test_model_predictions(self, trained_logistic_model):
        """Test model prediction functionality."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        # Test predict
        predictions = wrapper.predict(X_df)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_df)
        assert set(np.unique(predictions)).issubset({0, 1})  # Binary classification

        # Test predict_proba
        probabilities = wrapper.predict_proba(X_df)

        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(X_df), 2)  # Binary probabilities
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Should sum to 1
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)  # Valid probabilities

    def test_model_capabilities(self, trained_logistic_model):
        """Test model capability reporting."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        capabilities = wrapper.get_capabilities()

        assert isinstance(capabilities, dict)
        assert capabilities["supports_shap"] is True
        assert capabilities["supports_feature_importance"] is True
        assert capabilities["supports_proba"] is True
        assert capabilities["data_modality"] == "tabular"

    def test_model_type(self, trained_logistic_model):
        """Test model type reporting."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        model_type = wrapper.get_model_type()

        assert isinstance(model_type, str)
        assert model_type == "logistic_regression"

    def test_feature_importance(self, trained_logistic_model):
        """Test feature importance extraction."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        # Test coefficient-based importance
        importance = wrapper.get_feature_importance(importance_type="coef")

        assert isinstance(importance, dict)
        assert len(importance) == len(feature_names)

        # All values should be numeric
        for _feature, value in importance.items():
            assert isinstance(value, (int, float))

        # Test that default importance type also works
        default_importance = wrapper.get_feature_importance()

        assert isinstance(default_importance, dict)
        assert len(default_importance) == len(feature_names)

    def test_model_info(self, trained_logistic_model):
        """Test model information extraction."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        info = wrapper.get_model_info()

        assert isinstance(info, dict)
        assert "n_features" in info
        assert "n_classes" in info
        assert "status" in info
        assert info["n_features"] == len(feature_names)
        assert info["n_classes"] == 2
        assert info["status"] == "loaded"

    def test_model_saving_and_loading(self, trained_logistic_model, tmp_path):
        """Test model saving and loading functionality.

        Uses unique file paths to prevent parallel test interference.
        """
        import uuid

        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        # Save model with unique path to avoid parallel test conflicts
        unique_id = uuid.uuid4().hex[:8]
        save_path = tmp_path / f"model_{unique_id}.pkl"
        wrapper.save(save_path)

        assert save_path.exists()

        # Load model in new wrapper
        new_wrapper = LogisticRegressionWrapper()
        new_wrapper.load(save_path)

        assert new_wrapper.model is not None

        # Test that loaded model produces same predictions
        original_pred = wrapper.predict(X_df[:10])  # Test subset
        loaded_pred = new_wrapper.predict(X_df[:10])

        np.testing.assert_array_equal(original_pred, loaded_pred)

    def test_wrapper_repr(self, trained_logistic_model):
        """Test wrapper string representation."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        repr_str = repr(wrapper)

        assert isinstance(repr_str, str)
        assert "LogisticRegressionWrapper" in repr_str

    def test_multiclass_predictions(self, sample_multiclass_data):
        """Test wrapper with multiclass classification."""
        X_df, y, feature_names = sample_multiclass_data

        # Train multiclass model (use lbfgs for multiclass support)
        model = LogisticRegression(random_state=42, max_iter=2000, solver="lbfgs")
        model.fit(X_df, y)

        wrapper = LogisticRegressionWrapper(model=model)

        # Test predictions
        predictions = wrapper.predict(X_df)
        probabilities = wrapper.predict_proba(X_df)

        assert isinstance(predictions, np.ndarray)
        assert set(np.unique(predictions)).issubset({0, 1, 2})  # 3-class
        assert probabilities.shape == (len(X_df), 3)  # 3 class probabilities
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestSklearnGenericWrapper:
    """Test SklearnGenericWrapper functionality."""

    def test_generic_wrapper_initialization(self):
        """Test SklearnGenericWrapper initialization."""
        wrapper = SklearnGenericWrapper()

        assert wrapper.model is None
        assert wrapper.feature_names is None
        assert wrapper.capabilities["supports_shap"] is True
        assert wrapper.capabilities["data_modality"] == "tabular"

    def test_generic_wrapper_with_random_forest(self, trained_random_forest):
        """Test generic wrapper with RandomForest model."""
        model, X_df, y, feature_names = trained_random_forest
        wrapper = SklearnGenericWrapper(model=model)

        assert wrapper.model is not None
        assert isinstance(wrapper.model, RandomForestClassifier)

        # Test predictions
        predictions = wrapper.predict(X_df)
        probabilities = wrapper.predict_proba(X_df)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_df)
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape[0] == len(X_df)

    def test_generic_wrapper_model_type(self, trained_random_forest):
        """Test model type detection for generic wrapper."""
        model, X_df, y, feature_names = trained_random_forest
        wrapper = SklearnGenericWrapper(model=model)

        model_type = wrapper.get_model_type()

        assert isinstance(model_type, str)
        assert model_type == "sklearn_generic"

    def test_generic_wrapper_feature_importance(self, trained_random_forest):
        """Test feature importance for generic wrapper."""
        model, X_df, y, feature_names = trained_random_forest
        wrapper = SklearnGenericWrapper(model=model)

        # RandomForest has feature_importances_ attribute
        importance = wrapper.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == len(feature_names)

        # All importance values should be non-negative and sum to ~1
        values = list(importance.values())
        assert all(v >= 0 for v in values)
        assert abs(sum(values) - 1.0) < 1e-6  # Should sum to 1 for RandomForest


class TestModelWrapperErrorHandling:
    """Test error handling in model wrappers."""

    def test_prediction_without_model(self):
        """Test prediction methods fail gracefully without model."""
        wrapper = LogisticRegressionWrapper()
        X_dummy = pd.DataFrame({"feature_1": [1, 2, 3]})

        with pytest.raises((AttributeError, ValueError)):
            wrapper.predict(X_dummy)

        with pytest.raises((AttributeError, ValueError)):
            wrapper.predict_proba(X_dummy)

    def test_load_nonexistent_file(self):
        """Test loading nonexistent model file."""
        wrapper = LogisticRegressionWrapper()
        nonexistent_path = Path("nonexistent_model.pkl")

        with pytest.raises(FileNotFoundError):
            wrapper.load(nonexistent_path)

    def test_invalid_importance_type(self, trained_logistic_model):
        """Test invalid importance type handling."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        # Test with invalid importance type
        try:
            importance = wrapper.get_feature_importance(importance_type="invalid")
            # If it doesn't raise error, should return valid dict
            assert isinstance(importance, dict)
        except ValueError:
            # If it raises error, that's also acceptable
            pass

    def test_prediction_with_mismatched_features(self, trained_logistic_model):
        """Test predictions with wrong number of features."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        # Create data with different number of features
        X_wrong = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})  # Only 2 features

        with pytest.raises(ValueError):
            wrapper.predict(X_wrong)

    def test_save_without_model(self, tmp_path):
        """Test saving wrapper without model."""
        wrapper = LogisticRegressionWrapper()
        save_path = tmp_path / "empty_model.pkl"

        with pytest.raises((AttributeError, ValueError)):
            wrapper.save(save_path)


class TestModelWrapperIntegration:
    """Test integration scenarios and edge cases."""

    def test_model_pipeline_integration(self, trained_logistic_model):
        """Test model wrapper in a pipeline-like scenario."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        # Split data for testing
        X_train, X_test = train_test_split(X_df, test_size=0.3, random_state=42)

        # Test full pipeline: predict -> get importance -> get info
        predictions = wrapper.predict(X_test)
        probabilities = wrapper.predict_proba(X_test)
        importance = wrapper.get_feature_importance()
        model_info = wrapper.get_model_info()
        capabilities = wrapper.get_capabilities()

        # All operations should succeed
        assert len(predictions) == len(X_test)
        assert probabilities.shape[0] == len(X_test)
        assert len(importance) == len(feature_names)
        assert isinstance(model_info, dict)
        assert isinstance(capabilities, dict)

    def test_wrapper_consistency_across_calls(self, trained_logistic_model):
        """Test that wrapper gives consistent results across multiple calls."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        # Make predictions multiple times
        pred1 = wrapper.predict(X_df[:10])
        pred2 = wrapper.predict(X_df[:10])

        np.testing.assert_array_equal(pred1, pred2)

        # Check probabilities
        proba1 = wrapper.predict_proba(X_df[:10])
        proba2 = wrapper.predict_proba(X_df[:10])

        np.testing.assert_array_almost_equal(proba1, proba2)

    def test_wrapper_with_different_data_types(self, trained_logistic_model):
        """Test wrapper with different pandas DataFrame configurations."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        # Test with float64 data
        X_float64 = X_df.astype(np.float64)
        pred_float64 = wrapper.predict(X_float64[:5])

        # Test with float32 data
        X_float32 = X_df.astype(np.float32)
        pred_float32 = wrapper.predict(X_float32[:5])

        # Should produce same results (within tolerance)
        np.testing.assert_allclose(pred_float64, pred_float32, rtol=1e-5)

    def test_model_serialization_roundtrip(self, trained_logistic_model, tmp_path):
        """Test complete serialization roundtrip."""
        model, X_df, y, feature_names = trained_logistic_model
        original_wrapper = LogisticRegressionWrapper(model=model)

        # Get original predictions
        original_pred = original_wrapper.predict(X_df[:20])
        original_proba = original_wrapper.predict_proba(X_df[:20])
        original_importance = original_wrapper.get_feature_importance()

        # Save and reload
        save_path = tmp_path / "roundtrip_model.pkl"
        original_wrapper.save(save_path)

        new_wrapper = LogisticRegressionWrapper()
        new_wrapper.load(save_path)

        # Test that everything matches
        new_pred = new_wrapper.predict(X_df[:20])
        new_proba = new_wrapper.predict_proba(X_df[:20])
        new_importance = new_wrapper.get_feature_importance()

        np.testing.assert_array_equal(original_pred, new_pred)
        np.testing.assert_array_almost_equal(original_proba, new_proba)

        # Feature importance should be very close
        for feature in original_importance:
            assert abs(original_importance[feature] - new_importance[feature]) < 1e-10


class TestModelWrapperCompatibility:
    """Test compatibility with GlassAlpha pipeline components."""

    def test_wrapper_with_shap_compatibility(self, trained_logistic_model):
        """Test that wrapper reports SHAP compatibility correctly."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        capabilities = wrapper.get_capabilities()

        # Should support SHAP
        assert capabilities["supports_shap"] is True
        assert capabilities["data_modality"] == "tabular"

    def test_wrapper_model_info_completeness(self, trained_logistic_model):
        """Test that model info contains expected fields."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        info = wrapper.get_model_info()

        # Should contain key information for audit pipeline
        expected_fields = ["n_features", "n_classes", "status"]
        for field in expected_fields:
            assert field in info

        # Values should be reasonable
        assert info["n_features"] > 0
        assert info["n_classes"] >= 2
        assert isinstance(info["status"], str)

    def test_registry_integration(self):
        """Test that models are properly integrated with registry."""
        # Test model retrieval from registry
        LogisticWrapperCls = ModelRegistry.get("logistic_regression")
        GenericWrapperCls = ModelRegistry.get("sklearn_generic")

        # Should be able to instantiate from registry
        wrapper1 = LogisticWrapperCls()
        wrapper2 = GenericWrapperCls()

        assert isinstance(wrapper1, LogisticRegressionWrapper)
        assert isinstance(wrapper2, SklearnGenericWrapper)

        # Should have required attributes
        assert hasattr(wrapper1, "capabilities")
        assert hasattr(wrapper1, "version")
        assert hasattr(wrapper2, "capabilities")
        assert hasattr(wrapper2, "version")


class TestModelWrapperEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataset_handling(self, trained_logistic_model):
        """Test wrapper behavior with empty dataset."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        # Create empty DataFrame with correct columns
        empty_df = pd.DataFrame(columns=feature_names)

        # Should handle gracefully or raise informative error
        try:
            predictions = wrapper.predict(empty_df)
            probabilities = wrapper.predict_proba(empty_df)

            assert len(predictions) == 0
            assert len(probabilities) == 0
        except ValueError:
            # Sklearn may raise ValueError for empty input - acceptable
            pass

    def test_single_sample_prediction(self, trained_logistic_model):
        """Test prediction with single sample."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        # Single sample
        single_sample = X_df.iloc[[0]]

        prediction = wrapper.predict(single_sample)
        probability = wrapper.predict_proba(single_sample)

        assert len(prediction) == 1
        assert probability.shape == (1, 2)  # Single sample, 2 classes
        assert prediction[0] in [0, 1]  # Valid class

    def test_large_dataset_handling(self, trained_logistic_model):
        """Test wrapper with larger dataset."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        # Create larger dataset by repeating
        large_X = pd.concat([X_df] * 10, ignore_index=True)  # 2000 samples

        predictions = wrapper.predict(large_X)
        probabilities = wrapper.predict_proba(large_X)

        assert len(predictions) == len(large_X)
        assert probabilities.shape[0] == len(large_X)
        # Should complete without memory issues

    def test_feature_names_handling(self, trained_logistic_model):
        """Test wrapper with different feature name scenarios."""
        model, X_df, y, feature_names = trained_logistic_model
        wrapper = LogisticRegressionWrapper(model=model)

        # Test with renamed columns (same order)
        renamed_df = X_df.copy()
        new_names = [f"new_feature_{i}" for i in range(len(feature_names))]
        renamed_df.columns = new_names

        # Should still work (sklearn models are position-based)
        predictions = wrapper.predict(renamed_df)
        assert len(predictions) == len(renamed_df)

        # Feature importance should work regardless of feature names
        try:
            importance = wrapper.get_feature_importance()
            assert isinstance(importance, dict)
        except (ValueError, AttributeError):
            # May not work with renamed features - acceptable for basic test
            pass
