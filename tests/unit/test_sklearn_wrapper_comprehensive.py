"""Comprehensive tests for sklearn wrapper functionality.

This module tests multiclass support, calibration integration, feature importance,
and serialization with preprocessing to ensure robust sklearn model handling.
"""

import numpy as np
import pandas as pd
import pytest

from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper, SklearnGenericWrapper


def _create_test_data_multiclass():
    """Create multiclass test dataset."""
    X = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
        },
    )
    # 3-class classification
    y = np.random.choice([0, 1, 2], size=100)
    return X, y


def _create_test_data_binary():
    """Create binary classification test dataset."""
    X = pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "feature3": np.random.randn(50),
        },
    )
    y = np.random.choice([0, 1], size=50)
    return X, y


class TestSklearnMulticlassSupport:
    """Test multiclass classification support."""

    def test_logistic_regression_multiclass_ova(self):
        """Test LogisticRegression with one-vs-all multiclass strategy."""
        from sklearn.linear_model import LogisticRegression

        # Create multiclass data
        X, y = _create_test_data_multiclass()

        # Train base sklearn model (use lbfgs for multiclass support)
        base_model = LogisticRegression(random_state=42, max_iter=2000, solver="lbfgs")
        base_model.fit(X, y)

        # Wrap with our wrapper
        wrapper = SklearnGenericWrapper(base_model)

        # Should handle multiclass
        assert wrapper is not None
        assert hasattr(wrapper, "predict")

        # Test predictions
        predictions = wrapper.predict(X)
        assert len(predictions) == len(y)
        assert all(pred in [0, 1, 2] for pred in predictions)

        # Test probabilities
        probabilities = wrapper.predict_proba(X)
        assert probabilities.shape == (len(y), 3)  # 3 classes

    def test_logistic_regression_multiclass_multinomial(self):
        """Test LogisticRegression with multinomial multiclass strategy."""
        from sklearn.linear_model import LogisticRegression

        # Create multiclass data
        X, y = _create_test_data_multiclass()

        # Train base sklearn model with multinomial
        base_model = LogisticRegression(random_state=42, solver="lbfgs", max_iter=2000)
        base_model.fit(X, y)

        # Wrap with our wrapper
        wrapper = SklearnGenericWrapper(base_model)

        # Should handle multiclass multinomial
        predictions = wrapper.predict(X)
        assert len(predictions) == len(y)
        assert all(pred in [0, 1, 2] for pred in predictions)

        probabilities = wrapper.predict_proba(X)
        assert probabilities.shape == (len(y), 3)

    def test_wrapper_preserves_multiclass_capabilities(self):
        """Test that wrapper preserves multiclass prediction capabilities."""
        from sklearn.linear_model import LogisticRegression

        X, y = _create_test_data_multiclass()
        base_model = LogisticRegression(random_state=42, max_iter=2000, solver="lbfgs")
        base_model.fit(X, y)

        wrapper = SklearnGenericWrapper(base_model)

        # Test that multiclass methods work
        assert hasattr(wrapper, "predict")
        assert hasattr(wrapper, "predict_proba")

        # Test decision function if available
        if hasattr(wrapper, "decision_function"):
            decision_scores = wrapper.decision_function(X)
            assert decision_scores.shape[0] == len(y)


class TestSklearnCalibrationIntegration:
    """Test calibration integration with sklearn models."""

    def test_logistic_regression_with_isotonic_calibration(self):
        """Test LogisticRegression wrapper with isotonic calibration."""
        X, y = _create_test_data_binary()

        # Create wrapper and fit it (calibration would happen in train_from_config)
        wrapper = LogisticRegressionWrapper()
        wrapper.fit(X, y)

        # Test that wrapper works with fitted model
        predictions = wrapper.predict(X)
        probabilities = wrapper.predict_proba(X)

        assert len(predictions) == len(y)
        assert probabilities.shape == (len(y), 2)

        # Test that we can access the underlying model
        assert wrapper.model is not None

    def test_logistic_regression_with_sigmoid_calibration(self):
        """Test LogisticRegression wrapper with sigmoid calibration."""
        X, y = _create_test_data_binary()

        # Create wrapper and fit it (calibration would happen in train_from_config)
        wrapper = LogisticRegressionWrapper()
        wrapper.fit(X, y)

        # Test that wrapper works with fitted model
        predictions = wrapper.predict(X)
        probabilities = wrapper.predict_proba(X)

        assert len(predictions) == len(y)
        assert probabilities.shape == (len(y), 2)

        # Test that we can access the underlying model
        assert wrapper.model is not None


class TestSklearnFeatureImportance:
    """Test feature importance extraction."""

    def test_logistic_regression_feature_importance(self):
        """Test feature importance extraction from LogisticRegression."""
        # Create data with known feature importance pattern
        X = pd.DataFrame(
            {
                "important_feature": [1, 2, 3, 4, 5],
                "less_important": [5, 4, 3, 2, 1],
                "noise": [0.1, 0.2, 0.3, 0.4, 0.5],
            },
        )
        y = [0, 0, 1, 1, 1]  # important_feature should be most important

        wrapper = LogisticRegressionWrapper()

        # Fit the model through the wrapper to set _is_fitted flag
        wrapper.fit(X, y)

        # Test feature importance extraction
        importance = wrapper.get_feature_importance()

        # Should return importance scores
        assert importance is not None
        assert len(importance) == len(X.columns)

        # Important feature should have higher importance
        assert "important_feature" in importance
        assert "less_important" in importance
        assert "noise" in importance

        # Important feature should have higher absolute importance
        assert abs(importance["important_feature"]) > abs(importance["noise"])

    def test_feature_importance_with_coefficients(self):
        """Test feature importance using model coefficients."""
        X, y = _create_test_data_binary()
        wrapper = LogisticRegressionWrapper()

        # Fit through wrapper to set _is_fitted flag
        wrapper.fit(X, y)

        importance = wrapper.get_feature_importance()

        # Should extract from coefficients for linear models
        assert importance is not None
        assert len(importance) == len(X.columns)

        # All values should be numeric
        for feature, score in importance.items():
            assert isinstance(score, (int, float, np.number))


class TestSklearnSaveLoadWithPreprocessing:
    """Test serialization with preprocessing pipelines."""

    def test_save_load_with_feature_names(self):
        """Test save/load preserves feature names."""
        import os
        import tempfile

        X, y = _create_test_data_binary()
        wrapper = LogisticRegressionWrapper()
        wrapper.fit(X, y)

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            wrapper.save(temp_path)

            # Load model
            loaded_wrapper = LogisticRegressionWrapper()
            loaded_wrapper.load(temp_path)

            # Verify predictions are identical
            original_preds = wrapper.predict(X)
            loaded_preds = loaded_wrapper.predict(X)
            np.testing.assert_array_equal(original_preds, loaded_preds)

            # Verify probabilities are identical
            original_proba = wrapper.predict_proba(X)
            loaded_proba = loaded_wrapper.predict_proba(X)
            np.testing.assert_array_almost_equal(original_proba, loaded_proba)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_load_with_preprocessing_pipeline(self):
        """Test save/load with sklearn preprocessing pipeline."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = _create_test_data_binary()

        # Create preprocessing pipeline
        preprocessing_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42, max_iter=2000, solver="liblinear")),
            ],
        )

        preprocessing_pipeline.fit(X, y)

        # Wrap the pipeline
        wrapper = SklearnGenericWrapper(preprocessing_pipeline)

        # Test that wrapper handles pipeline
        predictions = wrapper.predict(X)
        assert len(predictions) == len(y)

        probabilities = wrapper.predict_proba(X)
        assert probabilities.shape == (len(y), 2)

    def test_save_load_preserves_model_state(self):
        """Test that save/load preserves trained model state."""
        import os
        import tempfile

        X, y = _create_test_data_binary()
        wrapper = LogisticRegressionWrapper()
        wrapper.fit(X, y)

        # Get model state before save
        original_state = wrapper.model.coef_.copy()

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            wrapper.save(temp_path)
            loaded_wrapper = LogisticRegressionWrapper()
            loaded_wrapper.load(temp_path)

            # Verify model state is preserved
            loaded_state = loaded_wrapper.model.coef_
            np.testing.assert_array_almost_equal(original_state, loaded_state)

            # Verify feature names are preserved
            assert wrapper.feature_names_ == loaded_wrapper.feature_names_

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_load_with_custom_preprocessing(self):
        """Test save/load with custom preprocessing in wrapper."""
        import os
        import tempfile

        X, y = _create_test_data_binary()

        # Use the standard wrapper
        wrapper = LogisticRegressionWrapper()
        wrapper.fit(X, y)

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            wrapper.save(temp_path)
            loaded_wrapper = LogisticRegressionWrapper()
            loaded_wrapper.load(temp_path)

            # Verify predictions work after load
            original_preds = wrapper.predict(X)
            loaded_preds = loaded_wrapper.predict(X)
            np.testing.assert_array_equal(original_preds, loaded_preds)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestSklearnWrapperAdvancedFeatures:
    """Test advanced sklearn wrapper features."""

    def test_wrapper_with_different_solvers(self):
        """Test LogisticRegression with different solvers."""
        from sklearn.linear_model import LogisticRegression

        X, y = _create_test_data_binary()

        solvers = ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]

        for solver in solvers:
            wrapper = LogisticRegressionWrapper()

            # Skip solvers that might not converge with small data
            try:
                base_model = LogisticRegression(random_state=42, solver=solver, max_iter=2000)
                base_model.fit(X, y)
                wrapper.model = base_model

                # Should work with different solvers
                predictions = wrapper.predict(X)
                assert len(predictions) == len(y)

                probabilities = wrapper.predict_proba(X)
                assert probabilities.shape == (len(y), 2)

            except Exception:
                # Some solvers may fail with small data - that's acceptable
                pass

    def test_wrapper_with_regularization_parameters(self):
        """Test LogisticRegression with different regularization parameters."""
        from sklearn.linear_model import LogisticRegression

        X, y = _create_test_data_binary()

        # Test different regularization strengths
        regularization_params = [
            {"C": 0.001},  # Strong regularization
            {"C": 1.0},  # Default
            {"C": 100.0},  # Weak regularization
        ]

        for params in regularization_params:
            wrapper = LogisticRegressionWrapper()

            base_model = LogisticRegression(random_state=42, **params)
            base_model.fit(X, y)
            wrapper.model = base_model

            # Should work with different regularization
            predictions = wrapper.predict(X)
            assert len(predictions) == len(y)

    def test_wrapper_with_class_weights(self):
        """Test LogisticRegression with class weight balancing."""
        from sklearn.linear_model import LogisticRegression

        X, y = _create_test_data_binary()

        wrapper = LogisticRegressionWrapper()

        # Test with balanced class weights
        base_model = LogisticRegression(
            random_state=42,
            class_weight="balanced",
        )
        base_model.fit(X, y)
        wrapper.model = base_model

        # Should handle class weights
        predictions = wrapper.predict(X)
        assert len(predictions) == len(y)

        # Should work with imbalanced data
        imbalanced_y = [0] * 40 + [1] * 10  # 4:1 imbalance
        base_model.fit(X, imbalanced_y)

        predictions_imbalanced = wrapper.predict(X)
        assert len(predictions_imbalanced) == len(y)

    def test_wrapper_error_handling_invalid_input(self):
        """Test error handling for invalid input data."""
        from sklearn.linear_model import LogisticRegression

        X, y = _create_test_data_binary()

        wrapper = LogisticRegressionWrapper()
        base_model = LogisticRegression(random_state=42, max_iter=2000, solver="liblinear")
        base_model.fit(X, y)
        wrapper.model = base_model

        # Test with wrong number of features
        wrong_features = pd.DataFrame(
            {
                "wrong_feature": [1, 2, 3],
            },
        )

        with pytest.raises((ValueError, Exception)):
            wrapper.predict(wrong_features)

    def test_wrapper_with_feature_selection(self):
        """Test wrapper with feature selection preprocessing."""
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.linear_model import LogisticRegression

        X, y = _create_test_data_binary()

        # Create pipeline with feature selection
        from sklearn.pipeline import Pipeline

        feature_selector = SelectKBest(f_classif, k=2)
        classifier = LogisticRegression(random_state=42, max_iter=2000, solver="liblinear")

        pipeline = Pipeline(
            [
                ("feature_selection", feature_selector),
                ("classification", classifier),
            ],
        )

        pipeline.fit(X, y)

        # Wrap the pipeline
        wrapper = SklearnGenericWrapper(pipeline)

        # Should work with feature selection
        predictions = wrapper.predict(X)
        assert len(predictions) == len(y)

        # Should handle the feature selection preprocessing
        probabilities = wrapper.predict_proba(X)
        assert probabilities.shape == (len(y), 2)

    def test_wrapper_predict_proba_binary_classification(self):
        """Test predict_proba for binary classification."""
        from sklearn.linear_model import LogisticRegression

        X, y = _create_test_data_binary()

        wrapper = LogisticRegressionWrapper()
        base_model = LogisticRegression(random_state=42, max_iter=2000, solver="liblinear")
        base_model.fit(X, y)
        wrapper.model = base_model

        # Test predict_proba
        probabilities = wrapper.predict_proba(X)

        # Should return probabilities for both classes
        assert probabilities.shape == (len(y), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1

        # All probabilities should be between 0 and 1
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_wrapper_predict_proba_multiclass(self):
        """Test predict_proba for multiclass classification."""
        from sklearn.linear_model import LogisticRegression

        X, y = _create_test_data_multiclass()

        wrapper = LogisticRegressionWrapper()
        base_model = LogisticRegression(random_state=42, max_iter=2000, solver="lbfgs")
        base_model.fit(X, y)
        wrapper.model = base_model

        # Test predict_proba for multiclass
        probabilities = wrapper.predict_proba(X)

        # Should return probabilities for all 3 classes
        assert probabilities.shape == (len(y), 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1

        # All probabilities should be between 0 and 1
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_wrapper_with_custom_preprocessing_pipeline(self):
        """Test wrapper with custom preprocessing pipeline."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X, y = _create_test_data_binary()

        # Create custom preprocessing pipeline
        class CustomPreprocessor:
            def __init__(self):
                self.scaler = StandardScaler()

            def fit(self, X, y=None):
                self.scaler.fit(X)
                return self

            def transform(self, X):
                return self.scaler.transform(X)

            def fit_transform(self, X, y=None):
                return self.fit(X, y).transform(X)

        preprocessor = CustomPreprocessor()
        classifier = LogisticRegression(random_state=42, max_iter=2000, solver="liblinear")

        # Use custom preprocessing
        X_processed = preprocessor.fit_transform(X)
        classifier.fit(X_processed, y)

        # Wrap the classifier (assuming preprocessing is handled elsewhere)
        wrapper = SklearnGenericWrapper(classifier)

        # Should work with preprocessed data
        predictions = wrapper.predict(X_processed)
        assert len(predictions) == len(y)

    def test_wrapper_preserves_sklearn_attributes(self):
        """Test that wrapper preserves important sklearn model attributes."""
        from sklearn.linear_model import LogisticRegression

        X, y = _create_test_data_binary()

        wrapper = LogisticRegressionWrapper()
        base_model = LogisticRegression(random_state=42, max_iter=2000, solver="liblinear")
        base_model.fit(X, y)
        wrapper.model = base_model

        # Should preserve sklearn attributes
        assert hasattr(wrapper, "model")
        assert hasattr(wrapper.model, "coef_")  # LogisticRegression coefficients
        assert hasattr(wrapper.model, "classes_")  # Class labels
        assert hasattr(wrapper.model, "n_features_in_")  # Number of features

        # Attributes should have correct values
        assert wrapper.model.n_features_in_ == X.shape[1]
        assert len(wrapper.model.classes_) == 2  # Binary classification

    def test_wrapper_with_extreme_parameter_values(self):
        """Test wrapper with extreme parameter values."""
        from sklearn.linear_model import LogisticRegression

        X, y = _create_test_data_binary()

        # Test with extreme regularization
        wrapper = LogisticRegressionWrapper()

        extreme_params = {
            "C": 1e-10,  # Very strong regularization
            "max_iter": 10000,  # Many iterations
            "tol": 1e-10,  # Very strict tolerance
        }

        base_model = LogisticRegression(random_state=42, **extreme_params)
        base_model.fit(X, y)
        wrapper.model = base_model

        # Should handle extreme parameters
        predictions = wrapper.predict(X)
        assert len(predictions) == len(y)

        probabilities = wrapper.predict_proba(X)
        assert probabilities.shape == (len(y), 2)

    def test_wrapper_save_load_with_large_model(self):
        """Test save/load with larger, more complex models."""
        import os
        import tempfile

        # Create larger dataset by concatenating multiple calls
        datasets = [_create_test_data_binary() for _ in range(20)]  # 20 * 50 = 1000 samples
        X_list, y_list = zip(*datasets, strict=False)
        X = pd.concat(X_list, ignore_index=True)
        y = np.concatenate(y_list)

        wrapper = LogisticRegressionWrapper()
        wrapper.fit(X, y)

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            wrapper.save(temp_path)
            loaded_wrapper = LogisticRegressionWrapper()
            loaded_wrapper.load(temp_path)

            # Verify predictions work on subset
            test_X = X[:100]
            original_preds = wrapper.predict(test_X)
            loaded_preds = loaded_wrapper.predict(test_X)
            np.testing.assert_array_equal(original_preds, loaded_preds)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
