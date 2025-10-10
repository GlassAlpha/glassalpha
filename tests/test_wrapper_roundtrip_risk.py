"""High-risk wrapper round-trip tests - save/load critical path validation.

These tests target the actual risk areas in model persistence:
- fit → save → load → predict pipeline must work
- feature_names_ and n_classes must survive round-trip
- Random state must be preserved for reproducibility
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.xdist_group(name="sklearn_pickle")
class TestWrapperRoundtripRisk:
    """Test wrapper save/load critical paths that prevent customer data loss.

    Note: sklearn pickle tests run in single worker due to pytest-xdist class identity issues.
    """

    def test_xgboost_fit_save_load_predict_roundtrip(self):
        """XGBoost fit → save → load → predict must work with all metadata intact."""
        from glassalpha.models.tabular.xgboost import XGBoostWrapper

        # Create training data
        rng = np.random.default_rng(42)
        X_train = pd.DataFrame(
            {
                "feature_1": rng.normal(size=50),
                "feature_2": rng.uniform(size=50),
                "feature_3": rng.exponential(size=50),
            },
        )
        y_train = (X_train.feature_1 + X_train.feature_2 > 0).astype(int)

        # Fit original model
        wrapper1 = XGBoostWrapper()
        wrapper1.fit(X_train, y_train, random_state=123)

        # Capture original state
        original_features = wrapper1.feature_names_
        original_classes = wrapper1.n_classes
        original_predictions = wrapper1.predict(X_train)
        original_probabilities = wrapper1.predict_proba(X_train)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.json"
            wrapper1.save(model_path)

            # Load into new wrapper
            wrapper2 = XGBoostWrapper()
            wrapper2.load(model_path)

            # Verify metadata survived
            assert wrapper2.feature_names_ == original_features, "Feature names must survive round-trip"
            assert wrapper2.n_classes == original_classes, "N classes must survive round-trip"
            assert wrapper2._is_fitted, "Loaded model must be marked as fitted"

            # Verify predictions are identical
            new_predictions = wrapper2.predict(X_train)
            new_probabilities = wrapper2.predict_proba(X_train)

            np.testing.assert_array_equal(
                original_predictions,
                new_predictions,
                "Predictions must be identical after round-trip",
            )

            np.testing.assert_array_almost_equal(
                original_probabilities,
                new_probabilities,
                decimal=6,
                err_msg="Probabilities must be nearly identical after round-trip",
            )

    def test_lightgbm_feature_names_preservation(self):
        """LightGBM feature names must survive save/load for SHAP compatibility."""
        pytest.importorskip("lightgbm")  # Skip if LightGBM not available
        from glassalpha.models.tabular.lightgbm import LightGBMWrapper

        # Training data with specific column names
        X_train = pd.DataFrame(
            {
                "revenue_growth": np.random.normal(0, 1, 30),
                "customer_satisfaction": np.random.uniform(0, 10, 30),
                "market_share": np.random.beta(2, 5, 30),
            },
        )
        y_train = np.random.binomial(1, 0.3, 30)

        # Fit and save
        wrapper1 = LightGBMWrapper()
        wrapper1.fit(X_train, y_train, random_state=456)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lgb_model.json"
            wrapper1.save(model_path)

            # Load and verify
            wrapper2 = LightGBMWrapper()
            wrapper2.load(model_path)

            assert wrapper2.feature_names_ == ["revenue_growth", "customer_satisfaction", "market_share"]
            assert wrapper2.n_classes == 2  # Binary classification

            # Test prediction with renamed columns (feature alignment test)
            X_renamed = X_train.copy()
            X_renamed.columns = ["col_a", "col_b", "col_c"]

            # Should work due to feature alignment
            predictions = wrapper2.predict(X_renamed)
            assert len(predictions) == len(y_train), "Feature alignment must work after load"

    def test_sklearn_logistic_regression_roundtrip(self):
        """Sklearn LogisticRegression must preserve random_state and coefficients."""
        from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper

        # Create linearly separable data for stable coefficients
        rng = np.random.default_rng(789)
        X_train = pd.DataFrame(
            {
                "x1": np.concatenate([rng.normal(-2, 1, 25), rng.normal(2, 1, 25)]),
                "x2": np.concatenate([rng.normal(-2, 1, 25), rng.normal(2, 1, 25)]),
            },
        )
        y_train = np.concatenate([np.zeros(25), np.ones(25)])

        # Fit with specific random state
        wrapper1 = LogisticRegressionWrapper()
        wrapper1.fit(X_train, y_train, random_state=999)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lr_model.json"
            wrapper1.save(model_path)

            wrapper2 = LogisticRegressionWrapper()
            wrapper2.load(model_path)

            # Verify model coefficients are preserved (critical for explainability)
            original_coef = wrapper1.model.coef_
            loaded_coef = wrapper2.model.coef_

            np.testing.assert_array_almost_equal(
                original_coef,
                loaded_coef,
                decimal=8,
                err_msg="Model coefficients must be preserved exactly",
            )

    def test_save_creates_parent_directories(self):
        """Save must create parent directories - prevents customer file errors."""
        from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper

        X_train = pd.DataFrame({"x": [1, 2, 3, 4]})
        y_train = [0, 0, 1, 1]

        wrapper = LogisticRegressionWrapper()
        wrapper.fit(X_train, y_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Deep nested path that doesn't exist
            deep_path = Path(tmpdir) / "level1" / "level2" / "level3" / "model.json"

            # Should create all parent directories
            wrapper.save(deep_path)
            assert deep_path.exists(), "Save must create parent directories"

            # Should be loadable
            wrapper2 = LogisticRegressionWrapper()
            wrapper2.load(deep_path)
            assert wrapper2._is_fitted, "Loaded model should be fitted"

    def test_unfitted_save_raises_exact_error_message(self):
        """Unfitted model save must raise exact error message for consistency."""
        from glassalpha.models.tabular.xgboost import XGBoostWrapper

        wrapper = XGBoostWrapper()
        # Don't fit the model

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "should_fail.json"

            with pytest.raises(ValueError, match="Model not loaded. Load a model first."):
                wrapper.save(model_path)
