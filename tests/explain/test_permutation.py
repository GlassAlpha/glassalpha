"""Contract tests for PermutationExplainer.

Tests verify that PermutationExplainer:
- Works with classifiers (using log loss)
- Works with regressors (using MSE)
- Requires y parameter
- Generates deterministic results with same seed
- Integrates with progress bar utilities
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from glassalpha.explain.permutation import PermutationExplainer


class TestPermutationExplainerCompatibility:
    """Test model compatibility checks."""

    def test_is_compatible_with_sklearn_model(self):
        """Verify is_compatible returns True for sklearn models."""
        model = LogisticRegression()
        assert PermutationExplainer.is_compatible(model=model)

    def test_is_compatible_with_any_predict_model(self):
        """Verify is_compatible works with any model having predict."""
        model = Mock()
        model.predict = Mock(return_value=np.zeros(10))
        assert PermutationExplainer.is_compatible(model=model)

    def test_is_compatible_without_model(self):
        """Verify is_compatible returns True when no model provided."""
        # Assumes compatible, will check at fit time
        assert PermutationExplainer.is_compatible()


class TestPermutationExplainerFit:
    """Test explainer fitting."""

    def test_fit_stores_model_and_features(self):
        """Verify fit() stores model wrapper and feature names."""
        explainer = PermutationExplainer(n_repeats=2, random_state=42)

        model = Mock()
        model.predict = Mock(return_value=np.zeros(10))

        X_bg = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])

        result = explainer.fit(model, X_bg, feature_names=["a", "b", "c"])

        assert result is explainer  # Returns self for chaining
        assert explainer.model is model
        assert explainer.feature_names == ["a", "b", "c"]
        assert explainer.is_fitted is True


class TestPermutationExplainerClassifier:
    """Test with classification models."""

    def test_classifier_requires_y_parameter(self):
        """Verify explainer raises error if y not provided."""
        # Create simple dataset
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
        y = np.random.randint(0, 2, size=50)

        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Fit explainer
        explainer = PermutationExplainer(n_repeats=2, random_state=42)
        explainer.fit(model, X[:10])

        # Should raise ValueError if y not provided
        with pytest.raises(ValueError, match="requires target values"):
            explainer.explain(X[:20])

    def test_classifier_with_y_parameter(self):
        """Verify explainer works when y is provided."""
        # Create simple dataset
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
        y = np.random.randint(0, 2, size=50)

        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Fit explainer
        explainer = PermutationExplainer(n_repeats=2, random_state=42)
        explainer.fit(model, X[:10], feature_names=["a", "b", "c"])

        # Should work with y provided
        result = explainer.explain(X[:20].values, y=y[:20])

        assert "local_explanations" in result
        assert "global_importance" in result
        assert "importances" in result
        assert len(result["importances"]) == 3  # 3 features

    def test_classifier_uses_log_loss_scorer(self):
        """Verify classifier uses log loss as scoring metric."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
        y = np.random.randint(0, 2, size=100)

        model = RandomForestClassifier(n_estimators=3, random_state=42, max_depth=3)
        model.fit(X, y)

        explainer = PermutationExplainer(n_repeats=3, random_state=42)
        explainer.fit(model, X[:20])

        result = explainer.explain(X[:30].values, y=y[:30])

        # Should have importances (log loss based)
        assert len(result["importances"]) == 3
        # All importances should be numeric
        assert all(isinstance(imp, (int, float)) for imp in result["importances"])


class TestPermutationExplainerRegressor:
    """Test with regression models."""

    def test_regressor_with_y_parameter(self):
        """Verify regressor works when y is provided."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
        y = np.random.randn(50)

        model = LinearRegression()
        model.fit(X, y)

        explainer = PermutationExplainer(n_repeats=2, random_state=42)
        explainer.fit(model, X[:10], feature_names=["a", "b", "c"])

        result = explainer.explain(X[:20].values, y=y[:20])

        assert "local_explanations" in result
        assert "global_importance" in result
        assert len(result["importances"]) == 3

    def test_regressor_uses_mse_scorer(self):
        """Verify regressor uses MSE as scoring metric."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
        y = X["a"] * 2 + X["b"] - X["c"] + np.random.randn(100) * 0.1

        model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5)
        model.fit(X, y)

        explainer = PermutationExplainer(n_repeats=5, random_state=42)
        explainer.fit(model, X[:20])

        result = explainer.explain(X[:30].values, y=y[:30])

        # Should have importances (MSE based)
        assert len(result["importances"]) == 3
        # At least one feature should have non-zero importance
        importances_dict = result["global_importance"]
        total_importance = sum(importances_dict.values())
        assert total_importance > 0  # Some features should be important


class TestPermutationExplainerDeterminism:
    """Test deterministic behavior."""

    def test_same_seed_same_results(self):
        """Verify same seed produces identical results."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
        y = np.random.randint(0, 2, size=50)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Run 1
        explainer1 = PermutationExplainer(n_repeats=3, random_state=123)
        explainer1.fit(model, X[:10])
        result1 = explainer1.explain(X[:20].values, y=y[:20])

        # Run 2 (same seed)
        explainer2 = PermutationExplainer(n_repeats=3, random_state=123)
        explainer2.fit(model, X[:10])
        result2 = explainer2.explain(X[:20].values, y=y[:20])

        # Results should be identical
        np.testing.assert_array_almost_equal(
            result1["importances"],
            result2["importances"],
        )

    def test_different_seed_different_results(self):
        """Verify different seeds can produce different results."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
        y = np.random.randint(0, 2, size=50)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Run 1
        explainer1 = PermutationExplainer(n_repeats=3, random_state=123)
        explainer1.fit(model, X[:10])
        result1 = explainer1.explain(X[:20].values, y=y[:20])

        # Run 2 (different seed)
        explainer2 = PermutationExplainer(n_repeats=3, random_state=456)
        explainer2.fit(model, X[:10])
        result2 = explainer2.explain(X[:20].values, y=y[:20])

        # Results may differ (due to randomness in permutation)
        # At least one importance should be different
        diffs = np.abs(np.array(result1["importances"]) - np.array(result2["importances"]))
        assert np.any(diffs > 1e-10)  # Some difference expected


class TestPermutationExplainerOutputFormat:
    """Test output format compliance."""

    def test_output_has_required_keys(self):
        """Verify output has all required keys."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
        y = np.random.randint(0, 2, size=30)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        explainer = PermutationExplainer(n_repeats=2, random_state=42)
        explainer.fit(model, X[:10], feature_names=["a", "b", "c"])

        result = explainer.explain(X[:15].values, y=y[:15])

        # Check required keys
        assert "local_explanations" in result
        assert "global_importance" in result
        assert "feature_names" in result
        assert "status" in result
        assert "importances" in result
        assert "importances_std" in result

    def test_local_explanations_format(self):
        """Verify local explanations have correct format."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
        y = np.random.randint(0, 2, size=30)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        explainer = PermutationExplainer(n_repeats=2, random_state=42)
        explainer.fit(model, X[:10], feature_names=["a", "b", "c"])

        result = explainer.explain(X[:15].values, y=y[:15])

        # Should have one dict per sample
        assert len(result["local_explanations"]) == 15
        # Each dict should have feature names as keys
        for local_exp in result["local_explanations"]:
            assert set(local_exp.keys()) == {"a", "b", "c"}
            # Values should be floats
            assert all(isinstance(v, float) for v in local_exp.values())

    def test_global_importance_format(self):
        """Verify global importance has correct format."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
        y = np.random.randint(0, 2, size=30)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        explainer = PermutationExplainer(n_repeats=2, random_state=42)
        explainer.fit(model, X[:10], feature_names=["a", "b", "c"])

        result = explainer.explain(X[:15].values, y=y[:15])

        # Should be dict with feature names as keys
        assert set(result["global_importance"].keys()) == {"a", "b", "c"}
        # Values should be floats (absolute importances)
        assert all(isinstance(v, float) for v in result["global_importance"].values())
        # All should be >= 0 (absolute values)
        assert all(v >= 0 for v in result["global_importance"].values())


class TestPermutationExplainerProgressIntegration:
    """Test progress bar integration."""

    def test_accepts_progress_parameters(self):
        """Verify explainer accepts progress parameters without error."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
        y = np.random.randint(0, 2, size=30)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        explainer = PermutationExplainer(n_repeats=2, random_state=42)
        explainer.fit(model, X[:10])

        # Should accept show_progress and strict_mode
        result = explainer.explain(
            X[:15].values,
            y=y[:15],
            show_progress=True,
            strict_mode=False,
        )

        assert result is not None
        assert "importances" in result

    def test_strict_mode_disables_progress(self):
        """Verify strict_mode=True disables progress."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
        y = np.random.randint(0, 2, size=30)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        explainer = PermutationExplainer(n_repeats=2, random_state=42)
        explainer.fit(model, X[:10])

        # Should work with strict_mode (no progress shown)
        result = explainer.explain(
            X[:15].values,
            y=y[:15],
            show_progress=True,
            strict_mode=True,  # Disables progress
        )

        assert result is not None


class TestPermutationExplainerErrorHandling:
    """Test error handling."""

    def test_explain_before_fit_raises_error(self):
        """Verify explain() before fit() raises RuntimeError."""
        explainer = PermutationExplainer()

        X = np.random.randn(10, 3)
        y = np.random.randint(0, 2, size=10)

        with pytest.raises(RuntimeError, match="must be fitted"):
            explainer.explain(X, y=y)

    def test_supports_model_check(self):
        """Verify supports_model() works correctly."""
        explainer = PermutationExplainer()

        # Model with predict
        model_with_predict = Mock()
        model_with_predict.predict = Mock()
        assert explainer.supports_model(model_with_predict)

        # Model without predict
        model_without_predict = Mock(spec=[])
        assert not explainer.supports_model(model_without_predict)

    def test_explain_local_passes_y_parameter(self):
        """Verify explain_local() passes y to explain()."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
        y = np.random.randint(0, 2, size=30)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        explainer = PermutationExplainer(n_repeats=2, random_state=42)
        explainer.fit(model, X[:10])

        # explain_local should work when y provided
        result = explainer.explain_local(X[:15].values, y=y[:15])

        # Should return local explanations
        assert isinstance(result, list)
        assert len(result) == 15
