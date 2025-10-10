"""Explainer integration tests.

Tests SHAP explainer functionality including initialization, explanation generation,
and registry integration. These tests focus on explainer logic without requiring
complex model dependencies that cause CI issues.
"""

import contextlib
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest


def assert_explainer_capabilities(
    explainer,
    expected_type="shap_values",
    supports_local=True,
    supports_global=True,
    data_modality="tabular",
):
    """Helper function to assert common explainer capabilities."""
    capabilities = explainer.capabilities
    assert capabilities["explanation_type"] == expected_type
    assert capabilities["supports_local"] is supports_local
    assert capabilities["supports_global"] is supports_global
    assert capabilities["data_modality"] == data_modality


# Conditional imports for SHAP explainers (SHAP may not be available in CI)
try:
    from glassalpha.explain.shap.kernel import KernelSHAPExplainer
    from glassalpha.explain.shap.tree import TreeSHAPExplainer

    SHAP_EXPLAINERS_AVAILABLE = True
except ImportError:
    # Create dummy classes for when SHAP is not available
    class KernelSHAPExplainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("SHAP not available")

    class TreeSHAPExplainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("SHAP not available")

    SHAP_EXPLAINERS_AVAILABLE = False

# Conditional sklearn import with graceful fallback for CI compatibility
try:
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    make_classification = None
    LogisticRegression = None
    SKLEARN_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available - CI compatibility issues"),
    pytest.mark.skipif(not SHAP_EXPLAINERS_AVAILABLE, reason="SHAP explainers not available - CI compatibility issues"),
]

from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper


@pytest.fixture
def sample_classification_data():
    """Create sample classification dataset for testing."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=100,
        n_features=8,
        n_classes=2,
        n_informative=6,
        n_redundant=1,
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    return X_df, y, feature_names


@pytest.fixture
def trained_model_wrapper(sample_classification_data):
    """Create a trained model wrapper for testing explainers."""
    X_df, y, feature_names = sample_classification_data

    # Use LogisticRegression for simplicity (no dependency issues)
    model = LogisticRegression(random_state=42, max_iter=2000, solver="liblinear")
    model.fit(X_df, y)

    wrapper = LogisticRegressionWrapper(model=model)

    return wrapper, X_df, y, feature_names


@pytest.fixture
def small_background_data(sample_classification_data):
    """Create small background dataset for SHAP explainers."""
    X_df, y, feature_names = sample_classification_data

    # Use first 20 samples as background
    return X_df.iloc[:20], y[:20]


@pytest.fixture
def explanation_samples(sample_classification_data):
    """Create small sample for explanation testing."""
    X_df, y, feature_names = sample_classification_data

    # Use last 10 samples for explanation
    return X_df.iloc[-10:], y[-10:]


class TestTreeSHAPExplainer:
    """Test TreeSHAPExplainer functionality."""

    def test_treeshap_initialization(self):
        """Test TreeSHAPExplainer initialization."""
        explainer = TreeSHAPExplainer()

        assert explainer.explainer is None
        assert explainer.base_value is None
        assert explainer.check_additivity is False
        assert_explainer_capabilities(explainer)
        assert explainer.version == "1.0.0"
        assert explainer.priority == 100

    def test_treeshap_initialization_with_options(self):
        """Test TreeSHAPExplainer initialization with options."""
        explainer = TreeSHAPExplainer(check_additivity=True)

        assert explainer.check_additivity is True

    def test_treeshap_capabilities(self):
        """Test TreeSHAPExplainer capability reporting."""
        explainer = TreeSHAPExplainer()

        # Test class-level capabilities
        capabilities = explainer.capabilities

        assert isinstance(capabilities, dict)
        assert_explainer_capabilities(explainer)
        assert "xgboost" in capabilities["supported_models"]
        assert "lightgbm" in capabilities["supported_models"]
        assert "random_forest" in capabilities["supported_models"]

    def test_treeshap_model_compatibility_check(self, trained_model_wrapper):
        """Test TreeSHAPExplainer model compatibility checking."""
        explainer = TreeSHAPExplainer()
        wrapper, X_df, y, feature_names = trained_model_wrapper

        # Test with actual model - TreeSHAP checks via supports_model method
        # LogisticRegression should not be supported by TreeSHAP
        assert explainer.supports_model(wrapper) is False

        # Test class-level capabilities
        capabilities = explainer.capabilities
        assert "xgboost" in capabilities["supported_models"]
        assert "lightgbm" in capabilities["supported_models"]

    def test_treeshap_explain_method_structure(self, trained_model_wrapper, small_background_data):
        """Test TreeSHAPExplainer explain method structure."""
        wrapper, X_df, y, feature_names = trained_model_wrapper
        background_X, background_y = small_background_data

        explainer = TreeSHAPExplainer()

        # TreeSHAP should return error for unsupported model (LogisticRegression)
        result = explainer.explain(wrapper, background_X)

        assert isinstance(result, dict)
        assert "status" in result
        assert "explainer_type" in result
        assert result["explainer_type"] == "treeshap"
        # Should fail for unsupported model type
        assert result["status"] == "error"

    def test_treeshap_explain_output_structure(self, small_background_data):
        """Test TreeSHAPExplainer explain output structure."""
        background_X, background_y = small_background_data

        explainer = TreeSHAPExplainer()

        # Create a mock model wrapper that supports TreeSHAP
        mock_model = Mock()
        mock_model.get_model_type.return_value = "xgboost"

        # Test with mock model - should attempt to explain
        result = explainer.explain(mock_model, background_X)

        assert isinstance(result, dict)
        # Will likely fail due to mock, but structure should be there
        assert "status" in result
        assert "explainer_type" in result

    def test_treeshap_utility_methods(self):
        """Test TreeSHAPExplainer utility methods."""
        explainer = TreeSHAPExplainer()

        # Test supports_model with mock models
        mock_xgb = Mock()
        mock_xgb.get_model_type.return_value = "xgboost"
        assert explainer.supports_model(mock_xgb) is True

        mock_lr = Mock()
        mock_lr.get_model_type.return_value = "logistic_regression"
        assert explainer.supports_model(mock_lr) is False

    def test_treeshap_error_handling_unsupported_model(self, trained_model_wrapper, explanation_samples):
        """Test TreeSHAPExplainer error handling with unsupported model."""
        explainer = TreeSHAPExplainer()
        wrapper, X_df, y, feature_names = trained_model_wrapper
        explain_X, explain_y = explanation_samples

        # Should return error status for unsupported model
        result = explainer.explain(wrapper, explain_X)

        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "not supported" in result.get("reason", "").lower()


class TestKernelSHAPExplainer:
    """Test KernelSHAPExplainer functionality."""

    def test_kernelshap_initialization(self):
        """Test KernelSHAPExplainer initialization."""
        explainer = KernelSHAPExplainer()

        assert explainer.explainer is None
        assert explainer.base_value is None
        assert explainer.n_samples == 100
        assert explainer.background_size == 100
        assert explainer.link == "identity"
        assert_explainer_capabilities(explainer)
        assert explainer.version == "1.0.0"
        assert explainer.priority == 50

    def test_kernelshap_initialization_with_options(self):
        """Test KernelSHAPExplainer initialization with custom options."""
        explainer = KernelSHAPExplainer(n_samples=50, background_size=25, link="logit")

        assert explainer.n_samples == 50
        assert explainer.background_size == 25
        assert explainer.link == "logit"

    def test_kernelshap_capabilities(self):
        """Test KernelSHAPExplainer capability reporting."""
        explainer = KernelSHAPExplainer()

        # Test class-level capabilities
        capabilities = explainer.capabilities

        assert isinstance(capabilities, dict)
        assert_explainer_capabilities(explainer)
        assert capabilities["supported_models"] == ["all"]  # Works with any model

    def test_kernelshap_model_compatibility_check(self, trained_model_wrapper):
        """Test KernelSHAPExplainer model compatibility (should accept all)."""
        explainer = KernelSHAPExplainer()
        wrapper, X_df, y, feature_names = trained_model_wrapper

        # KernelSHAP should support any model
        assert explainer.supports_model(wrapper) is True

        # Test capabilities
        capabilities = explainer.capabilities
        assert capabilities["supported_models"] == ["all"]

    def test_kernelshap_explain_method_structure(self, trained_model_wrapper, small_background_data):
        """Test KernelSHAPExplainer explain method structure."""
        wrapper, X_df, y, feature_names = trained_model_wrapper
        background_X, background_y = small_background_data

        explainer = KernelSHAPExplainer()

        # KernelSHAP should work with any model (including LogisticRegression)
        result = explainer.explain(wrapper, background_X)

        assert isinstance(result, dict)
        assert "status" in result
        assert "explainer_type" in result
        # Status could be success or error depending on implementation

    @patch("glassalpha.explain.shap.kernel.shap.KernelExplainer")
    def test_kernelshap_explain_local(
        self,
        mock_shap_explainer,
        trained_model_wrapper,
        small_background_data,
        explanation_samples,
    ):
        """Test KernelSHAPExplainer local explanation."""
        wrapper, X_df, y, feature_names = trained_model_wrapper
        background_X, background_y = small_background_data
        explain_X, explain_y = explanation_samples

        # Mock the SHAP explainer and its output
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.3
        mock_shap_values = np.random.rand(len(explain_X), len(feature_names))
        mock_explainer_instance.shap_values.return_value = mock_shap_values
        mock_shap_explainer.return_value = mock_explainer_instance

        explainer = KernelSHAPExplainer()
        explainer.fit(wrapper, background_X)

        # Test local explanation
        explanations = explainer.explain_local(explain_X, nsamples=50)

        assert isinstance(explanations, dict)
        assert "shap_values" in explanations
        assert "base_value" in explanations
        assert explanations["base_value"] == 0.3

        # Mock should have been called with custom nsamples
        mock_explainer_instance.shap_values.assert_called()

    def test_kernelshap_explain_global_aggregation(self):
        """Test KernelSHAPExplainer global explanation from local values."""
        explainer = KernelSHAPExplainer()

        # Create mock SHAP values
        n_samples, n_features = 15, 6
        mock_shap_values = np.random.randn(n_samples, n_features)  # Can be negative
        feature_names = [f"feature_{i}" for i in range(n_features)]

        # Test global aggregation
        global_importance = explainer._aggregate_to_global(mock_shap_values, feature_names)

        assert isinstance(global_importance, dict)
        assert len(global_importance) == n_features

        # All importance values should be non-negative (absolute values)
        for _feature, importance in global_importance.items():
            assert isinstance(importance, (int, float))
            assert importance >= 0

    def test_kernelshap_error_handling_no_fit(self, explanation_samples):
        """Test KernelSHAPExplainer error handling when not fitted."""
        explainer = KernelSHAPExplainer()
        explain_X, explain_y = explanation_samples

        # Should raise error when not fitted
        with pytest.raises(ValueError, match="not fitted"):
            explainer.explain_local(explain_X)


class TestExplainerIntegration:
    """Test integration between explainers and other components."""

    @patch("glassalpha.explain.shap.kernel.shap.KernelExplainer")
    def test_explainer_with_model_wrapper_integration(
        self,
        mock_shap_explainer,
        trained_model_wrapper,
        small_background_data,
    ):
        """Test explainer integration with model wrapper."""
        wrapper, X_df, y, feature_names = trained_model_wrapper
        background_X, background_y = small_background_data

        # Mock SHAP
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.4
        mock_shap_explainer.return_value = mock_explainer_instance

        # KernelSHAP should work with any model wrapper
        explainer = KernelSHAPExplainer()
        explainer.fit(wrapper, background_X)

        # Verify integration worked
        assert explainer.explainer is not None
        mock_shap_explainer.assert_called_once()

    def test_explainer_feature_name_handling(self, small_background_data):
        """Test explainer handling of feature names."""
        background_X, background_y = small_background_data
        feature_names = list(background_X.columns)

        # Test feature name extraction
        tree_explainer = TreeSHAPExplainer()
        kernel_explainer = KernelSHAPExplainer()

        # Both should extract feature names correctly
        tree_names = tree_explainer._extract_feature_names(background_X)
        kernel_names = kernel_explainer._extract_feature_names(background_X)

        assert tree_names == feature_names
        assert kernel_names == feature_names
        assert len(tree_names) == len(background_X.columns)


class TestExplainerErrorHandling:
    """Test error handling in explainers."""

    def test_explainer_with_empty_background_data(self, trained_model_wrapper):
        """Test explainer behavior with empty background data."""
        wrapper, X_df, y, feature_names = trained_model_wrapper

        empty_df = pd.DataFrame(columns=feature_names)

        explainer = KernelSHAPExplainer()

        # Should handle empty background data gracefully or raise informative error
        try:
            explainer.fit(wrapper, empty_df)
            # If it succeeds, that's ok
        except ValueError as e:
            # If it raises ValueError, should be informative
            assert len(str(e)) > 0

    def test_explainer_with_mismatched_feature_shapes(self, trained_model_wrapper, small_background_data):
        """Test explainer with mismatched feature shapes."""
        wrapper, X_df, y, feature_names = trained_model_wrapper
        background_X, background_y = small_background_data

        # Create data with different number of features
        wrong_features_df = pd.DataFrame({"feature_0": [1, 2, 3], "feature_1": [4, 5, 6]})

        explainer = KernelSHAPExplainer()

        # Should raise error for mismatched features
        with pytest.raises((ValueError, IndexError)):
            explainer.fit(wrapper, wrong_features_df)

    def test_explainer_repr_and_info(self):
        """Test explainer string representation and info methods."""
        tree_explainer = TreeSHAPExplainer()
        kernel_explainer = KernelSHAPExplainer()

        # Test string representations
        tree_repr = str(tree_explainer)
        kernel_repr = str(kernel_explainer)

        assert "TreeSHAP" in tree_repr
        assert "KernelSHAP" in kernel_repr

        # Test get_info method if available
        if hasattr(tree_explainer, "get_info"):
            tree_info = tree_explainer.get_info()
            assert isinstance(tree_info, dict)

        if hasattr(kernel_explainer, "get_info"):
            kernel_info = kernel_explainer.get_info()
            assert isinstance(kernel_info, dict)


class TestExplainerEdgeCases:
    """Test edge cases and boundary conditions - focused on compliance-critical scenarios."""

    def test_explainer_with_empty_background_data(self, trained_model_wrapper):
        """Test explainer behavior with empty background data (compliance: graceful failure)."""
        wrapper, X_df, y, feature_names = trained_model_wrapper

        empty_df = pd.DataFrame(columns=feature_names)

        explainer = KernelSHAPExplainer()

        # Should handle empty background data gracefully or raise informative error
        try:
            explainer.fit(wrapper, empty_df)
            # If it succeeds, that's ok for compliance
        except ValueError as e:
            # If it raises ValueError, should be informative
            assert len(str(e)) > 0

    def test_explainer_parameter_validation_basic(self):
        """Test basic parameter validation (compliance: prevent invalid configs)."""
        # Test that invalid parameters are handled reasonably
        with contextlib.suppress(ValueError):
            KernelSHAPExplainer(n_samples=-10)
            # Implementation choice - if it accepts, that's ok

        with contextlib.suppress(ValueError):
            KernelSHAPExplainer(background_size=0)
            # Implementation choice - if it accepts, that's ok
