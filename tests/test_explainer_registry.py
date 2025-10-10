"""Explainer registry and compatibility tests.

Tests for explainer registration, discovery, model compatibility checking,
and the new plugin registry functionality.
"""

import pytest

from glassalpha.core import ExplainerRegistry


@pytest.mark.xdist_group(name="explainer_registry")
class TestExplainerRegistry:
    """Test explainer registration and discovery.

    Note: Tests run serially to preserve registry state across test methods.
    """

    def test_shap_explainers_are_registered(self):
        """Test that SHAP explainers are properly registered."""
        components = ExplainerRegistry.get_all()

        # Should include SHAP explainers if SHAP is available
        assert "treeshap" in components or "kernelshap" in components

    def test_get_shap_explainer_classes(self):
        """Test that we can retrieve SHAP explainer classes."""
        # Try to get explainers - they may not be available if SHAP not installed
        try:
            tree_cls = ExplainerRegistry.get("treeshap")
            assert tree_cls is not None

            kernel_cls = ExplainerRegistry.get("kernelshap")
            assert kernel_cls is not None

        except (KeyError, ImportError):
            # Expected if SHAP not available
            pytest.skip("SHAP not available for testing")

    def test_explainer_priorities(self):
        """Test that explainers have expected priorities."""
        try:
            tree_cls = ExplainerRegistry.get("treeshap")
            kernel_cls = ExplainerRegistry.get("kernelshap")

            # TreeSHAP should have higher priority
            assert tree_cls.priority == 100
            assert kernel_cls.priority == 50
            assert tree_cls.priority > kernel_cls.priority

        except (KeyError, ImportError):
            # Expected if SHAP not available
            pytest.skip("SHAP not available for testing")

    def test_explainer_registry_decorator_support(self):
        """Test that the registry supports decorator registration."""

        @ExplainerRegistry.register("test_explainer", import_check="test_dep")
        class TestExplainer:
            @classmethod
            def is_compatible(cls, *, model=None, model_type=None, config=None):
                return True

        # Should be registered
        assert "test_explainer" in ExplainerRegistry.names()

    def test_explainer_registry_install_hint(self):
        """Test that explainer registry provides correct install hints."""
        hint = ExplainerRegistry.get_install_hint("kernelshap")
        assert hint == "pip install 'glassalpha[shap]'"

        hint = ExplainerRegistry.get_install_hint("treeshap")
        assert hint == "pip install 'glassalpha[shap]'"

    def test_explainer_registry_availability(self):
        """Test availability checking for explainers."""

        # Register a test explainer for this test
        @ExplainerRegistry.register("test_explainer_avail", import_check="test_dep")
        class TestExplainer:
            @classmethod
            def is_compatible(cls, *, model=None, model_type=None, config=None):
                return True

        available = ExplainerRegistry.available_plugins()

        # Should return dict with plugin names and availability status
        assert isinstance(available, dict)

        # Should include our test explainer
        assert "test_explainer_avail" in available

    def test_explainer_registry_lazy_import(self):
        """Test that SHAP import is lazy and doesn't crash registry."""
        # This should not raise ImportError due to circular imports
        # The registry should handle missing dependencies gracefully

        # Try to import explainer modules - should not crash
        try:
            from glassalpha.explain import shap
            # If we get here, the import succeeded
        except ImportError as e:
            # If there's an import error, it should be about SHAP specifically
            # not about circular imports or registry issues
            assert "shap" in str(e).lower() or "glassalpha" in str(e).lower()

    def test_explainer_registry_circular_import_fix(self):
        """Test that the circular import issue is resolved."""
        # This should not raise ImportError due to circular imports
        from glassalpha.explain import ExplainerRegistry as ImportedRegistry

        # The registry should be accessible and functional
        assert ImportedRegistry is not None

        # Should be able to list available plugins without issues
        try:
            names = ImportedRegistry.names()
            # If this succeeds, the circular import is fixed
            assert isinstance(names, list)
        except Exception as e:
            # If there's still an issue, it should not be a circular import
            # but something else (like missing SHAP)
            assert "circular" not in str(e).lower()
            assert "import" in str(e).lower()


class TestExplainerCompatibility:
    """Test explainer compatibility with different model types."""

    def test_explainer_selection_by_priority(self):
        """Test that explainers are selected by priority."""
        try:
            # TreeSHAP has higher priority than KernelSHAP
            tree_cls = ExplainerRegistry.get("treeshap")
            kernel_cls = ExplainerRegistry.get("kernelshap")

            assert tree_cls.priority > kernel_cls.priority
        except (KeyError, ImportError):
            # Expected if SHAP not available
            pytest.skip("SHAP not available for testing")

    def test_explainer_compatibility_filtering(self):
        """Test filtering explainers by model compatibility."""
        try:
            # Import explainer classes directly
            from glassalpha.explain.shap.kernel import KernelSHAPExplainer
            from glassalpha.explain.shap.tree import TreeSHAPExplainer

            tree_explainer = TreeSHAPExplainer()
            kernel_explainer = KernelSHAPExplainer()

            # For tree model, TreeSHAP should be compatible
            assert tree_explainer.is_compatible(model_type="xgboost") is True
            assert kernel_explainer.is_compatible(model_type="xgboost") is True

            # For linear model, only KernelSHAP should be compatible
            assert tree_explainer.is_compatible(model_type="logistic_regression") is False
            assert kernel_explainer.is_compatible(model_type="logistic_regression") is True

        except (ImportError, AttributeError):
            # Expected if SHAP not available
            pytest.skip("SHAP not available for testing")
