"""Contract tests for ModelRegistry auto-import capability.

Tests verify that:
1. Models auto-import on first access
2. Unknown models return appropriate errors
3. Auto-import is transparent to callers
"""

import pytest

from glassalpha.core.registry import ModelRegistry


class TestModelRegistryAutoImport:
    """Test auto-import functionality in ModelRegistry."""

    def test_auto_import_xgboost_on_first_access(self):
        """Registry auto-imports xgboost module when model is requested."""
        pytest.importorskip("xgboost")

        # Don't pre-import the module
        # Registry should auto-import when we call get()
        model_class = ModelRegistry.get("xgboost")

        assert model_class is not None
        assert hasattr(model_class, "capabilities")

    def test_auto_import_lightgbm_on_first_access(self):
        """Registry auto-imports lightgbm module when model is requested."""
        pytest.importorskip("lightgbm")

        model_class = ModelRegistry.get("lightgbm")

        assert model_class is not None
        assert hasattr(model_class, "capabilities")

    def test_auto_import_sklearn_models_on_first_access(self):
        """Registry auto-imports sklearn models when requested."""
        pytest.importorskip("sklearn")

        model_class = ModelRegistry.get("logistic_regression")

        assert model_class is not None
        assert hasattr(model_class, "capabilities")

    def test_unknown_model_raises_key_error(self):
        """Unknown models raise KeyError with clear message."""
        with pytest.raises(KeyError) as exc_info:
            ModelRegistry.get("unknown_model_type_xyz")

        assert "unknown_model_type_xyz" in str(exc_info.value)

    def test_auto_import_is_transparent(self):
        """Auto-import doesn't change registry behavior - works like normal get()."""
        pytest.importorskip("xgboost")

        # First call - triggers auto-import
        model_class_1 = ModelRegistry.get("xgboost")

        # Second call - uses cached version
        model_class_2 = ModelRegistry.get("xgboost")

        # Should be the same object
        assert model_class_1 is model_class_2

    # Note: parameter_rules feature removed - was over-engineered and redundant
    # Parameter validation is handled by underlying libraries (sklearn, xgboost, lightgbm)
    # Complex cross-parameter validation done imperatively in wrapper fit() methods
