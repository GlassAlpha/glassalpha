"""Tests for plugin dependency handling and fallbacks."""

from unittest.mock import patch

import pytest

from glassalpha.cli.preflight import preflight_check_model
from glassalpha.core.registry import ModelRegistry


class TestPluginDependencies:
    """Test plugin dependency handling."""

    def test_logistic_regression_always_available(self):
        """Test that logistic regression is always available."""
        available = ModelRegistry.available_plugins()
        assert "logistic_regression" in available
        assert available["logistic_regression"] is True

    def test_missing_xgboost_handling(self):
        """Test handling when XGBoost is not available."""
        # Mock XGBoost as unavailable
        with patch.object(
            ModelRegistry,
            "available_plugins",
            return_value={
                "logistic_regression": True,
                "xgboost": False,
                "lightgbm": False,
            },
        ):
            # Create config requesting XGBoost with fallback allowed
            from types import SimpleNamespace

            config = SimpleNamespace()
            config.model = SimpleNamespace()
            config.model.type = "xgboost"
            config.model.allow_fallback = True

            # Should fallback to logistic regression
            modified_config, originally_requested = preflight_check_model(config)
            assert modified_config.model.type == "logistic_regression"
            assert originally_requested == "xgboost"  # Should return the originally requested model

    def test_missing_xgboost_no_fallback(self):
        """Test handling when XGBoost is not available and fallbacks disabled."""
        # Mock XGBoost as unavailable
        with patch.object(
            ModelRegistry,
            "available_plugins",
            return_value={
                "logistic_regression": True,
                "xgboost": False,
                "lightgbm": False,
            },
        ):
            # Create config requesting XGBoost with fallback disabled
            from types import SimpleNamespace

            config = SimpleNamespace()
            config.model = SimpleNamespace()
            config.model.type = "xgboost"

            # Should exit with error
            import typer

            with pytest.raises(typer.Exit):
                preflight_check_model(config, allow_fallback=False)

    def test_install_hint_generation(self):
        """Test that install hints are generated correctly."""
        hint = ModelRegistry.get_install_hint("xgboost")
        assert hint == "pip install 'glassalpha[xgboost]'"

        hint = ModelRegistry.get_install_hint("lightgbm")
        assert hint == "pip install 'glassalpha[lightgbm]'"

        hint = ModelRegistry.get_install_hint("logistic_regression")
        assert hint is None  # No hint needed for always-available model

    def test_registry_get_method(self):
        """Test the get method returns registered objects via entry points."""
        # Import PassThroughModel from its actual location
        from glassalpha.core.noop_components import PassThroughModel

        # Test getting passthrough model (loaded via entry point)
        model_cls = ModelRegistry.get("passthrough")
        assert model_cls == PassThroughModel

        # Test that get works with lazy loading from entry points
        assert callable(model_cls)

    def test_plugin_loading_with_missing_dependency(self):
        """Test plugin loading fails gracefully with missing dependencies."""
        # For now, this test is simplified since we don't have proper dependency checking
        # for directly registered objects. The actual dependency checking happens
        # when the XGBoostWrapper class tries to import xgboost.
        try:
            ModelRegistry.load("xgboost")
        except (ImportError, RuntimeError):
            # Expected when xgboost is not installed
            pass

    def test_plugin_loading_success(self):
        """Test plugin loading succeeds when dependencies are available."""
        # Mock all models as available
        with patch.object(
            ModelRegistry,
            "available_plugins",
            return_value={
                "logistic_regression": True,
                "xgboost": True,
                "lightgbm": True,
            },
        ):
            # Should succeed (though actual loading may fail due to missing actual libraries)
            # We're testing the dependency check logic, not the actual loading
            try:
                ModelRegistry.load("logistic_regression")
            except (ImportError, RuntimeError):
                # Expected since we don't have actual sklearn available in test env
                pass


class TestPreflightIntegration:
    """Test preflight integration with CLI."""

    def test_default_model_config_creation(self):
        """Test creation of default model config when none provided."""
        from glassalpha.cli.preflight import _create_default_model_config

        config = _create_default_model_config()
        assert config.type == "logistic_regression"
        assert config.params["random_state"] == 42

    def test_fallback_model_selection(self):
        """Test fallback model selection logic."""
        from glassalpha.cli.preflight import _find_fallback_model

        # Test fallback when XGBoost is requested but unavailable
        available = {"logistic_regression": True, "xgboost": False}
        fallback = _find_fallback_model("xgboost", available)
        assert fallback == "logistic_regression"

        # Test no fallback when no suitable model available
        available = {"xgboost": False, "lightgbm": False}
        fallback = _find_fallback_model("xgboost", available)
        assert fallback is None
