"""Tests for enterprise feature gating.

These tests ensure that enterprise features are properly gated
and that the OSS/Enterprise boundary is enforced correctly.
"""

import os

import pytest

# Note: pandas/numpy mocking removed to avoid matplotlib import issues
from glassalpha.core import (
    ExplainerRegistry,
    FeatureNotAvailable,
    MetricRegistry,
    ModelRegistry,
    check_feature,
    is_enterprise,
)


class TestFeatureFlags:
    """Test basic feature flag functionality."""

    def setup_method(self):
        """Ensure clean environment for each test."""
        # Remove license key if present
        if "GLASSALPHA_LICENSE_KEY" in os.environ:
            del os.environ["GLASSALPHA_LICENSE_KEY"]

    def test_is_enterprise_without_license(self):
        """Test that is_enterprise returns False without license."""
        assert not is_enterprise(), "Should not be enterprise without license"

    def test_is_enterprise_with_license(self):
        """Test that is_enterprise returns True with license."""
        os.environ["GLASSALPHA_LICENSE_KEY"] = "test-key-123"
        try:
            assert is_enterprise(), "Should be enterprise with license"
        finally:
            del os.environ["GLASSALPHA_LICENSE_KEY"]

    def test_is_enterprise_with_empty_license(self):
        """Test that empty license key is treated as no license."""
        os.environ["GLASSALPHA_LICENSE_KEY"] = ""
        try:
            assert not is_enterprise(), "Empty license should not enable enterprise"
        finally:
            del os.environ["GLASSALPHA_LICENSE_KEY"]

    def test_check_feature_decorator_blocks_without_license(self):
        """Test that check_feature decorator blocks access without license."""

        @check_feature("test_feature")
        def enterprise_function():
            return "enterprise_result"

        with pytest.raises(FeatureNotAvailable) as exc_info:
            enterprise_function()

        assert "test_feature" in str(exc_info.value)
        assert "enterprise license" in str(exc_info.value).lower()

    def test_check_feature_decorator_allows_with_license(self):
        """Test that check_feature decorator allows access with license."""

        @check_feature("test_feature")
        def enterprise_function():
            return "enterprise_result"

        os.environ["GLASSALPHA_LICENSE_KEY"] = "valid-key"
        try:
            result = enterprise_function()
            assert result == "enterprise_result"
        finally:
            del os.environ["GLASSALPHA_LICENSE_KEY"]

    def test_check_feature_with_custom_message(self):
        """Test check_feature with custom error message."""
        custom_msg = "Custom error: This feature requires payment"

        @check_feature("custom_feature", message=custom_msg)
        def enterprise_function():
            return "result"

        with pytest.raises(FeatureNotAvailable) as exc_info:
            enterprise_function()

        assert str(exc_info.value) == custom_msg

    def test_nested_enterprise_functions(self):
        """Test that nested enterprise functions are properly gated."""

        @check_feature("outer_feature")
        def outer_function():
            return inner_function()

        @check_feature("inner_feature")
        def inner_function():
            return "inner_result"

        # Without license, should fail at outer
        with pytest.raises(FeatureNotAvailable) as exc_info:
            outer_function()
        assert "outer_feature" in str(exc_info.value)

        # With license, both should work
        os.environ["GLASSALPHA_LICENSE_KEY"] = "key"
        try:
            result = outer_function()
            assert result == "inner_result"
        finally:
            del os.environ["GLASSALPHA_LICENSE_KEY"]


class TestEnterpriseComponentFiltering:
    """Test that enterprise components are properly filtered."""

    def test_enterprise_component_registration(self):
        """Test registering enterprise components."""

        # Register an enterprise explainer
        @ExplainerRegistry.register("enterprise_explainer", enterprise=True, priority=100)
        class EnterpriseExplainer:
            capabilities = {"supported_models": ["all"]}
            version = "1.0.0"
            priority = 100

            @classmethod
            def is_compatible(cls, *, model=None, model_type=None, config=None):
                return True  # Test explainer accepts all models

        # Register an OSS explainer
        @ExplainerRegistry.register("oss_explainer", enterprise=False, priority=50)
        class OSSExplainer:
            capabilities = {"supported_models": ["all"]}
            version = "1.0.0"
            priority = 50

            @classmethod
            def is_compatible(cls, *, model=None, model_type=None, config=None):
                return True  # Test explainer accepts all models

        # Without enterprise flag, should not see enterprise component
        oss_components = ExplainerRegistry.get_all(include_enterprise=False)
        assert "enterprise_explainer" not in oss_components
        assert "oss_explainer" in oss_components
        assert "noop" in oss_components  # NoOp should always be there

        # With enterprise flag, should see all
        all_components = ExplainerRegistry.get_all(include_enterprise=True)
        assert "enterprise_explainer" in all_components
        assert "oss_explainer" in all_components
        assert "noop" in all_components

    def test_enterprise_component_selection_without_license(self):
        """Test that enterprise components are not selected without license."""

        # Register enterprise component with high priority
        @ExplainerRegistry.register("enterprise_best", enterprise=True, priority=1000)
        class EnterpriseBest:
            capabilities = {"supported_models": ["xgboost"]}
            version = "1.0.0"
            priority = 1000

            @classmethod
            def is_compatible(cls, *, model=None, model_type=None, config=None):
                return model_type in cls.capabilities.get("supported_models", [])

        config = {"explainers": {"priority": ["enterprise_best", "noop"]}}

        # Without license, should skip enterprise and select noop
        from glassalpha.core import select_explainer

        selected = select_explainer("xgboost", config)
        assert selected == "noop", "Should skip enterprise component without license"

    def test_enterprise_component_selection_with_license(self):
        """Test that enterprise components are selected with license."""

        # Register enterprise component
        @ExplainerRegistry.register("enterprise_premium", enterprise=True, priority=500)
        class EnterprisePremium:
            capabilities = {"supported_models": ["xgboost"]}
            version = "1.0.0"
            priority = 500

            @classmethod
            def is_compatible(cls, *, model=None, model_type=None, config=None):
                return model_type in cls.capabilities.get("supported_models", [])

        config = {"explainers": {"priority": ["enterprise_premium", "noop"]}}

        # With license, should select enterprise component
        os.environ["GLASSALPHA_LICENSE_KEY"] = "premium-key"
        try:
            from glassalpha.core import select_explainer

            selected = select_explainer("xgboost", config)
            assert selected == "enterprise_premium", "Should select enterprise component with license"
        finally:
            del os.environ["GLASSALPHA_LICENSE_KEY"]

    def test_mixed_oss_enterprise_priority(self):
        """Test selection with mixed OSS and enterprise components."""

        # Register mixed components
        @MetricRegistry.register("metric_oss", enterprise=False, priority=50)
        class MetricOSS:
            metric_type = "performance"
            version = "1.0.0"

        @MetricRegistry.register("metric_enterprise", enterprise=True, priority=100)
        class MetricEnterprise:
            metric_type = "performance"
            version = "2.0.0"

        # Without license
        oss_metrics = MetricRegistry.get_all(include_enterprise=False)
        assert "metric_oss" in oss_metrics
        assert "metric_enterprise" not in oss_metrics

        # Simulate component selection (simplified)
        available = [name for name in ["metric_enterprise", "metric_oss", "noop"] if name in oss_metrics]
        assert "metric_enterprise" not in available
        assert "metric_oss" in available


class TestEnterpriseFeatureIsolation:
    """Test that enterprise features are properly isolated."""

    def test_enterprise_module_import_gating(self):
        """Test that enterprise modules can be gated at import."""

        # Simulate an enterprise module
        def create_enterprise_module():
            @check_feature("advanced_module")
            def initialize():
                return "initialized"

            # Module initialization would call this
            return initialize()

        # Should fail without license
        with pytest.raises(FeatureNotAvailable):
            create_enterprise_module()

        # Should work with license
        os.environ["GLASSALPHA_LICENSE_KEY"] = "key"
        try:
            result = create_enterprise_module()
            assert result == "initialized"
        finally:
            del os.environ["GLASSALPHA_LICENSE_KEY"]

    def test_registry_metadata_for_enterprise(self):
        """Test that registry properly tracks enterprise metadata."""

        # Register with enterprise flag
        @ModelRegistry.register("enterprise_model", enterprise=True)
        class EnterpriseModel:
            capabilities = {"advanced": True}
            version = "1.0.0"

        # Check metadata
        metadata = ModelRegistry.get_metadata("enterprise_model")
        assert metadata.get("enterprise") is True

        # Register OSS model
        @ModelRegistry.register("oss_model", enterprise=False)
        class OSSModel:
            capabilities = {"basic": True}
            version = "1.0.0"

        metadata = ModelRegistry.get_metadata("oss_model")
        assert metadata.get("enterprise") is False

    def test_license_key_formats(self):
        """Test different license key formats are handled."""
        test_keys = [
            "simple-key",
            "UPPERCASE-KEY",
            "key-with-numbers-123",
            "key_with_underscores",
            "very-long-key-" + "x" * 100,
        ]

        for key in test_keys:
            os.environ["GLASSALPHA_LICENSE_KEY"] = key
            try:
                assert is_enterprise(), f"Should accept license key: {key[:20]}..."
            finally:
                del os.environ["GLASSALPHA_LICENSE_KEY"]

    def test_enterprise_boundary_in_config(self):
        """Test that configs can specify enterprise requirements."""

        # This would be used in config validation
        def validate_config_features(config, is_enterprise):
            """Validate that config doesn't use enterprise features in OSS mode."""
            issues = []

            # Check for enterprise-only report templates
            if config.get("report", {}).get("template") == "eu_ai_act" and not is_enterprise:
                issues.append("EU AI Act template requires enterprise license")

            # Check for enterprise-only explainers
            explainer_priority = config.get("explainers", {}).get("priority", [])
            enterprise_explainers = ["deep_shap", "gradient_shap"]
            for exp in explainer_priority:
                if exp in enterprise_explainers and not is_enterprise:
                    issues.append(f"Explainer '{exp}' requires enterprise license")

            return issues

        # Test OSS config
        oss_config = {
            "report": {"template": "standard_audit"},
            "explainers": {"priority": ["treeshap", "kernelshap"]},
        }
        issues = validate_config_features(oss_config, is_enterprise=False)
        assert len(issues) == 0, "OSS config should have no issues"

        # Test enterprise config without license
        enterprise_config = {
            "report": {"template": "eu_ai_act"},
            "explainers": {"priority": ["deep_shap", "treeshap"]},
        }
        issues = validate_config_features(enterprise_config, is_enterprise=False)
        assert len(issues) == 2, "Should detect enterprise features"
        assert any("EU AI Act" in issue for issue in issues)
        assert any("deep_shap" in issue for issue in issues)
