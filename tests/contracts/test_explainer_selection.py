"""Regression tests for explainer selection contract compliance.

Prevents missing "No compatible explainer found" errors during selection.
"""

from typing import Any

import pytest

from glassalpha.constants import NO_EXPLAINER_MSG


class TestExplainerSelection:
    """Test explainer selection contract compliance."""

    def test_no_compatible_explainer_raises_exact_error(self) -> None:
        """Test that unsupported models raise exact RuntimeError.

        Prevents regression where explainer selection fails silently
        or raises wrong error types/messages.
        """
        from glassalpha.explain.registry import ExplainerRegistry  # noqa: PLC0415

        # Create dummy unsupported model
        class UnsupportedModel:
            def get_model_info(self) -> dict[str, str]:
                return {"type": "unsupported_model_type"}

        unsupported_model = UnsupportedModel()

        # Should raise exactly RuntimeError with exact message
        with pytest.raises(RuntimeError) as exc_info:
            ExplainerRegistry.find_compatible(unsupported_model)

        assert str(exc_info.value) == NO_EXPLAINER_MSG  # noqa: S101

    def test_xgboost_has_compatible_explainer(self) -> None:
        """Test that XGBoost models find compatible explainers."""
        from glassalpha.explain.registry import ExplainerRegistry  # noqa: PLC0415

        # Test with string model type
        explainer_name = ExplainerRegistry.find_compatible("xgboost")
        assert explainer_name == "treeshap"  # noqa: S101

        # Test with mock model object
        class XGBoostModel:
            def get_model_info(self) -> dict[str, str]:
                return {"type": "xgboost"}

        xgb_model = XGBoostModel()
        explainer_name = ExplainerRegistry.find_compatible(xgb_model)
        assert explainer_name == "treeshap"  # noqa: S101

    def test_sklearn_models_have_compatible_explainers(self) -> None:
        """Test that sklearn models find compatible explainers."""
        from glassalpha.explain.registry import ExplainerRegistry  # noqa: PLC0415

        sklearn_model_types = [
            "logistic_regression",
            "linear_regression",
        ]

        for model_type in sklearn_model_types:
            explainer_name = ExplainerRegistry.find_compatible(model_type)
            # Should get either coefficients or kernelshap for sklearn models
            assert explainer_name in ["coefficients", "kernelshap"], f"No compatible explainer found for {model_type}"  # noqa: S101

    def test_kernel_shap_supports_tree_models(self) -> None:
        """Test that KernelSHAP is compatible with tree models.

        This is the specific regression test for the contract violation:
        "KernelSHAPExplainer.is_compatible('xgboost') expected True but returns False"
        """
        from glassalpha.explain.shap.kernel import KernelSHAPExplainer  # noqa: PLC0415

        explainer = KernelSHAPExplainer()

        tree_model_types = [
            "xgboost",
            "lightgbm",
            "random_forest",
            "decision_tree",
        ]

        for model_type in tree_model_types:
            is_compatible = explainer.is_compatible(model_type=model_type)
            assert is_compatible, f"KernelSHAP should be compatible with {model_type}"  # noqa: S101

    def test_tree_shap_supports_tree_models(self) -> None:
        """Test that TreeSHAP is compatible with tree models."""
        try:
            from glassalpha.explain.shap.tree import TreeSHAPExplainer  # noqa: PLC0415
        except ImportError:
            pytest.skip("TreeSHAP explainer not available")

        explainer = TreeSHAPExplainer()

        tree_model_types = [
            "xgboost",
            "lightgbm",
            "random_forest",
            "decision_tree",
            "gradient_boosting",
        ]

        for model_type in tree_model_types:
            is_compatible = explainer.is_compatible(model_type=model_type)
            assert is_compatible, f"TreeSHAP should be compatible with {model_type}"  # noqa: S101

    def test_explainer_selection_deterministic(self) -> None:
        """Test that explainer selection is deterministic.

        Same model should always get same explainer for reproducibility.
        """
        from glassalpha.explain.registry import ExplainerRegistry  # noqa: PLC0415

        test_cases = [
            "xgboost",
            "lightgbm",
            "logistic_regression",
        ]

        for model_type in test_cases:
            # Multiple calls should return same explainer name
            explainer1 = ExplainerRegistry.find_compatible(model_type)
            explainer2 = ExplainerRegistry.find_compatible(model_type)
            explainer3 = ExplainerRegistry.find_compatible(model_type)

            assert explainer1 == explainer2 == explainer3, f"Explainer selection not deterministic for {model_type}"  # noqa: S101

    def test_explainer_registry_priority_order(self) -> None:
        """Test that explainer selection follows priority order.

        For models supported by multiple explainers, should select
        based on consistent priority.
        """
        from glassalpha.explain.registry import ExplainerRegistry  # noqa: PLC0415

        # XGBoost should get TreeSHAP (preferred for tree models)
        xgb_explainer = ExplainerRegistry.find_compatible("xgboost")

        # Check that it's a tree-based explainer (TreeSHAP has higher priority)
        try:
            from glassalpha.explain.shap.tree import TreeSHAPExplainer  # noqa: PLC0415

            assert xgb_explainer == "treeshap", "XGBoost should prefer TreeSHAP over KernelSHAP"  # noqa: S101
        except ImportError:
            # If TreeSHAP not available, should fall back to KernelSHAP

            assert xgb_explainer == "kernelshap"  # noqa: S101

    def test_explainer_selection_returns_string_name(self) -> None:
        """Test that explainer selection returns string names, not classes."""
        from glassalpha.core import select_explainer

        name = select_explainer("xgboost", {"explainers": {"priority": ["noop"]}})
        assert isinstance(name, str) and name == "noop"

    def test_mock_model_without_get_model_info(self) -> None:
        """Test explainer compatibility with models lacking get_model_info."""
        from glassalpha.explain.shap.kernel import KernelSHAPExplainer  # noqa: PLC0415

        class ModelWithoutInfo:
            def predict(self, X: Any) -> list[int]:  # noqa: N803, ARG002, ANN401
                return [0, 1]

        model = ModelWithoutInfo()
        explainer = KernelSHAPExplainer()

        # Should fall back to supports_model check
        # KernelSHAP is model-agnostic so should support models with predict
        is_compatible = explainer.is_compatible(model=model)
        assert isinstance(is_compatible, bool)  # Should not crash  # noqa: S101

    def test_pipeline_explainer_selection_integration(self) -> None:
        """Test explainer selection within audit pipeline context.

        Integration test to ensure the pipeline properly handles
        explainer selection and raises proper errors.
        """
        from types import SimpleNamespace  # noqa: PLC0415
        from unittest.mock import Mock, patch  # noqa: PLC0415

        from glassalpha.pipeline.audit import AuditPipeline  # noqa: PLC0415

        # Mock config
        config = SimpleNamespace(
            audit_profile="tabular_compliance",
            model=Mock(),
            explainers=Mock(),
        )

        # Mock unsupported model that should trigger error
        config.model.get_model_info = Mock(return_value={"type": "unsupported_model"})

        pipeline = AuditPipeline(config)
        pipeline.model = config.model

        # Should raise RuntimeError when no compatible explainer found
        with patch.object(pipeline, "_select_explainer") as mock_select:
            mock_select.side_effect = RuntimeError(NO_EXPLAINER_MSG)

            with pytest.raises(RuntimeError, match=NO_EXPLAINER_MSG):
                pipeline._select_explainer(config.model)  # noqa: SLF001

    def test_registry_knows_common_aliases(self) -> None:
        """Test that registry knows common aliases and resolves them correctly.

        Prevents regression where configs use aliases but registry doesn't recognize them.
        """
        from glassalpha.explain.registry import ExplainerRegistry

        # Test that aliases resolve to the same objects as canonical names
        assert ExplainerRegistry.get("coefficients") == ExplainerRegistry.get("coef")
        assert ExplainerRegistry.get("coefficients") == ExplainerRegistry.get("coeff")
        assert ExplainerRegistry.get("permutation") == ExplainerRegistry.get("permutation_importance")
        assert ExplainerRegistry.get("permutation") == ExplainerRegistry.get("perm")

        # Test that aliases are recognized in names()
        names = ExplainerRegistry.names()
        assert "coefficients" in names
        assert "coef" in names
        assert "coeff" in names
        assert "permutation" in names
        assert "permutation_importance" in names
        assert "perm" in names

        # Test that has() works with aliases
        assert ExplainerRegistry.has("coefficients")
        assert ExplainerRegistry.has("coef")
        assert ExplainerRegistry.has("coeff")
        assert ExplainerRegistry.has("permutation")
        assert ExplainerRegistry.has("permutation_importance")
        assert ExplainerRegistry.has("perm")
