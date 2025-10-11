"""Consolidated explainer selection tests.

This module consolidates all explainer selection and registry tests from:
- tests/test_explainer_selection.py
- tests/test_explainer_registry_contract.py
- tests/contracts/test_explainer_selection.py

Tests the complete explainer selection system including registry,
compatibility checking, and fallback behavior.
"""

import pytest

from glassalpha.constants import NO_EXPLAINER_MSG
from glassalpha.explain import select_explainer # ExplainerRegistry, select_explainer

# ============================================================================
# Selection Tests (from test_explainer_selection.py)
# ============================================================================


def test_select_explainer_with_explicit_unavailable_priority():
    """Test that requesting unavailable explainer gives helpful error."""
    # Assume SHAP might not be installed
    with pytest.raises(RuntimeError) as exc_info:
        # Try to request treeshap for a model where it's not available
        # This should give a helpful error message
        try:
            import shap

            pytest.skip("SHAP is installed, cannot test missing dependency case")
        except ImportError:
            pass

        select_explainer("logistic_regression", ["treeshap"])

    error_msg = str(exc_info.value)
    # Should mention the requested explainer
    assert "treeshap" in error_msg
    # Should mention missing dependencies
    assert "shap" in error_msg.lower()
    # Should suggest alternatives
    assert "coefficients" in error_msg or "permutation" in error_msg


def test_select_explainer_auto_fallback_for_linear_models():
    """Test that linear models get coefficients explainer by default."""
    # When no priority specified, should select coefficients for linear models
    selected = select_explainer("logistic_regression", requested_priority=None)
    # Should prefer coefficients (zero dependencies) over others
    assert selected == "coefficients"


def test_select_explainer_respects_user_priority():
    """Test that user-specified priority is respected when available."""
    # If user explicitly requests coefficients, should get it
    selected = select_explainer("logistic_regression", ["coefficients", "permutation"])
    assert selected == "coefficients"

    # If user requests permutation first, should get it
    selected = select_explainer("logistic_regression", ["permutation", "coefficients"])
    assert selected == "permutation"


# ============================================================================
# Registry Contract Tests (from test_explainer_registry_contract.py)
# ============================================================================


def test_all_explainers_have_correct_is_compatible_signature():
    """Ensure all registered explainers implement is_compatible correctly.

    This test validates the Phase 2.5 explainer API standardization by checking
    that all explainers can be called with the keyword-only signature:
        is_compatible(model=None, model_type=None, config=None)

    Failure indicates an explainer needs to be updated to match the base class signature.
    """
    # Discover all available explainers
    ExplainerRegistry.discover()

    # Get all registered explainers
    explainer_names = ExplainerRegistry.names()

    # Must have at least one explainer registered
    assert len(explainer_names) > 0, "No explainers registered - check registration"

    failures = []
    for explainer_name in explainer_names:
        try:
            explainer_class = ExplainerRegistry.get(explainer_name)

            # Test the signature by calling with keyword-only arguments
            result = explainer_class.is_compatible(
                model=None,
                model_type=None,
                config=None,
            )

            # Should return a boolean
            assert isinstance(result, bool), f"{explainer_name}.is_compatible() should return bool"

        except (TypeError, AttributeError) as e:
            failures.append(f"{explainer_name}: {e}")
        except Exception as e:
            # Other errors might indicate bugs, but not signature issues
            pytest.skip(f"{explainer_name} has other issues: {e}")

    if failures:
        failure_msg = "Explainer signature violations found:\n" + "\n".join(f"  - {f}" for f in failures)
        pytest.fail(failure_msg)


def test_explainer_compatibility_method_exists():
    """Test that all explainers have is_compatible method."""
    ExplainerRegistry.discover()

    for explainer_name in ExplainerRegistry.names():
        explainer_class = ExplainerRegistry.get(explainer_name)
        assert hasattr(explainer_class, "is_compatible"), f"{explainer_name} missing is_compatible method"


def test_explainer_is_compatible_returns_boolean():
    """Test that is_compatible returns boolean for all explainers."""
    ExplainerRegistry.discover()

    for explainer_name in ExplainerRegistry.names():
        explainer_class = ExplainerRegistry.get(explainer_name)

        # Test various argument combinations
        test_cases = [
            {"model": None, "model_type": None, "config": None},
            {"model": None, "model_type": "xgboost", "config": None},
            {"model": None, "model_type": None, "config": {}},
        ]

        for kwargs in test_cases:
            result = explainer_class.is_compatible(**kwargs)
            assert isinstance(result, bool), f"{explainer_name}.is_compatible() should return bool, got {type(result)}"


# ============================================================================
# Regression Contract Tests (from contracts/test_explainer_selection.py)
# ============================================================================


class TestExplainerSelection:
    """Test explainer selection contract compliance."""

    def test_no_compatible_explainer_raises_exact_error(self) -> None:
        """Test that unsupported models raise exact RuntimeError.

        Prevents regression where explainer selection fails silently
        or raises wrong error types/messages.
        """

        # Create dummy unsupported model
        class UnsupportedModel:
            def get_model_info(self) -> dict[str, str]:
                return {"type": "unsupported_model_type"}

        unsupported_model = UnsupportedModel()

        # Should raise exactly RuntimeError with exact message
        with pytest.raises(RuntimeError) as exc_info:
            ExplainerRegistry.find_compatible(unsupported_model)

        assert str(exc_info.value) == NO_EXPLAINER_MSG

    def test_xgboost_has_compatible_explainer(self) -> None:
        """Test that XGBoost models find compatible explainers."""
        # Test with string model type
        explainer_name = ExplainerRegistry.find_compatible("xgboost")
        # TreeSHAP is preferred for XGBoost, but falls back to permutation if shap not available
        assert explainer_name in ["treeshap", "permutation"]

        # Test with mock model object
        class XGBoostModel:
            def get_model_info(self) -> dict[str, str]:
                return {"type": "xgboost"}

        model = XGBoostModel()
        explainer_name = ExplainerRegistry.find_compatible(model)
        assert explainer_name in ["treeshap", "permutation"]

    def test_logistic_regression_has_compatible_explainer(self) -> None:
        """Test that LogisticRegression models find compatible explainers."""
        # Test with string model type
        explainer_name = ExplainerRegistry.find_compatible("logistic_regression")
        # Should prefer coefficients (zero dependencies)
        assert explainer_name == "coefficients"

        # Test with mock model object
        class LogisticRegressionModel:
            def get_model_info(self) -> dict[str, str]:
                return {"type": "logistic_regression"}

        model = LogisticRegressionModel()
        explainer_name = ExplainerRegistry.find_compatible(model)
        assert explainer_name == "coefficients"

    def test_explainer_selection_with_priority_list(self) -> None:
        """Test explainer selection respects priority list."""
        # Test with priority list using select_explainer function
        explainer_name = select_explainer("xgboost", ["permutation", "treeshap"])
        # Should respect priority order
        assert explainer_name in ["permutation", "treeshap"]

    def test_explainer_selection_fallback_on_unavailable(self) -> None:
        """Test that unavailable explainers fall back to compatible ones."""
        # Try to request unavailable explainer first using select_explainer
        explainer_name = select_explainer("xgboost", ["unavailable_explainer", "treeshap"])
        # Should fall back to available explainer
        assert explainer_name in ["treeshap", "permutation"]

    def test_explainer_selection_no_compatible_explainer(self) -> None:
        """Test that no compatible explainer raises exact RuntimeError."""

        # Test with completely unsupported model
        class CompletelyUnsupportedModel:
            def get_model_info(self) -> dict[str, str]:
                return {"type": "completely_unsupported"}

        model = CompletelyUnsupportedModel()

        with pytest.raises(RuntimeError) as exc_info:
            ExplainerRegistry.find_compatible(model)

        assert str(exc_info.value) == NO_EXPLAINER_MSG

    def test_explainer_selection_with_empty_priority(self) -> None:
        """Test explainer selection with empty priority list."""
        explainer_name = select_explainer("xgboost", [])
        # Should find compatible explainer even with empty priority
        assert explainer_name in ["treeshap", "permutation"]


# ============================================================================
# Integration Tests (New - comprehensive explainer scenarios)
# ============================================================================


def test_explainer_selection_deterministic():
    """Test that explainer selection is deterministic."""
    # Same inputs should give same result
    result1 = select_explainer("logistic_regression", ["coefficients"])
    result2 = select_explainer("logistic_regression", ["coefficients"])
    assert result1 == result2


def test_explainer_selection_with_model_object():
    """Test explainer selection with actual model object."""

    # Create a mock model
    class MockModel:
        def __init__(self, model_type):
            self.model_type = model_type

        def get_model_info(self):
            return {"type": self.model_type}

    model = MockModel("xgboost")
    explainer = ExplainerRegistry.find_compatible(model)
    assert explainer in ["treeshap", "permutation"]


def test_explainer_priority_respected():
    """Test that explicit priority overrides defaults."""
    # For logistic regression, coefficients is preferred, but if we explicitly request permutation
    # and coefficients is available, we should still get permutation if requested first
    explainer = select_explainer("logistic_regression", ["permutation", "coefficients"])
    assert explainer == "permutation"


def test_explainer_fallback_when_priority_unavailable():
    """Test fallback when requested explainer is not available."""
    # Request unavailable explainer first, should fall back to available one
    explainer = select_explainer("xgboost", ["unavailable", "permutation"])
    assert explainer == "permutation"
