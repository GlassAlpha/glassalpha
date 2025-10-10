"""Tests for explainer selection logic with missing dependencies."""

import pytest

from glassalpha.explain.registry import select_explainer


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


def test_select_explainer_tree_models_default():
    """Test that tree models get appropriate explainer."""
    # XGBoost should prefer treeshap if available, otherwise fallback
    selected = select_explainer("xgboost", requested_priority=None)
    # Should be one of the tree-compatible explainers
    assert selected in ["treeshap", "permutation", "kernelshap"]


def test_select_explainer_unknown_model_type():
    """Test that unknown model types raise clear error."""
    with pytest.raises(RuntimeError) as exc_info:
        select_explainer("unknown_model_type_xyz123", requested_priority=None)

    error_msg = str(exc_info.value)
    # Should mention that no explainer is available
    assert "No compatible explainer" in error_msg or "not available" in error_msg.lower()
