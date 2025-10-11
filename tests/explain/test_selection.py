"""Explainer selection tests (simplified after registry removal).

Tests the explicit dispatch-based explainer selection system.
"""

import pytest

from glassalpha.explain import select_explainer

# ============================================================================
# Selection Tests
# ============================================================================


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


def test_explainer_selection_deterministic():
    """Test that explainer selection is deterministic."""
    # Same inputs should give same result
    result1 = select_explainer("logistic_regression", ["coefficients"])
    result2 = select_explainer("logistic_regression", ["coefficients"])
    assert result1 == result2


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
