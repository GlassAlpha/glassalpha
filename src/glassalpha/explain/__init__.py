"""Explainer selection with explicit dispatch for AI maintainability.

Replaces registry pattern with simple if/elif dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from glassalpha.explain.base import ExplainerBase


def select_explainer(model_type: str, config: dict | None = None) -> ExplainerBase:
    """Select appropriate explainer based on model type.

    Args:
        model_type: Model type string (xgboost, lightgbm, logistic_regression)
        config: Optional configuration dict (unused currently)

    Returns:
        Explainer instance ready to fit

    Raises:
        ImportError: If SHAP not installed for tree models

    Examples:
        >>> explainer = select_explainer("xgboost")
        >>> explainer = select_explainer("logistic_regression")

    """
    model_type = model_type.lower()

    # Tree models → TreeSHAP (requires SHAP library)
    if model_type in ("xgboost", "lightgbm", "xgb", "lgb", "lgbm"):
        try:
            from glassalpha.explain.shap import TreeSHAPExplainer

            return TreeSHAPExplainer()
        except ImportError:
            # Fall back to KernelSHAP if TreeSHAP fails
            from glassalpha.explain.shap import KernelSHAPExplainer

            return KernelSHAPExplainer()

    # Linear models → Coefficients explainer
    if model_type in ("logistic_regression", "logistic", "linear"):
        from glassalpha.explain.coefficients import CoefficientsExplainer

        return CoefficientsExplainer()

    # Default fallback → KernelSHAP (model-agnostic)
    try:
        from glassalpha.explain.shap import KernelSHAPExplainer

        return KernelSHAPExplainer()
    except ImportError as e:
        raise ImportError(
            "SHAP library required for explanations. Install with: pip install 'glassalpha[shap]' or pip install shap",
        ) from e


__all__ = [
    "select_explainer",
]
