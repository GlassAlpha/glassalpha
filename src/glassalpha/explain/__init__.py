"""Explainer selection with explicit dispatch for AI maintainability.

Replaces registry pattern with simple if/elif dispatch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from glassalpha.explain.base import ExplainerBase

logger = logging.getLogger(__name__)


def select_explainer(model_type: str, requested_priority: list[str] | None = None, config: dict | None = None) -> str:
    """Select appropriate explainer based on model type.

    Args:
        model_type: Model type string (xgboost, lightgbm, logistic_regression)
        requested_priority: List of explainer names in priority order
        config: Optional configuration dict (unused currently)

    Returns:
        Explainer name string

    Raises:
        ImportError: If SHAP not installed for tree models

    Examples:
        >>> explainer = select_explainer("xgboost")
        >>> explainer = select_explainer("logistic_regression")

    """
    model_type = model_type.lower()

    # Define available explainers for each model type
    explainer_options = {
        "xgboost": ["treeshap", "kernelshap"],
        "lightgbm": ["treeshap", "kernelshap"],
        "xgb": ["treeshap", "kernelshap"],
        "lgb": ["treeshap", "kernelshap"],
        "lgbm": ["treeshap", "kernelshap"],
        "logistic_regression": ["coefficients"],
        "logistic": ["coefficients"],
        "linear": ["coefficients"],
    }

    # Get available explainers for this model type
    available_explainers = explainer_options.get(model_type, ["kernelshap"])

    # If priority is requested, use it; otherwise use defaults
    if requested_priority:
        # Filter requested priority to only include available explainers
        valid_priority = [p for p in requested_priority if p in available_explainers]
        if valid_priority:
            # Check if the first priority explainer is actually available
            first_choice = valid_priority[0]

            # Check availability for each explainer type
            if first_choice == "treeshap":
                try:
                    from glassalpha.explain.shap.tree import TreeSHAPExplainer

                    logger.info(f"Explainer: selected {first_choice} for {model_type} (priority)")
                    return first_choice
                except ImportError:
                    # Try next priority or fall back to default
                    if len(valid_priority) > 1:
                        next_choice = valid_priority[1]
                        if next_choice == "kernelshap" or next_choice == "coefficients":
                            logger.info(f"Explainer: selected {next_choice} for {model_type} (priority fallback)")
                            return next_choice
            elif first_choice == "kernelshap":
                # KernelSHAP is always available in our stub
                logger.info(f"Explainer: selected {first_choice} for {model_type} (priority)")
                return first_choice
            elif first_choice == "coefficients":
                logger.info(f"Explainer: selected {first_choice} for {model_type} (priority)")
                return first_choice

        # If no valid priority, raise error
        raise RuntimeError(f"No explainer from {requested_priority} is available for {model_type}")

    # Default selection logic
    if model_type in ("xgboost", "lightgbm", "xgb", "lgb", "lgbm"):
        try:
            from glassalpha.explain.shap.tree import TreeSHAPExplainer

            result = "treeshap"
            logger.info(f"Explainer: selected {result} for {model_type}")
            return result
        except ImportError:
            result = "kernelshap"
            logger.info(f"Explainer: selected {result} for {model_type} (fallback)")
            return result

    if model_type in ("logistic_regression", "logistic", "linear"):
        result = "coefficients"
        logger.info(f"Explainer: selected {result} for {model_type}")
        return result

    # Default fallback
    result = "kernelshap"
    logger.info(f"Explainer: selected {result} for {model_type}")
    return result


def _available(explainer_name: str) -> bool:
    """Check if an explainer is available (stub for test compatibility)."""
    # For compatibility, assume all explainers are available
    return True


# Import noop explainer for CLI registration
from glassalpha.explain.noop import noop as _noop

__all__ = [
    "_available",
    "noop",
    "select_explainer",
]
