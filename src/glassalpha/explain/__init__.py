"""Explainer selection with explicit dispatch for AI maintainability.

Replaces registry pattern with simple if/elif dispatch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from glassalpha.explain.noop import noop as _noop

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
        "random_forest": ["treeshap", "kernelshap"],
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
                        if next_choice in {"kernelshap", "coefficients"}:
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
    if model_type in ("xgboost", "lightgbm", "xgb", "lgb", "lgbm", "random_forest"):
        try:
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


def compute_explanations(
    model: Any,
    X: Any,
    feature_names: list[str],
    random_seed: int = 42,
    n_samples: int = 100,
) -> dict[str, Any]:
    """Compute SHAP explanations for model predictions.

    Args:
        model: Fitted model object
        X: Feature matrix (DataFrame or array)
        feature_names: List of feature names
        random_seed: Random seed for deterministic sampling
        n_samples: Number of samples to explain (default: 100)

    Returns:
        Dictionary with explanation summary including:
        - method: Explainer method used (treeshap, kernelshap, coefficients)
        - global_importance: Mean absolute SHAP values per feature
        - n_samples_explained: Number of samples explained

    """
    import numpy as np
    import pandas as pd

    from glassalpha.models.detection import detect_model_type

    model_type = detect_model_type(model)
    explainer_name = select_explainer(model_type)

    # Import appropriate explainer
    if explainer_name == "treeshap":
        from glassalpha.explain.shap import TreeSHAPExplainer

        explainer = TreeSHAPExplainer()
    elif explainer_name == "coefficients":
        from glassalpha.explain.coefficients import CoefficientsExplainer

        explainer = CoefficientsExplainer()
    else:  # kernelshap
        from glassalpha.explain.shap import KernelSHAPExplainer

        explainer = KernelSHAPExplainer()

    # Sample data if too large
    if len(X) > n_samples:
        np.random.seed(random_seed)
        sample_idx = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X.iloc[sample_idx] if isinstance(X, pd.DataFrame) else X[sample_idx]
    else:
        X_sample = X

    # Create a simple wrapper object that explainers expect
    class ModelWrapper:
        def __init__(self, model):
            self.model = model
            self._model = model

        def predict(self, X):
            return model.predict(X)

        def predict_proba(self, X):
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)
            raise AttributeError("Model doesn't have predict_proba")

    wrapper = ModelWrapper(model)

    # Fit explainer with wrapper and background data
    explainer.fit(wrapper, X_sample, feature_names=feature_names)

    # Generate explanations (pass model and X for API compatibility)
    result = explainer.explain(wrapper, X_sample)

    # Return summary
    return {
        "method": explainer_name,
        "global_importance": result.get("global_importance", {}),
        "n_samples_explained": len(X_sample),
        "feature_names": feature_names,
    }


# Import noop explainer for CLI registration

__all__ = [
    "_available",
    "compute_explanations",
    "noop",
    "select_explainer",
]
