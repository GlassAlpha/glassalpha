"""NoOp explainer for testing and fallback purposes.

This explainer provides placeholder values and can be used when
no real explainers are available or for testing the explainer system.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class NoOpExplainer:
    """NoOp explainer that returns empty explanations."""

    name = "noop"
    priority = -100
    version = "1.0.0"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize NoOp explainer."""

    @classmethod
    def is_compatible(cls, *, model: Any = None, model_type: str | None = None, config: dict | None = None) -> bool:
        """Check if model is compatible with NoOp explainer.

        Args:
            model: Model instance (optional)
            model_type: String model type identifier (optional)
            config: Configuration dict (optional, unused)

        Returns:
            Always True - NoOp explainer is compatible with all models

        Note:
            All arguments are keyword-only. NoOp is a fallback that works with any model.

        """
        return True  # NoOp always works as fallback

    def fit(self, model: Any, background_X: Any, **kwargs: Any) -> NoOpExplainer:
        """Fit the explainer (no-op)."""
        return self

    def explain(self, model: Any, X: Any, y: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Generate explanations (returns empty dict)."""
        return {
            "status": "no_explanation",
            "shap_values": np.zeros((X.shape[0], X.shape[1])),
            "base_value": 0.0,
            "data": X,
        }

    def explain_local(self, X: Any, **kwargs: Any) -> np.ndarray:
        """Generate local explanations (returns zeros)."""
        if hasattr(X, "shape"):
            n_samples, n_features = X.shape
            return np.zeros((n_samples, n_features))
        return np.array([])

    def supports_model(self, model: Any) -> bool:
        """Check if model is supported (always True for NoOp)."""
        return True
