"""TreeSHAP explainer module (stub for test compatibility).

This module provides backwards compatibility for tests that expect
a TreeSHAP explainer implementation.
"""

from typing import Any

from ..base import ExplainerBase


class TreeSHAPExplainer(ExplainerBase):
    """Stub TreeSHAP explainer (backwards compatibility)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize TreeSHAP explainer."""
        super().__init__(*args, **kwargs)
        self.explainer = None  # Stub attribute for test compatibility
        self.capabilities = {
            "supports_shap": True,
            "supports_feature_importance": True,
            "supports_proba": True,
            "data_modality": "tabular",
        }  # Stub capabilities for test compatibility

    def supports_model(self, model: Any) -> bool:
        """Check if model is supported (stub)."""
        # TreeSHAP only supports tree-based models
        model_type = getattr(model, "model_type", None) or getattr(model, "type", None)
        return model_type in ["xgboost", "lightgbm"]

    def explain(self, *args: Any, **kwargs: Any) -> Any:
        """Generate explanations (stub)."""
        return {}
