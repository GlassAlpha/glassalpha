"""KernelSHAP explainer module (stub for test compatibility).

This module provides backwards compatibility for tests that expect
a KernelSHAP explainer implementation.
"""

from typing import Any

from ..base import ExplainerBase


class KernelSHAPExplainer(ExplainerBase):
    """Stub KernelSHAP explainer (backwards compatibility)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize KernelSHAP explainer."""
        super().__init__(*args, **kwargs)
        self.explainer = None  # Stub attribute for test compatibility
        self.capabilities = {
            "supports_shap": True,
            "supports_feature_importance": False,
            "supports_proba": True,
            "data_modality": "tabular",
        }  # Stub capabilities for test compatibility

    def supports_model(self, model: Any) -> bool:
        """Check if model is supported (stub)."""
        # KernelSHAP supports any model type
        return True

    def explain(self, *args: Any, **kwargs: Any) -> Any:
        """Generate explanations (stub)."""
        return {}
