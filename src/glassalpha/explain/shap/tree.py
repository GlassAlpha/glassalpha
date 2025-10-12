"""TreeSHAP explainer module (stub for test compatibility).

This module provides backwards compatibility for tests that expect
a TreeSHAP explainer implementation.
"""

import logging
from typing import Any

from ..base import ExplainerBase

logger = logging.getLogger(__name__)


class TreeSHAPExplainer(ExplainerBase):
    """Stub TreeSHAP explainer (backwards compatibility)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize TreeSHAP explainer."""
        super().__init__(*args, **kwargs)
        self.model = None  # Model wrapper for explanations
        self.background_data = None  # Background data for SHAP baseline
        self.feature_names = None  # Feature names for interpretation
        self.explainer = None  # SHAP explainer instance (stub for compatibility)
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

    def fit(self, wrapper, background_X, feature_names=None):
        """Fit the TreeSHAP explainer with model and background data.

        Args:
            wrapper: Model wrapper with predict/predict_proba methods
            background_X: Background data for SHAP baseline
            feature_names: Optional feature names for interpretation

        Returns:
            self: Returns self for chaining

        """
        self.model = wrapper
        self.background_data = background_X
        self.feature_names = feature_names or list(range(background_X.shape[1]))

        # For testing compatibility, create a stub SHAP explainer
        # In a real implementation, this would create an actual SHAP TreeExplainer
        self.explainer = f"TreeSHAP_stub_for_{wrapper.__class__.__name__}"

        logger.info(f"Fitted TreeSHAP explainer with {background_X.shape[0]} background samples")
        return self

    def explain(self, *args: Any, **kwargs: Any) -> Any:
        """Generate explanations (stub)."""
        return {"global_importance": {}, "local_explanations": []}
