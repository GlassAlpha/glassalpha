"""NoOp implementations for testing and partial pipelines.

These minimal implementations allow the pipeline to run even when
some components are not yet available or need to be disabled.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PassThroughModel:
    """Minimal model that returns constant predictions."""

    capabilities = {
        "supports_shap": True,  # Works with any explainer
        "supports_proba": True,
        "data_modality": "any",
    }
    version = "1.0.0"

    def __init__(self, default_value: float = 0.5, **kwargs):
        """Initialize with default prediction value.

        Args:
            default_value: Value to return for all predictions
            **kwargs: Ignored (for compatibility with sklearn-style API)

        """
        self.default_value = default_value
        self.is_fitted = False
        logger.info("PassThroughModel initialized")

    def fit(self, X, y=None, **kwargs):
        """Fit the model (no-op, just marks as fitted).

        Args:
            X: Training features (ignored)
            y: Training labels (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            self: Returns self for sklearn compatibility

        """
        self.is_fitted = True
        logger.debug("PassThroughModel fit() called (no-op)")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return constant predictions.

        Args:
            X: Input features (shape determines output size)

        Returns:
            Array of default values

        """
        n_samples = len(X)
        logger.debug(f"PassThroughModel predicting for {n_samples} samples")
        return np.full(n_samples, self.default_value)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return constant probabilities.

        Args:
            X: Input features

        Returns:
            Array of probabilities (binary classification assumed)

        """
        n_samples = len(X)
        proba = np.zeros((n_samples, 2))
        proba[:, 0] = 1 - self.default_value
        proba[:, 1] = self.default_value
        return proba

    def get_model_type(self) -> str:
        """Return model type."""
        return "passthrough"

    def get_capabilities(self) -> dict[str, Any]:
        """Return model capabilities."""
        return self.capabilities


class NoOpMetric:
    """Minimal metric that returns placeholder values."""

    metric_type = "noop"
    version = "1.0.0"

    def __init__(self):
        """Initialize NoOp metric."""
        logger.info("NoOpMetric initialized")

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Return placeholder metrics.

        Args:
            y_true: Ground truth (used for basic stats)
            y_pred: Predictions (used for basic stats)
            sensitive_features: Ignored

        Returns:
            Dictionary with placeholder metrics

        """
        logger.debug("NoOpMetric computing placeholder metrics")

        return {
            "noop_metric": 0.0,
            "samples_processed": len(y_true),
            "unique_predictions": len(np.unique(y_pred)),
            "status": "placeholder",
        }

    def get_metric_names(self) -> list[str]:
        """Return metric names."""
        return ["noop_metric", "samples_processed", "unique_predictions", "status"]

    def requires_sensitive_features(self) -> bool:
        """No sensitive features required."""
        return False
