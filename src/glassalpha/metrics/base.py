"""Base metric interfaces and common utilities.

This module provides base interfaces and utilities for implementing
metrics in GlassAlpha. It follows the same pattern as models and explainers
with protocol-based interfaces and registry system.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from .registry import MetricRegistry

logger = logging.getLogger(__name__)


class BaseMetric:
    """Base class providing common functionality for metrics.

    This class provides common utilities that metric implementations
    can inherit from, while still following the MetricInterface protocol.
    """

    def __init__(self, name: str, metric_type: str, version: str = "1.0.0"):
        """Initialize base metric.

        Args:
            name: Name of the metric
            metric_type: Type category (performance, fairness, drift)
            version: Version string

        """
        self.name = name
        self.metric_type = metric_type
        self.version = version
        logger.debug(f"BaseMetric {name} initialized")

    def validate_inputs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Validate common metric inputs.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sensitive_features: Optional sensitive features for fairness metrics

        Returns:
            Dictionary with validation results and processed inputs

        """
        # Convert to numpy arrays for consistent handling
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Basic shape validation
        if len(y_true) != len(y_pred):
            raise ValueError(f"Shape mismatch: y_true has {len(y_true)} samples, y_pred has {len(y_pred)}")

        # Check for NaN values
        if np.any(np.isnan(y_true)):
            logger.warning("y_true contains NaN values")
        if np.any(np.isnan(y_pred)):
            logger.warning("y_pred contains NaN values")

        # Sensitive features validation
        if sensitive_features is not None:
            sensitive_features = np.asarray(sensitive_features)
            if len(sensitive_features) != len(y_true):
                raise ValueError(
                    f"Sensitive features length {len(sensitive_features)} doesn't match target length {len(y_true)}",
                )

        return {
            "n_samples": len(y_true),
            "y_true_type": str(y_true.dtype),
            "y_pred_type": str(y_pred.dtype),
            "has_sensitive_features": sensitive_features is not None,
            "sensitive_feature_columns": None,  # Not applicable for numpy arrays
        }

    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safely divide two numbers, handling division by zero.

        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if denominator is zero

        Returns:
            Result of division or default value

        """
        if denominator == 0:
            logger.warning(f"Division by zero in metric calculation, returning {default}")
            return default
        return numerator / denominator

    def get_metric_names(self) -> list[str]:
        """Return list of metric names computed by this metric.

        Base implementation returns the metric name. Override for multi-metric implementations.
        """
        return [self.name]

    def requires_sensitive_features(self) -> bool:
        """Check if metric requires sensitive features.

        Base implementation returns False. Override for fairness metrics.
        """
        return False


@MetricRegistry.register("noop_metric", priority=-100)
class NoOpMetric(BaseMetric):
    """NoOp metric for testing and fallback purposes.

    This metric provides placeholder values and can be used when
    no real metrics are available or for testing the metric system.
    """

    metric_type = "performance"

    def __init__(self):
        """Initialize NoOp metric."""
        super().__init__("noop_metric", "performance", "1.0.0")
        logger.info("NoOpMetric initialized")

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute placeholder metric values.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sensitive_features: Ignored for NoOp metric

        Returns:
            Dictionary with placeholder metrics

        """
        validation = self.validate_inputs(y_true, y_pred, sensitive_features)

        logger.debug(f"NoOpMetric computing placeholder for {validation['n_samples']} samples")

        return {
            "noop_metric": 0.5,  # Placeholder value
            "n_samples": float(validation["n_samples"]),
            "status": "noop",
        }

    def get_metric_names(self) -> list[str]:
        """Return metric names."""
        return ["noop_metric", "n_samples", "status"]

    def requires_sensitive_features(self) -> bool:
        """NoOp metric doesn't require sensitive features."""
        return False


# Auto-register the NoOp metric
logger.debug("Base metrics registered")
