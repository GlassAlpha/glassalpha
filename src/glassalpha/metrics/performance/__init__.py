"""Performance metrics for model evaluation.

This module contains metrics for evaluating model performance such as
accuracy, precision, recall, F1-score, and AUC-ROC.
"""

from typing import Any

import numpy as np


def compute_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute standard performance metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for AUC)

    Returns:
        Dictionary with accuracy, precision, recall, f1, and optionally auc_roc

    """
    from .classification import (
        AccuracyMetric,
        AUCROCMetric,
        F1Metric,
        PrecisionMetric,
        RecallMetric,
    )

    results = {}

    # Compute core metrics
    results.update(AccuracyMetric().compute(y_true=y_true, y_pred=y_pred))
    results.update(PrecisionMetric().compute(y_true=y_true, y_pred=y_pred))
    results.update(RecallMetric().compute(y_true=y_true, y_pred=y_pred))
    results.update(F1Metric().compute(y_true=y_true, y_pred=y_pred))

    # Compute AUC if probabilities provided
    if y_proba is not None:
        try:
            results.update(AUCROCMetric().compute(y_true=y_true, y_pred=y_pred, y_proba=y_proba))
        except Exception:
            # Skip AUC if it fails (e.g., single class)
            pass

    return results


__all__ = [
    "compute_performance_metrics",
]
