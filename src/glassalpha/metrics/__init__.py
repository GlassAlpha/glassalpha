"""Metrics computation with explicit dispatch for AI maintainability.

Provides simple functions for computing performance, fairness, and calibration metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    protected_attributes: dict[str, np.ndarray] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute all available metrics based on provided data.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for calibration/AUC)
        protected_attributes: Dict of protected attribute arrays (optional, for fairness)
        config: Configuration dict (optional)

    Returns:
        Dictionary with performance, fairness, and calibration metrics

    Examples:
        >>> metrics = compute_all_metrics(y_true, y_pred, y_proba, {"gender": gender})
        >>> metrics["performance"]["accuracy"]
        0.847

    """
    results: dict[str, Any] = {}

    # Performance metrics (always computed)
    from glassalpha.metrics.performance import compute_performance_metrics

    results["performance"] = compute_performance_metrics(y_true, y_pred, y_proba)

    # Fairness metrics (if protected attributes provided)
    if protected_attributes:
        from glassalpha.metrics.fairness.runner import compute_fairness_metrics

        results["fairness"] = compute_fairness_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            protected_attributes=protected_attributes,
            config=config or {},
        )

    # Calibration metrics (if probabilities provided)
    if y_proba is not None:
        from glassalpha.metrics.calibration.quality import assess_calibration_quality

        results["calibration"] = assess_calibration_quality(
            y_true=y_true,
            y_prob_pos=y_proba,
            n_bins=10,
            compute_confidence_intervals=False,
        )

    return results


__all__ = [
    "compute_all_metrics",
]
