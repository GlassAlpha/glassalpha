"""Metrics runner for demographic shift analysis.

This module computes fairness, calibration, and performance metrics before and
after applying demographic shifts, then detects degradation for gating decisions.

Example:
    >>> from glassalpha.metrics.shift import run_shift_analysis
    >>> result = run_shift_analysis(
    ...     model=model,
    ...     X_test=X_test,
    ...     y_test=y_test,
    ...     y_proba=y_proba,
    ...     sensitive_features=sensitive_df,
    ...     attribute="gender",
    ...     shift=0.1,
    ...     threshold=0.05,
    ... )
    >>> print(result.gate_status)  # "PASS" | "WARNING" | "FAIL"
    >>> print(result.violations)   # List of degraded metrics

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .reweighting import ShiftSpecification, compute_shifted_weights

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShiftAnalysisResult:
    """Results from demographic shift analysis.

    Attributes:
        shift_spec: Specification of the applied shift
        baseline_metrics: Metrics computed without shift (original distribution)
        shifted_metrics: Metrics computed with shift applied
        degradation: Absolute change in each metric (shifted - baseline)
        gate_status: Overall status ("PASS" | "WARNING" | "FAIL")
        threshold: Degradation threshold for failure (optional)
        violations: List of metrics exceeding degradation threshold

    Example:
        >>> result.gate_status
        'FAIL'
        >>> result.violations
        ['demographic_parity degraded by 0.08 (threshold: 0.05)']

    """

    shift_spec: ShiftSpecification
    baseline_metrics: dict[str, Any]
    shifted_metrics: dict[str, Any]
    degradation: dict[str, float]
    gate_status: str
    threshold: float | None = None
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary for JSON serialization."""
        return {
            "shift_specification": self.shift_spec.to_dict(),
            "baseline_metrics": self.baseline_metrics,
            "shifted_metrics": self.shifted_metrics,
            "degradation": self.degradation,
            "gate_status": self.gate_status,
            "threshold": self.threshold,
            "violations": self.violations,
        }


def _compute_fairness_metrics_weighted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: pd.Series,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute basic fairness metrics with optional weights.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive: Binary sensitive attribute
        weights: Optional sample weights

    Returns:
        Dictionary with demographic_parity, equal_opportunity, equalized_odds

    """
    if weights is None:
        weights = np.ones(len(y_true))

    # Ensure binary attribute
    unique_vals = sensitive.unique()
    if len(unique_vals) != 2:
        raise ValueError(f"Sensitive attribute must be binary, found: {unique_vals}")

    # Get groups
    group_0 = sensitive == unique_vals[0]
    group_1 = sensitive == unique_vals[1]

    # Weighted selection rates
    selection_rate_0 = np.average(y_pred[group_0], weights=weights[group_0])
    selection_rate_1 = np.average(y_pred[group_1], weights=weights[group_1])

    # Demographic parity: |selection_rate_1 - selection_rate_0|
    demographic_parity = abs(selection_rate_1 - selection_rate_0)

    # True positive rates (for equal opportunity)
    positives_0 = y_true[group_0] == 1
    positives_1 = y_true[group_1] == 1

    if positives_0.sum() > 0:
        tpr_0 = np.average(
            y_pred[group_0][positives_0],
            weights=weights[group_0][positives_0],
        )
    else:
        tpr_0 = 0.0

    if positives_1.sum() > 0:
        tpr_1 = np.average(
            y_pred[group_1][positives_1],
            weights=weights[group_1][positives_1],
        )
    else:
        tpr_1 = 0.0

    # Equal opportunity: |TPR_1 - TPR_0|
    equal_opportunity = abs(tpr_1 - tpr_0)

    # False positive rates (for equalized odds)
    negatives_0 = y_true[group_0] == 0
    negatives_1 = y_true[group_1] == 0

    if negatives_0.sum() > 0:
        fpr_0 = np.average(
            y_pred[group_0][negatives_0],
            weights=weights[group_0][negatives_0],
        )
    else:
        fpr_0 = 0.0

    if negatives_1.sum() > 0:
        fpr_1 = np.average(
            y_pred[group_1][negatives_1],
            weights=weights[group_1][negatives_1],
        )
    else:
        fpr_1 = 0.0

    # Equalized odds: max(|TPR_1 - TPR_0|, |FPR_1 - FPR_0|)
    equalized_odds = max(abs(tpr_1 - tpr_0), abs(fpr_1 - fpr_0))

    return {
        "demographic_parity": demographic_parity,
        "equal_opportunity": equal_opportunity,
        "equalized_odds": equalized_odds,
    }


def _compute_calibration_metrics_weighted(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    weights: np.ndarray | None = None,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute calibration metrics with optional weights.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        weights: Optional sample weights
        n_bins: Number of bins for ECE

    Returns:
        Dictionary with ece (Expected Calibration Error) and brier_score

    """
    if weights is None:
        weights = np.ones(len(y_true))

    # Brier score: weighted mean squared error between probabilities and labels
    brier_score = np.average((y_proba - y_true) ** 2, weights=weights)

    # Expected Calibration Error (ECE)
    # Bin predictions by probability
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    total_weight = weights.sum()

    for i in range(n_bins):
        bin_mask = bin_indices == i
        if not bin_mask.any():
            continue

        bin_weights = weights[bin_mask]
        bin_weight_sum = bin_weights.sum()

        # Weighted mean predicted probability in bin
        bin_pred = np.average(y_proba[bin_mask], weights=bin_weights)

        # Weighted mean observed outcome in bin
        bin_true = np.average(y_true[bin_mask], weights=bin_weights)

        # Contribution to ECE: (bin_weight / total_weight) * |pred - true|
        ece += (bin_weight_sum / total_weight) * abs(bin_pred - bin_true)

    return {
        "ece": ece,
        "brier_score": brier_score,
    }


def _compute_performance_metrics_weighted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute performance metrics with optional weights.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        weights: Optional sample weights

    Returns:
        Dictionary with accuracy, precision, recall, f1

    """
    if weights is None:
        weights = np.ones(len(y_true))

    # Weighted accuracy
    correct = (y_true == y_pred).astype(float)
    accuracy = np.average(correct, weights=weights)

    # Weighted precision and recall
    tp = np.sum(weights[(y_true == 1) & (y_pred == 1)])
    fp = np.sum(weights[(y_true == 0) & (y_pred == 1)])
    fn = np.sum(weights[(y_true == 1) & (y_pred == 0)])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    sensitive: pd.Series,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute all metrics (fairness, calibration, performance).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for calibration)
        sensitive: Binary sensitive attribute
        weights: Optional sample weights

    Returns:
        Dictionary with nested metrics: {fairness: {...}, calibration: {...}, performance: {...}}

    """
    metrics = {}

    # Fairness metrics
    try:
        metrics["fairness"] = _compute_fairness_metrics_weighted(
            y_true,
            y_pred,
            sensitive,
            weights,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to compute fairness metrics: {e}")
        metrics["fairness"] = {}

    # Calibration metrics (requires probabilities)
    if y_proba is not None:
        try:
            metrics["calibration"] = _compute_calibration_metrics_weighted(
                y_true,
                y_proba,
                weights,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to compute calibration metrics: {e}")
            metrics["calibration"] = {}
    else:
        metrics["calibration"] = {}

    # Performance metrics
    try:
        metrics["performance"] = _compute_performance_metrics_weighted(
            y_true,
            y_pred,
            weights,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to compute performance metrics: {e}")
        metrics["performance"] = {}

    return metrics


def _compute_degradation(
    baseline: dict[str, Any],
    shifted: dict[str, Any],
) -> dict[str, float]:
    """Compute metric degradation (shifted - baseline).

    Args:
        baseline: Baseline metrics (nested dict)
        shifted: Shifted metrics (nested dict)

    Returns:
        Flat dictionary of degradation values

    Example:
        >>> baseline = {"fairness": {"demographic_parity": 0.05}}
        >>> shifted = {"fairness": {"demographic_parity": 0.12}}
        >>> _compute_degradation(baseline, shifted)
        {'demographic_parity': 0.07}

    """
    degradation = {}

    for category, metrics in baseline.items():
        if category not in shifted:
            continue

        for metric_name, baseline_value in metrics.items():
            if metric_name not in shifted[category]:
                continue

            shifted_value = shifted[category][metric_name]

            # Compute absolute degradation
            # For fairness metrics: increase is bad (more disparity)
            # For calibration: increase is bad (worse calibration)
            # For performance: decrease is bad (worse performance)
            if category == "performance":
                # Performance metrics: decrease is degradation
                degradation[metric_name] = baseline_value - shifted_value
            else:
                # Fairness/calibration: increase is degradation
                degradation[metric_name] = shifted_value - baseline_value

    return degradation


def _assess_gate_status(
    degradation: dict[str, float],
    threshold: float | None,
) -> tuple[str, list[str]]:
    """Assess gate status based on degradation.

    Args:
        degradation: Dictionary of metric degradation values
        threshold: Degradation threshold for failure (None = no gating)

    Returns:
        Tuple of (gate_status, violations)
        - gate_status: "PASS" | "WARNING" | "FAIL"
        - violations: List of violation descriptions

    Logic:
        - If no threshold: status = "PASS" (informational only)
        - If any metric degrades > threshold: status = "FAIL"
        - If any metric degrades > threshold/2: status = "WARNING"
        - Otherwise: status = "PASS"

    """
    if threshold is None:
        return "PASS", []

    violations = []
    max_degradation = 0.0

    for metric_name, deg_value in degradation.items():
        if deg_value > threshold:
            violations.append(
                f"{metric_name} degraded by {deg_value:.3f} (threshold: {threshold:.3f})",
            )
        max_degradation = max(max_degradation, deg_value)

    # Determine status
    if violations:
        status = "FAIL"
    elif max_degradation > threshold / 2:
        status = "WARNING"
    else:
        status = "PASS"

    return status, violations


def run_shift_analysis(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive_features: pd.DataFrame,
    attribute: str,
    shift: float,
    *,
    y_proba: np.ndarray | pd.Series | None = None,
    threshold: float | None = None,
) -> ShiftAnalysisResult:
    """Run demographic shift analysis with before/after metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        sensitive_features: DataFrame with protected attributes
        attribute: Column name of attribute to shift
        shift: Shift value in percentage points (e.g., 0.1 = +10pp)
        y_proba: Predicted probabilities for calibration metrics (optional)
        threshold: Degradation threshold for gate failure (optional)

    Returns:
        ShiftAnalysisResult with baseline/shifted metrics and gate status

    Example:
        >>> result = run_shift_analysis(
        ...     y_true=y_test,
        ...     y_pred=predictions,
        ...     sensitive_features=sensitive_df,
        ...     attribute="gender",
        ...     shift=0.1,
        ...     y_proba=probabilities,
        ...     threshold=0.05,
        ... )
        >>> print(result.gate_status)
        'FAIL'
        >>> for violation in result.violations:
        ...     print(violation)

    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_proba is not None:
        y_proba = np.asarray(y_proba)

    # Validate attribute exists
    if attribute not in sensitive_features.columns:
        raise ValueError(
            f"Attribute '{attribute}' not found in sensitive_features. Available: {list(sensitive_features.columns)}",
        )

    sensitive = sensitive_features[attribute]

    # Compute shift weights
    weights, shift_spec = compute_shifted_weights(
        sensitive_features,
        attribute,
        shift,
    )

    logger.info(
        f"Computing shift analysis: {attribute} "
        f"{shift_spec.original_proportion:.3f} → {shift_spec.shifted_proportion:.3f}",
    )

    # Compute baseline metrics (no weights)
    baseline_metrics = _compute_all_metrics(
        y_true,
        y_pred,
        y_proba,
        sensitive,
        weights=None,
    )

    # Compute shifted metrics (with weights)
    shifted_metrics = _compute_all_metrics(
        y_true,
        y_pred,
        y_proba,
        sensitive,
        weights=weights,
    )

    # Compute degradation
    degradation = _compute_degradation(baseline_metrics, shifted_metrics)

    # Assess gate status
    gate_status, violations = _assess_gate_status(degradation, threshold)

    logger.info(
        f"Shift analysis complete: {gate_status} "
        f"({len(violations)} violations, max degradation: {max(degradation.values(), default=0.0):.3f})",
    )

    return ShiftAnalysisResult(
        shift_spec=shift_spec,
        baseline_metrics=baseline_metrics,
        shifted_metrics=shifted_metrics,
        degradation=degradation,
        gate_status=gate_status,
        threshold=threshold,
        violations=violations,
    )
