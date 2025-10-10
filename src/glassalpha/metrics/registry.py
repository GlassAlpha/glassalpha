"""Metric Registry: Metadata for all metrics.

Provides metadata for each metric including:
- Display name
- Description
- Higher is better (for comparisons)
- Compute requirements (e.g., requires probabilities)
- Default tolerance for equality checks
- Aggregation method (for multi-group metrics)
- Fairness definition reference
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from glassalpha.core.decor_registry import DecoratorFriendlyRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricSpec:
    """Specification for a metric.

    Attributes:
        key: Metric key (e.g., "accuracy", "demographic_parity_diff")
        display_name: Human-readable name
        description: One-line description
        higher_is_better: True if higher values are better
        requires_proba: True if metric requires predicted probabilities
        default_rtol: Default relative tolerance for equality checks
        default_atol: Default absolute tolerance for equality checks
        aggregation: How to aggregate across groups (for fairness metrics)
        fairness_definition: Reference to fairness definition (if applicable)

    """

    key: str
    display_name: str
    description: str
    higher_is_better: bool
    requires_proba: bool = False
    default_rtol: float = 1e-5
    default_atol: float = 1e-8
    aggregation: Literal["max_diff", "mean", "none"] = "none"
    fairness_definition: str | None = None


# Performance Metrics Registry

PERFORMANCE_METRICS = {
    "accuracy": MetricSpec(
        key="accuracy",
        display_name="Accuracy",
        description="Fraction of correct predictions",
        higher_is_better=True,
        default_rtol=1e-5,
        default_atol=1e-8,
    ),
    "precision": MetricSpec(
        key="precision",
        display_name="Precision",
        description="Fraction of positive predictions that are correct",
        higher_is_better=True,
        default_rtol=1e-5,
        default_atol=1e-8,
    ),
    "recall": MetricSpec(
        key="recall",
        display_name="Recall",
        description="Fraction of actual positives correctly predicted",
        higher_is_better=True,
        default_rtol=1e-5,
        default_atol=1e-8,
    ),
    "f1": MetricSpec(
        key="f1",
        display_name="F1 Score",
        description="Harmonic mean of precision and recall",
        higher_is_better=True,
        default_rtol=1e-5,
        default_atol=1e-8,
    ),
    "roc_auc": MetricSpec(
        key="roc_auc",
        display_name="ROC AUC",
        description="Area under ROC curve (requires probabilities)",
        higher_is_better=True,
        requires_proba=True,
        default_rtol=1e-5,
        default_atol=1e-8,
    ),
    "pr_auc": MetricSpec(
        key="pr_auc",
        display_name="PR AUC",
        description="Area under precision-recall curve (requires probabilities)",
        higher_is_better=True,
        requires_proba=True,
        default_rtol=1e-5,
        default_atol=1e-8,
    ),
    "brier_score": MetricSpec(
        key="brier_score",
        display_name="Brier Score",
        description="Mean squared error of probability predictions",
        higher_is_better=False,
        requires_proba=True,
        default_rtol=1e-5,
        default_atol=1e-8,
    ),
    "log_loss": MetricSpec(
        key="log_loss",
        display_name="Log Loss",
        description="Negative log-likelihood of probability predictions",
        higher_is_better=False,
        requires_proba=True,
        default_rtol=1e-5,
        default_atol=1e-8,
    ),
}

# Fairness Metrics Registry

FAIRNESS_METRICS = {
    "demographic_parity_diff": MetricSpec(
        key="demographic_parity_diff",
        display_name="Demographic Parity Difference",
        description="Max difference in positive prediction rate across groups",
        higher_is_better=False,
        default_rtol=1e-5,
        default_atol=1e-8,
        aggregation="max_diff",
        fairness_definition="Demographic parity: P(Ŷ=1|A=a) equal across groups",
    ),
    "equalized_odds_max_diff": MetricSpec(
        key="equalized_odds_max_diff",
        display_name="Equalized Odds Difference",
        description="Max difference in TPR or FPR across groups",
        higher_is_better=False,
        default_rtol=1e-5,
        default_atol=1e-8,
        aggregation="max_diff",
        fairness_definition="Equalized odds: TPR and FPR equal across groups",
    ),
    "equal_opportunity_diff": MetricSpec(
        key="equal_opportunity_diff",
        display_name="Equal Opportunity Difference",
        description="Max difference in TPR (recall) across groups",
        higher_is_better=False,
        default_rtol=1e-5,
        default_atol=1e-8,
        aggregation="max_diff",
        fairness_definition="Equal opportunity: TPR equal across groups",
    ),
}

# Calibration Metrics Registry

CALIBRATION_METRICS = {
    "ece": MetricSpec(
        key="ece",
        display_name="Expected Calibration Error",
        description="Expected difference between predicted probabilities and actual rates",
        higher_is_better=False,
        requires_proba=True,
        default_rtol=1e-4,  # Calibration metrics: looser tolerance
        default_atol=1e-6,
        aggregation="mean",
    ),
    "mce": MetricSpec(
        key="mce",
        display_name="Maximum Calibration Error",
        description="Maximum difference between predicted probabilities and actual rates",
        higher_is_better=False,
        requires_proba=True,
        default_rtol=1e-4,
        default_atol=1e-6,
        aggregation="max_diff",
    ),
}

# Stability Metrics Registry

STABILITY_METRICS = {
    "monotonicity_violations": MetricSpec(
        key="monotonicity_violations",
        display_name="Monotonicity Violations",
        description="Number of monotonicity constraint violations",
        higher_is_better=False,
        default_rtol=0.0,  # Count metrics: exact match
        default_atol=0.0,
        aggregation="none",
    ),
}

# Combined Registry

ALL_METRICS = {
    **PERFORMANCE_METRICS,
    **FAIRNESS_METRICS,
    **CALIBRATION_METRICS,
    **STABILITY_METRICS,
}


def get_metric_spec(key: str) -> MetricSpec | None:
    """Get metric specification by key.

    Args:
        key: Metric key (e.g., "accuracy")

    Returns:
        MetricSpec or None if not found

    """
    return ALL_METRICS.get(key)


def requires_probabilities(key: str) -> bool:
    """Check if metric requires predicted probabilities.

    Args:
        key: Metric key

    Returns:
        True if metric requires y_proba

    """
    spec = get_metric_spec(key)
    return spec.requires_proba if spec else False


def get_default_tolerance(key: str) -> tuple[float, float]:
    """Get default tolerance for metric.

    Args:
        key: Metric key

    Returns:
        Tuple of (rtol, atol)

    """
    spec = get_metric_spec(key)
    if spec:
        return (spec.default_rtol, spec.default_atol)
    # Fallback: standard tolerance
    return (1e-5, 1e-8)


class MetricRegistryClass(DecoratorFriendlyRegistry):
    """Registry for metric classes with metadata support.

    Follows the same pattern as ExplainerRegistry and ModelRegistry.
    Supports decorator-based registration and priority-based selection.
    """

    def get_metric_type(self, name: str) -> str | None:
        """Get metric type (performance, fairness, drift, calibration).

        Args:
            name: Metric name

        Returns:
            Metric type string or None if not found

        """
        meta = self._meta.get(name, {})
        return meta.get("metric_type")


# Create the real metric registry instance
MetricRegistry = MetricRegistryClass(group="glassalpha.metrics")
MetricRegistry.discover()  # Safe discovery without heavy imports
