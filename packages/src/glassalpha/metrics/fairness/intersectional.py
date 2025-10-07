"""Intersectional fairness metrics for GlassAlpha.

This module implements E5.1: Basic Intersectional Fairness.

Key features:
1. Two-way intersectional analysis (e.g., gender*race, age*income)
2. Cartesian product of protected attribute values
3. Reuses E10 bootstrap CI infrastructure
4. Sample size warnings (n<30 WARNING, n<10 ERROR)
5. Deterministic computation (seeded for reproducibility)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from .confidence import (
    check_sample_size_adequacy,
    compute_bootstrap_ci,
    compute_statistical_power,
)

logger = logging.getLogger(__name__)


def parse_intersection_spec(spec: str) -> tuple[str, str]:
    """Parse intersection specification into attribute names.

    Args:
        spec: Intersection specification (e.g., "gender*race")

    Returns:
        Tuple of (attr1, attr2)

    Raises:
        ValueError: If spec is not in valid format
    """
    parts = spec.split('*')
    if len(parts) != 2:
        raise ValueError(
            f"Invalid intersection format '{spec}'. "
            "Must be 'attr1*attr2' for two-way intersections."
        )

    attr1, attr2 = [p.strip() for p in parts]

    if not attr1 or not attr2:
        raise ValueError(
            f"Invalid intersection '{spec}'. "
            "Both attributes must be non-empty."
        )

    return attr1, attr2


def create_intersectional_groups(
    X: pd.DataFrame,
    attr1: str,
    attr2: str,
) -> pd.Series:
    """Create intersectional groups from two attributes.

    Args:
        X: DataFrame with protected attributes
        attr1: First attribute name
        attr2: Second attribute name

    Returns:
        Series with intersectional group labels (e.g., "gender_male*race_white")

    Raises:
        ValueError: If attributes not found in DataFrame
    """
    if attr1 not in X.columns:
        raise ValueError(f"Attribute '{attr1}' not found in data")
    if attr2 not in X.columns:
        raise ValueError(f"Attribute '{attr2}' not found in data")

    # Create intersectional group labels
    # Format: "attr1_value1*attr2_value2"
    groups = X[attr1].astype(str) + '*' + X[attr2].astype(str)
    groups.name = f"{attr1}*{attr2}"

    return groups


def compute_intersectional_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: pd.DataFrame,
    intersection_spec: str,
    compute_confidence_intervals: bool = True,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute fairness metrics for intersectional groups.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        X: DataFrame with protected attributes
        intersection_spec: Intersection specification (e.g., "gender*race")
        compute_confidence_intervals: Whether to compute bootstrap CIs (default True)
        n_bootstrap: Number of bootstrap samples (default 1000)
        confidence_level: Confidence level for CIs (default 0.95)
        seed: Random seed for deterministic bootstrap

    Returns:
        Dictionary with intersectional fairness metrics including:
        - Group-level metrics (TPR, FPR, precision, recall)
        - Sample size warnings
        - Statistical power analysis
        - Bootstrap confidence intervals (if requested)
    """
    # Parse intersection spec
    attr1, attr2 = parse_intersection_spec(intersection_spec)

    # Create intersectional groups
    intersectional_groups = create_intersectional_groups(X, attr1, attr2)

    # Get unique groups
    unique_groups = np.unique(intersectional_groups)

    # Initialize results
    results = {
        "intersection_spec": intersection_spec,
        "attr1": attr1,
        "attr2": attr2,
        "n_groups": len(unique_groups),
    }

    # Compute metrics for each intersectional group
    group_metrics = {}
    sample_size_warnings = {}
    power_analysis = {}

    for group in unique_groups:
        group_mask = intersectional_groups == group
        n_samples = int(np.sum(group_mask))

        # Sample size adequacy check
        group_key = f"{attr1}*{attr2}_{group}"
        warning_result = check_sample_size_adequacy(
            n=n_samples,
            group_name=group_key
        )
        sample_size_warnings[group_key] = warning_result

        # Statistical power for detecting 10% disparity
        power = compute_statistical_power(
            n_samples=n_samples,
            effect_size=0.10,
            alpha=0.05,
        )
        power_analysis[group_key] = {
            "n_samples": n_samples,
            "power": float(power),
            "effect_size": 0.10,
            "alpha": 0.05,
            "interpretation": _interpret_power(power),
        }

        # Compute group metrics
        y_true_group = y_true[group_mask]
        y_pred_group = y_pred[group_mask]

        if len(y_true_group) > 0:
            # Confusion matrix
            tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
            fp = np.sum((y_true_group == 0) & (y_pred_group == 1))
            tn = np.sum((y_true_group == 0) & (y_pred_group == 0))
            fn = np.sum((y_true_group == 1) & (y_pred_group == 0))

            # Metrics
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tpr  # TPR = recall
            selection_rate = np.mean(y_pred_group)

            group_metrics[f"{group_key}_tpr"] = float(tpr)
            group_metrics[f"{group_key}_fpr"] = float(fpr)
            group_metrics[f"{group_key}_precision"] = float(precision)
            group_metrics[f"{group_key}_recall"] = float(recall)
            group_metrics[f"{group_key}_selection_rate"] = float(selection_rate)
            group_metrics[f"{group_key}_n_samples"] = int(n_samples)
        else:
            # Empty group - set to NaN
            group_metrics[f"{group_key}_tpr"] = np.nan
            group_metrics[f"{group_key}_fpr"] = np.nan
            group_metrics[f"{group_key}_precision"] = np.nan
            group_metrics[f"{group_key}_recall"] = np.nan
            group_metrics[f"{group_key}_selection_rate"] = np.nan
            group_metrics[f"{group_key}_n_samples"] = 0

    # Add group metrics to results
    results.update(group_metrics)
    results["sample_size_warnings"] = sample_size_warnings
    results["statistical_power"] = power_analysis

    # Compute disparity metrics across intersectional groups
    disparity_metrics = _compute_disparity_metrics(group_metrics, intersection_spec)
    results.update(disparity_metrics)

    # Compute confidence intervals if requested
    if compute_confidence_intervals:
        ci_results = _compute_intersectional_cis(
            y_true,
            y_pred,
            intersectional_groups,
            unique_groups,
            intersection_spec,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed,
        )
        results["confidence_intervals"] = ci_results

    return results


def _interpret_power(power: float) -> str:
    """Interpret statistical power value."""
    if power < 0.5:
        return "Very low power - insufficient to detect disparity reliably"
    elif power < 0.8:
        return "Low power - may miss true disparities"
    elif power < 0.9:
        return "Adequate power - reasonable detection capability"
    else:
        return "High power - strong detection capability"


def _compute_disparity_metrics(
    group_metrics: dict[str, float],
    intersection_spec: str,
) -> dict[str, float]:
    """Compute disparity metrics across intersectional groups.

    Calculates max-min differences and ratios for each metric type.
    """
    results = {}

    # Extract metric types
    metric_types = ["tpr", "fpr", "precision", "recall", "selection_rate"]

    for metric_type in metric_types:
        # Get all values for this metric type
        values = []
        for key, value in group_metrics.items():
            if key.endswith(f"_{metric_type}") and not np.isnan(value):
                values.append(value)

        if len(values) >= 2:
            max_val = max(values)
            min_val = min(values)

            # Disparity difference
            results[f"{intersection_spec}_{metric_type}_disparity_difference"] = float(max_val - min_val)

            # Disparity ratio
            ratio = min_val / max_val if max_val > 0 else 1.0
            results[f"{intersection_spec}_{metric_type}_disparity_ratio"] = float(ratio)

    return results


def _compute_intersectional_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    intersectional_groups: pd.Series,
    unique_groups: np.ndarray,
    intersection_spec: str,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute bootstrap confidence intervals for intersectional metrics.

    Reuses E10 bootstrap infrastructure.
    """
    ci_results = {}

    # Parse spec for naming
    attr1, attr2 = parse_intersection_spec(intersection_spec)

    for group in unique_groups:
        group_key = f"{attr1}*{attr2}_{group}"

        # Create metric functions for this specific group
        def make_metric_fn(target_group: str, metric_type: str):
            """Factory for creating group-specific metric functions."""
            def metric_fn(yt, yp, sens):
                # IMPORTANT: Use 'sens' parameter (resampled intersectional groups)
                # not the outer scope 'intersectional_groups' variable
                # Filter to specific intersectional group
                mask = (sens == target_group)

                if not np.any(mask):
                    return np.nan

                yt_group = yt[mask]
                yp_group = yp[mask]

                if len(yt_group) == 0:
                    return np.nan

                # Calculate metric
                if metric_type == "tpr":
                    tp = np.sum((yt_group == 1) & (yp_group == 1))
                    fn = np.sum((yt_group == 1) & (yp_group == 0))
                    return tp / (tp + fn) if (tp + fn) > 0 else 0.0
                elif metric_type == "fpr":
                    fp = np.sum((yt_group == 0) & (yp_group == 1))
                    tn = np.sum((yt_group == 0) & (yp_group == 0))
                    return fp / (fp + tn) if (fp + tn) > 0 else 0.0
                elif metric_type == "precision":
                    tp = np.sum((yt_group == 1) & (yp_group == 1))
                    fp = np.sum((yt_group == 0) & (yp_group == 1))
                    return tp / (tp + fp) if (tp + fp) > 0 else 0.0
                elif metric_type == "recall":
                    tp = np.sum((yt_group == 1) & (yp_group == 1))
                    fn = np.sum((yt_group == 1) & (yp_group == 0))
                    return tp / (tp + fn) if (tp + fn) > 0 else 0.0
                elif metric_type == "selection_rate":
                    return np.mean(yp_group)
                else:
                    return np.nan

            return metric_fn

        # Compute CIs for each metric type
        metric_types = ["tpr", "fpr", "precision", "recall", "selection_rate"]

        for metric_type in metric_types:
            metric_fn = make_metric_fn(group, metric_type)

            try:
                ci = compute_bootstrap_ci(
                    y_true,
                    y_pred,
                    intersectional_groups.values,  # Dummy sensitive array (not used in metric_fn)
                    metric_fn=metric_fn,
                    n_bootstrap=n_bootstrap,
                    confidence_level=confidence_level,
                    seed=seed,
                )

                ci_key = f"{group_key}_{metric_type}"
                ci_results[ci_key] = ci.to_dict()

            except Exception as e:  # noqa: BLE001
                logger.debug(f"Could not compute CI for {group_key}_{metric_type}: {e}")
                continue

    return ci_results
