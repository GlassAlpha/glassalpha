"""Fairness metrics runner for GlassAlpha.

This module provides utilities for running fairness metrics with proper
handling of numpy arrays and pandas DataFrames.

Includes E10: Statistical Confidence & Uncertainty support:
- Bootstrap confidence intervals for all fairness metrics
- Small sample size warnings (n<30 WARNING, n<10 ERROR)
- Statistical power calculations

Includes E5.1: Basic Intersectional Fairness support:
- Two-way intersectional analysis (e.g., gender*race)
- Cartesian product of protected attribute values
- Sample size warnings and bootstrap CIs for intersectional groups
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .confidence import (
    check_sample_size_adequacy,
    compute_bootstrap_ci,
    compute_statistical_power,
)
from .intersectional import compute_intersectional_fairness

logger = logging.getLogger(__name__)


def _as_series(v: Any, name: str) -> pd.Series:
    """Convert input to pandas Series for consistent handling."""
    if isinstance(v, pd.Series):
        return v
    return pd.Series(v, name=name)


def run_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray | pd.DataFrame,
    metrics: list[Any],
    compute_confidence_intervals: bool = True,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
    intersections: list[str] | None = None,
) -> dict[str, Any]:
    """Run fairness metrics with proper input handling.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        sensitive: Sensitive features (numpy array or DataFrame)
        metrics: List of metric classes to run
        compute_confidence_intervals: Whether to compute bootstrap CIs (default True)
        n_bootstrap: Number of bootstrap samples (default 1000)
        confidence_level: Confidence level for CIs (default 0.95)
        seed: Random seed for deterministic bootstrap
        intersections: List of intersection specs for E5.1 (e.g., ["gender*race"])

    Returns:
        Dictionary with results from all metrics, including CIs and warnings

    """
    # Convert inputs to consistent format
    y_true = _as_series(y_true, "y_true")
    y_pred = _as_series(y_pred, "y_pred")

    # Handle sensitive features
    if isinstance(sensitive, pd.DataFrame):
        sensitive_df = sensitive
    else:
        # Single sensitive feature as numpy array
        sensitive_df = pd.DataFrame({"sensitive": np.asarray(sensitive)})

    # Validate inputs
    if len(y_true) != len(y_pred):
        msg = f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})"
        raise ValueError(msg)

    if len(y_true) != len(sensitive_df):
        msg = f"Length mismatch: y_true ({len(y_true)}) vs sensitive ({len(sensitive_df)})"
        raise ValueError(msg)

    results = {}

    # Compute sample size warnings for each sensitive attribute and group
    sample_size_warnings = _compute_sample_size_warnings(sensitive_df)
    results["sample_size_warnings"] = sample_size_warnings

    # Compute statistical power for detecting 10% disparity
    power_analysis = _compute_power_analysis(sensitive_df, effect_size=0.10)
    results["statistical_power"] = power_analysis

    for metric_class in metrics:
        try:
            metric = metric_class()
            metric_name = metric_class.__name__.lower().replace("metric", "")

            # Run metric with appropriate sensitive features format
            if isinstance(sensitive, pd.DataFrame):
                # Multiple sensitive features
                metric_result = metric.compute(y_true.values, y_pred.values, sensitive)
            else:
                # Single sensitive feature
                metric_result = metric.compute(y_true.values, y_pred.values, np.asarray(sensitive))

            results[metric_name] = metric_result
            logger.debug(f"Computed {metric_name}: {list(metric_result.keys())}")

            # Compute confidence intervals if requested
            if compute_confidence_intervals:
                ci_results = _compute_metric_confidence_intervals(
                    y_true.values,
                    y_pred.values,
                    sensitive_df,
                    metric_name,
                    metric_result,
                    n_bootstrap=n_bootstrap,
                    confidence_level=confidence_level,
                    seed=seed,
                )
                results[f"{metric_name}_ci"] = ci_results

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to compute {metric_class.__name__}: {e}")
            results[metric_class.__name__.lower().replace("metric", "")] = {"error": str(e)}

    # Compute intersectional fairness metrics if specified (E5.1)
    if intersections and len(intersections) > 0:
        intersectional_results = {}

        for intersection_spec in intersections:
            try:
                logger.debug(f"Computing intersectional metrics for: {intersection_spec}")

                intersectional_metrics = compute_intersectional_fairness(
                    y_true.values,
                    y_pred.values,
                    sensitive_df,
                    intersection_spec=intersection_spec,
                    compute_confidence_intervals=compute_confidence_intervals,
                    n_bootstrap=n_bootstrap,
                    confidence_level=confidence_level,
                    seed=seed,
                )

                # Store under intersection spec key
                intersectional_results[intersection_spec] = intersectional_metrics

            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to compute intersectional metrics for {intersection_spec}: {e}")
                intersectional_results[intersection_spec] = {"error": str(e)}

        if intersectional_results:
            results["intersectional"] = intersectional_results

    return results


def _compute_sample_size_warnings(sensitive_df: pd.DataFrame) -> dict[str, Any]:
    """Compute sample size warnings for each group."""
    warnings = {}

    for attr_name in sensitive_df.columns:
        attr_values = sensitive_df[attr_name].values
        unique_groups = np.unique(attr_values)

        for group in unique_groups:
            group_mask = attr_values == group
            n_samples = np.sum(group_mask)

            group_key = f"{attr_name}_{group}"
            warning_result = check_sample_size_adequacy(
                n=int(n_samples),
                group_name=group_key,
            )
            warnings[group_key] = warning_result

    return warnings


def _compute_power_analysis(
    sensitive_df: pd.DataFrame,
    effect_size: float = 0.10,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute statistical power for each group."""
    power_results = {}

    for attr_name in sensitive_df.columns:
        attr_values = sensitive_df[attr_name].values
        unique_groups = np.unique(attr_values)

        for group in unique_groups:
            group_mask = attr_values == group
            n_samples = int(np.sum(group_mask))

            power = compute_statistical_power(
                n_samples=n_samples,
                effect_size=effect_size,
                alpha=alpha,
            )

            group_key = f"{attr_name}_{group}"
            power_results[group_key] = {
                "n_samples": n_samples,
                "power": float(power),
                "effect_size": effect_size,
                "alpha": alpha,
                "interpretation": _interpret_power(power),
            }

    return power_results


def _interpret_power(power: float) -> str:
    """Interpret statistical power value."""
    if power < 0.5:
        return "Very low power - insufficient to detect disparity reliably"
    if power < 0.8:
        return "Low power - may miss true disparities"
    if power < 0.9:
        return "Adequate power - reasonable detection capability"
    return "High power - strong detection capability"


def _compute_metric_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_df: pd.DataFrame,
    metric_name: str,  # noqa: ARG001
    metric_result: dict[str, Any],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute bootstrap CIs for fairness metric results.

    This extracts group-level metrics (TPR, FPR, selection rates, etc.)
    and computes bootstrap CIs for each.
    """
    ci_results = {}

    # Identify which keys in metric_result are group-level metrics
    # Pattern: "{attr_name}_{group}_{metric_type}" or "{attr_name}_{group}_rate"
    for key, value in metric_result.items():
        # Skip non-numeric values and summary statistics
        if not isinstance(value, (int, float, np.number)):
            continue
        if key in ["n_samples", "tolerance", "n_sensitive_attributes"]:
            continue
        if key.endswith("_is_fair"):
            continue

        # Extract attribute name and metric type from key
        # E.g., "gender_0_tpr" -> compute CI for TPR of gender group 0
        parts = key.split("_")
        if len(parts) < 2:
            continue

        # Try to parse as {attr}_{group}_{metric} format
        try:
            # Determine metric function based on key suffix
            if "_tpr" in key:
                metric_fn = _make_tpr_metric(key, sensitive_df)
            elif "_fpr" in key:
                metric_fn = _make_fpr_metric(key, sensitive_df)
            elif "_ppv" in key or "_precision" in key:
                metric_fn = _make_ppv_metric(key, sensitive_df)
            elif "parity" in key.lower() or "rate" in key:
                # Skip derived metrics like parity_difference, parity_ratio
                if "difference" in key or "ratio" in key or key.endswith("_is_fair"):
                    continue
                # Selection rate for demographic parity or group rates
                metric_fn = _make_selection_rate_metric(key, sensitive_df)
            # Try to match pattern: {attr}_{group} (e.g., gender_0, gender_1)
            # This is a selection rate for the group
            elif len(parts) == 2 and parts[1].isdigit():
                metric_fn = _make_selection_rate_metric(key, sensitive_df)
            else:
                # Generic metric - skip CI computation
                continue

            # Compute bootstrap CI
            ci = compute_bootstrap_ci(
                y_true,
                y_pred,
                sensitive_df.values[:, 0],  # Use first sensitive attribute for now
                metric_fn=metric_fn,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                seed=seed,
            )

            ci_results[key] = ci.to_dict()

        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not compute CI for {key}: {e}")
            continue

    return ci_results


def _make_tpr_metric(key: str, sensitive_df: pd.DataFrame) -> Callable:
    """Create TPR metric function for specific group."""
    # Parse key like "gender_0_tpr"
    parts = key.replace("_tpr", "").split("_")
    attr_name = parts[0]
    group_value = "_".join(parts[1:])  # Handle multi-part group names

    def tpr_metric(yt, yp, sens):  # noqa: ARG001
        # Filter to specific group
        attr_values = sensitive_df[attr_name].values
        mask = attr_values.astype(str) == str(group_value)

        if not np.any(mask):
            return np.nan

        yt_group = yt[mask]
        yp_group = yp[mask]

        # Calculate TPR
        tp = np.sum((yt_group == 1) & (yp_group == 1))
        fn = np.sum((yt_group == 1) & (yp_group == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return tpr_metric


def _make_fpr_metric(key: str, sensitive_df: pd.DataFrame) -> Callable:
    """Create FPR metric function for specific group."""
    parts = key.replace("_fpr", "").split("_")
    attr_name = parts[0]
    group_value = "_".join(parts[1:])

    def fpr_metric(yt, yp, sens):  # noqa: ARG001
        attr_values = sensitive_df[attr_name].values
        mask = attr_values.astype(str) == str(group_value)

        if not np.any(mask):
            return np.nan

        yt_group = yt[mask]
        yp_group = yp[mask]

        # Calculate FPR
        fp = np.sum((yt_group == 0) & (yp_group == 1))
        tn = np.sum((yt_group == 0) & (yp_group == 0))
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return fpr_metric


def _make_ppv_metric(key: str, sensitive_df: pd.DataFrame) -> Callable:
    """Create PPV/precision metric function for specific group."""
    parts = key.replace("_ppv", "").replace("_precision", "").split("_")
    attr_name = parts[0]
    group_value = "_".join(parts[1:])

    def ppv_metric(yt, yp, sens):  # noqa: ARG001
        attr_values = sensitive_df[attr_name].values
        mask = attr_values.astype(str) == str(group_value)

        if not np.any(mask):
            return np.nan

        yt_group = yt[mask]
        yp_group = yp[mask]

        # Calculate PPV
        tp = np.sum((yt_group == 1) & (yp_group == 1))
        fp = np.sum((yt_group == 0) & (yp_group == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return ppv_metric


def _make_selection_rate_metric(key: str, sensitive_df: pd.DataFrame) -> Callable:
    """Create selection rate metric function for specific group."""
    # Parse key - it might be like "gender_0" or "gender_0_rate"
    parts = key.replace("_rate", "").split("_")
    attr_name = parts[0]
    group_value = "_".join(parts[1:])

    def selection_rate_metric(yt, yp, sens):  # noqa: ARG001
        attr_values = sensitive_df[attr_name].values
        mask = attr_values.astype(str) == str(group_value)

        if not np.any(mask):
            return np.nan

        yp_group = yp[mask]
        return np.mean(yp_group)

    return selection_rate_metric
