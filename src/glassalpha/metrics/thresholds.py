"""Threshold selection policies for binary classification decisions.

This module provides deterministic threshold selection strategies that are
more principled than arbitrary 0.5 cutoffs. These policies are essential
for regulatory compliance where decision thresholds must be explainable
and defensible.
"""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

logger = logging.getLogger(__name__)


def pick_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    policy: str = "youden",
    **policy_kwargs: Any,
) -> dict[str, Any]:
    """Select optimal threshold using specified policy.

    This function implements various threshold selection strategies that provide
    principled, explainable alternatives to arbitrary 0.5 cutoffs. The chosen
    threshold and policy rationale are returned for audit trail purposes.

    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for positive class
        policy: Threshold selection policy ('youden', 'fixed', 'prevalence', 'cost_sensitive')
        **policy_kwargs: Policy-specific parameters

    Returns:
        Dictionary containing threshold, policy info, and performance metrics

    Raises:
        ValueError: If policy is not recognized or parameters are invalid

    Examples:
        >>> # Youden's J statistic (maximize sensitivity + specificity)
        >>> result = pick_threshold(y_true, y_proba, policy='youden')
        >>> threshold = result['threshold']

        >>> # Fixed threshold
        >>> result = pick_threshold(y_true, y_proba, policy='fixed', threshold=0.3)

        >>> # Match dataset prevalence
        >>> result = pick_threshold(y_true, y_proba, policy='prevalence')

        >>> # Cost-sensitive threshold
        >>> result = pick_threshold(y_true, y_proba, policy='cost_sensitive',
        ...                        cost_fp=1.0, cost_fn=5.0)

    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Validate inputs
    if len(y_true) != len(y_proba):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_proba={len(y_proba)}")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1 values")

    if not np.all((y_proba >= 0) & (y_proba <= 1)):
        raise ValueError("y_proba must contain probabilities in [0, 1]")

    logger.info(f"Selecting threshold using policy: {policy}")

    # Dispatch to policy-specific implementation
    if policy == "youden":
        return _youden_threshold(y_true, y_proba, **policy_kwargs)
    if policy == "fixed":
        return _fixed_threshold(y_true, y_proba, **policy_kwargs)
    if policy == "prevalence":
        return _prevalence_threshold(y_true, y_proba, **policy_kwargs)
    if policy == "cost_sensitive":
        return _cost_sensitive_threshold(y_true, y_proba, **policy_kwargs)
    raise ValueError(f"Unknown threshold policy: '{policy}'. Must be one of: youden, fixed, prevalence, cost_sensitive")


def _youden_threshold(y_true: np.ndarray, y_proba: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    """Select threshold using Youden's J statistic (maximize sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    # Youden's J = Sensitivity + Specificity - 1 = TPR - FPR
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Compute performance at optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    result = {
        "threshold": float(optimal_threshold),
        "policy": "youden",
        "policy_description": "Youden's J statistic: maximizes (Sensitivity + Specificity - 1)",
        "j_statistic": float(j_scores[optimal_idx]),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "rationale": f"Selected threshold {optimal_threshold:.3f} that maximizes Youden's J = {j_scores[optimal_idx]:.3f}",
    }

    logger.info(
        f"Youden threshold selected: {optimal_threshold:.3f} (J={j_scores[optimal_idx]:.3f}, Sens={sensitivity:.3f}, Spec={specificity:.3f})",
    )

    return result


def _fixed_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5, **kwargs: Any) -> dict[str, Any]:
    """Use a fixed threshold (explicit business rule or regulatory requirement)."""
    if not isinstance(threshold, (int, float)):
        raise ValueError(f"Fixed threshold must be a number, got: {type(threshold)}")

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Fixed threshold must be in [0, 1], got: {threshold}")

    # Compute performance at fixed threshold
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    result = {
        "threshold": float(threshold),
        "policy": "fixed",
        "policy_description": f"Fixed threshold: {threshold} (business rule or regulatory requirement)",
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "rationale": f"Using fixed threshold {threshold} as specified by business rule or regulatory requirement",
    }

    logger.info(
        f"Fixed threshold used: {threshold:.3f} (Sens={sensitivity:.3f}, Spec={specificity:.3f}, Prec={precision:.3f})",
    )

    return result


def _prevalence_threshold(y_true: np.ndarray, y_proba: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    """Select threshold to match the dataset prevalence (positive rate)."""
    prevalence = np.mean(y_true)

    # Find threshold that gives prediction rate closest to prevalence
    thresholds = np.linspace(0, 1, 1000)
    prediction_rates = [np.mean(y_proba >= t) for t in thresholds]

    # Find threshold with prediction rate closest to prevalence
    differences = np.abs(np.array(prediction_rates) - prevalence)
    optimal_idx = np.argmin(differences)
    optimal_threshold = thresholds[optimal_idx]

    # Compute performance at optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    actual_prediction_rate = np.mean(y_pred)

    result = {
        "threshold": float(optimal_threshold),
        "policy": "prevalence",
        "policy_description": "Prevalence matching: threshold selected to match dataset positive rate",
        "target_prevalence": float(prevalence),
        "actual_prediction_rate": float(actual_prediction_rate),
        "prevalence_difference": float(abs(actual_prediction_rate - prevalence)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "rationale": f"Selected threshold {optimal_threshold:.3f} to match dataset prevalence {prevalence:.3f} (achieved {actual_prediction_rate:.3f})",
    }

    logger.info(
        f"Prevalence threshold selected: {optimal_threshold:.3f} (target={prevalence:.3f}, achieved={actual_prediction_rate:.3f})",
    )

    return result


def _cost_sensitive_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Select threshold that minimizes expected cost given false positive and false negative costs."""
    if not isinstance(cost_fp, (int, float)) or cost_fp < 0:
        raise ValueError(f"cost_fp must be a non-negative number, got: {cost_fp}")

    if not isinstance(cost_fn, (int, float)) or cost_fn < 0:
        raise ValueError(f"cost_fn must be a non-negative number, got: {cost_fn}")

    if cost_fp == 0 and cost_fn == 0:
        raise ValueError("At least one cost must be positive")

    # Evaluate costs at different thresholds
    thresholds = np.linspace(0, 1, 1000)
    costs = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        total_cost = cost_fp * fp + cost_fn * fn
        costs.append(total_cost)

    # Find threshold with minimum cost
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    min_cost = costs[optimal_idx]

    # Compute performance at optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    result = {
        "threshold": float(optimal_threshold),
        "policy": "cost_sensitive",
        "policy_description": f"Cost-sensitive: minimizes expected cost (FP cost={cost_fp}, FN cost={cost_fn})",
        "cost_fp": float(cost_fp),
        "cost_fn": float(cost_fn),
        "total_cost": float(min_cost),
        "fp_cost_contribution": float(cost_fp * fp),
        "fn_cost_contribution": float(cost_fn * fn),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "rationale": f"Selected threshold {optimal_threshold:.3f} that minimizes total cost {min_cost:.1f} (FP cost: {cost_fp}, FN cost: {cost_fn})",
    }

    logger.info(
        f"Cost-sensitive threshold selected: {optimal_threshold:.3f} (cost={min_cost:.1f}, FP_cost={cost_fp:.1f}, FN_cost={cost_fn:.1f})",
    )

    return result


def validate_threshold_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize threshold configuration.

    Args:
        config: Raw threshold configuration dictionary

    Returns:
        Validated and normalized configuration

    Raises:
        ValueError: If configuration is invalid

    """
    if not config:
        return {"policy": "youden"}  # Default policy

    policy = config.get("policy", "youden").lower()
    valid_policies = {"youden", "fixed", "prevalence", "cost_sensitive"}

    if policy not in valid_policies:
        raise ValueError(f"Invalid threshold policy: '{policy}'. Must be one of: {valid_policies}")

    validated_config = {"policy": policy}

    # Policy-specific validation
    if policy == "fixed":
        threshold = config.get("threshold", 0.5)
        if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Fixed threshold must be a number in [0, 1], got: {threshold}")
        validated_config["threshold"] = float(threshold)

    elif policy == "cost_sensitive":
        cost_fp = config.get("cost_fp", 1.0)
        cost_fn = config.get("cost_fn", 1.0)

        if not isinstance(cost_fp, (int, float)) or cost_fp < 0:
            raise ValueError(f"cost_fp must be a non-negative number, got: {cost_fp}")
        if not isinstance(cost_fn, (int, float)) or cost_fn < 0:
            raise ValueError(f"cost_fn must be a non-negative number, got: {cost_fn}")
        if cost_fp == 0 and cost_fn == 0:
            raise ValueError("At least one cost must be positive")

        validated_config["cost_fp"] = float(cost_fp)
        validated_config["cost_fn"] = float(cost_fn)

    logger.debug(f"Validated threshold config: {validated_config}")
    return validated_config


def get_threshold_summary(threshold_result: dict[str, Any]) -> str:
    """Format threshold selection result into a human-readable summary.

    Args:
        threshold_result: Result from pick_threshold()

    Returns:
        Formatted summary string

    """
    policy = threshold_result["policy"]
    threshold = threshold_result["threshold"]
    cm = threshold_result["confusion_matrix"]

    lines = []
    lines.append(f"Threshold Selection: {policy.title()} Policy")
    lines.append(f"Selected Threshold: {threshold:.3f}")
    lines.append(f"Policy: {threshold_result['policy_description']}")
    lines.append("")

    # Performance metrics
    if "sensitivity" in threshold_result:
        lines.append(f"Sensitivity (Recall): {threshold_result['sensitivity']:.3f}")
    if "specificity" in threshold_result:
        lines.append(f"Specificity: {threshold_result['specificity']:.3f}")
    if "precision" in threshold_result:
        lines.append(f"Precision: {threshold_result['precision']:.3f}")

    # Policy-specific metrics
    if policy == "youden" and "j_statistic" in threshold_result:
        lines.append(f"Youden's J Statistic: {threshold_result['j_statistic']:.3f}")
    elif policy == "prevalence" and "prevalence_difference" in threshold_result:
        lines.append(f"Prevalence Match Error: {threshold_result['prevalence_difference']:.3f}")
    elif policy == "cost_sensitive" and "total_cost" in threshold_result:
        lines.append(f"Total Cost: {threshold_result['total_cost']:.1f}")

    lines.append("")
    lines.append("Confusion Matrix:")
    lines.append(f"  True Negatives:  {cm['tn']:,}")
    lines.append(f"  False Positives: {cm['fp']:,}")
    lines.append(f"  False Negatives: {cm['fn']:,}")
    lines.append(f"  True Positives:  {cm['tp']:,}")

    lines.append("")
    lines.append(f"Rationale: {threshold_result['rationale']}")

    return "\n".join(lines)


def recommend_threshold_policy(
    dataset_size: int,
    class_balance: float,
    use_case: str = "general",
) -> str:
    """Recommend threshold policy based on dataset characteristics and use case.

    Args:
        dataset_size: Number of samples in dataset
        class_balance: Ratio of positive to negative samples
        use_case: Application context ('medical', 'financial', 'general')

    Returns:
        Recommended threshold policy name

    """
    # For medical applications, sensitivity is often critical
    if use_case.lower() in {"medical", "healthcare", "clinical"}:
        if class_balance < 0.1:  # Rare disease/condition
            return "cost_sensitive"  # High cost for false negatives
        return "youden"  # Balance sensitivity and specificity

    # For financial applications, precision is often critical
    if use_case.lower() in {"financial", "fraud", "credit"}:
        if class_balance < 0.05:  # Very rare fraud
            return "cost_sensitive"  # High cost for false positives
        return "prevalence"  # Match expected fraud rate

    # For general applications
    if dataset_size < 1000:
        return "fixed"  # Simple and stable for small datasets
    if abs(class_balance - 0.5) > 0.3:  # Imbalanced
        return "prevalence"  # Account for class imbalance
    return "youden"  # Balanced dataset, optimize both metrics
