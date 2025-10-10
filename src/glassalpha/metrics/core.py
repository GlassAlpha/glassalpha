"""Core metrics computation with automatic problem type detection and safe defaults.

This module provides intelligent metrics computation that automatically detects
binary vs multiclass problems and applies appropriate averaging strategies.
Designed to eliminate common "average='binary' with multiclass" errors.
"""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def infer_problem_type(y_true: np.ndarray) -> str:
    """Infer whether this is a binary or multiclass classification problem.

    Args:
        y_true: True labels array

    Returns:
        "binary" or "multiclass"

    """
    y_true = np.asarray(y_true)
    n_classes = len(np.unique(y_true))
    return "multiclass" if n_classes > 2 else "binary"


def choose_averaging(problem_type: str, average: str | None = None) -> str | None:
    """Choose appropriate averaging strategy based on problem type.

    For binary problems, returns None (which gives the standard binary metrics).
    For multiclass problems, defaults to 'macro' for fairness-focused metrics.

    Args:
        problem_type: "binary" or "multiclass"
        average: Override averaging strategy, or None for auto-selection

    Returns:
        Averaging strategy to use with sklearn metrics

    """
    if problem_type == "binary":
        # For binary, None gives standard binary metrics
        return None if average in (None, "binary") else average

    # For multiclass, default to macro averaging for fairness
    return average or "macro"


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    average: str | None = None,
    pos_label: int = 1,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics with automatic problem detection.

    This function automatically detects binary vs multiclass problems and applies
    appropriate averaging strategies. It gracefully handles missing probabilities
    and provides detailed error information when metrics cannot be computed.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, shape (n_samples, n_classes))
        average: Override averaging strategy (None for auto-selection)
        pos_label: Positive class label for binary classification

    Returns:
        Dictionary containing computed metrics and metadata

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Detect problem type and choose averaging
    problem_type = infer_problem_type(y_true)
    avg_strategy = choose_averaging(problem_type, average)

    logger.info(f"Computing {problem_type} classification metrics with averaging={avg_strategy}")

    # Initialize results
    results: dict[str, Any] = {
        "problem_type": problem_type,
        "n_classes": len(np.unique(y_true)),
        "averaging_strategy": avg_strategy,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "roc_auc": None,
        "log_loss": None,
        "confusion_matrix": None,
        "errors": [],
        "warnings": [],
    }

    try:
        # Accuracy (always computable)
        results["accuracy"] = float(accuracy_score(y_true, y_pred))

        # Confusion matrix (always computable)
        results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        # Precision, Recall, F1 with appropriate averaging
        if problem_type == "binary":
            try:
                results["precision"] = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
                results["recall"] = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
                results["f1_score"] = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
            except Exception as e:
                results["errors"].append(f"Binary metrics error: {e}")
        else:
            # Multiclass with averaging
            try:
                results["precision"] = float(precision_score(y_true, y_pred, average=avg_strategy, zero_division=0))
                results["recall"] = float(recall_score(y_true, y_pred, average=avg_strategy, zero_division=0))
                results["f1_score"] = float(f1_score(y_true, y_pred, average=avg_strategy, zero_division=0))
            except Exception as e:
                results["errors"].append(f"Multiclass metrics error: {e}")

        # Probability-based metrics (ROC-AUC, Log Loss)
        if y_proba is not None:
            try:
                if problem_type == "binary":
                    # For binary, use probabilities of positive class
                    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                        proba_pos = y_proba[:, 1]
                    elif y_proba.ndim == 1:
                        proba_pos = y_proba
                    else:
                        raise ValueError(f"Invalid probability shape for binary: {y_proba.shape}")

                    results["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
                    results["log_loss"] = float(log_loss(y_true, y_proba))

                else:
                    # Multiclass with one-vs-rest
                    if y_proba.ndim != 2 or y_proba.shape[1] != results["n_classes"]:
                        raise ValueError(f"Invalid probability shape for multiclass: {y_proba.shape}")

                    results["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
                    results["log_loss"] = float(log_loss(y_true, y_proba))

            except Exception as e:
                results["errors"].append(f"Probability-based metrics error: {e}")
                results["warnings"].append("ROC-AUC and Log Loss skipped due to probability issues")
        else:
            results["warnings"].append("ROC-AUC and Log Loss skipped: probabilities not provided")

    except Exception as e:
        results["errors"].append(f"Unexpected error in metrics computation: {e}")
        logger.error(f"Metrics computation failed: {e}")

    # Log summary
    computed_metrics = [
        k
        for k, v in results.items()
        if k not in ["problem_type", "n_classes", "averaging_strategy", "errors", "warnings"] and v is not None
    ]

    logger.info(f"Successfully computed {len(computed_metrics)} metrics: {computed_metrics}")

    if results["errors"]:
        logger.warning(f"Metrics computation had {len(results['errors'])} errors: {results['errors']}")

    return results


def format_metrics_summary(metrics: dict[str, Any]) -> str:
    """Format metrics results into a human-readable summary.

    Args:
        metrics: Results from compute_classification_metrics()

    Returns:
        Formatted summary string

    """
    lines = []
    lines.append(f"Classification Metrics ({metrics['problem_type']}, {metrics['n_classes']} classes)")
    lines.append(f"Averaging Strategy: {metrics['averaging_strategy']}")
    lines.append("")

    # Core metrics
    if metrics["accuracy"] is not None:
        lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
    if metrics["precision"] is not None:
        lines.append(f"Precision: {metrics['precision']:.4f}")
    if metrics["recall"] is not None:
        lines.append(f"Recall: {metrics['recall']:.4f}")
    if metrics["f1_score"] is not None:
        lines.append(f"F1-Score: {metrics['f1_score']:.4f}")

    # Probability-based metrics
    if metrics["roc_auc"] is not None:
        lines.append(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    if metrics["log_loss"] is not None:
        lines.append(f"Log Loss: {metrics['log_loss']:.4f}")

    # Warnings and errors
    if metrics["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for warning in metrics["warnings"]:
            lines.append(f"  • {warning}")

    if metrics["errors"]:
        lines.append("")
        lines.append("Errors:")
        for error in metrics["errors"]:
            lines.append(f"  • {error}")

    return "\n".join(lines)


def validate_inputs(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> None:
    """Validate inputs for metrics computation.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)

    Raises:
        ValueError: If inputs are invalid

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    if len(y_true) == 0:
        raise ValueError("Empty input arrays")

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if len(y_proba) != len(y_true):
            raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_proba={len(y_proba)}")

        # Check probability constraints
        if y_proba.ndim == 1:
            if not np.all((y_proba >= 0) & (y_proba <= 1)):
                raise ValueError("Probabilities must be in [0, 1]")
        elif y_proba.ndim == 2:
            if not np.all((y_proba >= 0) & (y_proba <= 1)):
                raise ValueError("Probabilities must be in [0, 1]")
            if not np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError("Probability rows must sum to 1")
        else:
            raise ValueError(f"Invalid probability array shape: {y_proba.shape}")
