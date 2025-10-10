"""Classification performance metrics for GlassAlpha.

This module implements standard classification performance metrics including
accuracy, precision, recall, F1-score, and AUC-ROC. All metrics follow
the BaseMetric pattern and integrate with the registry system.
"""

import logging

import numpy as np
import pandas as pd

# Conditional sklearn import with graceful fallback for CI compatibility
try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback stubs when sklearn unavailable (CI environment issues)
    SKLEARN_AVAILABLE = False

    def _sklearn_unavailable(*args, **kwargs):
        raise ImportError("sklearn not available - install scikit-learn or fix CI environment")

    accuracy_score = _sklearn_unavailable
    classification_report = _sklearn_unavailable
    f1_score = _sklearn_unavailable
    precision_score = _sklearn_unavailable
    recall_score = _sklearn_unavailable
    roc_auc_score = _sklearn_unavailable

from ..base import BaseMetric
from ..registry import MetricRegistry

logger = logging.getLogger(__name__)


@MetricRegistry.register("accuracy", priority=100)
class AccuracyMetric(BaseMetric):
    """Accuracy metric for classification tasks.

    Computes the fraction of predictions that match the ground truth labels.
    Works for both binary and multiclass classification.
    """

    metric_type = "performance"
    version = "1.0.0"

    def __init__(self):
        """Initialize accuracy metric."""
        super().__init__("accuracy", "performance", "1.0.0")
        logger.info("AccuracyMetric initialized")

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute accuracy metric.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sensitive_features: Optional sensitive features (not used for accuracy)

        Returns:
            Dictionary with accuracy score

        """
        validation = self.validate_inputs(y_true, y_pred, sensitive_features)

        try:
            accuracy = accuracy_score(y_true, y_pred)

            logger.debug(f"Computed accuracy: {accuracy:.4f} for {validation['n_samples']} samples")

            return {
                "accuracy": float(accuracy),
                "n_samples": float(validation["n_samples"]),
            }

        except Exception as e:
            logger.error(f"Error computing accuracy: {e}")
            return {"accuracy": 0.0, "error": str(e)}

    def get_metric_names(self) -> list[str]:
        """Return metric names."""
        return ["accuracy", "n_samples"]


@MetricRegistry.register("precision", priority=90)
class PrecisionMetric(BaseMetric):
    """Precision metric for classification tasks.

    Computes precision (positive predictive value) which is the fraction of
    predicted positive instances that are actually positive.
    """

    metric_type = "performance"
    version = "1.0.0"

    def __init__(self, average: str = "binary"):
        """Initialize precision metric.

        Args:
            average: Averaging method ('binary', 'macro', 'micro', 'weighted')

        """
        super().__init__("precision", "performance", "1.0.0")
        self.average = average
        logger.info(f"PrecisionMetric initialized (average={average})")

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute precision metric.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sensitive_features: Optional sensitive features (not used for precision)

        Returns:
            Dictionary with precision score

        """
        validation = self.validate_inputs(y_true, y_pred, sensitive_features)

        try:
            # Determine if binary or multiclass
            n_classes = len(np.unique(y_true))
            if n_classes == 2 and self.average == "binary":
                # Binary classification
                precision = precision_score(y_true, y_pred, average="binary", zero_division=0)

            else:
                # Multiclass classification
                precision = precision_score(y_true, y_pred, average=self.average, zero_division=0)

            result = {
                "precision": float(precision),
                "n_samples": float(validation["n_samples"]),
                "n_classes": float(n_classes),
                "average": self.average,
            }

            logger.debug(f"Computed precision ({self.average}): {precision:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error computing precision: {e}")
            return {"precision": 0.0, "error": str(e)}

    def get_metric_names(self) -> list[str]:
        """Return metric names."""
        return ["precision", "n_samples", "n_classes", "average"]


@MetricRegistry.register("recall", priority=89)
class RecallMetric(BaseMetric):
    """Recall (sensitivity) metric for classification tasks.

    Computes recall which is the fraction of actual positive instances
    that are correctly predicted as positive.
    """

    metric_type = "performance"
    version = "1.0.0"

    def __init__(self, average: str = "binary"):
        """Initialize recall metric.

        Args:
            average: Averaging method ('binary', 'macro', 'micro', 'weighted')

        """
        super().__init__("recall", "performance", "1.0.0")
        self.average = average
        logger.info(f"RecallMetric initialized (average={average})")

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute recall metric.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sensitive_features: Optional sensitive features (not used for recall)

        Returns:
            Dictionary with recall score

        """
        validation = self.validate_inputs(y_true, y_pred, sensitive_features)

        try:
            # Determine if binary or multiclass
            n_classes = len(np.unique(y_true))
            if n_classes == 2 and self.average == "binary":
                # Binary classification
                recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
            else:
                # Multiclass classification
                recall = recall_score(y_true, y_pred, average=self.average, zero_division=0)

            result = {
                "recall": float(recall),
                "n_samples": float(validation["n_samples"]),
                "n_classes": float(n_classes),
                "average": self.average,
            }

            logger.debug(f"Computed recall ({self.average}): {recall:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error computing recall: {e}")
            return {"recall": 0.0, "error": str(e)}

    def get_metric_names(self) -> list[str]:
        """Return metric names."""
        return ["recall", "n_samples", "n_classes", "average"]


@MetricRegistry.register("f1", priority=88)
class F1Metric(BaseMetric):
    """F1-score metric for classification tasks.

    Computes the F1-score which is the harmonic mean of precision and recall.
    Provides a balanced measure of both precision and recall.
    """

    metric_type = "performance"
    version = "1.0.0"

    def __init__(self, average: str = "binary"):
        """Initialize F1 metric.

        Args:
            average: Averaging method ('binary', 'macro', 'micro', 'weighted')

        """
        super().__init__("f1", "performance", "1.0.0")
        self.average = average
        logger.info(f"F1Metric initialized (average={average})")

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute F1-score metric.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sensitive_features: Optional sensitive features (not used for F1)

        Returns:
            Dictionary with F1-score

        """
        validation = self.validate_inputs(y_true, y_pred, sensitive_features)

        try:
            # Determine if binary or multiclass
            n_classes = len(np.unique(y_true))
            if n_classes == 2 and self.average == "binary":
                # Binary classification
                f1 = f1_score(y_true, y_pred, average="binary")
            else:
                # Multiclass classification
                f1 = f1_score(y_true, y_pred, average=self.average, zero_division=0)

            result = {
                "f1": float(f1),
                "n_samples": float(validation["n_samples"]),
                "n_classes": float(n_classes),
                "average": self.average,
            }

            logger.debug(f"Computed F1-score ({self.average}): {f1:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error computing F1-score: {e}")
            return {"f1": 0.0, "error": str(e)}

    def get_metric_names(self) -> list[str]:
        """Return metric names."""
        return ["f1", "n_samples", "n_classes", "average"]


@MetricRegistry.register("auc_roc", priority=87)
class AUCROCMetric(BaseMetric):
    """AUC-ROC metric for classification tasks.

    Computes the Area Under the Receiver Operating Characteristic curve.
    Measures the model's ability to distinguish between classes.
    """

    metric_type = "performance"
    version = "1.0.0"

    def __init__(self, multiclass: str = "ovr"):
        """Initialize AUC-ROC metric.

        Args:
            multiclass: Strategy for multiclass ('ovr', 'ovo')

        """
        super().__init__("auc_roc", "performance", "1.0.0")
        self.multiclass = multiclass
        logger.info(f"AUCROCMetric initialized (multiclass={multiclass})")

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute AUC-ROC metric.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities (not labels!)
            sensitive_features: Optional sensitive features (not used for AUC)

        Returns:
            Dictionary with AUC-ROC score

        """
        validation = self.validate_inputs(y_true, y_pred, sensitive_features)

        try:
            n_classes = len(np.unique(y_true))

            # Handle different input formats
            if n_classes == 2:
                # Binary classification
                if y_pred.ndim == 2:
                    # Probabilities for both classes - use positive class
                    y_proba = y_pred[:, 1]
                elif np.all((y_pred >= 0) & (y_pred <= 1)):
                    # Already probabilities
                    y_proba = y_pred
                else:
                    # Hard predictions - convert to probabilities (not ideal for AUC)
                    logger.warning("AUC-ROC computed with hard predictions instead of probabilities")
                    y_proba = y_pred.astype(float)

                auc_roc = roc_auc_score(y_true, y_proba)

            else:
                # Multiclass classification
                if y_pred.ndim == 1:
                    logger.error("AUC-ROC for multiclass requires probability predictions, got hard predictions")
                    return {"auc_roc": 0.0, "error": "Multiclass AUC requires probabilities"}

                auc_roc = roc_auc_score(y_true, y_pred, multi_class=self.multiclass, average="macro")

            result = {
                "auc_roc": float(auc_roc),
                "n_samples": float(validation["n_samples"]),
                "n_classes": float(n_classes),
                "multiclass": self.multiclass,
            }

            logger.debug(f"Computed AUC-ROC: {auc_roc:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error computing AUC-ROC: {e}")
            return {"auc_roc": 0.0, "error": str(e)}

    def get_metric_names(self) -> list[str]:
        """Return metric names."""
        return ["auc_roc", "n_samples", "n_classes", "multiclass"]


@MetricRegistry.register("classification_report", priority=85)
class ClassificationReportMetric(BaseMetric):
    """Comprehensive classification report metric.

    Computes a comprehensive report including precision, recall, F1-score
    for each class, plus macro and weighted averages.
    """

    metric_type = "performance"
    version = "1.0.0"

    def __init__(self):
        """Initialize classification report metric."""
        super().__init__("classification_report", "performance", "1.0.0")
        logger.info("ClassificationReportMetric initialized")

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute comprehensive classification report.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sensitive_features: Optional sensitive features (not used)

        Returns:
            Dictionary with comprehensive metrics

        """
        validation = self.validate_inputs(y_true, y_pred, sensitive_features)

        try:
            # Get basic metrics
            accuracy = accuracy_score(y_true, y_pred)

            # Get detailed report as dict
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

            # Extract key metrics
            result = {
                "accuracy": float(accuracy),
                "macro_precision": float(report["macro avg"]["precision"]),
                "macro_recall": float(report["macro avg"]["recall"]),
                "macro_f1": float(report["macro avg"]["f1-score"]),
                "weighted_precision": float(report["weighted avg"]["precision"]),
                "weighted_recall": float(report["weighted avg"]["recall"]),
                "weighted_f1": float(report["weighted avg"]["f1-score"]),
                "n_samples": float(validation["n_samples"]),
                "n_classes": float(len([k for k in report if k.isdigit() or k in ["0", "1"]])),
            }

            logger.debug("Computed comprehensive classification report")

            return result

        except Exception as e:
            logger.error(f"Error computing classification report: {e}")
            return {"accuracy": 0.0, "error": str(e)}

    def get_metric_names(self) -> list[str]:
        """Return metric names."""
        return [
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "n_samples",
            "n_classes",
        ]


# Auto-register metrics
logger.debug("Performance classification metrics registered")
