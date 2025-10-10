"""Basic metrics functionality tests.

Tests performance metrics including accuracy, precision, recall, F1, and AUC-ROC.
These tests focus on covering the metrics computation logic and registry integration.
"""

import numpy as np
import pandas as pd
import pytest

# Conditional sklearn import for CI compatibility
try:
    from sklearn.datasets import make_classification

    SKLEARN_AVAILABLE = True
except ImportError:
    make_classification = None
    SKLEARN_AVAILABLE = False

# Skip all tests if sklearn not available
pytestmark = pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available - CI compatibility issues")

from glassalpha.core import MetricRegistry
from glassalpha.metrics.performance.classification import (
    AccuracyMetric,
    AUCROCMetric,
    ClassificationReportMetric,
    F1Metric,
    PrecisionMetric,
    RecallMetric,
)


@pytest.fixture
def binary_classification_data():
    """Create binary classification test data."""
    # Generate deterministic data
    np.random.seed(42)
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=2,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )

    # Create predictions (with some errors)
    y_pred = y.copy()
    # Add some prediction errors
    error_indices = np.random.choice(len(y), size=20, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]

    # Create probabilities
    y_proba = np.random.random(len(y))
    y_proba[y == 1] += 0.3  # Bias toward correct class
    y_proba = np.clip(y_proba, 0, 1)

    return y, y_pred, y_proba


@pytest.fixture
def multiclass_classification_data():
    """Create multiclass classification test data."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=3,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )

    # Create predictions with some errors
    y_pred = y.copy()
    error_indices = np.random.choice(len(y), size=15, replace=False)
    y_pred[error_indices] = (y_pred[error_indices] + 1) % 3

    # Create probability matrix
    y_proba = np.random.random((len(y), 3))
    for i in range(len(y)):
        y_proba[i, y[i]] += 0.5  # Bias toward correct class
    # Normalize to sum to 1
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    return y, y_pred, y_proba


@pytest.fixture
def sensitive_features_data():
    """Create sensitive features for fairness testing."""
    return pd.DataFrame(
        {
            "gender": np.random.choice(["M", "F"], 100),
            "age_group": np.random.choice(["young", "old"], 100),
            "ethnicity": np.random.choice(["A", "B", "C"], 100),
        },
    )


class TestMetricRegistry:
    """Test metric registration and discovery."""

    def test_metrics_are_registered(self):
        """Test that metrics are properly registered."""
        components = MetricRegistry.get_all()

        # Should include our performance metrics
        assert "accuracy" in components
        assert "precision" in components
        assert "recall" in components
        assert "f1" in components
        assert "auc_roc" in components
        assert "classification_report" in components

    def test_get_metric_classes(self):
        """Test that we can retrieve metric classes."""
        accuracy_cls = MetricRegistry.get("accuracy")
        assert accuracy_cls == AccuracyMetric

        precision_cls = MetricRegistry.get("precision")
        assert precision_cls == PrecisionMetric

        f1_cls = MetricRegistry.get("f1")
        assert f1_cls == F1Metric


class TestAccuracyMetric:
    """Test AccuracyMetric functionality."""

    def test_accuracy_initialization(self):
        """Test AccuracyMetric initialization."""
        metric = AccuracyMetric()

        assert metric.name == "accuracy"
        assert metric.metric_type == "performance"
        assert metric.version == "1.0.0"

    def test_accuracy_binary_classification(self, binary_classification_data):
        """Test accuracy computation for binary classification."""
        y_true, y_pred, y_proba = binary_classification_data

        metric = AccuracyMetric()
        result = metric.compute(y_true, y_pred)

        assert isinstance(result, dict)
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

        # Manual verification
        expected_accuracy = np.mean(y_true == y_pred)
        assert abs(result["accuracy"] - expected_accuracy) < 1e-10

    def test_accuracy_multiclass_classification(self, multiclass_classification_data):
        """Test accuracy computation for multiclass classification."""
        y_true, y_pred, y_proba = multiclass_classification_data

        metric = AccuracyMetric()
        result = metric.compute(y_true, y_pred)

        assert isinstance(result, dict)
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_accuracy_perfect_predictions(self):
        """Test accuracy with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = y_true.copy()

        metric = AccuracyMetric()
        result = metric.compute(y_true, y_pred)

        assert result["accuracy"] == 1.0

    def test_accuracy_with_sensitive_features(self, binary_classification_data, sensitive_features_data):
        """Test accuracy computation with sensitive features."""
        y_true, y_pred, y_proba = binary_classification_data

        metric = AccuracyMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_features_data)

        assert isinstance(result, dict)
        assert "accuracy" in result
        # Should still compute overall accuracy
        assert 0.0 <= result["accuracy"] <= 1.0


class TestPrecisionMetric:
    """Test PrecisionMetric functionality."""

    def test_precision_initialization(self):
        """Test PrecisionMetric initialization."""
        metric = PrecisionMetric()

        assert metric.name == "precision"
        assert metric.metric_type == "performance"
        assert metric.version == "1.0.0"

    def test_precision_binary_classification(self, binary_classification_data):
        """Test precision computation for binary classification."""
        y_true, y_pred, y_proba = binary_classification_data

        metric = PrecisionMetric()
        result = metric.compute(y_true, y_pred)

        assert isinstance(result, dict)
        assert "precision" in result
        assert 0.0 <= result["precision"] <= 1.0

    def test_precision_multiclass_classification(self, multiclass_classification_data):
        """Test precision computation for multiclass classification."""
        y_true, y_pred, y_proba = multiclass_classification_data

        metric = PrecisionMetric()
        result = metric.compute(y_true, y_pred)

        assert isinstance(result, dict)
        assert "precision" in result
        assert 0.0 <= result["precision"] <= 1.0

    def test_precision_zero_division(self):
        """Test precision with zero division case."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])  # All wrong predictions

        metric = PrecisionMetric()
        result = metric.compute(y_true, y_pred)

        assert isinstance(result, dict)
        assert "precision" in result
        # Should handle zero division gracefully


class TestRecallMetric:
    """Test RecallMetric functionality."""

    def test_recall_initialization(self):
        """Test RecallMetric initialization."""
        metric = RecallMetric()

        assert metric.name == "recall"
        assert metric.metric_type == "performance"
        assert metric.version == "1.0.0"

    def test_recall_binary_classification(self, binary_classification_data):
        """Test recall computation for binary classification."""
        y_true, y_pred, y_proba = binary_classification_data

        metric = RecallMetric()
        result = metric.compute(y_true, y_pred)

        assert isinstance(result, dict)
        assert "recall" in result
        assert 0.0 <= result["recall"] <= 1.0

    def test_recall_multiclass_classification(self, multiclass_classification_data):
        """Test recall computation for multiclass classification."""
        y_true, y_pred, y_proba = multiclass_classification_data

        metric = RecallMetric()
        result = metric.compute(y_true, y_pred)

        assert isinstance(result, dict)
        assert "recall" in result
        assert 0.0 <= result["recall"] <= 1.0

    def test_recall_perfect_predictions(self):
        """Test recall with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = y_true.copy()

        metric = RecallMetric()
        result = metric.compute(y_true, y_pred)

        assert result["recall"] == 1.0


class TestF1Metric:
    """Test F1Metric functionality."""

    def test_f1_initialization(self):
        """Test F1Metric initialization."""
        metric = F1Metric()

        assert metric.name == "f1"
        assert metric.metric_type == "performance"
        assert metric.version == "1.0.0"

    def test_f1_binary_classification(self, binary_classification_data):
        """Test F1 computation for binary classification."""
        y_true, y_pred, y_proba = binary_classification_data

        metric = F1Metric()
        result = metric.compute(y_true, y_pred)

        assert isinstance(result, dict)
        assert "f1" in result
        assert 0.0 <= result["f1"] <= 1.0

    def test_f1_multiclass_classification(self, multiclass_classification_data):
        """Test F1 computation for multiclass classification."""
        y_true, y_pred, y_proba = multiclass_classification_data

        metric = F1Metric()
        result = metric.compute(y_true, y_pred)

        assert isinstance(result, dict)
        assert "f1" in result
        assert 0.0 <= result["f1"] <= 1.0

    def test_f1_harmonic_mean_property(self):
        """Test that F1 is harmonic mean of precision and recall."""
        # Create data where we can calculate precision/recall manually
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])

        # Manual calculation
        tp = np.sum((y_true == 1) & (y_pred == 1))  # 2
        fp = np.sum((y_true == 0) & (y_pred == 1))  # 1
        fn = np.sum((y_true == 1) & (y_pred == 0))  # 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        expected_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metric = F1Metric()
        result = metric.compute(y_true, y_pred)

        assert abs(result["f1"] - expected_f1) < 1e-10


class TestAUCROCMetric:
    """Test AUCROCMetric functionality."""

    def test_auc_roc_initialization(self):
        """Test AUCROCMetric initialization."""
        metric = AUCROCMetric()

        assert metric.name == "auc_roc"
        assert metric.metric_type == "performance"
        assert metric.version == "1.0.0"

    def test_auc_roc_binary_classification(self, binary_classification_data):
        """Test AUC-ROC computation for binary classification."""
        y_true, y_pred, y_proba = binary_classification_data

        metric = AUCROCMetric()
        # AUC-ROC expects probabilities as y_pred, not discrete predictions
        result = metric.compute(y_true, y_proba)

        assert isinstance(result, dict)
        assert "auc_roc" in result
        assert 0.0 <= result["auc_roc"] <= 1.0

    def test_auc_roc_without_probabilities(self, binary_classification_data):
        """Test AUC-ROC with discrete predictions (should work but less accurate)."""
        y_true, y_pred, y_proba = binary_classification_data

        metric = AUCROCMetric()
        result = metric.compute(y_true, y_pred.astype(float))  # Convert to float for ROC

        # Should handle gracefully
        assert isinstance(result, dict)
        assert "auc_roc" in result

    def test_auc_roc_perfect_classifier(self):
        """Test AUC-ROC with perfect classifier."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])  # Perfect separation

        metric = AUCROCMetric()
        result = metric.compute(y_true, y_proba)

        assert result["auc_roc"] == 1.0

    def test_auc_roc_multiclass_classification(self, multiclass_classification_data):
        """Test AUC-ROC computation for multiclass classification."""
        y_true, y_pred, y_proba = multiclass_classification_data

        metric = AUCROCMetric()
        # For multiclass, use probability matrix
        result = metric.compute(y_true, y_proba)

        assert isinstance(result, dict)
        assert "auc_roc" in result
        assert 0.0 <= result["auc_roc"] <= 1.0


class TestClassificationReportMetric:
    """Test ClassificationReportMetric functionality."""

    def test_classification_report_initialization(self):
        """Test ClassificationReportMetric initialization."""
        metric = ClassificationReportMetric()

        assert metric.name == "classification_report"
        assert metric.metric_type == "performance"
        assert metric.version == "1.0.0"

    def test_classification_report_binary(self, binary_classification_data):
        """Test classification report for binary classification."""
        y_true, y_pred, y_proba = binary_classification_data

        metric = ClassificationReportMetric()
        result = metric.compute(y_true, y_pred)

        assert isinstance(result, dict)
        # Classification report returns individual metrics, not nested dict
        assert "accuracy" in result
        assert "macro_f1" in result or "f1" in result

    def test_classification_report_multiclass(self, multiclass_classification_data):
        """Test classification report for multiclass classification."""
        y_true, y_pred, y_proba = multiclass_classification_data

        metric = ClassificationReportMetric()
        result = metric.compute(y_true, y_pred)

        assert isinstance(result, dict)
        # Should contain multiple metrics
        assert len(result) > 1


class TestMetricsIntegration:
    """Test integration between different metrics."""

    def test_multiple_metrics_computation(self, binary_classification_data):
        """Test computing multiple metrics together."""
        y_true, y_pred, y_proba = binary_classification_data

        metrics = [AccuracyMetric(), PrecisionMetric(), RecallMetric(), F1Metric()]

        results = {}
        for metric in metrics:
            result = metric.compute(y_true, y_pred)
            results.update(result)

        # Should have all metrics
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results

        # All should be valid values (some might be percentages 0-100)
        for _key, value in results.items():
            if isinstance(value, (int, float)):
                assert value >= 0.0  # Allow values > 1 for percentages

    def test_metrics_consistency(self):
        """Test that metrics are consistent with each other."""
        # Perfect predictions
        y_true = np.array([1, 1, 0, 0])
        y_pred = y_true.copy()

        accuracy_metric = AccuracyMetric()
        precision_metric = PrecisionMetric()
        recall_metric = RecallMetric()
        f1_metric = F1Metric()

        accuracy_result = accuracy_metric.compute(y_true, y_pred)
        precision_result = precision_metric.compute(y_true, y_pred)
        recall_result = recall_metric.compute(y_true, y_pred)
        f1_result = f1_metric.compute(y_true, y_pred)

        # With perfect predictions, all metrics should be 1.0
        assert accuracy_result["accuracy"] == 1.0
        assert precision_result["precision"] == 1.0
        assert recall_result["recall"] == 1.0
        assert f1_result["f1"] == 1.0


class TestMetricsErrorHandling:
    """Test error handling in metrics computation."""

    def test_empty_arrays(self):
        """Test metrics with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])

        metric = AccuracyMetric()
        result = metric.compute(y_true, y_pred)

        # Should handle empty arrays gracefully, might return NaN
        assert isinstance(result, dict)
        # Check that result contains expected keys
        assert "accuracy" in result

    def test_mismatched_array_lengths(self):
        """Test metrics with mismatched array lengths."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([1, 0])  # Different length

        metric = AccuracyMetric()

        with pytest.raises(ValueError):
            metric.compute(y_true, y_pred)

    def test_invalid_class_labels(self):
        """Test metrics with invalid class labels."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 5])  # Invalid class 5

        metric = AccuracyMetric()
        result = metric.compute(y_true, y_pred)

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_single_class_case(self):
        """Test metrics with single class."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])

        precision_metric = PrecisionMetric()
        recall_metric = RecallMetric()

        precision_result = precision_metric.compute(y_true, y_pred)
        recall_result = recall_metric.compute(y_true, y_pred)

        # Should handle single class case
        assert isinstance(precision_result, dict)
        assert isinstance(recall_result, dict)


class TestMetricsWithSensitiveFeatures:
    """Test metrics computation with sensitive features."""

    def test_metrics_ignore_sensitive_features_for_performance(
        self,
        binary_classification_data,
        sensitive_features_data,
    ):
        """Test that performance metrics ignore sensitive features."""
        y_true, y_pred, y_proba = binary_classification_data

        accuracy_metric = AccuracyMetric()

        # Compute with and without sensitive features
        result_without = accuracy_metric.compute(y_true, y_pred)
        result_with = accuracy_metric.compute(y_true, y_pred, sensitive_features=sensitive_features_data)

        # Performance metrics should be the same regardless of sensitive features
        assert result_without["accuracy"] == result_with["accuracy"]
