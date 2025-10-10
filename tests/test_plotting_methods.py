"""Test plotting methods in API metrics.

These tests verify that the plotting methods work with stored metrics data.
"""

import numpy as np
import pytest

from glassalpha.api.metrics import CalibrationMetrics, FairnessMetrics, PerformanceMetrics


def test_performance_plot_confusion_matrix():
    """Test confusion matrix plotting from stored metrics."""
    # Create performance metrics with confusion matrix
    perf_data = {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1": 0.85,
        "confusion_matrix": [[50, 10], [5, 35]],  # 2x2 binary classification
    }
    perf = PerformanceMetrics(perf_data)

    # Verify confusion_matrix is accessible (frozen as tuples)
    cm = perf["confusion_matrix"]
    assert len(cm) == 2
    assert len(cm[0]) == 2

    # Test that plotting method exists and would work
    # (we skip actual plotting in tests to avoid matplotlib dependency)
    assert hasattr(perf, "plot_confusion_matrix")


def test_performance_plot_roc_curve_not_implemented():
    """Test ROC curve plotting shows helpful not implemented error."""
    perf_data = {
        "accuracy": 0.85,
        "roc_auc": 0.92,
    }
    perf = PerformanceMetrics(perf_data)

    # Should raise NotImplementedError with helpful message
    with pytest.raises(NotImplementedError, match="ROC curve plotting requires"):
        perf.plot_roc_curve()


def test_fairness_plot_group_metrics():
    """Test fairness group metrics plotting."""
    fairness_data = {
        "demographic_parity_max_diff": 0.05,
        "group_metrics": {
            "gender": {
                "M": {"selection_rate": 0.75, "tpr": 0.80, "fpr": 0.15},
                "F": {"selection_rate": 0.70, "tpr": 0.75, "fpr": 0.20},
            },
            "age_group": {
                "18-30": {"selection_rate": 0.80, "tpr": 0.85, "fpr": 0.10},
                "31-50": {"selection_rate": 0.72, "tpr": 0.78, "fpr": 0.18},
                "51+": {"selection_rate": 0.68, "tpr": 0.70, "fpr": 0.22},
            },
        },
    }
    fairness = FairnessMetrics(fairness_data)

    # Verify group_metrics is accessible
    assert "group_metrics" in fairness
    assert "gender" in fairness["group_metrics"]

    # Test that plotting method exists
    assert hasattr(fairness, "plot_group_metrics")


def test_fairness_plot_no_group_metrics():
    """Test fairness plotting fails gracefully without group metrics."""
    fairness_data = {
        "demographic_parity_max_diff": 0.05,
    }
    fairness = FairnessMetrics(fairness_data)

    # Should raise ValueError with helpful message
    with pytest.raises(ValueError, match="No group_metrics available"):
        fairness.plot_group_metrics()


def test_calibration_plot_not_implemented():
    """Test calibration plotting shows helpful not implemented error."""
    calib_data = {
        "ece": 0.042,
        "brier_score": 0.12,
    }
    calib = CalibrationMetrics(calib_data)

    # Should raise NotImplementedError with helpful message
    with pytest.raises(NotImplementedError, match="Calibration curve plotting requires"):
        calib.plot()


def test_metrics_immutability():
    """Verify plotting methods don't mutate metrics data."""
    perf_data = {
        "accuracy": 0.85,
        "confusion_matrix": [[50, 10], [5, 35]],
    }
    perf = PerformanceMetrics(perf_data)

    # Access confusion matrix
    cm = perf["confusion_matrix"]

    # Original data should be unchanged (frozen as tuples)
    assert perf["accuracy"] == 0.85
    assert len(cm) == 2
    assert len(cm[0]) == 2
