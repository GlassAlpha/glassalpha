"""Tests for fairness metrics with numpy array tolerance."""

import numpy as np
import pandas as pd
import pytest

from glassalpha.metrics.fairness.bias_detection import (
    DemographicParityMetric,
    EqualOpportunityMetric,
)
from glassalpha.metrics.fairness.runner import run_fairness_metrics


def test_demographic_parity_with_numpy():
    """Test demographic parity metric with numpy arrays."""
    # Create test data
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 0])  # Higher positive rate for group 1
    sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # Two groups

    metric = DemographicParityMetric()
    result = metric.compute(y_true, y_pred, sensitive)

    # Should have results for both groups
    assert "sensitive_0" in result
    assert "sensitive_1" in result
    assert "demographic_parity" in result
    assert result["demographic_parity"] < 1.0  # Should detect bias


def test_equal_opportunity_with_numpy():
    """Test equal opportunity metric with numpy arrays."""
    # Create test data with different TPRs
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array(
        [1, 1, 0, 0, 0, 0, 0, 0],
    )  # TPR=0.5 for group 0, TPR=0 for group 1
    sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # Two groups

    metric = EqualOpportunityMetric()
    result = metric.compute(y_true, y_pred, sensitive)

    # Should detect the difference in TPR
    assert "equal_opportunity" in result
    assert result["equal_opportunity"] < 1.0  # Should detect bias


def test_fairness_runner_with_numpy():
    """Test fairness runner with numpy arrays."""
    # Create test data
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    sensitive = np.array([0, 0, 1, 1, 1, 0])

    metrics = [DemographicParityMetric, EqualOpportunityMetric]
    results = run_fairness_metrics(y_true, y_pred, sensitive, metrics)

    # Should have results for both metrics
    assert "demographicparity" in results
    assert "equalopportunity" in results

    # Each should have the expected structure
    dp_result = results["demographicparity"]
    assert "demographic_parity" in dp_result
    assert "sensitive_0" in dp_result
    assert "sensitive_1" in dp_result


def test_fairness_runner_with_dataframe():
    """Test fairness runner with pandas DataFrame."""
    # Create test data
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    sensitive_df = pd.DataFrame(
        {"gender": [0, 0, 1, 1, 1, 0], "age": [25, 30, 35, 40, 45, 50]},
    )

    metrics = [DemographicParityMetric, EqualOpportunityMetric]
    results = run_fairness_metrics(y_true, y_pred, sensitive_df, metrics)

    # Should have results for both metrics
    assert "demographicparity" in results
    assert "equalopportunity" in results

    # Should process all columns
    dp_result = results["demographicparity"]
    assert "gender_0" in dp_result
    assert "gender_1" in dp_result
    assert "age_25" in dp_result


def test_fairness_metrics_length_mismatch():
    """Test that length mismatches are caught."""
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 1, 1])
    sensitive = np.array([0, 0])  # Wrong length

    metrics = [DemographicParityMetric]

    with pytest.raises(ValueError, match="Length mismatch"):
        run_fairness_metrics(y_true, y_pred, sensitive, metrics)


def test_fairness_metrics_with_edge_cases():
    """Test fairness metrics with edge cases."""
    # Test with single class
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    sensitive = np.array([0, 0, 1, 1])

    metric = DemographicParityMetric()
    result = metric.compute(y_true, y_pred, sensitive)

    # Should handle single class gracefully
    assert "demographic_parity" in result

    # Test with empty groups
    y_true = np.array([1, 0])
    y_pred = np.array([1, 0])
    sensitive = np.array([0, 1])  # Each group has one sample

    metric = DemographicParityMetric()
    result = metric.compute(y_true, y_pred, sensitive)

    # Should handle small groups
    assert "demographic_parity" in result
