"""Test metrics auto-detection engine with multiclass defaults."""

import numpy as np
import pytest

from glassalpha.metrics.core import (
    choose_averaging,
    compute_classification_metrics,
    format_metrics_summary,
    infer_problem_type,
    validate_inputs,
)


def test_infer_problem_type():
    """Test problem type detection."""
    # Binary case
    y_binary = np.array([0, 1, 0, 1, 1])
    assert infer_problem_type(y_binary) == "binary"

    # Multiclass case
    y_multiclass = np.array([0, 1, 2, 3, 1, 0])
    assert infer_problem_type(y_multiclass) == "multiclass"

    # Edge case: single class (should be binary for sklearn compatibility)
    y_single = np.array([1, 1, 1, 1])
    assert infer_problem_type(y_single) == "binary"


def test_choose_averaging():
    """Test averaging strategy selection."""
    # Binary cases
    assert choose_averaging("binary", None) is None
    assert choose_averaging("binary", "binary") is None
    assert choose_averaging("binary", "macro") == "macro"  # Override

    # Multiclass cases
    assert choose_averaging("multiclass", None) == "macro"  # Default
    assert choose_averaging("multiclass", "weighted") == "weighted"  # Override
    assert choose_averaging("multiclass", "micro") == "micro"  # Override


def test_binary_classification_metrics():
    """Test metrics computation for binary classification."""
    # Simple binary case
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    y_proba = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.6, 0.4]])

    results = compute_classification_metrics(y_true, y_pred, y_proba)

    # Check basic structure
    assert results["problem_type"] == "binary"
    assert results["n_classes"] == 2
    assert results["averaging_strategy"] is None

    # Check that all metrics are computed
    assert results["accuracy"] is not None
    assert results["precision"] is not None
    assert results["recall"] is not None
    assert results["f1_score"] is not None
    assert results["roc_auc"] is not None
    assert results["log_loss"] is not None
    assert results["confusion_matrix"] is not None

    # Check no errors
    assert len(results["errors"]) == 0

    # Validate ranges
    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["f1_score"] <= 1
    assert 0 <= results["roc_auc"] <= 1


def test_multiclass_classification_metrics():
    """Test metrics computation for multiclass classification."""
    # 4-class case (like customer segmentation)
    np.random.seed(42)
    y_true = np.random.choice([0, 1, 2, 3], size=100)
    y_pred = np.random.choice([0, 1, 2, 3], size=100)
    y_proba = np.random.dirichlet([1, 1, 1, 1], size=100)  # Proper probability matrix

    results = compute_classification_metrics(y_true, y_pred, y_proba)

    # Check basic structure
    assert results["problem_type"] == "multiclass"
    assert results["n_classes"] == 4
    assert results["averaging_strategy"] == "macro"  # Default for multiclass

    # Check that all metrics are computed without errors
    assert results["accuracy"] is not None
    assert results["precision"] is not None
    assert results["recall"] is not None
    assert results["f1_score"] is not None
    assert results["roc_auc"] is not None
    assert results["log_loss"] is not None
    assert results["confusion_matrix"] is not None

    # This is the key test - no "average='binary'" errors
    assert len(results["errors"]) == 0

    # Validate ranges
    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["f1_score"] <= 1
    assert 0 <= results["roc_auc"] <= 1


def test_multiclass_with_averaging_override():
    """Test multiclass metrics with custom averaging."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])

    # Test weighted averaging
    results = compute_classification_metrics(y_true, y_pred, average="weighted")
    assert results["averaging_strategy"] == "weighted"
    assert results["precision"] is not None
    assert len(results["errors"]) == 0

    # Test micro averaging
    results = compute_classification_metrics(y_true, y_pred, average="micro")
    assert results["averaging_strategy"] == "micro"
    assert results["precision"] is not None
    assert len(results["errors"]) == 0


def test_metrics_without_probabilities():
    """Test metrics computation when probabilities are not provided."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])

    results = compute_classification_metrics(y_true, y_pred, y_proba=None)

    # Basic metrics should still work
    assert results["accuracy"] is not None
    assert results["precision"] is not None
    assert results["recall"] is not None
    assert results["f1_score"] is not None

    # Probability-based metrics should be None
    assert results["roc_auc"] is None
    assert results["log_loss"] is None

    # Should have warning about missing probabilities
    assert any("probabilities not provided" in w for w in results["warnings"])
    assert len(results["errors"]) == 0


def test_invalid_probability_shapes():
    """Test handling of invalid probability shapes."""
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 0])

    # Wrong shape for multiclass
    y_proba_wrong = np.array([[0.5, 0.5], [0.3, 0.7], [0.2, 0.8], [0.6, 0.4]])  # 2 cols for 3 classes

    results = compute_classification_metrics(y_true, y_pred, y_proba_wrong)

    # Should have error about probability shape
    assert any("probability" in error.lower() for error in results["errors"])
    assert results["roc_auc"] is None
    assert results["log_loss"] is None


def test_validate_inputs():
    """Test input validation."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    y_proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.9, 0.1]])

    # Valid inputs should not raise
    validate_inputs(y_true, y_pred, y_proba)

    # Length mismatch should raise
    with pytest.raises(ValueError, match="Length mismatch"):
        validate_inputs(y_true, y_pred[:-1])

    # Empty arrays should raise
    with pytest.raises(ValueError, match="Empty input"):
        validate_inputs(np.array([]), np.array([]))

    # Invalid probabilities should raise
    with pytest.raises(ValueError, match="Probabilities must be in"):
        bad_proba = np.array([[1.5, -0.5], [0.2, 0.8], [0.4, 0.6], [0.9, 0.1]])
        validate_inputs(y_true, y_pred, bad_proba)


def test_format_metrics_summary():
    """Test metrics summary formatting."""
    # Create sample metrics
    metrics = {
        "problem_type": "multiclass",
        "n_classes": 3,
        "averaging_strategy": "macro",
        "accuracy": 0.8333,
        "precision": 0.7500,
        "recall": 0.6667,
        "f1_score": 0.7059,
        "roc_auc": 0.9000,
        "log_loss": 0.4500,
        "confusion_matrix": [[2, 0, 1], [0, 2, 1], [1, 0, 2]],
        "errors": [],
        "warnings": ["Some warning"],
    }

    summary = format_metrics_summary(metrics)

    # Check that key information is included
    assert "multiclass" in summary
    assert "3 classes" in summary
    assert "macro" in summary
    assert "Accuracy: 0.8333" in summary
    assert "Some warning" in summary


def test_edge_cases():
    """Test edge cases and error conditions."""
    # Perfect predictions
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])  # Perfect match

    results = compute_classification_metrics(y_true, y_pred)
    assert results["accuracy"] == 1.0
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0
    assert results["f1_score"] == 1.0

    # All wrong predictions
    y_pred_wrong = np.array([2, 0, 1, 2, 0, 1])  # All wrong
    results = compute_classification_metrics(y_true, y_pred_wrong)
    assert results["accuracy"] == 0.0
    assert len(results["errors"]) == 0  # Should still compute without errors


if __name__ == "__main__":
    pytest.main([__file__])
