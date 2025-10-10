"""Tests for from_predictions() entry point."""

import numpy as np
import pandas as pd
import pytest

import glassalpha as ga
from glassalpha.exceptions import (
    InvalidProtectedAttributesError,
    LengthMismatchError,
    NonBinaryClassificationError,
)


def test_from_predictions_basic():
    """Test basic from_predictions with labels only."""
    # Binary classification data
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])

    result = ga.audit.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        calibration=False,  # No y_proba
    )

    # Check basic metrics
    assert result.performance["accuracy"] == 0.75  # 6/8 correct
    assert "precision" in result.performance
    assert "recall" in result.performance
    assert "f1" in result.performance

    # No probability-based metrics
    assert "roc_auc" not in result.performance
    assert "brier_score" not in result.performance

    # No fairness (no protected attributes)
    assert len(result.fairness) == 0

    # Check result ID is deterministic
    assert len(result.id) == 64
    assert result.schema_version == "0.2.0"


def test_from_predictions_with_probabilities():
    """Test from_predictions with predicted probabilities."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.6, 0.7])

    result = ga.audit.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    # Probability-based metrics should be present
    assert "roc_auc" in result.performance
    assert "pr_auc" in result.performance
    assert "brier_score" in result.performance
    assert "log_loss" in result.performance

    # Calibration metrics should be present
    assert "brier_score" in result.calibration
    assert "expected_calibration_error" in result.calibration
    assert "maximum_calibration_error" in result.calibration


def test_from_predictions_with_protected_attributes():
    """Test from_predictions with protected attributes."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    gender = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    result = ga.audit.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        protected_attributes={"gender": gender},
        calibration=False,
    )

    # Fairness metrics should be present
    assert len(result.fairness) > 0
    # Should have group-level metrics
    assert "gender_0" in result.fairness or "demographic_parity" in str(result.fairness)


def test_from_predictions_length_mismatch():
    """Test that length mismatch raises error."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0])  # Missing one

    with pytest.raises(LengthMismatchError) as exc_info:
        ga.audit.from_predictions(y_true=y_true, y_pred=y_pred)

    assert "GAE1003" in str(exc_info.value)
    assert "y_pred" in str(exc_info.value)


def test_from_predictions_non_binary():
    """Test that non-binary classification raises error."""
    y_true = np.array([0, 1, 2, 0, 1, 2])  # 3 classes
    y_pred = np.array([0, 1, 2, 0, 1, 2])

    with pytest.raises(NonBinaryClassificationError) as exc_info:
        ga.audit.from_predictions(y_true=y_true, y_pred=y_pred)

    assert "GAE1004" in str(exc_info.value)


def test_from_predictions_invalid_protected_attributes():
    """Test that invalid protected_attributes format raises error."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])

    # Pass list instead of dict
    with pytest.raises(InvalidProtectedAttributesError) as exc_info:
        ga.audit.from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            protected_attributes=[0, 1, 0, 1],  # type: ignore[arg-type]
        )

    assert "GAE1001" in str(exc_info.value)


def test_from_predictions_protected_attributes_length_mismatch():
    """Test that protected_attributes length mismatch raises error."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    gender = np.array([0, 1, 0])  # Missing one

    with pytest.raises(LengthMismatchError) as exc_info:
        ga.audit.from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            protected_attributes={"gender": gender},
        )

    assert "GAE1003" in str(exc_info.value)
    assert "gender" in str(exc_info.value)


def test_from_predictions_with_nan_in_protected_attributes():
    """Test that NaN in protected_attributes is mapped to 'Unknown'."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    gender = np.array([0, 1, np.nan, 0, 1, np.nan])

    result = ga.audit.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        protected_attributes={"gender": gender},
        calibration=False,
    )

    # Should succeed (NaN mapped to "Unknown")
    assert len(result.fairness) > 0
    # Manifest should show Unknown as a category
    assert "Unknown" in result.manifest["protected_attributes_categories"]["gender"]


def test_from_predictions_deterministic():
    """Test that result ID is deterministic for same inputs."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8])

    result1 = ga.audit.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        random_seed=42,
    )

    result2 = ga.audit.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        random_seed=42,
    )

    # Result IDs should match (byte-identical)
    assert result1.id == result2.id


def test_from_predictions_pandas_series():
    """Test from_predictions with pandas Series inputs."""
    y_true = pd.Series([0, 1, 0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 0, 0, 0, 1])
    y_proba = pd.Series([0.1, 0.9, 0.2, 0.4, 0.3, 0.8])
    gender = pd.Series([0, 0, 1, 1, 0, 1])

    result = ga.audit.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        protected_attributes={"gender": gender},
    )

    # Should work with pandas Series
    assert result.performance["accuracy"] > 0
    assert len(result.fairness) > 0


def test_from_predictions_2d_probabilities():
    """Test from_predictions with 2D probability array."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    # 2D array with probabilities for both classes
    y_proba = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.3, 0.7],
        ],
    )

    result = ga.audit.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    # Should extract positive class (column 1)
    assert "roc_auc" in result.performance
    assert result.performance["roc_auc"] == 1.0  # Perfect predictions
