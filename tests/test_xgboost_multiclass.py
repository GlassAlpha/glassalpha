"""Tests for XGBoost multiclass support and parameter handling.

NOTE: These tests were moved from /tests/ and may need API updates.
Currently skipped pending XGBoostWrapper API review.
"""

import numpy as np
import pandas as pd
import pytest

from glassalpha.models.tabular.xgboost import XGBoostWrapper

# Tests re-enabled - API appears compatible


def test_multiclass_training():
    """Test that XGBoost wrapper handles multiclass training correctly."""
    # Create test data with 4 classes
    X = pd.DataFrame(np.random.randn(40, 3), columns=["a", "b", "c"])
    y = np.array([0, 1, 2, 3] * 10)

    # Train model with multiclass config
    model = XGBoostWrapper()
    model.fit(X, y, objective="multi:softprob", num_class=4, max_depth=3)

    # Verify model info
    info = model.get_model_info()
    assert info["n_classes"] == 4
    assert len(info["classes"]) == 4
    assert list(info["classes"]) == [0, 1, 2, 3]


def test_multiclass_auto_inference():
    """Test that XGBoost wrapper auto-infers multiclass objective."""
    # Create test data with 3 classes
    X = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
    y = np.array([0, 1, 2] * 10)

    # Train model without specifying objective
    model = XGBoostWrapper()
    model.fit(X, y, max_depth=3)

    # Should auto-infer multiclass
    info = model.get_model_info()
    assert info["n_classes"] == 3


def test_binary_classification():
    """Test that XGBoost wrapper handles binary classification correctly."""
    # Create binary test data
    X = pd.DataFrame(np.random.randn(40, 3), columns=["a", "b", "c"])
    y = np.array([0, 1] * 20)

    # Train model with binary config
    model = XGBoostWrapper()
    model.fit(X, y, objective="binary:logistic", max_depth=3)

    # Verify binary classification
    info = model.get_model_info()
    assert info["n_classes"] == 2


def test_binary_auto_inference():
    """Test that XGBoost wrapper auto-infers binary objective."""
    # Create binary test data
    X = pd.DataFrame(np.random.randn(40, 3), columns=["a", "b", "c"])
    y = np.array([0, 1] * 20)

    # Train model without specifying objective
    model = XGBoostWrapper()
    model.fit(X, y, max_depth=3)

    # Should auto-infer binary
    info = model.get_model_info()
    assert info["n_classes"] == 2


def test_predict_proba_shape():
    """Test that predict_proba returns correct shape for both binary and multiclass."""
    # Test binary case
    X_binary = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    y_binary = np.array([0, 1] * 10)

    model_binary = XGBoostWrapper()
    model_binary.fit(X_binary, y_binary, max_depth=3)

    proba_binary = model_binary.predict_proba(X_binary)
    assert proba_binary.shape == (20, 2)  # (n_samples, 2) for binary

    # Test multiclass case
    X_multi = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    y_multi = np.array([0, 1, 2] * 6 + [0, 1])  # 3 classes

    model_multi = XGBoostWrapper()
    model_multi.fit(
        X_multi,
        y_multi,
        objective="multi:softprob",
        num_class=3,
        max_depth=3,
    )

    proba_multi = model_multi.predict_proba(X_multi)
    assert proba_multi.shape == (20, 3)  # (n_samples, 3) for multiclass


def test_predict_returns_original_labels():
    """Test that predict returns class labels in the expected range."""
    # Create test data with sequential labels (XGBoost requirement)
    X = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
    y = np.array([0, 1, 2] * 10)  # Sequential labels 0, 1, 2

    model = XGBoostWrapper()
    model.fit(X, y, objective="multi:softprob", num_class=3, max_depth=3)

    # Get predictions
    predictions = model.predict(X)

    # Should return labels in the range [0, num_class)
    unique_predictions = np.unique(predictions)
    assert set(unique_predictions).issubset({0, 1, 2}), (
        f"Predictions should be in range [0, 3), got {set(unique_predictions)}"
    )


def test_inconsistent_objective_num_class_raises():
    """Test that inconsistent objective/num_class raises ValueError."""
    X = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    y = np.array([0, 1, 2] * 6 + [0, 1])  # 3 classes (multiclass data)

    model = XGBoostWrapper()

    # Binary objective with 3 classes should fail
    with pytest.raises(
        ValueError,
        match="Binary objective .* incompatible with .* classes",
    ):
        model.fit(X, y, objective="binary:logistic", num_class=3)


def test_predict_before_fit_raises():
    """Test that predict before fit raises appropriate error."""
    X = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
    model = XGBoostWrapper()

    with pytest.raises(ValueError, match="Model not loaded"):
        model.predict(X)


def test_feature_alignment():
    """Test that feature alignment works correctly."""
    # Train with certain column names
    X_train = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    y_train = np.array([0, 1] * 10)

    model = XGBoostWrapper()
    model.fit(X_train, y_train, max_depth=3)

    # Predict with different column order
    X_test = pd.DataFrame(np.random.randn(10, 3), columns=["c", "a", "b"])

    # Should handle column reordering
    predictions = model.predict(X_test)
    assert len(predictions) == 10

    # Predict with missing columns (should fill with 0)
    X_test_missing = pd.DataFrame(np.random.randn(10, 2), columns=["a", "b"])

    predictions_missing = model.predict(X_test_missing)
    assert len(predictions_missing) == 10
