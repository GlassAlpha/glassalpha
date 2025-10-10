"""Test XGBoost wrapper validation for multi-class objectives."""

import numpy as np
import pandas as pd
import pytest

from glassalpha.models.tabular.xgboost import XGBoostWrapper


def test_binary_objective_with_multiclass_data_raises_error():
    """Test that binary objective with >2 classes raises clear ValueError."""
    # Create 4-class dataset
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
        },
    )
    y = np.random.choice([0, 1, 2, 3], size=100)  # 4 classes

    wrapper = XGBoostWrapper()

    # Try to fit with binary objective - should fail
    with pytest.raises(ValueError) as exc_info:
        wrapper.fit(X, y, objective="binary:logistic", n_estimators=10)

    error_msg = str(exc_info.value)
    assert "Binary objective 'binary:logistic' incompatible with 4 classes" in error_msg


def test_multiclass_objective_with_wrong_num_class_raises_error():
    """Test that multi-class objective with wrong num_class raises clear ValueError."""
    # Create 4-class dataset
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
        },
    )
    y = np.random.choice([0, 1, 2, 3], size=100)  # 4 classes

    wrapper = XGBoostWrapper()

    # Try to fit with wrong num_class - should fail
    with pytest.raises(ValueError) as exc_info:
        wrapper.fit(X, y, objective="multi:softprob", num_class=3, n_estimators=10)

    error_msg = str(exc_info.value)
    assert "num_class=3 does not match observed classes=4" in error_msg


def test_multiclass_objective_with_correct_num_class_succeeds():
    """Test that multi-class objective with correct num_class succeeds."""
    # Create 3-class dataset
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
        },
    )
    y = np.random.choice([0, 1, 2], size=50)  # 3 classes

    wrapper = XGBoostWrapper()

    # This should succeed
    wrapper.fit(X, y, objective="multi:softprob", num_class=3, n_estimators=5)

    assert wrapper._is_fitted
    assert wrapper.n_classes_ == 3

    # Test predictions
    preds = wrapper.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1, 2})

    # Test probabilities
    proba = wrapper.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_binary_objective_with_binary_data_succeeds():
    """Test that binary objective with 2 classes succeeds."""
    # Create binary dataset
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
        },
    )
    y = np.random.choice([0, 1], size=50)  # 2 classes

    wrapper = XGBoostWrapper()

    # This should succeed
    wrapper.fit(X, y, objective="binary:logistic", n_estimators=5)

    assert wrapper._is_fitted
    assert wrapper.n_classes_ == 2

    # Test predictions
    preds = wrapper.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})

    # Test probabilities - should be shape (n, 2) even for binary
    proba = wrapper.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_auto_inferred_objective_multiclass():
    """Test that objective is auto-inferred correctly for multi-class."""
    # Create 4-class dataset
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
        },
    )
    y = np.random.choice([0, 1, 2, 3], size=50)  # 4 classes

    wrapper = XGBoostWrapper()

    # Don't specify objective - should auto-infer multi:softprob
    wrapper.fit(X, y, n_estimators=5)

    assert wrapper._is_fitted
    assert wrapper.n_classes_ == 4

    # Test probabilities
    proba = wrapper.predict_proba(X)
    assert proba.shape == (len(X), 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_auto_inferred_objective_binary():
    """Test that objective is auto-inferred correctly for binary."""
    # Create binary dataset
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
        },
    )
    y = np.random.choice([0, 1], size=50)  # 2 classes

    wrapper = XGBoostWrapper()

    # Don't specify objective - should auto-infer binary:logistic
    wrapper.fit(X, y, n_estimators=5)

    assert wrapper._is_fitted
    assert wrapper.n_classes_ == 2

    # Test probabilities - should be shape (n, 2) even for binary
    proba = wrapper.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
