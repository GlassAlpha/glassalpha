"""Test XGBoost wrapper coercion from multi:softmax to multi:softprob."""

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from glassalpha.models.tabular.xgboost import XGBoostWrapper


def test_softmax_coerced_to_softprob_with_require_proba():
    """Test that multi:softmax is coerced to multi:softprob when require_proba=True."""
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

    # Capture log messages
    with patch("glassalpha.models.tabular.xgboost.logger") as mock_logger:
        # Fit with multi:softmax and require_proba=True (default)
        wrapper.fit(X, y, objective="multi:softmax", n_estimators=5)

        # Verify warning was logged
        mock_logger.warning.assert_called_with(
            "Coercing multi:softmax to multi:softprob for audit compatibility (predict_proba required)",
        )

    assert wrapper._is_fitted
    assert wrapper.n_classes_ == 3

    # Test that probabilities work (proving softprob was used)
    proba = wrapper.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    # All probabilities should be in [0, 1]
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)


def test_softmax_not_coerced_with_require_proba_false():
    """Test that multi:softmax is not coerced when require_proba=False."""
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

    # Capture log messages
    with patch("glassalpha.models.tabular.xgboost.logger") as mock_logger:
        # Fit with multi:softmax and require_proba=False
        wrapper.fit(X, y, objective="multi:softmax", require_proba=False, n_estimators=5)

        # Verify no coercion warning was logged
        mock_logger.warning.assert_not_called()

    assert wrapper._is_fitted
    assert wrapper.n_classes_ == 3

    # Predictions should still work
    preds = wrapper.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1, 2})


def test_softprob_not_coerced():
    """Test that multi:softprob is not coerced (already correct)."""
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

    # Capture log messages
    with patch("glassalpha.models.tabular.xgboost.logger") as mock_logger:
        # Fit with multi:softprob - should not be coerced
        wrapper.fit(X, y, objective="multi:softprob", n_estimators=5)

        # Verify no coercion warning was logged
        mock_logger.warning.assert_not_called()

    assert wrapper._is_fitted
    assert wrapper.n_classes_ == 3

    # Test probabilities
    proba = wrapper.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_binary_objective_not_coerced():
    """Test that binary objectives are not coerced."""
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

    # Capture log messages
    with patch("glassalpha.models.tabular.xgboost.logger") as mock_logger:
        # Fit with binary:logistic - should not be coerced
        wrapper.fit(X, y, objective="binary:logistic", n_estimators=5)

        # Verify no coercion warning was logged
        mock_logger.warning.assert_not_called()

    assert wrapper._is_fitted
    assert wrapper.n_classes_ == 2

    # Test probabilities - should be shape (n, 2) for binary
    proba = wrapper.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_proba_shape_consistency():
    """Test that predict_proba always returns (n, k) shape."""
    # Test binary case
    np.random.seed(42)
    X_binary = pd.DataFrame(
        {
            "feature1": np.random.randn(30),
            "feature2": np.random.randn(30),
        },
    )
    y_binary = np.random.choice([0, 1], size=30)

    wrapper_binary = XGBoostWrapper()
    wrapper_binary.fit(X_binary, y_binary, n_estimators=5)

    proba_binary = wrapper_binary.predict_proba(X_binary)
    assert proba_binary.shape == (30, 2)  # Always (n, 2) for binary
    assert np.allclose(proba_binary.sum(axis=1), 1.0, atol=1e-6)

    # Test multi-class case
    X_multi = pd.DataFrame(
        {
            "feature1": np.random.randn(30),
            "feature2": np.random.randn(30),
        },
    )
    y_multi = np.random.choice([0, 1, 2, 3], size=30)

    wrapper_multi = XGBoostWrapper()
    wrapper_multi.fit(X_multi, y_multi, n_estimators=5)

    proba_multi = wrapper_multi.predict_proba(X_multi)
    assert proba_multi.shape == (30, 4)  # (n, k) for k classes
    assert np.allclose(proba_multi.sum(axis=1), 1.0, atol=1e-6)


def test_coercion_logged_only_once():
    """Test that coercion warning is logged only once per fit."""
    # Create 3-class dataset
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
        },
    )
    y = np.random.choice([0, 1, 2], size=50)

    wrapper = XGBoostWrapper()

    # Use real logger to test actual behavior
    logger = logging.getLogger("glassalpha.models.tabular.xgboost")

    with patch.object(logger, "warning") as mock_warning:
        # Fit with multi:softmax
        wrapper.fit(X, y, objective="multi:softmax", n_estimators=5)

        # Should be called exactly once
        assert mock_warning.call_count == 1
        mock_warning.assert_called_with(
            "Coercing multi:softmax to multi:softprob for audit compatibility (predict_proba required)",
        )


if __name__ == "__main__":
    pytest.main([__file__])
