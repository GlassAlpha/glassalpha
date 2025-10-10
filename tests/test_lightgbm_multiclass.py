"""Regression test for LightGBM multiclass support."""

import numpy as np
import pandas as pd
import pytest

try:
    import lightgbm

    from glassalpha.models.tabular.lightgbm import LightGBMWrapper

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LightGBMWrapper = None  # type: ignore

pytestmark = pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")


def test_lightgbm_multiclass_auto_num_class():
    """Test that LightGBM automatically sets num_class for multiclass problems.

    This is a regression test for the bug where multiclass training would fail
    with: "Number of classes should be specified and greater than 1 for multiclass training"
    """
    # Create 3-class dataset
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature_0": np.random.randn(100),
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
        },
    )
    y = np.array([0, 1, 2] * 33 + [0])  # 3 classes, 100 samples

    # This should work without explicitly setting num_class
    model = LightGBMWrapper()
    model.fit(X, y, random_state=42)

    # Verify predictions work
    predictions = model.predict(X[:10])
    assert len(predictions) == 10
    assert all(pred in [0, 1, 2] for pred in predictions)

    # Verify probabilities have correct shape
    probabilities = model.predict_proba(X[:10])
    assert probabilities.shape == (10, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_lightgbm_binary_classification():
    """Test that binary classification still works correctly."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature_0": np.random.randn(100),
            "feature_1": np.random.randn(100),
        },
    )
    y = np.array([0, 1] * 50)

    model = LightGBMWrapper()
    model.fit(X, y, random_state=42)

    predictions = model.predict(X[:10])
    assert len(predictions) == 10
    assert all(pred in [0, 1] for pred in predictions)

    probabilities = model.predict_proba(X[:10])
    assert probabilities.shape == (10, 2)


def test_lightgbm_four_class_classification():
    """Test LightGBM with 4 classes to ensure num_class parameter scales correctly."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature_0": np.random.randn(120),
            "feature_1": np.random.randn(120),
            "feature_2": np.random.randn(120),
            "feature_3": np.random.randn(120),
        },
    )
    y = np.array([0, 1, 2, 3] * 30)  # 4 classes, 120 samples

    model = LightGBMWrapper()
    model.fit(X, y, random_state=42)

    # Verify predictions work with 4 classes
    predictions = model.predict(X[:20])
    assert len(predictions) == 20
    assert all(pred in [0, 1, 2, 3] for pred in predictions)

    # Verify probabilities have correct shape for 4 classes
    probabilities = model.predict_proba(X[:20])
    assert probabilities.shape == (20, 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_lightgbm_multiclass_save_load():
    """Test that multiclass LightGBM models can be saved and loaded correctly."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature_0": np.random.randn(100),
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
        },
    )
    y = np.array([0, 1, 2] * 33 + [0])

    # Train and save
    model = LightGBMWrapper()
    model.fit(X, y, random_state=42)

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        model.save(f.name)

        # Load and verify
        loaded_model = LightGBMWrapper()
        loaded_model.load(f.name)

        # Compare predictions
        original_pred = model.predict(X[:10])
        loaded_pred = loaded_model.predict(X[:10])

        assert np.array_equal(original_pred, loaded_pred)

        # Compare probabilities
        original_proba = model.predict_proba(X[:10])
        loaded_proba = loaded_model.predict_proba(X[:10])

        assert np.allclose(original_proba, loaded_proba, atol=1e-6)
