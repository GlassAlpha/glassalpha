"""Tests to ensure multiclass objective and num_class configuration threads correctly."""

import types

import numpy as np
import pandas as pd
import pytest

from glassalpha.pipeline.train import train_from_config


def test_multiclass_objective_and_num_class_threaded():
    """Test that multiclass config reaches estimator and probabilities exist."""
    # Create test data with 4 classes
    X = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
    y = np.array([0, 1, 2, 3] * 12 + [0, 1])

    # Config with explicit multiclass parameters
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.type = "xgboost"
    cfg.model.params = {"objective": "multi:softprob", "num_class": 4, "max_depth": 3}
    cfg.reproducibility = types.SimpleNamespace()
    cfg.reproducibility.random_seed = 42

    # Train model
    model = train_from_config(cfg, X, y)

    # Verify model info shows correct number of classes
    info = model.get_model_info()
    assert info["n_classes"] == 4, f"Expected 4 classes, got {info['n_classes']}"

    # Verify predict_proba returns correct shape
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 4), f"Expected shape ({len(X)}, 4), got {proba.shape}"

    # Verify probabilities sum to 1 (for softprob)
    prob_sums = np.sum(proba, axis=1)
    np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-6)


def test_multiclass_auto_inference():
    """Test that multiclass objective is auto-inferred when not specified."""
    # Create test data with 3 classes
    X = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
    y = np.array([0, 1, 2] * 10)

    # Config without objective (should auto-infer)
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.type = "xgboost"
    cfg.model.params = {"max_depth": 3}
    cfg.reproducibility = types.SimpleNamespace()
    cfg.reproducibility.random_seed = 42

    # Train model
    model = train_from_config(cfg, X, y)

    # Should auto-infer multiclass
    info = model.get_model_info()
    assert info["n_classes"] == 3, f"Expected 3 classes, got {info['n_classes']}"

    # Verify predict_proba works
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 3), f"Expected shape ({len(X)}, 3), got {proba.shape}"


def test_binary_classification_threading():
    """Test that binary classification config threads correctly."""
    # Create binary test data
    X = pd.DataFrame(np.random.randn(40, 3), columns=["a", "b", "c"])
    y = np.array([0, 1] * 20)

    # Config with binary parameters
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.type = "xgboost"
    cfg.model.params = {"objective": "binary:logistic", "max_depth": 4}
    cfg.reproducibility = types.SimpleNamespace()
    cfg.reproducibility.random_seed = 42

    # Train model
    model = train_from_config(cfg, X, y)

    # Verify binary classification
    info = model.get_model_info()
    assert info["n_classes"] == 2, f"Expected 2 classes, got {info['n_classes']}"

    # Verify predict_proba returns (n, 2) shape
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2), f"Expected shape ({len(X)}, 2), got {proba.shape}"


def test_inconsistent_num_class_raises():
    """Test that inconsistent num_class raises clear error."""
    # Create test data with 3 classes
    X = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
    y = np.array([0, 1, 2] * 10)

    # Config with wrong num_class
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.type = "xgboost"
    cfg.model.params = {
        "num_class": 5,  # Wrong! Data has 3 classes
        "max_depth": 3,
    }
    cfg.reproducibility = types.SimpleNamespace()
    cfg.reproducibility.random_seed = 42

    # Should raise ValueError
    with pytest.raises(ValueError, match="num_class.*does not match observed classes"):
        train_from_config(cfg, X, y)


def test_binary_objective_with_multiclass_raises():
    """Test that binary objective with multiclass data raises clear error."""
    # Create test data with 3 classes
    X = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
    y = np.array([0, 1, 2] * 10)

    # Config with binary objective but multiclass data
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.type = "xgboost"
    cfg.model.params = {
        "objective": "binary:logistic",  # Wrong for 3 classes
        "max_depth": 3,
    }
    cfg.reproducibility = types.SimpleNamespace()
    cfg.reproducibility.random_seed = 42

    # Should raise ValueError
    with pytest.raises(
        ValueError,
        match="Binary objective .* incompatible with .* classes",
    ):
        train_from_config(cfg, X, y)
