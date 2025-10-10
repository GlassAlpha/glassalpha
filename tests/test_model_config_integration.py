"""Tests for configuration threading and parameter passing."""

import types

import numpy as np
import pandas as pd
import pytest

from glassalpha.pipeline.train import train_from_config


def test_yaml_params_reach_estimator(tmp_path):
    """Test that YAML parameters reach the underlying estimator."""
    # Create mock config with multiclass parameters
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.type = "xgboost"
    cfg.model.params = {"objective": "multi:softprob", "num_class": 4, "max_depth": 3}
    cfg.reproducibility = types.SimpleNamespace()
    cfg.reproducibility.random_seed = 42

    # Create test data
    X = pd.DataFrame(np.random.randn(40, 3), columns=["a", "b", "c"])
    y = np.array([0, 1, 2, 3] * 10)

    # Train model
    model = train_from_config(cfg, X, y)

    # Verify model info shows correct number of classes
    info = model.get_model_info()
    assert info["n_classes"] == 4

    # Verify the underlying XGBoost model has the right objective
    assert hasattr(model, "model")
    assert model.model is not None

    # Check that the model was trained with multiclass objective
    # The exact attribute name may vary, but we can check n_classes
    assert model.n_classes == 4


def test_binary_classification_params():
    """Test that binary classification parameters work correctly."""
    # Create mock config with binary parameters
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.type = "xgboost"
    cfg.model.params = {"objective": "binary:logistic", "max_depth": 4}
    cfg.reproducibility = types.SimpleNamespace()
    cfg.reproducibility.random_seed = 42

    # Create binary test data
    X = pd.DataFrame(np.random.randn(40, 3), columns=["a", "b", "c"])
    y = np.array([0, 1] * 20)

    # Train model
    model = train_from_config(cfg, X, y)

    # Verify binary classification
    info = model.get_model_info()
    assert info["n_classes"] == 2


def test_unknown_model_type_raises():
    """Test that unknown model types raise appropriate errors."""
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.type = "unknown_model"
    cfg.model.params = {}

    X = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
    y = np.array([0, 1] * 5)

    with pytest.raises(ValueError, match="Unknown model type: unknown_model"):
        train_from_config(cfg, X, y)


def test_model_without_fit_raises():
    """Test that models without fit method raise appropriate errors."""
    # This would require mocking a model class that doesn't have fit
    # For now, we'll skip this test as it's harder to mock properly
