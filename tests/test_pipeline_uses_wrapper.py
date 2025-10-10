"""Tests to ensure pipeline uses wrapper instead of direct estimator training."""

import types

import numpy as np
import pandas as pd

from glassalpha.pipeline.train import train_from_config


def test_pipeline_trains_via_wrapper():
    """Test that train_from_config works correctly and returns proper model."""
    # Create minimal config
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.type = "xgboost"
    cfg.model.params = {"max_depth": 3}
    cfg.reproducibility = types.SimpleNamespace()
    cfg.reproducibility.random_seed = 42

    # Create test data
    X = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    y = np.array([0, 1] * 10)

    # Call train_from_config
    model = train_from_config(cfg, X, y)

    # Verify model was created successfully
    assert model is not None, "Model should be created"

    # Verify model has required methods for audits
    assert hasattr(model, "predict"), "Model must have predict method"
    assert hasattr(model, "predict_proba"), "Model must have predict_proba method"
    assert hasattr(model, "get_capabilities"), "Model must have get_capabilities method"
    assert hasattr(model, "get_model_info"), "Model must have get_model_info method"

    # Verify model has required capabilities for audits
    caps = model.get_capabilities()
    assert caps.get("supports_proba", False), "Model must support predict_proba for audits"

    # Verify model info is correct
    info = model.get_model_info()
    assert "n_classes" in info, "Model info should include n_classes"
    assert info["n_classes"] == 2, f"Expected 2 classes, got {info['n_classes']}"

    # Verify predictions work
    predictions = model.predict(X)
    assert len(predictions) == len(X), "Predictions should have same length as input"

    # Verify probabilities work
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), 2), f"Expected shape ({len(X)}, 2), got {probabilities.shape}"
