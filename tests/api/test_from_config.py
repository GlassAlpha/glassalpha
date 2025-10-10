"""Tests for from_config() entry point."""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import glassalpha as ga
from glassalpha.exceptions import DataHashMismatchError, ResultIDMismatchError


def test_from_config_basic(tmp_path):
    """Test from_config with basic YAML config."""
    from sklearn.linear_model import LogisticRegression

    # Generate data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 3), columns=["f1", "f2", "f3"])
    y = pd.Series((X["f1"] > 0).astype(int))

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Save model and data
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    X.to_csv(tmp_path / "X.csv", index=False)
    y.to_csv(tmp_path / "y.csv", index=False, header=False)

    # Create config
    config = {
        "model": {"path": "model.pkl"},
        "data": {
            "X_path": "X.csv",
            "y_path": "y.csv",
        },
        "audit": {
            "random_seed": 42,
            "explain": False,
            "calibration": True,
        },
    }

    config_path = tmp_path / "audit.yaml"
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run audit from config
    result = ga.audit.from_config(config_path)

    # Verify result
    assert result.performance["accuracy"] > 0.5
    assert result.manifest["model_type"] == "logistic_regression"
    assert result.manifest["n_features"] == 3


def test_from_config_with_hash_validation(tmp_path):
    """Test from_config with data hash validation."""
    from sklearn.linear_model import LogisticRegression

    from glassalpha.core.canonicalization import hash_data_for_manifest

    # Generate data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 3), columns=["f1", "f2", "f3"])
    y = pd.Series((X["f1"] > 0).astype(int))

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Save model and data
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    X.to_csv(tmp_path / "X.csv", index=False)
    y.to_csv(tmp_path / "y.csv", index=False, header=False)

    # Compute hashes after reloading (to match what from_config will see)
    X_reloaded = pd.read_csv(tmp_path / "X.csv")
    y_reloaded = pd.read_csv(tmp_path / "y.csv", header=None).iloc[:, 0]
    X_hash = hash_data_for_manifest(X_reloaded)
    y_hash = hash_data_for_manifest(y_reloaded)

    # Create config with hashes
    config = {
        "model": {"path": "model.pkl"},
        "data": {
            "X_path": "X.csv",
            "y_path": "y.csv",
            "expected_hashes": {
                "X": X_hash,
                "y": y_hash,
            },
        },
        "audit": {
            "random_seed": 42,
        },
    }

    config_path = tmp_path / "audit.yaml"
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Should succeed with correct hashes
    result = ga.audit.from_config(config_path)
    assert result.performance["accuracy"] > 0


def test_from_config_hash_mismatch(tmp_path):
    """Test from_config fails with wrong data hash."""
    from sklearn.linear_model import LogisticRegression

    # Generate data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 3), columns=["f1", "f2", "f3"])
    y = pd.Series((X["f1"] > 0).astype(int))

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Save model and data
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    X.to_csv(tmp_path / "X.csv", index=False)
    y.to_csv(tmp_path / "y.csv", index=False, header=False)

    # Create config with wrong hash
    config = {
        "model": {"path": "model.pkl"},
        "data": {
            "X_path": "X.csv",
            "y_path": "y.csv",
            "expected_hashes": {
                "X": "sha256:wrong_hash",
            },
        },
        "audit": {
            "random_seed": 42,
        },
    }

    config_path = tmp_path / "audit.yaml"
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Should fail with hash mismatch
    with pytest.raises(DataHashMismatchError) as exc_info:
        ga.audit.from_config(config_path)

    assert "GAE2003" in str(exc_info.value)
    assert "wrong_hash" in str(exc_info.value)


def test_from_config_with_protected_attributes(tmp_path):
    """Test from_config with protected attributes."""
    from sklearn.linear_model import LogisticRegression

    # Generate data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 3), columns=["f1", "f2", "f3"])
    y = pd.Series((X["f1"] > 0).astype(int))
    gender = pd.Series(np.random.randint(0, 2, size=50))

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Save model and data
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    X.to_csv(tmp_path / "X.csv", index=False)
    y.to_csv(tmp_path / "y.csv", index=False, header=False)
    gender.to_csv(tmp_path / "gender.csv", index=False, header=False)

    # Create config
    config = {
        "model": {"path": "model.pkl"},
        "data": {
            "X_path": "X.csv",
            "y_path": "y.csv",
            "protected_attributes": {
                "gender": "gender.csv",
            },
        },
        "audit": {
            "random_seed": 42,
        },
    }

    config_path = tmp_path / "audit.yaml"
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run audit
    result = ga.audit.from_config(config_path)

    # Verify fairness metrics are present
    assert len(result.fairness) > 0
    assert "demographic_parity_max_diff" in result.fairness


def test_from_config_deterministic(tmp_path):
    """Test from_config produces deterministic results."""
    from sklearn.linear_model import LogisticRegression

    # Generate data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 3), columns=["f1", "f2", "f3"])
    y = pd.Series((X["f1"] > 0).astype(int))

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Save model and data
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    X.to_csv(tmp_path / "X.csv", index=False)
    y.to_csv(tmp_path / "y.csv", index=False, header=False)

    # Create config
    config = {
        "model": {"path": "model.pkl"},
        "data": {
            "X_path": "X.csv",
            "y_path": "y.csv",
        },
        "audit": {
            "random_seed": 42,
        },
    }

    config_path = tmp_path / "audit.yaml"
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run twice
    result1 = ga.audit.from_config(config_path)
    result2 = ga.audit.from_config(config_path)

    # Should have same ID
    assert result1.id == result2.id
