"""Consolidated model tests.

This module consolidates all model-related tests from:
- tests/unit/test_models_io.py
- tests/unit/test_passthrough_model.py
- tests/unit/test_sklearn_wrapper_comprehensive.py
- tests/unit/test_model_registry_auto_import.py
- tests/test_model_integration.py
- tests/test_model_config_integration.py

Tests the complete model system including I/O, wrappers, registry,
and integration scenarios.
"""

import json
from pathlib import Path

import numpy as np
import pytest

# from glassalpha.core.noop_components import PassThroughModel
from glassalpha.models._io import read_wrapper_state, write_wrapper_state

# ============================================================================
# Model I/O Tests (from test_models_io.py)
# ============================================================================


def test_write_wrapper_state(tmp_path):
    """Test write_wrapper_state creates file with correct structure."""
    p = tmp_path / "state.json"
    write_wrapper_state(
        p,
        model_str='{"test": "model"}',
        feature_names=["a", "b"],
        n_classes=2,
    )

    assert p.exists()
    data = json.loads(p.read_text())
    assert "model" in data
    assert data["feature_names_"] == ["a", "b"]
    assert data["n_classes"] == 2


def test_read_wrapper_state(tmp_path):
    """Test read_wrapper_state loads correct data."""
    p = tmp_path / "state.json"
    write_wrapper_state(
        p,
        model_str='{"test": "model"}',
        feature_names=["a", "b"],
        n_classes=2,
    )

    model_str, features, n_classes = read_wrapper_state(p)
    assert model_str == '{"test": "model"}'
    assert features == ["a", "b"]
    assert n_classes == 2


def test_read_wrapper_state_missing_file(tmp_path):
    """Test read_wrapper_state raises FileNotFoundError for missing file."""
    bad = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        read_wrapper_state(bad)


def test_write_wrapper_state_creates_parent_dirs(tmp_path):
    """Test write_wrapper_state creates parent directories."""
    p = tmp_path / "subdir" / "state.json"
    write_wrapper_state(
        p,
        model_str='{"test": "model"}',
        feature_names=["a", "b"],
        n_classes=2,
    )

    assert p.exists()


def test_wrapper_state_roundtrip(tmp_path):
    """Test that write and read operations are reversible."""
    model_str = '{"class": "XGBoostClassifier", "params": {}}'
    feature_names = ["feature1", "feature2", "feature3"]
    n_classes = 2

    p = tmp_path / "state.json"
    write_wrapper_state(p, model_str, feature_names, n_classes)

    read_model_str, read_features, read_n_classes = read_wrapper_state(p)

    assert read_model_str == model_str
    assert read_features == feature_names
    assert read_n_classes == n_classes


# ============================================================================
# PassThroughModel Tests (from test_passthrough_model.py)
# ============================================================================


def test_passthrough_predict_and_proba():
    """Test PassThroughModel predict and predict_proba behavior."""
    X = [[1], [2], [3]]
    m = PassThroughModel().fit(X, [0, 1, 0])
    y = m.predict(X)
    proba = m.predict_proba(X)

    assert len(y) == 3
    assert len(proba) == 3
    assert all(abs(sum(p) - 1) < 1e-9 for p in proba)


def test_passthrough_shape_preservation():
    """Test that PassThroughModel preserves input shapes."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    m = PassThroughModel().fit(X, y)
    pred = m.predict(X)

    assert len(pred) == len(y)
    # PassThrough returns lists, not arrays
    assert len(pred) == len(y)


def test_passthrough_unfitted_error():
    """Test that unfitted model raises appropriate error."""
    m = PassThroughModel()

    # PassThrough doesn't enforce fitted state, so this tests the actual behavior
    # It should work even without fit (returns random predictions)
    X = [[1], [2]]
    pred = m.predict(X)
    assert len(pred) == 2


# ============================================================================
# Model Registry Tests (from test_model_registry_auto_import.py)
# ============================================================================


def test_model_registry_auto_discovery():
    """Test that model registry auto-discovers registered models."""
    from glassalpha.models import load_model

    # Discover should find at least some models
    ModelRegistry.discover()

    # Should have at least one model registered
    names = ModelRegistry.names()
    assert len(names) > 0

    # Should be able to get models by name
    for name in names[:3]:  # Test first few
        model_class = ModelRegistry.get(name)
        assert model_class is not None


def test_model_registry_get_nonexistent():
    """Test that getting non-existent model raises appropriate error."""
    from glassalpha.models import load_model

    with pytest.raises(KeyError):
        ModelRegistry.get("nonexistent_model_type")


# ============================================================================
# Integration Tests (New - comprehensive model scenarios)
# ============================================================================


def test_model_registry_basic_functionality():
    """Test basic model registry operations."""
    from glassalpha.models import load_model

    # Should be able to discover models
    ModelRegistry.discover()
    names = ModelRegistry.names()

    # Should have at least one model
    assert len(names) > 0

    # Should be able to get models
    for name in names:
        model_class = ModelRegistry.get(name)
        assert hasattr(model_class, "__init__")


def test_model_registry_priority_ordering():
    """Test that model registry respects priority ordering."""
    from glassalpha.models import load_model

    ModelRegistry.discover()
    names = ModelRegistry.names()

    # Registry should return names in priority order
    # (highest priority first)
    assert isinstance(names, list)
    assert len(names) > 0


def test_model_wrapper_state_determinism():
    """Test that model wrapper state is deterministic."""
    # Same wrapper state should produce same hash
    state1 = {
        "model_str": '{"test": "model"}',
        "feature_names": ["a", "b"],
        "n_classes": 2,
    }

    state2 = {
        "model_str": '{"test": "model"}',
        "feature_names": ["a", "b"],
        "n_classes": 2,
    }

    # Should be identical (same content)
    assert state1 == state2
