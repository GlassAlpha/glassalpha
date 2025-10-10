"""Unit tests for model I/O operations."""

import json

import pytest

from glassalpha.models._io import read_wrapper_state, write_wrapper_state


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
    p = tmp_path / "nested" / "dir" / "state.json"
    write_wrapper_state(
        p,
        model_str='{"test": "model"}',
        feature_names=None,
        n_classes=None,
    )

    assert p.exists()
    assert p.parent.exists()
