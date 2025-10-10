"""Unit tests for feature utilities."""

import numpy as np
import pandas as pd
import pytest

from glassalpha.data.utils import ensure_feature_frame


def test_ensure_feature_frame_dataframe_order():
    """Test that DataFrame columns are reordered correctly."""
    df = pd.DataFrame({"b": [2, 3], "a": [1, 4]})
    out = ensure_feature_frame(df, ["a", "b"])
    assert list(out.columns) == ["a", "b"]
    assert list(out["a"]) == [1, 4]
    assert list(out["b"]) == [2, 3]


def test_ensure_feature_frame_array_wrap():
    """Test that numpy arrays are wrapped with feature names."""
    out = ensure_feature_frame(np.array([[1, 2]]), ["a", "b"])
    assert list(out.columns) == ["a", "b"]
    assert out.shape == (1, 2)


def test_ensure_feature_frame_array_2d():
    """Test 2D array wrapping with multiple samples."""
    X = np.array([[1, 2], [3, 4]])
    out = ensure_feature_frame(X, ["a", "b"])
    assert list(out.columns) == ["a", "b"]
    assert out.shape == (2, 2)


def test_ensure_feature_frame_wrong_length_raises():
    """Test that wrong number of features raises ValueError."""
    X = np.array([[1, 2, 3]])
    with pytest.raises(ValueError, match="Expected 2 features, got 3"):
        ensure_feature_frame(X, ["a", "b"])


def test_ensure_feature_frame_missing_features_raises():
    """Test that DataFrame with missing features raises ValueError."""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="missing required features"):
        ensure_feature_frame(df, ["a", "b", "c"])


def test_ensure_feature_frame_single_sample():
    """Test 1D array wrapping for single sample."""
    X = np.array([1, 2])
    out = ensure_feature_frame(X, ["a", "b"])
    assert list(out.columns) == ["a", "b"]
    assert out.shape == (1, 2)
