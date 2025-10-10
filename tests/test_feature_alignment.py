"""Tests for feature alignment utilities."""

import numpy as np
import pandas as pd
import pytest

from glassalpha.utils.features import align_features


def test_positional_rename():
    """Test that positional column renaming works correctly."""
    # Create DataFrame with columns in different order
    X = pd.DataFrame(np.random.randn(10, 3), columns=["c", "a", "b"])
    expected_names = ["a", "b", "c"]

    # Align features
    X_aligned = align_features(X, expected_names)

    # Check that columns are renamed correctly
    assert list(X_aligned.columns) == expected_names

    # Check that data is preserved
    assert X_aligned.shape == (10, 3)


def test_reindex_with_fill():
    """Test that reindexing with missing columns fills with zeros."""
    # Create DataFrame with missing columns
    X = pd.DataFrame(np.random.randn(10, 2), columns=["a", "b"])
    expected_names = ["a", "b", "c", "d"]  # Missing 'c' and 'd'

    # Align features
    X_aligned = align_features(X, expected_names)

    # Check that shape is correct
    assert X_aligned.shape == (10, 4)

    # Check that missing columns are filled with zeros
    assert np.all(X_aligned["c"] == 0)
    assert np.all(X_aligned["d"] == 0)

    # Check that existing columns are preserved
    pd.testing.assert_frame_equal(X_aligned[["a", "b"]], X)


def test_reindex_with_extra_columns():
    """Test that reindexing drops extra columns."""
    # Create DataFrame with extra columns
    X = pd.DataFrame(np.random.randn(10, 4), columns=["a", "b", "c", "d"])
    expected_names = ["a", "b"]  # Only keep first 2

    # Align features
    X_aligned = align_features(X, expected_names)

    # Check that shape is correct
    assert X_aligned.shape == (10, 2)

    # Check that only expected columns remain
    assert list(X_aligned.columns) == expected_names

    # Check that kept columns are preserved
    pd.testing.assert_frame_equal(X_aligned, X[["a", "b"]])


def test_no_change_needed():
    """Test that no alignment is performed when columns already match."""
    X = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
    expected_names = ["a", "b", "c"]

    # Align features (should be no-op)
    X_aligned = align_features(X, expected_names)

    # Should be identical
    pd.testing.assert_frame_equal(X_aligned, X)


def test_invalid_input_type():
    """Test that non-DataFrame input raises ValueError."""
    with pytest.raises(ValueError, match="Expected DataFrame"):
        align_features([1, 2, 3], ["a", "b", "c"])


def test_empty_dataframe():
    """Test alignment with empty DataFrame."""
    X = pd.DataFrame(columns=["a", "b"])
    expected_names = ["a", "b", "c"]

    X_aligned = align_features(X, expected_names)

    assert X_aligned.shape == (0, 3)
    assert list(X_aligned.columns) == expected_names
    assert np.all(X_aligned["c"] == 0)


def test_complex_alignment_scenario():
    """Test a complex scenario with renaming and missing columns."""
    # Create DataFrame with mixed scenario
    X = pd.DataFrame(np.random.randn(10, 3), columns=["z", "a", "b"])
    expected_names = ["a", "b", "c", "d"]  # Different order, missing columns

    X_aligned = align_features(X, expected_names)

    # Check final shape
    assert X_aligned.shape == (10, 4)

    # Check that columns are in correct order
    assert list(X_aligned.columns) == expected_names

    # Check that existing data is preserved in correct positions
    pd.testing.assert_frame_equal(X_aligned[["a", "b"]], X[["a", "b"]])

    # Check that missing columns are zeros
    assert np.all(X_aligned["c"] == 0)
    assert np.all(X_aligned["d"] == 0)
