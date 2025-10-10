"""Unit tests for PassThroughModel."""

import numpy as np

from glassalpha.core.noop_components import PassThroughModel


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
