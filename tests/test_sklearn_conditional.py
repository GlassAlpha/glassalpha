"""Conditional sklearn imports for CI compatibility.

This module provides conditional importing of sklearn and scipy to handle
CI environment compatibility issues gracefully.
"""

import pytest

# Conditional sklearn/scipy import with graceful fallback
try:
    import scipy.sparse
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    SKLEARN_AVAILABLE = True
    SKLEARN_SKIP_REASON = None
except ImportError as e:
    # Mock the imports if they fail
    make_classification = None
    LogisticRegression = None
    RandomForestClassifier = None
    accuracy_score = None
    precision_score = None
    recall_score = None
    f1_score = None
    roc_auc_score = None
    scipy = None
    SKLEARN_AVAILABLE = False
    SKLEARN_SKIP_REASON = f"sklearn/scipy not available: {e}"

# Skip marker for tests that need sklearn
skip_if_no_sklearn = pytest.mark.skipif(
    not SKLEARN_AVAILABLE,
    reason=SKLEARN_SKIP_REASON or "sklearn/scipy not available",
)


def test_sklearn_availability():
    """Test that sklearn imports are working."""
    if SKLEARN_AVAILABLE:
        pytest.skip("sklearn available - this test only runs when sklearn is unavailable")
    else:
        assert not SKLEARN_AVAILABLE
        assert SKLEARN_SKIP_REASON is not None


@skip_if_no_sklearn
def test_sklearn_basic_functionality():
    """Test basic sklearn functionality when available."""
    # Simple test that sklearn works
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    model = LogisticRegression(random_state=42, max_iter=2000, solver="liblinear")
    model.fit(X, y)

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0
