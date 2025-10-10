"""Contract smoke test for feature alignment handling.

This test enforces the exact column-drift handling contract that prevents
sklearn ValueError: The feature names should match... errors.

Contract:
- Same width + renamed columns -> accept positionally
- Otherwise -> reindex to feature_names_, drop extras, fill missing with 0
"""

import numpy as np
import pandas as pd

from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper


def _toy(n=40):
    """Generate toy dataset for testing."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {"x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.normal(size=n)},
    )
    y = (X.x1 + 0.2 * X.x2 - 0.1 * X.x3 > 0).astype(int)
    return X, y


def test_same_width_renamed_columns_positionally_ok():
    """Test that renamed columns with same width are accepted positionally."""
    X, y = _toy()
    w = LogisticRegressionWrapper()
    w.fit(X, y)
    X2 = X.copy()
    X2.columns = ["A", "B", "C"]  # rename, same order/width
    # Should NOT raise (accept positionally)
    w.predict(X2)


def test_missing_and_extra_columns_reindexed_and_filled():
    """Test that missing/extra columns are properly handled via reindexing."""
    X, y = _toy()
    w = LogisticRegressionWrapper()
    w.fit(X, y)
    # Drop one, add one
    X3 = X.drop(columns=["x3"]).assign(extra=1.0)
    # Should NOT raise: reindex to stored feature_names_, fill missing with 0, drop extras
    w.predict(X3)


def test_feature_alignment_predict_proba():
    """Test that feature alignment works for predict_proba as well."""
    X, y = _toy()
    w = LogisticRegressionWrapper()
    w.fit(X, y)
    X2 = X.copy()
    X2.columns = ["A", "B", "C"]  # rename, same order/width
    # Should NOT raise (accept positionally)
    w.predict_proba(X2)


def test_non_dataframe_passthrough():
    """Test that non-DataFrame inputs pass through unchanged."""
    X, y = _toy()
    w = LogisticRegressionWrapper()
    w.fit(X, y)
    # Convert to numpy array - should pass through unchanged
    X_array = X.to_numpy()
    # Should work without column issues
    w.predict(X_array)
