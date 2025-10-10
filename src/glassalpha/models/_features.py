"""Feature alignment utilities for GlassAlpha wrappers.

This module provides centralized feature handling to prevent
column drift errors and ensure consistent behavior across all
model wrappers.
"""

from typing import Any

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def align_features(X: Any, feature_names: list[str] | None) -> Any:  # noqa: ANN401
    """Align DataFrame features to match training feature names.

    Contract compliance: Implements the exact feature drift handling
    specified in wheel-first CI triage.

    Args:
        X: Input features (DataFrame, array, or other)
        feature_names: Expected feature names from training

    Returns:
        Aligned features following contract:
        - Same width + different names -> positional mapping (DataFrame with correct names)
        - Otherwise -> reindex with fill_value=0 for missing, drop extras
        - Non-DataFrame or no feature_names -> pass through unchanged

    """
    if not PANDAS_AVAILABLE or not isinstance(X, pd.DataFrame) or not feature_names:
        return X

    # Same width but different names: accept positionally with correct column names
    if X.shape[1] == len(feature_names) and list(X.columns) != list(feature_names):
        return pd.DataFrame(X.to_numpy(), columns=feature_names, index=X.index)

    # Otherwise: reindex to stored feature_names_, drop extras, fill missing with 0
    return X.reindex(columns=feature_names, fill_value=0)


def extract_feature_names(X: Any) -> list[str] | None:  # noqa: ANN401
    """Extract feature names from input data.

    Args:
        X: Input data (DataFrame or other)

    Returns:
        List of column names if DataFrame, None otherwise

    """
    if PANDAS_AVAILABLE and isinstance(X, pd.DataFrame):
        return list(X.columns)
    return None


def validate_feature_alignment(X: Any, expected_features: list[str] | None) -> bool:  # noqa: ANN401
    """Validate if features are properly aligned.

    Args:
        X: Input features to validate
        expected_features: Expected feature names

    Returns:
        True if features are aligned or validation not applicable

    """
    if not PANDAS_AVAILABLE or not isinstance(X, pd.DataFrame) or not expected_features:
        return True

    # Check if same width (allows positional mapping)
    if X.shape[1] == len(expected_features):
        return True

    # Check if all expected features are present (allows reindexing)
    return set(expected_features).issubset(set(X.columns))
