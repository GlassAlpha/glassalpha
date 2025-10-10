"""Feature alignment utilities for GlassAlpha.

This module provides utilities for aligning features between training and prediction,
handling column renaming and missing columns gracefully.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Align features to match expected column names and order.

    This function handles two common scenarios:
    1. Column renaming: Same columns but different names/order
    2. Missing columns: Fewer columns than expected, fill with zeros

    Args:
        X: Input DataFrame to align
        feature_names: Expected feature names in correct order

    Returns:
        Aligned DataFrame with correct column names and all expected features

    Raises:
        ValueError: If input is not a DataFrame

    """
    if not isinstance(X, pd.DataFrame):
        msg = f"Expected DataFrame, got {type(X)}"
        raise ValueError(msg)

    original_columns = list(X.columns)
    expected_columns = list(feature_names)

    # Case 1: Same number of columns, just different names/order
    if len(original_columns) == len(expected_columns):
        if original_columns != expected_columns:
            logger.debug(f"feature_align: renamed {len(original_columns)} columns positionally")
            X_aligned = X.copy()
            X_aligned.columns = expected_columns
            return X_aligned
        # Columns already match
        return X

    # Case 2: Different number of columns - need to reindex
    missing_cols = set(expected_columns) - set(original_columns)
    extra_cols = set(original_columns) - set(expected_columns)

    X_aligned = X.reindex(columns=expected_columns, fill_value=0)

    if missing_cols:
        logger.debug(
            f"feature_align: added {len(missing_cols)} missing columns with fill_value=0: {sorted(missing_cols)}",
        )

    if extra_cols:
        logger.debug(f"feature_align: dropped {len(extra_cols)} extra columns: {sorted(extra_cols)}")

    return X_aligned
