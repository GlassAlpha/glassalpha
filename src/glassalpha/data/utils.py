"""Data utilities for GlassAlpha.

This module provides utilities for data handling and normalization.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ensure_feature_frame(X: pd.DataFrame | np.ndarray | list, feature_names: list[str]) -> pd.DataFrame:
    """Ensure input data is a DataFrame with correct feature names and order.

    This is the single source of truth for feature alignment across the pipeline.
    Use this in:
    - Data loading (after reading CSV)
    - Model fit and predict
    - Explainer input preparation

    Args:
        X: Input data (DataFrame, array, or list)
        feature_names: Expected feature names in correct order

    Returns:
        DataFrame with features in the specified order

    Raises:
        ValueError: If DataFrame has missing features or wrong shape

    """
    if isinstance(X, pd.DataFrame):
        # Enforce order and subset to expected features
        missing_features = set(feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(
                f"DataFrame is missing required features: {sorted(missing_features)}",
            )

        # Return subset in correct order
        return X.loc[:, feature_names]

    # Wrap ndarray or list
    if isinstance(X, (np.ndarray, list)):
        X_arr = np.asarray(X)

        # Check shape
        if X_arr.ndim == 1:
            # Single sample
            if len(X_arr) != len(feature_names):
                raise ValueError(
                    f"Expected {len(feature_names)} features, got {len(X_arr)}",
                )
            return pd.DataFrame([X_arr], columns=feature_names)

        if X_arr.ndim == 2:
            # Multiple samples
            if X_arr.shape[1] != len(feature_names):
                raise ValueError(
                    f"Expected {len(feature_names)} features, got {X_arr.shape[1]}",
                )
            return pd.DataFrame(X_arr, columns=feature_names)

        raise ValueError(f"Expected 1D or 2D array, got {X_arr.ndim}D")

    raise TypeError(f"Expected DataFrame, array, or list; got {type(X)}")
