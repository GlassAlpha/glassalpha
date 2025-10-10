"""Shared preprocessing utilities for GlassAlpha.

This module provides common preprocessing functions used across
the audit pipeline and CLI commands to avoid code duplication.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_auto(X: pd.DataFrame) -> pd.DataFrame:
    """Auto preprocessing for categorical features.

    Handles categorical features (like German Credit strings "< 0 DM") with OneHotEncoder
    to prevent ValueError: could not convert string to float during training.

    Args:
        X: Raw features DataFrame

    Returns:
        Processed features DataFrame suitable for training

    """
    from sklearn.compose import ColumnTransformer  # noqa: PLC0415
    from sklearn.preprocessing import OneHotEncoder  # noqa: PLC0415

    # Log at appropriate level based on context
    # In development/testing, this is INFO (expected behavior)
    # In strict mode audits, this should be WARNING (if it even runs)
    logger.info(
        "Using AUTO preprocessing (development mode). "
        "This is fine for experimentation and testing. "
        "For production/regulatory audits: "
        "(1) Save preprocessing with scripts/create_preprocessing_artifacts.py, "
        "(2) Use 'preprocessing: artifact: path/to/artifact.joblib' in config, "
        "(3) Run with --strict flag for validation.",
    )

    # Identify categorical and numeric columns
    categorical_cols = list(X.select_dtypes(include=["object"]).columns)
    numeric_cols = list(X.select_dtypes(exclude=["object"]).columns)

    logger.debug(f"Categorical columns: {categorical_cols}")
    logger.debug(f"Numeric columns: {numeric_cols}")

    if not categorical_cols:
        # No categorical columns, return as-is
        logger.debug("No categorical columns detected, returning original DataFrame")
        return X

    # Build ColumnTransformer with OneHotEncoder for categorical features
    transformers = []

    if categorical_cols:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None),
                categorical_cols,
            ),
        )

    if numeric_cols:
        transformers.append(
            (
                "numeric",
                "passthrough",  # Pass numeric columns through unchanged
                numeric_cols,
            ),
        )

    # Create and fit the ColumnTransformer
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    try:
        # Fit and transform the data
        X_transformed = preprocessor.fit_transform(X)

        # Get feature names after transformation
        feature_names = []

        # Add categorical feature names (one-hot encoded)
        if categorical_cols:
            cat_transformer = preprocessor.named_transformers_["categorical"]
            if hasattr(cat_transformer, "get_feature_names_out"):
                cat_features = cat_transformer.get_feature_names_out(categorical_cols)
            else:
                # Fallback for older sklearn versions
                cat_features = []
                for i, col in enumerate(categorical_cols):
                    unique_vals = cat_transformer.categories_[i]
                    cat_features.extend([f"{col}_{val}" for val in unique_vals])
            feature_names.extend(cat_features)

        # Add numeric feature names
        if numeric_cols:
            feature_names.extend(numeric_cols)

        # Sanitize feature names for XGBoost compatibility (no [, ], <, >)
        sanitized_feature_names = []
        for name in feature_names:
            # Replace problematic characters with underscores
            sanitized = name.replace("<", "lt").replace(">", "gt").replace("[", "_").replace("]", "_").replace(" ", "_")
            # Ensure no double underscores
            sanitized = "_".join(filter(None, sanitized.split("_")))
            sanitized_feature_names.append(sanitized)

        # Convert back to DataFrame with sanitized feature names
        X_processed = pd.DataFrame(X_transformed, columns=sanitized_feature_names, index=X.index)

        logger.info(f"Preprocessed {len(categorical_cols)} categorical columns with OneHotEncoder")
        logger.info(f"Final feature count: {len(sanitized_feature_names)} (from {len(X.columns)} original)")
        logger.debug(f"Sanitized feature names: {sanitized_feature_names[:5]}...")

        return X_processed

    except Exception as e:
        logger.exception(f"Preprocessing failed: {e}")
        logger.warning("Falling back to simple preprocessing")

        # Fallback: simple label encoding as before
        X_processed = X.copy()
        for col in categorical_cols:
            if X_processed[col].dtype == "object":
                unique_values = X_processed[col].unique()
                value_map = {val: idx for idx, val in enumerate(unique_values)}
                X_processed[col] = X_processed[col].map(value_map)
                logger.debug(f"Label encoded column '{col}': {value_map}")

        return X_processed
