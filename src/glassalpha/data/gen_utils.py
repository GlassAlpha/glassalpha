"""Data generation utilities with type safety and health checks.

This module provides centralized helpers for categorical construction and
label generation to prevent script-by-script type errors and ensure
data quality consistency across all synthetic datasets.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def safe_categorical_select(
    conditions: list[np.ndarray],
    choices: list[str],
    default: str = "Unknown",
) -> pd.Series:
    """Type-safe wrapper for np.select with consistent string output.

    This function ensures that all categorical selections result in consistent
    string dtypes, preventing the common TypeError from mixed integer/string
    choices in np.select.

    Args:
        conditions: List of boolean arrays for selection conditions
        choices: List of string choices corresponding to conditions
        default: Default value when no conditions are met

    Returns:
        pandas Series with string dtype

    Example:
        >>> age = np.array([25, 45, 65])
        >>> conditions = [age < 30, age < 50, age < 70]
        >>> choices = ["Young", "Middle", "Senior"]
        >>> age_group = safe_categorical_select(conditions, choices, "Elder")
        >>> age_group.dtype
        string

    """
    # Ensure all choices are strings
    str_choices = [str(choice) for choice in choices]

    # Use np.select with string default to ensure consistent dtype
    result = np.select(conditions, str_choices, default=str(default))

    # Convert to pandas Series with explicit string dtype
    return pd.Series(result, dtype="string")


def safe_numeric_binning(
    values: np.ndarray | pd.Series,
    bins: list[float],
    labels: list[str],
    include_lowest: bool = True,
) -> pd.Series:
    """Type-safe numeric binning with consistent categorical output.

    Args:
        values: Numeric values to bin
        bins: Bin edges (must have len(labels) + 1 elements)
        labels: Labels for each bin
        include_lowest: Whether to include the lowest value in the first bin

    Returns:
        pandas Series with categorical dtype

    """
    if len(bins) != len(labels) + 1:
        raise ValueError(f"bins must have {len(labels) + 1} elements for {len(labels)} labels")

    # Use pandas cut for consistent categorical output
    result = pd.cut(
        values,
        bins=bins,
        labels=labels,
        include_lowest=include_lowest,
        ordered=True,
    )

    # Convert to Series to handle categorical operations properly
    result_series = pd.Series(result, dtype="category")

    # Fill any NaN values with "Unknown"
    result_series = result_series.cat.add_categories("Unknown").fillna("Unknown")

    return result_series


def generate_correlated_features(
    n_samples: int,
    feature_specs: list[dict],
    correlation_matrix: np.ndarray | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Generate correlated numeric features with specified distributions.

    Args:
        n_samples: Number of samples to generate
        feature_specs: List of feature specifications, each containing:
            - name: Feature name
            - mean: Mean value (default: 0)
            - std: Standard deviation (default: 1)
            - min_val: Minimum value (optional)
            - max_val: Maximum value (optional)
        correlation_matrix: Correlation matrix (if None, features are independent)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with generated features

    """
    if random_state is not None:
        np.random.seed(random_state)

    n_features = len(feature_specs)

    # Generate base correlated normal variables
    if correlation_matrix is not None:
        if correlation_matrix.shape != (n_features, n_features):
            raise ValueError(f"Correlation matrix shape {correlation_matrix.shape} doesn't match {n_features} features")

        # Generate multivariate normal with specified correlation
        base_features = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=correlation_matrix,
            size=n_samples,
        )
    else:
        # Generate independent normal variables
        base_features = np.random.normal(0, 1, size=(n_samples, n_features))

    # Transform each feature according to its specification
    result_data = {}

    for i, spec in enumerate(feature_specs):
        name = spec["name"]
        mean = spec.get("mean", 0)
        std = spec.get("std", 1)
        min_val = spec.get("min_val")
        max_val = spec.get("max_val")

        # Transform to desired mean and std
        feature_values = base_features[:, i] * std + mean

        # Apply bounds if specified
        if min_val is not None:
            feature_values = np.maximum(feature_values, min_val)
        if max_val is not None:
            feature_values = np.minimum(feature_values, max_val)

        result_data[name] = feature_values

    return pd.DataFrame(result_data)


def assert_dataset_health(
    df: pd.DataFrame,
    expect_cols: list[str],
    target_col: str | None = None,
    protected_cols: list[str] | None = None,
) -> None:
    """Validate dataset health after generation.

    This function performs comprehensive health checks on generated datasets
    to catch common issues early and ensure data quality.

    Args:
        df: Generated dataset DataFrame
        expect_cols: List of columns that must be present
        target_col: Target column name (if specified, will check distribution)
        protected_cols: Protected attribute columns (if specified, will check for nulls)

    Raises:
        ValueError: If dataset fails health checks

    """
    issues = []

    # Check for missing required columns
    missing_cols = [col for col in expect_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Dataset missing required columns: {missing_cols}")

    # Check for null values in existing required columns only
    existing_cols = [col for col in expect_cols if col in df.columns]
    if existing_cols:
        null_cols = df[existing_cols].columns[df[existing_cols].isnull().any()].tolist()
        if null_cols:
            null_counts = {col: df[col].isnull().sum() for col in null_cols}
            issues.append(f"Nulls in required columns: {null_counts}")

    # Check target column distribution
    if target_col and target_col in df.columns:
        target_counts = df[target_col].value_counts()
        n_classes = len(target_counts)

        if n_classes < 2:
            issues.append(f"Target column '{target_col}' has only {n_classes} unique value(s)")

        # Check for severe class imbalance (smallest class < 5% of data)
        min_class_pct = (target_counts.min() / len(df)) * 100
        if min_class_pct < 5.0:
            issues.append(f"Severe class imbalance in '{target_col}': smallest class is {min_class_pct:.1f}%")

    # Check protected attributes for nulls (critical for fairness analysis)
    if protected_cols:
        for col in protected_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    null_pct = (null_count / len(df)) * 100
                    issues.append(f"Protected attribute '{col}' has {null_count} nulls ({null_pct:.1f}%)")

    # Check for constant features (no variation)
    constant_features = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_features.append(col)

    if constant_features:
        issues.append(f"Constant features (no variation): {constant_features}")

    # Check data types
    dtype_issues = []
    for col in df.columns:
        if df[col].dtype == "object":
            # Skip cardinality check for likely ID/date columns
            if col.lower().endswith(("_id", "_date", "id", "date")):
                continue

            # Check if object columns have reasonable cardinality
            n_unique = df[col].nunique()
            if n_unique > len(df) * 0.8:  # More than 80% unique values
                dtype_issues.append(f"Column '{col}' has high cardinality ({n_unique} unique values)")

    if dtype_issues:
        issues.append(f"Data type concerns: {dtype_issues}")

    # Raise error if any issues found
    if issues:
        error_msg = f"Dataset health check failed for {df.shape} dataset:\n"
        error_msg += "\n".join(f"  • {issue}" for issue in issues)
        raise ValueError(error_msg)

    # Log success
    logger.info(
        f"Dataset health check passed: {df.shape[0]} rows, {df.shape[1]} columns, "
        f"{df.select_dtypes(include=[np.number]).shape[1]} numeric, "
        f"{df.select_dtypes(include=['object', 'string', 'category']).shape[1]} categorical",
    )


def validate_feature_correlations(
    df: pd.DataFrame,
    numeric_cols: list[str],
    max_correlation: float = 0.95,
) -> None:
    """Validate that numeric features don't have excessive correlation.

    Args:
        df: Dataset DataFrame
        numeric_cols: List of numeric column names to check
        max_correlation: Maximum allowed absolute correlation

    Raises:
        ValueError: If any feature pairs exceed maximum correlation

    """
    if len(numeric_cols) < 2:
        return  # Nothing to check

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()

    # Find highly correlated pairs (excluding diagonal)
    high_corr_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > max_correlation:
                high_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_val))

    if high_corr_pairs:
        error_msg = f"High correlation detected (>{max_correlation}):\n"
        for col1, col2, corr in high_corr_pairs:
            error_msg += f"  • {col1} <-> {col2}: {corr:.3f}\n"
        error_msg += "Consider removing one feature from each highly correlated pair."
        raise ValueError(error_msg)


def generate_realistic_ids(
    n_samples: int,
    prefix: str = "ID",
    id_length: int = 8,
    random_state: int | None = None,
) -> pd.Series:
    """Generate realistic-looking ID strings.

    Args:
        n_samples: Number of IDs to generate
        prefix: Prefix for each ID
        id_length: Length of numeric part
        random_state: Random seed for reproducibility

    Returns:
        Series of unique ID strings

    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate unique numeric IDs
    max_id = 10**id_length - 1
    if n_samples > max_id:
        raise ValueError(f"Cannot generate {n_samples} unique IDs with length {id_length}")

    # Generate random unique integers
    ids = np.random.choice(max_id, size=n_samples, replace=False)

    # Format as strings with prefix and zero-padding
    id_strings = [f"{prefix}{id_num:0{id_length}d}" for id_num in ids]

    return pd.Series(id_strings, dtype="string")


def add_realistic_noise(
    df: pd.DataFrame,
    numeric_cols: list[str],
    noise_level: float = 0.01,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Add realistic noise to numeric columns to prevent perfect correlations.

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric columns to add noise to
        noise_level: Noise level as fraction of column standard deviation
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with noise added to specified columns

    """
    if random_state is not None:
        np.random.seed(random_state)

    result_df = df.copy()

    for col in numeric_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            col_std = df[col].std()
            # Use column std if > 0, otherwise use column mean as base for noise scale
            noise_scale = col_std if col_std > 0 else abs(df[col].mean()) if df[col].mean() != 0 else 1.0
            noise = np.random.normal(0, noise_scale * noise_level, size=len(df))
            result_df[col] = df[col] + noise

    return result_df
