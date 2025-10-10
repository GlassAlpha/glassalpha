"""Test data generation utilities and health checks."""

import numpy as np
import pandas as pd
import pytest

from glassalpha.data.gen_utils import (
    add_realistic_noise,
    assert_dataset_health,
    generate_correlated_features,
    generate_realistic_ids,
    safe_categorical_select,
    safe_numeric_binning,
    validate_feature_correlations,
)


def test_safe_categorical_select():
    """Test safe categorical selection with consistent string output."""
    # Create test data
    age = np.array([20, 35, 50, 65, 80])

    conditions = [
        age < 30,
        (age >= 30) & (age < 40),
        (age >= 40) & (age < 60),
        age >= 60,
    ]
    choices = ["Young", "Adult", "Middle", "Senior"]

    # Test safe categorical selection
    result = safe_categorical_select(conditions, choices, "Unknown")

    # Check output type and values
    assert isinstance(result, pd.Series)
    assert result.dtype == "string"
    assert list(result) == ["Young", "Adult", "Middle", "Senior", "Senior"]


def test_safe_categorical_select_with_mixed_types():
    """Test that safe categorical selection handles mixed input types."""
    values = np.array([1, 2, 3, 4, 5])
    conditions = [values < 2, values < 4, values >= 4]
    choices = ["Low", "Medium", "High"]  # String choices

    result = safe_categorical_select(conditions, choices, "Unknown")

    assert result.dtype == "string"
    assert list(result) == ["Low", "Medium", "Medium", "High", "High"]


def test_safe_numeric_binning():
    """Test safe numeric binning with categorical output."""
    values = np.array([10, 25, 35, 45, 55, 65, 75])
    bins = [0, 30, 50, 70, 100]
    labels = ["Young", "Adult", "Middle", "Senior"]

    result = safe_numeric_binning(values, bins, labels)

    assert isinstance(result, pd.Series)
    assert result.dtype.name == "category"
    assert list(result) == ["Young", "Young", "Adult", "Adult", "Middle", "Middle", "Senior"]


def test_safe_numeric_binning_with_nans():
    """Test that numeric binning handles NaN values."""
    values = np.array([10, np.nan, 35, 45])
    bins = [0, 30, 50, 100]
    labels = ["Young", "Adult", "Senior"]

    result = safe_numeric_binning(values, bins, labels)

    assert "Unknown" in result.cat.categories
    assert result.isna().sum() == 0  # NaNs should be filled with "Unknown"


def test_generate_correlated_features():
    """Test generation of correlated numeric features."""
    feature_specs = [
        {"name": "feature1", "mean": 0, "std": 1},
        {"name": "feature2", "mean": 10, "std": 2},
        {"name": "feature3", "mean": -5, "std": 0.5},
    ]

    # Test without correlation
    df = generate_correlated_features(100, feature_specs, random_state=42)

    assert df.shape == (100, 3)
    assert list(df.columns) == ["feature1", "feature2", "feature3"]

    # Check approximate means and stds
    assert abs(df["feature1"].mean() - 0) < 0.5
    assert abs(df["feature2"].mean() - 10) < 1.0
    assert abs(df["feature3"].mean() - (-5)) < 0.5


def test_generate_correlated_features_with_correlation():
    """Test generation of correlated features with specified correlation matrix."""
    feature_specs = [
        {"name": "x", "mean": 0, "std": 1},
        {"name": "y", "mean": 0, "std": 1},
    ]

    # High positive correlation
    correlation_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])

    df = generate_correlated_features(1000, feature_specs, correlation_matrix, random_state=42)

    # Check that correlation is approximately as specified
    actual_corr = df.corr().iloc[0, 1]
    assert abs(actual_corr - 0.8) < 0.1  # Allow some tolerance


def test_assert_dataset_health_passes():
    """Test that dataset health check passes for valid dataset."""
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "age": np.random.randint(18, 80, 100),
            "target": np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        },
    )

    required_cols = ["feature1", "feature2", "age", "target"]

    # Should pass without error
    assert_dataset_health(df, required_cols, target_col="target", protected_cols=["age"])


def test_assert_dataset_health_missing_columns():
    """Test that dataset health check fails for missing columns."""
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "target": np.random.choice([0, 1], 50),
        },
    )

    required_cols = ["feature1", "feature2", "target"]  # feature2 is missing

    with pytest.raises(ValueError) as exc_info:
        assert_dataset_health(df, required_cols)

    error_msg = str(exc_info.value)
    assert "Dataset missing required columns: ['feature2']" in error_msg


def test_assert_dataset_health_null_values():
    """Test that dataset health check detects null values in critical columns."""
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "age": [None] * 10 + list(range(40)),  # 10 null values
            "target": np.random.choice([0, 1], 50),
        },
    )

    required_cols = ["feature1", "age", "target"]

    with pytest.raises(ValueError) as exc_info:
        assert_dataset_health(df, required_cols, protected_cols=["age"])

    error_msg = str(exc_info.value)
    assert "Protected attribute 'age' has 10 nulls" in error_msg


def test_assert_dataset_health_class_imbalance():
    """Test that dataset health check detects severe class imbalance."""
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "target": [0] * 98 + [1] * 2,  # Severe imbalance: 2% minority class
        },
    )

    required_cols = ["feature1", "target"]

    with pytest.raises(ValueError) as exc_info:
        assert_dataset_health(df, required_cols, target_col="target")

    error_msg = str(exc_info.value)
    assert "Severe class imbalance" in error_msg
    assert "smallest class is 2.0%" in error_msg


def test_assert_dataset_health_constant_features():
    """Test that dataset health check detects constant features."""
    df = pd.DataFrame(
        {
            "feature1": [1] * 50,  # Constant feature
            "feature2": np.random.randn(50),
            "target": np.random.choice([0, 1], 50),
        },
    )

    required_cols = ["feature1", "feature2", "target"]

    with pytest.raises(ValueError) as exc_info:
        assert_dataset_health(df, required_cols)

    error_msg = str(exc_info.value)
    assert "Constant features (no variation): ['feature1']" in error_msg


def test_validate_feature_correlations():
    """Test feature correlation validation."""
    # Create highly correlated features
    x = np.random.randn(100)
    df = pd.DataFrame(
        {
            "feature1": x,
            "feature2": x + np.random.randn(100) * 0.01,  # Almost identical to feature1
            "feature3": np.random.randn(100),  # Independent
        },
    )

    numeric_cols = ["feature1", "feature2", "feature3"]

    with pytest.raises(ValueError) as exc_info:
        validate_feature_correlations(df, numeric_cols, max_correlation=0.95)

    error_msg = str(exc_info.value)
    assert "High correlation detected" in error_msg
    assert "feature1 <-> feature2" in error_msg


def test_validate_feature_correlations_passes():
    """Test that correlation validation passes for uncorrelated features."""
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
        },
    )

    numeric_cols = ["feature1", "feature2", "feature3"]

    # Should pass without error
    validate_feature_correlations(df, numeric_cols, max_correlation=0.95)


def test_generate_realistic_ids():
    """Test realistic ID generation."""
    ids = generate_realistic_ids(100, prefix="CUST", id_length=6, random_state=42)

    assert len(ids) == 100
    assert ids.dtype == "string"
    assert all(id_str.startswith("CUST") for id_str in ids)
    assert all(len(id_str) == 10 for id_str in ids)  # CUST + 6 digits
    assert len(set(ids)) == 100  # All unique


def test_generate_realistic_ids_too_many():
    """Test that ID generation fails when requesting too many unique IDs."""
    with pytest.raises(ValueError) as exc_info:
        generate_realistic_ids(1000, id_length=2)  # Only 100 possible 2-digit IDs

    error_msg = str(exc_info.value)
    assert "Cannot generate 1000 unique IDs with length 2" in error_msg


def test_add_realistic_noise():
    """Test adding realistic noise to numeric columns."""
    df = pd.DataFrame(
        {
            "feature1": [1.0] * 100,  # Constant values
            "feature2": np.arange(100),
            "categorical": ["A"] * 100,
        },
    )

    numeric_cols = ["feature1", "feature2"]

    noisy_df = add_realistic_noise(df, numeric_cols, noise_level=0.1, random_state=42)

    # Original constant feature should now have variation
    assert noisy_df["feature1"].std() > 0

    # Feature2 should be similar but not identical
    assert not np.array_equal(df["feature2"], noisy_df["feature2"])
    assert np.corrcoef(df["feature2"], noisy_df["feature2"])[0, 1] > 0.9

    # Categorical column should be unchanged
    assert (df["categorical"] == noisy_df["categorical"]).all()


def test_data_generation_integration():
    """Integration test using multiple utilities together."""
    # Generate base features
    feature_specs = [
        {"name": "income", "mean": 50000, "std": 15000, "min_val": 20000, "max_val": 150000},
        {"name": "age", "mean": 40, "std": 12, "min_val": 18, "max_val": 80},
        {"name": "score", "mean": 650, "std": 100, "min_val": 300, "max_val": 850},
    ]

    df = generate_correlated_features(1000, feature_specs, random_state=42)

    # Add categorical features using safe selection
    age_conditions = [df["age"] < 30, df["age"] < 50, df["age"] >= 50]
    age_choices = ["Young", "Middle", "Senior"]
    df["age_group"] = safe_categorical_select(age_conditions, age_choices)

    # Add income bins
    df["income_bracket"] = safe_numeric_binning(
        df["income"],
        bins=[0, 35000, 65000, 100000, 200000],
        labels=["Low", "Medium", "High", "Very_High"],
    )

    # Add target variable
    df["target"] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])

    # Add realistic IDs
    df["customer_id"] = generate_realistic_ids(len(df), "CUST", 8, random_state=42)

    # Add noise to numeric features
    df = add_realistic_noise(df, ["income", "score"], noise_level=0.02, random_state=42)

    # Validate the complete dataset
    required_cols = ["income", "age", "score", "age_group", "income_bracket", "target", "customer_id"]

    assert_dataset_health(
        df,
        expect_cols=required_cols,
        target_col="target",
        protected_cols=["age_group", "income_bracket"],
    )

    # Check data types
    assert df["age_group"].dtype == "string"
    assert df["income_bracket"].dtype.name == "category"
    assert df["customer_id"].dtype == "string"

    # Check realistic ranges (accounting for noise variation)
    # Income bounds are specified in feature generation, but noise can cause slight violations
    assert df["income"].min() >= 19000  # Allow some tolerance for noise
    assert df["income"].max() <= 160000  # Allow some tolerance for noise
    assert df["age"].min() >= 18
    assert df["age"].max() <= 80


if __name__ == "__main__":
    pytest.main([__file__])
