"""Test schema validation with actionable error messages."""

import numpy as np
import pandas as pd
import pytest

from glassalpha.data.schema import get_schema_summary, validate_config_schema, validate_data_quality


def test_valid_schema_passes():
    """Test that valid schema configuration passes validation."""
    # Create valid dataset
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "age": np.random.randint(18, 80, 100),
            "income": np.random.randint(20000, 100000, 100),
            "target": np.random.randint(0, 2, 100),
        },
    )

    # Valid configuration
    config = {
        "data": {
            "target_column": "target",
            "protected_attributes": ["age", "income"],
        },
    }

    # Should pass without error
    schema = validate_config_schema(df, config)

    assert schema.target == "target"
    assert "age" in schema.protected_attributes
    assert "income" in schema.protected_attributes
    assert "feature1" in schema.features
    assert "feature2" in schema.features
    assert "target" not in schema.features  # Target should not be in features
    assert "age" not in schema.features  # Protected attrs should not be in features
    assert "income" not in schema.features


def test_missing_target_column_error():
    """Test actionable error message for missing target column."""
    # Dataset without target column
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "age": np.random.randint(18, 80, 50),
        },
    )

    # Config references missing target
    config = {
        "data": {
            "target_column": "missing_target",
            "protected_attributes": ["age"],
        },
    }

    with pytest.raises(ValueError) as exc_info:
        validate_config_schema(df, config)

    error_msg = str(exc_info.value)
    assert "Missing target column: 'missing_target'" in error_msg
    assert "Available columns:" in error_msg
    assert "Dataset shape:" in error_msg


def test_missing_protected_attributes_error():
    """Test actionable error message for missing protected attributes."""
    # Dataset without protected attributes
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "target": np.random.randint(0, 2, 50),
        },
    )

    # Config references missing protected attributes
    config = {
        "data": {
            "target_column": "target",
            "protected_attributes": ["missing_age", "missing_gender"],
        },
    }

    with pytest.raises(ValueError) as exc_info:
        validate_config_schema(df, config)

    error_msg = str(exc_info.value)
    assert "Missing protected attribute columns: ['missing_age', 'missing_gender']" in error_msg
    assert "Available columns:" in error_msg


def test_actionable_suggestions_for_similar_columns():
    """Test that validation provides suggestions for similar column names."""
    # Dataset with target-like and protected-like columns
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "outcome": np.random.randint(0, 2, 50),  # target-like
            "customer_age": np.random.randint(18, 80, 50),  # protected-like
            "region": np.random.choice(["North", "South"], 50),  # protected-like
        },
    )

    # Config with wrong column names
    config = {
        "data": {
            "target_column": "target",  # Should be "outcome"
            "protected_attributes": ["age", "gender"],  # Should be "customer_age", "region"
        },
    }

    with pytest.raises(ValueError) as exc_info:
        validate_config_schema(df, config)

    error_msg = str(exc_info.value)
    assert "Missing target column: 'target'" in error_msg
    assert "Available target-like columns:" in error_msg
    assert "outcome" in error_msg
    assert "Missing protected attribute columns:" in error_msg
    assert "Available protected-like columns:" in error_msg
    assert "customer_age" in error_msg


def test_metadata_columns_excluded_from_features():
    """Test that metadata columns are properly excluded from features."""
    # Dataset with metadata columns
    df = pd.DataFrame(
        {
            "customer_id": range(50),  # metadata
            "created_at": pd.date_range("2023-01-01", periods=50),  # metadata
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "age": np.random.randint(18, 80, 50),
            "target": np.random.randint(0, 2, 50),
        },
    )

    config = {
        "data": {
            "target_column": "target",
            "protected_attributes": ["age"],
        },
    }

    schema = validate_config_schema(df, config)

    # Metadata columns should not be in features
    assert "customer_id" not in schema.features
    assert "created_at" not in schema.features

    # Regular features should be included
    assert "feature1" in schema.features
    assert "feature2" in schema.features

    # Target and protected attributes should not be in features
    assert "target" not in schema.features
    assert "age" not in schema.features


def test_data_quality_validation():
    """Test data quality validation warnings."""
    # Dataset with quality issues
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": [1] * 100,  # Constant feature
            "age": [None] * 10 + list(range(90)),  # Missing values in protected attr
            "target": [0] * 95 + [1] * 5,  # Severe class imbalance
        },
    )

    config = {
        "data": {
            "target_column": "target",
            "protected_attributes": ["age"],
        },
    }

    schema = validate_config_schema(df, config)

    # Data quality validation should not raise error, but log warnings
    # We can't easily test logging in unit tests, so just ensure it doesn't crash
    validate_data_quality(df, schema)


def test_schema_summary():
    """Test schema summary generation."""
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "age": np.random.randint(18, 80, 100),
            "target": np.random.choice([0, 1, 2], 100),  # 3 classes
        },
    )

    config = {
        "data": {
            "target_column": "target",
            "protected_attributes": ["age"],
        },
    }

    schema = validate_config_schema(df, config)
    summary = get_schema_summary(df, schema)

    assert summary["dataset_shape"] == (100, 4)
    assert summary["n_features"] == 2  # feature1, feature2
    assert summary["n_protected_attributes"] == 1  # age
    assert summary["target_column"] == "target"
    assert summary["n_classes"] == 3
    assert "target_distribution" in summary
    assert isinstance(summary["target_distribution"], dict)


def test_empty_protected_attributes():
    """Test validation with no protected attributes."""
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "target": np.random.randint(0, 2, 50),
        },
    )

    config = {
        "data": {
            "target_column": "target",
            "protected_attributes": [],  # Empty list
        },
    }

    # Should pass without error
    schema = validate_config_schema(df, config)

    assert schema.target == "target"
    assert len(schema.protected_attributes) == 0
    assert len(schema.features) == 2


def test_multiple_errors_in_single_message():
    """Test that multiple validation errors are combined in single message."""
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
        },
    )

    # Config with multiple missing columns
    config = {
        "data": {
            "target_column": "missing_target",
            "protected_attributes": ["missing_age", "missing_gender"],
        },
    }

    with pytest.raises(ValueError) as exc_info:
        validate_config_schema(df, config)

    error_msg = str(exc_info.value)
    assert "Missing target column: 'missing_target'" in error_msg
    assert "Missing protected attribute columns: ['missing_age', 'missing_gender']" in error_msg
    assert "Schema validation failed:" in error_msg


if __name__ == "__main__":
    pytest.main([__file__])
