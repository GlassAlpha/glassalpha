"""Data schema validation tests.

Tests for DataSchema and TabularDataSchema classes including
validation logic, feature categorization, and constraint checking.
"""

import pytest

from glassalpha.data.base import DataSchema
from glassalpha.data.tabular import TabularDataSchema


class TestDataSchema:
    """Test base DataSchema functionality."""

    def test_data_schema_initialization(self):
        """Test DataSchema initialization."""
        schema = DataSchema(target="target", features=["feature1", "feature2"], sensitive_features=["sensitive_attr"])

        assert schema.target == "target"
        assert schema.features == ["feature1", "feature2"]
        assert schema.sensitive_features == ["sensitive_attr"]

    def test_data_schema_minimal(self):
        """Test DataSchema with minimal required fields."""
        schema = DataSchema(target="target", features=["feature1"])

        assert schema.target == "target"
        assert schema.features == ["feature1"]
        assert schema.sensitive_features is None


class TestTabularDataSchema:
    """Test TabularDataSchema functionality."""

    def test_tabular_schema_initialization(self):
        """Test TabularDataSchema initialization."""
        schema = TabularDataSchema(
            target="target",
            features=["feature1", "feature2"],
            sensitive_features=["sensitive_attr"],
            categorical_features=["feature1"],
            numeric_features=["feature2"],
        )

        assert schema.target == "target"
        assert schema.features == ["feature1", "feature2"]
        assert schema.sensitive_features == ["sensitive_attr"]
        assert schema.categorical_features == ["feature1"]
        assert schema.numeric_features == ["feature2"]

    def test_tabular_schema_target_in_features_validation(self):
        """Test that target cannot be in features list."""
        with pytest.raises(ValueError, match="Target .* cannot be in features list"):
            TabularDataSchema(
                target="target",
                features=["feature1", "target"],  # Target in features - should fail
            )

    def test_tabular_schema_sensitive_features_warning(self):
        """Test warning for sensitive features not in feature/target columns."""
        # Should not raise error but may log warning
        schema = TabularDataSchema(
            target="target",
            features=["feature1", "feature2"],
            sensitive_features=["unknown_feature"],  # Not in features or target
        )

        # Schema should still be created
        assert schema.sensitive_features == ["unknown_feature"]

    def test_tabular_schema_empty_features_validation(self):
        """Test that features list cannot be empty."""
        with pytest.raises(ValueError):
            TabularDataSchema(
                target="target",
                features=[],  # Empty features list
            )
