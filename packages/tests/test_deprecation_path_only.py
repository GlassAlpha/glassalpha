"""Tests for clean dataset policy enforcement."""

import tempfile
from pathlib import Path

import pytest

from glassalpha.config.loader import load_config_from_file
from glassalpha.config.schema import DataConfig


class TestDatasetPolicy:
    """Test clean dataset policy: dataset required, custom enables path."""

    def test_path_only_config_validation(self):
        """Test that using path without dataset='custom' triggers validation."""
        # Create a config that only has path, no dataset
        config_dict = {
            "data": {
                "path": "/tmp/test.csv",
                "fetch": "if_missing",
                "offline": False,
                "target_column": "target",
                "feature_columns": ["feature1", "feature2"],
                "protected_attributes": [],
            },
            "model": {
                "type": "xgboost",
            },
            "audit_profile": "tabular_compliance",
            "reproducibility": {
                "random_seed": 42,
            },
        }

        # Should raise validation error for missing dataset
        with pytest.raises(ValueError, match="data.dataset"):
            DataConfig(**config_dict["data"])

    def test_path_with_registry_dataset_fails(self):
        """Test that using path with a registry dataset fails."""
        config_dict = {
            "data": {
                "dataset": "german_credit",
                "path": "/tmp/custom.csv",
                "fetch": "if_missing",
                "offline": False,
                "target_column": "target",
                "feature_columns": ["feature1", "feature2"],
                "protected_attributes": [],
            },
            "model": {
                "type": "xgboost",
            },
            "audit_profile": "tabular_compliance",
            "reproducibility": {
                "random_seed": 42,
            },
        }

        # Should raise validation error - registry datasets don't allow path
        with pytest.raises(ValueError, match="data.path must be omitted"):
            DataConfig(**config_dict["data"])

    def test_dataset_only_no_warning(self):
        """Test that using only dataset doesn't trigger warnings."""
        config_dict = {
            "data": {
                "dataset": "german_credit",
                "fetch": "if_missing",
                "offline": False,
                "target_column": "target",
                "feature_columns": ["feature1", "feature2"],
                "protected_attributes": [],
            },
            "model": {
                "type": "xgboost",
            },
            "audit_profile": "tabular_compliance",
            "reproducibility": {
                "random_seed": 42,
            },
        }

        # Should not raise any validation errors
        data_config = DataConfig(**config_dict["data"])
        assert data_config.dataset == "german_credit"
        assert data_config.path is None

    def test_custom_dataset_with_path_works(self):
        """Test that dataset='custom' with path works correctly."""
        config_dict = {
            "data": {
                "dataset": "custom",
                "path": "/tmp/test.csv",
                "fetch": "if_missing",
                "offline": False,
                "target_column": "target",
                "feature_columns": ["feature1", "feature2"],
                "protected_attributes": [],
            },
        }

        # Should work fine
        data_config = DataConfig(**config_dict["data"])
        assert data_config.dataset == "custom"
        assert data_config.path == "/tmp/test.csv"

    def test_custom_dataset_without_path_fails(self):
        """Test that dataset='custom' requires path."""
        config_dict = {
            "data": {
                "dataset": "custom",
                "fetch": "if_missing",
                "offline": False,
                "target_column": "target",
                "feature_columns": ["feature1", "feature2"],
                "protected_attributes": [],
            },
        }

        # Should fail validation - custom requires path
        with pytest.raises(ValueError, match="data.path.*custom"):
            DataConfig(**config_dict["data"])

    def test_config_loader_path_only_error(self):
        """Test that config loader rejects path-only configs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
data:
  path: /tmp/test.csv
  fetch: if_missing
  offline: false
  target_column: target
  feature_columns: [feature1, feature2]
  protected_attributes: []
model:
  type: xgboost
audit_profile: tabular_compliance
reproducibility:
  random_seed: 42
""")
            config_path = f.name

        try:
            # Should fail validation
            with pytest.raises(ValueError, match="data.dataset"):
                load_config_from_file(config_path)
        finally:
            Path(config_path).unlink()
