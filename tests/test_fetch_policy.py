"""Tests for fetch policy behavior with custom datasets.

Note: These tests use dataset='custom' because fetch policies primarily
apply to custom user-provided files. Registry datasets have their own
fetch logic tested in test_concurrency_fetch.py.
"""

import tempfile
from pathlib import Path

import pytest

from glassalpha.config.schema import DataConfig
from glassalpha.pipeline.audit import AuditPipeline


class TestFetchPolicyCustomDatasets:
    """Test fetch policy behavior for custom datasets."""

    def test_custom_fetch_never_with_existing_file(self):
        """Test fetch=never with custom dataset when file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_file = Path(temp_dir) / "custom_data.csv"
            existing_file.write_text("test,data\n1,2\n")

            config = DataConfig(
                dataset="custom",
                path=str(existing_file),
                fetch="never",
                offline=False,
            )

            pipeline = AuditPipeline.__new__(AuditPipeline)
            pipeline.config = type("Config", (), {"data": config})()

            # Should return existing file without any fetch attempt
            result = pipeline._ensure_dataset_availability(existing_file)
            assert result == existing_file
            assert result.exists()

    def test_custom_fetch_never_with_missing_file(self):
        """Test fetch=never with custom dataset when file is missing."""
        config = DataConfig(
            dataset="custom",
            path="/nonexistent/custom_data.csv",
            fetch="never",
            offline=False,
        )

        pipeline = AuditPipeline.__new__(AuditPipeline)
        pipeline.config = type("Config", (), {"data": config})()

        missing_file = Path("/nonexistent/custom_data.csv")

        # Should raise error for missing custom file
        with pytest.raises(FileNotFoundError, match="Custom data file not found"):
            pipeline._ensure_dataset_availability(missing_file)

    def test_custom_fetch_if_missing_with_existing_file(self):
        """Test fetch=if_missing with custom dataset when file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_file = Path(temp_dir) / "custom_data.csv"
            existing_file.write_text("test,data\n1,2\n")

            config = DataConfig(
                dataset="custom",
                path=str(existing_file),
                fetch="if_missing",
                offline=False,
            )

            pipeline = AuditPipeline.__new__(AuditPipeline)
            pipeline.config = type("Config", (), {"data": config})()

            # Should return existing file without fetching
            result = pipeline._ensure_dataset_availability(existing_file)
            assert result == existing_file
            assert result.exists()

    def test_custom_offline_with_missing_file(self):
        """Test offline=true with custom dataset raises clear error."""
        config = DataConfig(
            dataset="custom",
            path="/nonexistent/custom_data.csv",
            fetch="if_missing",
            offline=True,
        )

        pipeline = AuditPipeline.__new__(AuditPipeline)
        pipeline.config = type("Config", (), {"data": config})()

        missing_file = Path("/nonexistent/custom_data.csv")

        # Should raise error mentioning offline mode
        with pytest.raises(FileNotFoundError, match="offline is true"):
            pipeline._ensure_dataset_availability(missing_file)

    def test_fetch_policy_validation(self):
        """Test that invalid fetch policies are rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="String should match pattern"):
            DataConfig(
                dataset="custom",
                path="/tmp/test.csv",
                fetch="invalid_policy",
                offline=False,
            )

    def test_fetch_policy_defaults(self):
        """Test that fetch policy defaults to 'if_missing'."""
        config = DataConfig(
            dataset="custom",
            path="/tmp/test.csv",
            offline=False,
        )

        # Should default to "if_missing"
        assert config.fetch == "if_missing"

    def test_registry_dataset_forbids_path(self):
        """Test that registry datasets cannot specify custom paths."""
        # This enforces the clean policy: registry datasets use cache paths
        with pytest.raises(ValueError, match="data.path must be omitted"):
            DataConfig(
                dataset="german_credit",  # Registry dataset
                path="/tmp/my_custom_path.csv",  # Not allowed
                fetch="if_missing",
            )
