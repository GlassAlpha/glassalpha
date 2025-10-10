"""Tests for offline mode behavior in dataset fetching."""

import tempfile
from pathlib import Path

import pytest

from glassalpha.config.schema import DataConfig
from glassalpha.datasets.registry import REGISTRY, DatasetSpec
from glassalpha.pipeline.audit import AuditPipeline


class TestOfflineMode:
    """Test offline mode behavior."""

    def setup_method(self):
        """Set up test environment."""
        # Clear registry for clean tests
        REGISTRY.clear()

        # Add test dataset
        def mock_fetch():
            return Path("/tmp/test_dataset.csv")

        REGISTRY["test_dataset"] = DatasetSpec(
            key="test_dataset",
            default_relpath="test_dataset.csv",
            fetch_fn=mock_fetch,
            schema_version="v1",
        )

    def teardown_method(self):
        """Clean up after tests."""
        REGISTRY.clear()

    def test_offline_true_with_existing_file(self):
        """Test offline=true when file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_file = Path(temp_dir) / "test_dataset.csv"
            existing_file.write_text("test data")

            config = DataConfig(
                dataset="custom",
                path=str(existing_file),
                fetch="if_missing",
                offline=True,
            )

            pipeline = AuditPipeline.__new__(AuditPipeline)
            pipeline.config = type("Config", (), {"data": config})()

            # Should return existing file without attempting network
            result = pipeline._ensure_dataset_availability(existing_file)
            assert result == existing_file

    def test_offline_true_with_missing_file(self):
        """Test offline=true when file is missing."""
        config = DataConfig(
            dataset="custom",
            path="/nonexistent/test_dataset.csv",
            fetch="if_missing",
            offline=True,
        )

        pipeline = AuditPipeline.__new__(AuditPipeline)
        pipeline.config = type("Config", (), {"data": config})()

        missing_file = Path("/nonexistent/test_dataset.csv")

        # Should raise clear error about offline mode
        with pytest.raises(FileNotFoundError, match="offline is true"):
            pipeline._ensure_dataset_availability(missing_file)

    def test_offline_false_with_missing_file(self):
        """Test offline=false when file is missing (should attempt fetch)."""
        config = DataConfig(
            dataset="custom",
            path="/tmp/missing_test_dataset.csv",
            fetch="if_missing",
            offline=False,
        )

        pipeline = AuditPipeline.__new__(AuditPipeline)
        pipeline.config = type("Config", (), {"data": config})()

        missing_file = Path("/tmp/missing_test_dataset.csv")

        # Should attempt to fetch when offline=false
        with pytest.raises(Exception):  # Will fail due to mock, but should attempt fetch
            pipeline._ensure_dataset_availability(missing_file)

    def test_offline_mode_inheritance(self):
        """Test that offline mode is properly inherited from config."""
        config = DataConfig(
            dataset="custom",
            path="/tmp/test.csv",
            offline=True,
        )

        # Should default fetch to "if_missing" but respect offline=true
        assert config.fetch == "if_missing"
        assert config.offline is True

    def test_offline_mode_error_message_quality(self):
        """Test that offline mode error messages are clear and actionable."""
        config = DataConfig(
            dataset="custom",
            path="/missing/file.csv",
            fetch="if_missing",
            offline=True,
        )

        pipeline = AuditPipeline.__new__(AuditPipeline)
        pipeline.config = type("Config", (), {"data": config})()

        missing_file = Path("/missing/file.csv")

        with pytest.raises(FileNotFoundError) as exc_info:
            pipeline._ensure_dataset_availability(missing_file)

        error_msg = str(exc_info.value)

        # Should mention offline mode
        assert "offline is true" in error_msg

        # Should show the effective path
        assert "/missing/file.csv" in error_msg

        # Should suggest how to fix
        assert "set offline: false" in error_msg
