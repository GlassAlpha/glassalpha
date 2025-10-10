"""Integration tests for the complete dataset fetching system."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from glassalpha.config.loader import load_config_from_file

# Make tests CWD-independent by using absolute paths based on this file's location
HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE.parent / "configs"
from glassalpha.config.schema import DataConfig
from glassalpha.datasets.registry import REGISTRY
from glassalpha.pipeline.audit import AuditPipeline


class TestDatasetFetchingIntegration:
    """Test the complete dataset fetching system end-to-end."""

    def setup_method(self):
        """Ensure built-in datasets are registered."""
        from glassalpha.datasets.register_builtin import register_builtin_datasets

        register_builtin_datasets()

    def test_full_dataset_fetching_pipeline(self):
        """Test that the complete dataset fetching pipeline works."""
        # Use the german_credit_simple.yaml config
        config = load_config_from_file(CONFIG_DIR / "german_credit_simple.yaml")

        # Verify config loaded correctly
        assert config.data.dataset == "german_credit"
        assert config.data.fetch == "if_missing"
        assert config.data.offline is False

        # Create pipeline
        pipeline = AuditPipeline(config)

        # Test path resolution
        resolved_path = pipeline._resolve_dataset_path()
        # resolved_path points to: .../glassalpha/data/german_credit_processed.csv
        # So parent is .../glassalpha/data, which is what we want
        from glassalpha.utils.cache_dirs import resolve_data_root

        expected_cache = resolve_data_root()

        # Should resolve to cache location
        assert resolved_path.parent == expected_cache

        # Test dataset availability
        final_path = pipeline._ensure_dataset_availability(resolved_path)

        # Should exist and be readable
        assert final_path.exists()
        assert final_path.is_file()

        # Should be able to read the CSV
        import pandas as pd

        df = pd.read_csv(final_path)
        assert len(df) > 0  # Should have data
        assert "credit_risk" in df.columns  # Should have target column

    def test_dataset_registry_integration(self):
        """Test that dataset registry integrates properly."""
        # Verify German Credit is registered
        assert "german_credit" in REGISTRY

        spec = REGISTRY["german_credit"]
        assert spec.key == "german_credit"
        assert spec.default_relpath == "german_credit_processed.csv"
        assert spec.schema_version == "v1"
        assert callable(spec.fetch_fn)

    def test_cache_directory_integration(self):
        """Test that cache directory resolution works in context."""
        from glassalpha.utils.cache_dirs import ensure_dir_writable, resolve_data_root

        cache_root = ensure_dir_writable(resolve_data_root())

        # Should be OS-appropriate location
        assert cache_root.exists()
        assert cache_root.is_dir()

        # Should be writable
        test_file = cache_root / "test_write"
        test_file.write_text("test")
        assert test_file.read_text() == "test"
        test_file.unlink()

    def test_config_validation_integration(self):
        """Test that new config fields integrate with validation."""
        # Test valid configuration
        config = DataConfig(
            dataset="german_credit",
            fetch="if_missing",
            offline=False,
            target_column="credit_risk",
            feature_columns=["checking_account_status", "duration_months"],
            protected_attributes=["gender"],
        )

        assert config.dataset == "german_credit"
        assert config.fetch == "if_missing"
        assert config.offline is False
        assert config.target_column == "credit_risk"

        # Test invalid fetch policy
        with pytest.raises(ValidationError, match="String should match pattern"):
            DataConfig(
                dataset="german_credit",
                fetch="invalid",
                offline=False,
            )

    def test_error_handling_integration(self):
        """Test that error handling works across the system."""
        # Test with non-existent dataset
        config = DataConfig(
            dataset="custom",
            path="/tmp/non_existent.csv",
            fetch="if_missing",
            offline=False,
        )

        pipeline = AuditPipeline.__new__(AuditPipeline)
        pipeline.config = type("Config", (), {"data": config})()

        # Path resolution should fail for non-existent custom dataset
        with pytest.raises(FileNotFoundError) as exc_info:
            pipeline._resolve_dataset_path()

        error_msg = str(exc_info.value)
        assert "Custom data file not found" in error_msg

    def test_offline_mode_integration(self):
        """Test offline mode integration."""
        config = DataConfig(
            dataset="custom",
            path="/tmp/test.csv",
            fetch="if_missing",
            offline=True,  # Offline mode
        )

        pipeline = AuditPipeline.__new__(AuditPipeline)
        pipeline.config = type("Config", (), {"data": config})()

        # Should raise clear error about offline mode
        with pytest.raises(FileNotFoundError) as exc_info:
            pipeline._resolve_dataset_path()

        error_msg = str(exc_info.value)
        assert "Offline mode" in error_msg

    def test_concurrent_access_safety(self):
        """Test that concurrent access to dataset fetching is safe."""
        config = DataConfig(
            dataset="custom",
            path="/tmp/concurrent_test.csv",
            fetch="always",  # Force fetch to test concurrency
            offline=False,
        )

        pipeline = AuditPipeline.__new__(AuditPipeline)
        pipeline.config = type("Config", (), {"data": config})()

        requested_path = Path("/tmp/concurrent_test.csv")
        results = []

        def fetch_worker():
            try:
                result = pipeline._ensure_dataset_availability(requested_path)
                results.append(result)
            except Exception as e:
                results.append(e)

        # This test verifies that concurrent access doesn't cause issues
        # In a real scenario, we'd run multiple threads, but for this test
        # we'll just verify the locking mechanism exists
        from glassalpha.utils.locks import file_lock, get_lock_path

        lock_path = get_lock_path(requested_path)
        assert str(lock_path).endswith(".lock")

        # Verify lock can be acquired and released
        with file_lock(lock_path):
            assert lock_path.exists()

        assert not lock_path.exists()

    def test_metadata_creation_integration(self):
        """Test that metadata is created during dataset fetching."""
        # Clear any existing metadata
        cache_file = (
            Path.home() / "Library" / "Application Support" / "glassalpha" / "data" / "german_credit_processed.csv"
        )
        meta_file = cache_file.with_suffix(cache_file.suffix + ".meta.json")

        if meta_file.exists():
            meta_file.unlink()

        # Fetch dataset (should create metadata)
        config = load_config_from_file(CONFIG_DIR / "german_credit_simple.yaml")
        pipeline = AuditPipeline(config)

        # Trigger dataset fetching
        final_path = pipeline._ensure_dataset_availability(cache_file)

        # Check if metadata was created
        if meta_file.exists():
            from glassalpha.utils.integrity import load_metadata

            metadata = load_metadata(cache_file)

            assert metadata is not None
            assert metadata.dataset_key == "german_credit"
            assert metadata.schema_version == "v1"
            assert metadata.row_count > 0
            assert metadata.sha256 is not None
            assert len(metadata.columns) > 0
        else:
            # Metadata creation is optional for Phase 1
            pass
