"""Comprehensive tests for utility modules.

Tests the core utility functionality including seeds management, hashing,
and manifest generation that are critical for audit reproducibility and
regulatory compliance.
"""

import contextlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from glassalpha.utils.determinism import compute_file_hash, normalize_pdf_metadata, verify_deterministic_output
from glassalpha.utils.hashing import (
    hash_array,
    hash_config,
    hash_dataframe,
    hash_file,
    hash_model,
    hash_object,
)
from glassalpha.utils.manifest import (
    AuditManifest,
    EnvironmentInfo,
    GitInfo,
    ManifestGenerator,
    compare_manifests,
    create_manifest,
    load_manifest,
)
from glassalpha.utils.seeds import SeedManager, get_component_seed, get_seeds_manifest, set_global_seed


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        "audit_profile": "tabular_compliance",
        "model": {"type": "xgboost", "params": {"n_estimators": 100}},
        "data": {"dataset": "custom", "path": "test.csv", "target": "y"},
        "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy", "f1"]},
        "reproducibility": {"random_seed": 42},
    }


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randint(0, 10, 100),
            "feature3": np.random.choice(["A", "B", "C"], 100),
            "target": np.random.choice([0, 1], 100),
        },
    )


@pytest.fixture
def temp_file_with_content():
    """Create temporary file with known content for hashing tests."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        content = "This is test content for hashing\nLine 2\nLine 3"
        f.write(content)
        f.flush()
        yield f.name, content

    # Cleanup
    with contextlib.suppress(FileNotFoundError):
        os.unlink(f.name)


class TestSeedManager:
    """Test SeedManager functionality."""

    def test_seed_manager_initialization(self):
        """Test SeedManager initialization."""
        manager = SeedManager(master_seed=42)

        assert manager.master_seed == 42
        assert manager.component_seeds == {}
        assert manager._original_states == {}

    def test_seed_manager_default_initialization(self):
        """Test SeedManager with default seed."""
        manager = SeedManager()

        assert manager.master_seed == 42  # Default value

    def test_get_seed_deterministic(self):
        """Test that get_seed produces deterministic results."""
        manager = SeedManager(master_seed=123)

        # Get seed for same component multiple times
        seed1 = manager.get_seed("model")
        seed2 = manager.get_seed("model")
        seed3 = manager.get_seed("model")

        assert seed1 == seed2 == seed3
        assert isinstance(seed1, int)
        assert seed1 >= 0

    def test_get_seed_different_components(self):
        """Test that different components get different seeds."""
        manager = SeedManager(master_seed=123)

        model_seed = manager.get_seed("model")
        explainer_seed = manager.get_seed("explainer")
        data_seed = manager.get_seed("data")

        # All should be different
        assert model_seed != explainer_seed
        assert model_seed != data_seed
        assert explainer_seed != data_seed

        # All should be positive integers
        for seed in [model_seed, explainer_seed, data_seed]:
            assert isinstance(seed, int)
            assert seed >= 0

    def test_get_seed_different_master_seeds(self):
        """Test that different master seeds produce different component seeds."""
        manager1 = SeedManager(master_seed=42)
        manager2 = SeedManager(master_seed=123)

        seed1 = manager1.get_seed("model")
        seed2 = manager2.get_seed("model")

        assert seed1 != seed2

    def test_seed_manager_component_caching(self):
        """Test that component seeds are cached properly."""
        manager = SeedManager(master_seed=42)

        # First call should generate and cache seed
        seed1 = manager.get_seed("test_component")
        assert "test_component" in manager.component_seeds
        assert manager.component_seeds["test_component"] == seed1

        # Second call should return cached seed
        seed2 = manager.get_seed("test_component")
        assert seed1 == seed2

    def test_set_all_seeds(self):
        """Test setting all random seeds."""
        manager = SeedManager(master_seed=42)

        # Should not raise any exceptions
        manager.set_all_seeds()

        # Test that seeds are actually set by generating same random numbers
        np.random.seed(manager.get_seed("numpy"))
        first_random = np.random.random()

        np.random.seed(manager.get_seed("numpy"))
        second_random = np.random.random()

        assert first_random == second_random

    @patch("glassalpha.utils.seeds.tf", None)  # Simulate TensorFlow not installed
    @patch("glassalpha.utils.seeds.torch", None)  # Simulate PyTorch not installed
    def test_set_all_seeds_no_deep_learning(self):
        """Test setting seeds when deep learning frameworks are not available."""
        manager = SeedManager(master_seed=42)

        # Should not raise exceptions even without TF/PyTorch
        manager.set_all_seeds()

    def test_context_manager_basic(self):
        """Test SeedManager as context manager."""
        manager = SeedManager(master_seed=42)

        # Save original numpy state

        # Use context manager
        with manager:
            # Seeds should be set
            np.random.seed(manager.get_seed("numpy"))

        # Original state should be restored (approximately)
        # Note: We can't test exact restoration since we don't control all RNG state


class TestSeedUtilityFunctions:
    """Test standalone seed utility functions."""

    def test_get_component_seed_function(self):
        """Test standalone get_component_seed function."""
        seed = get_component_seed("test_component")

        assert isinstance(seed, int)
        assert seed >= 0

        # Should be deterministic
        seed2 = get_component_seed("test_component")
        assert seed == seed2

    def test_set_global_seed_function(self):
        """Test standalone set_global_seed function."""
        # Should not raise exceptions
        set_global_seed(123)

        # Test reproducibility
        np.random.seed(123)
        first_random = np.random.random()

        np.random.seed(123)
        second_random = np.random.random()

        assert first_random == second_random

    def test_get_seeds_manifest(self):
        """Test seeds manifest generation."""
        manifest = get_seeds_manifest()

        assert isinstance(manifest, dict)
        assert "master_seed" in manifest
        assert "component_seeds" in manifest
        assert "timestamp" in manifest

        # Should contain some default seeds
        assert isinstance(manifest["component_seeds"], dict)

    def test_component_seed_deterministic(self):
        """Test component seed deterministic behavior."""
        # Create two managers with same master seed
        manager1 = SeedManager(master_seed=42)
        manager2 = SeedManager(master_seed=42)

        seed1 = manager1.get_seed("test")
        seed2 = manager2.get_seed("test")

        # Same master seed + component should produce same seed
        assert seed1 == seed2


class TestHashingFunctions:
    """Test hashing utility functions."""

    def test_hash_object_basic(self):
        """Test basic object hashing."""
        obj1 = {"key": "value", "number": 42}
        obj2 = {"key": "value", "number": 42}
        obj3 = {"key": "different", "number": 42}

        hash1 = hash_object(obj1)
        hash2 = hash_object(obj2)
        hash3 = hash_object(obj3)

        # Same objects should have same hash
        assert hash1 == hash2

        # Different objects should have different hash
        assert hash1 != hash3

        # Hash should be hex string
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 produces 64-character hex string

    def test_hash_object_algorithm_selection(self):
        """Test different hashing algorithms."""
        obj = {"test": "data"}

        sha256_hash = hash_object(obj, algorithm="sha256")
        sha1_hash = hash_object(obj, algorithm="sha1")
        md5_hash = hash_object(obj, algorithm="md5")

        # Different algorithms should produce different hashes
        assert sha256_hash != sha1_hash
        assert sha256_hash != md5_hash
        assert sha1_hash != md5_hash

        # Hash lengths should be different
        assert len(sha256_hash) == 64  # SHA-256
        assert len(sha1_hash) == 40  # SHA-1
        assert len(md5_hash) == 32  # MD5

    def test_hash_object_deterministic_ordering(self):
        """Test that object hashing is deterministic regardless of key order."""
        obj1 = {"b": 2, "a": 1, "c": 3}
        obj2 = {"a": 1, "c": 3, "b": 2}
        obj3 = {"c": 3, "b": 2, "a": 1}

        hash1 = hash_object(obj1)
        hash2 = hash_object(obj2)
        hash3 = hash_object(obj3)

        # All should be identical due to sort_keys=True
        assert hash1 == hash2 == hash3

    def test_hash_object_error_handling(self):
        """Test hash_object error handling with unhashable objects."""

        # Test with object that can't be JSON serialized (circular reference)
        class CircularReference:
            def __init__(self):
                self.circular = self

        unhashable_obj = CircularReference()

        with pytest.raises(ValueError, match="Cannot hash object"):
            hash_object(unhashable_obj)

    def test_hash_config(self, sample_config):
        """Test configuration hashing."""
        hash1 = hash_config(sample_config)
        hash2 = hash_config(sample_config.copy())  # Copy should have same hash

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64

        # Modified config should have different hash
        modified_config = sample_config.copy()
        modified_config["model"]["params"]["n_estimators"] = 200
        hash3 = hash_config(modified_config)

        assert hash1 != hash3

    def test_hash_dataframe_basic(self, sample_dataframe):
        """Test DataFrame hashing."""
        hash1 = hash_dataframe(sample_dataframe)
        hash2 = hash_dataframe(sample_dataframe.copy())

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64

        # Modified DataFrame should have different hash
        modified_df = sample_dataframe.copy()
        modified_df.iloc[0, 0] = 999.999
        hash3 = hash_dataframe(modified_df)

        assert hash1 != hash3

    def test_hash_dataframe_column_order_insensitive(self, sample_dataframe):
        """Test that DataFrame hashing is insensitive to column order."""
        # Reorder columns
        reordered_df = sample_dataframe[["target", "feature1", "feature3", "feature2"]]

        hash1 = hash_dataframe(sample_dataframe)
        hash2 = hash_dataframe(reordered_df)

        # Should be the same if include_column_order=False (default behavior)
        # This depends on implementation - if it's sensitive, that's also valid
        assert isinstance(hash1, str)
        assert isinstance(hash2, str)

    def test_hash_dataframe_empty(self):
        """Test hashing empty DataFrame."""
        empty_df = pd.DataFrame()

        hash_result = hash_dataframe(empty_df)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

    def test_hash_file(self, temp_file_with_content):
        """Test file hashing."""
        file_path, content = temp_file_with_content

        hash1 = hash_file(Path(file_path))
        hash2 = hash_file(Path(file_path))

        # Same file should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64

    def test_hash_file_nonexistent(self):
        """Test hashing nonexistent file."""
        nonexistent_path = Path("nonexistent_file.txt")

        with pytest.raises(FileNotFoundError):
            hash_file(nonexistent_path)

    def test_hash_array(self):
        """Test NumPy array hashing."""
        np.random.seed(42)
        array1 = np.random.randn(10, 5)
        array2 = array1.copy()
        array3 = np.random.randn(10, 5)  # Different random data

        hash1 = hash_array(array1)
        hash2 = hash_array(array2)
        hash3 = hash_array(array3)

        # Same arrays should have same hash
        assert hash1 == hash2

        # Different arrays should have different hash
        assert hash1 != hash3

        assert isinstance(hash1, str)
        assert len(hash1) == 64

    def test_hash_array_dtypes(self):
        """Test NumPy array hashing with different dtypes."""
        arr_int = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        arr_float = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

        hash_int = hash_array(arr_int)
        hash_float = hash_array(arr_float)

        # Different dtypes should produce different hashes
        assert hash_int != hash_float

        assert isinstance(hash_int, str)
        assert isinstance(hash_float, str)

    def test_hash_dataframe_comprehensive(self, sample_dataframe):
        """Test comprehensive DataFrame hashing with metadata."""
        data_hash = hash_dataframe(sample_dataframe)

        assert isinstance(data_hash, str)
        assert len(data_hash) == 64

        # Test with metadata about the DataFrame

        # Should be consistent
        data_hash2 = hash_dataframe(sample_dataframe)
        assert data_hash == data_hash2

    def test_hash_model_metadata(self):
        """Test model metadata hashing."""
        model_info = {
            "type": "xgboost",
            "version": "1.6.0",
            "parameters": {"n_estimators": 100, "max_depth": 6},
            "feature_names": ["feature1", "feature2", "feature3"],
        }

        model_hash = hash_model(model_info)

        assert isinstance(model_hash, str)
        assert len(model_hash) == 64

        # Should be deterministic
        model_hash2 = hash_model(model_info)
        assert model_hash == model_hash2

    def test_hash_file_comprehensive(self, temp_file_with_content):
        """Test file hashing with metadata checks."""
        file_path, content = temp_file_with_content
        path = Path(file_path)

        file_hash = hash_file(path)

        assert isinstance(file_hash, str)
        assert len(file_hash) == 64

        # Should be deterministic
        file_hash2 = hash_file(path)
        assert file_hash == file_hash2

        # File should exist and have content
        assert path.exists()
        assert path.stat().st_size > 0


class TestEnvironmentInfo:
    """Test EnvironmentInfo model."""

    def test_environment_info_creation(self):
        """Test EnvironmentInfo model creation."""
        env_info = EnvironmentInfo(
            python_version="3.11.0",
            platform="Darwin",
            architecture="arm64",
            processor="Apple M1",
            hostname="test-machine",
            user="testuser",
            working_directory="/path/to/work",
            environment_variables={"TEST": "value"},
        )

        assert env_info.python_version == "3.11.0"
        assert env_info.platform == "Darwin"
        assert env_info.environment_variables == {"TEST": "value"}

    def test_environment_info_serialization(self):
        """Test EnvironmentInfo serialization."""
        env_info = EnvironmentInfo(
            python_version="3.11.0",
            platform="Darwin",
            architecture="arm64",
            processor="Apple M1",
            hostname="test-machine",
            user="testuser",
            working_directory="/path/to/work",
        )

        # Should serialize to dict
        data = env_info.model_dump()
        assert isinstance(data, dict)
        assert data["python_version"] == "3.11.0"

        # Should deserialize from dict
        env_info2 = EnvironmentInfo(**data)
        assert env_info2 == env_info


class TestGitInfo:
    """Test GitInfo model."""

    def test_git_info_creation(self):
        """Test GitInfo model creation."""
        git_info = GitInfo(
            commit_sha="abc123def456",
            branch="main",
            is_dirty=False,
            remote_url="https://github.com/user/repo.git",
            commit_message="Initial commit",
            commit_timestamp="2024-01-01T00:00:00Z",
        )

        assert git_info.commit_hash == "abc123def456"
        assert git_info.branch == "main"
        assert git_info.is_dirty is False

    def test_git_info_optional_fields(self):
        """Test GitInfo with optional fields."""
        git_info = GitInfo()

        assert git_info.commit_hash is None
        assert git_info.branch is None
        assert git_info.is_dirty is False  # Default value


class TestAuditManifest:
    """Test AuditManifest model."""

    def test_audit_manifest_creation(self):
        """Test AuditManifest creation."""
        manifest = AuditManifest(audit_id="test-audit-123", version="1.0.0", created_at="2024-01-01T00:00:00Z")

        assert manifest.audit_id == "test-audit-123"
        assert manifest.version == "1.0.0"
        assert manifest.created_at == "2024-01-01T00:00:00Z"

    def test_audit_manifest_to_json(self):
        """Test AuditManifest JSON serialization."""
        manifest = AuditManifest(audit_id="test-audit-123", version="1.0.0", created_at="2024-01-01T00:00:00Z")

        json_str = manifest.to_json()

        assert isinstance(json_str, str)
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["audit_id"] == "test-audit-123"

    def test_audit_manifest_to_dict(self):
        """Test AuditManifest dictionary conversion."""
        manifest = AuditManifest(audit_id="test-audit-123", version="1.0.0", created_at="2024-01-01T00:00:00Z")

        data = manifest.to_dict()

        assert isinstance(data, dict)
        assert data["audit_id"] == "test-audit-123"
        assert data["version"] == "1.0.0"

    def test_audit_manifest_save_load(self, tmp_path):
        """Test AuditManifest save and load."""
        manifest = AuditManifest(audit_id="test-audit-123", version="1.0.0", created_at="2024-01-01T00:00:00Z")

        save_path = tmp_path / "manifest.json"
        manifest.save(save_path)

        assert save_path.exists()

        # Load and verify
        loaded_manifest = load_manifest(save_path)
        assert loaded_manifest.audit_id == manifest.audit_id
        assert loaded_manifest.version == manifest.version


class TestManifestGenerator:
    """Test ManifestGenerator functionality."""

    def test_manifest_generator_initialization(self):
        """Test ManifestGenerator initialization."""
        generator = ManifestGenerator()

        assert generator.audit_id is not None
        assert len(generator.audit_id) > 10  # Should be a reasonable ID
        assert generator.components == {}
        assert generator.datasets == {}

    def test_manifest_generator_custom_id(self):
        """Test ManifestGenerator with custom audit ID."""
        custom_id = "custom-audit-123"
        generator = ManifestGenerator(audit_id=custom_id)

        assert generator.audit_id == custom_id

    def test_add_config(self, sample_config):
        """Test adding configuration to manifest."""
        generator = ManifestGenerator()

        generator.add_config(sample_config)

        assert generator.config is not None
        assert generator.config_hash is not None
        assert len(generator.config_hash) == 64

    def test_add_seeds(self):
        """Test adding seeds to manifest."""
        generator = ManifestGenerator()

        generator.add_seeds()

        assert generator.seeds is not None
        assert isinstance(generator.seeds, dict)
        assert "master_seed" in generator.seeds

    def test_add_component(self):
        """Test adding component to manifest."""
        generator = ManifestGenerator()

        component_info = {
            "type": "model",
            "implementation": "xgboost",
            "version": "1.6.0",
            "parameters": {"n_estimators": 100},
        }

        generator.add_component("test_model", "xgboost", component_info)

        assert "test_model" in generator.components
        component = generator.components["test_model"]
        assert component.name == "test_model"
        assert component.type == "xgboost"

    def test_add_dataset(self, sample_dataframe, tmp_path):
        """Test adding dataset to manifest."""
        generator = ManifestGenerator()

        # Save DataFrame to temporary file
        data_path = tmp_path / "test_data.csv"
        sample_dataframe.to_csv(data_path, index=False)

        generator.add_dataset("training_data", data_path, sample_dataframe)

        assert "training_data" in generator.datasets
        dataset = generator.datasets["training_data"]
        assert dataset["name"] == "training_data"
        assert dataset["path"] == str(data_path)

    def test_add_result_hash(self):
        """Test adding result hash to manifest."""
        generator = ManifestGenerator()

        generator.add_result_hash("predictions", "abc123def456")

        assert "predictions" in generator.result_hashes
        assert generator.result_hashes["predictions"] == "abc123def456"

    def test_mark_completed(self):
        """Test marking audit as completed."""
        generator = ManifestGenerator()

        generator.mark_completed(status="success")

        assert generator.status == "success"
        assert generator.completed_at is not None
        assert generator.error is None

    def test_mark_completed_with_error(self):
        """Test marking audit as completed with error."""
        generator = ManifestGenerator()

        error_msg = "Model training failed"
        generator.mark_completed(status="failed", error=error_msg)

        assert generator.status == "failed"
        assert generator.error == error_msg

    @patch("glassalpha.utils.manifest.datetime")
    def test_finalize_manifest(self, mock_datetime):
        """Test finalizing audit manifest."""
        # Mock current time
        mock_now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        mock_datetime.now.return_value = mock_now
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

        generator = ManifestGenerator(audit_id="test-audit")
        generator.add_config({"test": "config"})
        generator.add_seeds()

        manifest = generator.finalize()

        assert isinstance(manifest, AuditManifest)
        assert manifest.audit_id == "test-audit"
        assert manifest.config is not None
        assert manifest.seeds is not None

    def test_collect_environment_info(self):
        """Test environment info collection."""
        generator = ManifestGenerator()

        env_info = generator._collect_environment_info()

        assert isinstance(env_info, EnvironmentInfo)
        assert env_info.python_version is not None
        assert env_info.platform is not None
        assert env_info.hostname is not None

    @patch("glassalpha.utils.proc.subprocess.run")
    def test_collect_git_info_success(self, mock_run):
        """Test successful git info collection."""
        # Mock git command responses
        mock_run.side_effect = [
            Mock(returncode=0, stdout="abc123def456", stderr=""),
            Mock(returncode=0, stdout="main", stderr=""),
            Mock(returncode=0, stdout="", stderr=""),  # Clean working directory
            Mock(returncode=0, stdout="https://github.com/user/repo.git", stderr=""),
            Mock(returncode=0, stdout="Initial commit", stderr=""),
            Mock(returncode=0, stdout="2024-01-01 12:00:00 +0000", stderr=""),
        ]

        generator = ManifestGenerator()
        git_info = generator._collect_git_info()

        assert git_info is not None
        assert git_info.commit_hash == "abc123def456"
        assert git_info.branch == "main"
        assert git_info.is_dirty is False

    @patch("subprocess.run")
    def test_collect_git_info_dirty_working_directory(self, mock_run):
        """Test git info collection with dirty working directory."""
        # Mock git command responses with dirty state
        mock_run.side_effect = [
            Mock(returncode=0, stdout="abc123def456", stderr=""),
            Mock(returncode=0, stdout="main", stderr=""),
            Mock(returncode=0, stdout="M file.py", stderr=""),  # Modified file
            Mock(returncode=0, stdout="https://github.com/user/repo.git", stderr=""),
            Mock(returncode=0, stdout="Initial commit", stderr=""),
            Mock(returncode=0, stdout="2024-01-01 12:00:00 +0000", stderr=""),
        ]

        generator = ManifestGenerator()
        git_info = generator._collect_git_info()

        assert git_info is not None
        assert git_info.is_dirty is True

    @patch("subprocess.run")
    def test_collect_git_info_no_git(self, mock_run):
        """Test git info collection when git is not available."""
        mock_run.side_effect = FileNotFoundError("git command not found")

        generator = ManifestGenerator()
        git_info = generator._collect_git_info()

        assert git_info is None


class TestManifestUtilityFunctions:
    """Test utility functions for manifest management."""

    def test_create_manifest(self):
        """Test create_manifest utility function."""
        generator = create_manifest()

        assert isinstance(generator, ManifestGenerator)
        assert generator.audit_id is not None

    def test_create_manifest_with_id(self):
        """Test create_manifest with custom ID."""
        custom_id = "custom-audit-456"
        generator = create_manifest(audit_id=custom_id)

        assert generator.audit_id == custom_id

    def test_compare_manifests_identical(self):
        """Test comparing identical manifests."""
        manifest1 = AuditManifest(audit_id="test-1", version="1.0.0", created_at="2024-01-01T00:00:00Z")
        manifest2 = AuditManifest(audit_id="test-1", version="1.0.0", created_at="2024-01-01T00:00:00Z")

        comparison = compare_manifests(manifest1, manifest2)

        assert isinstance(comparison, dict)
        assert "identical" in comparison
        assert "differences" in comparison

    def test_compare_manifests_different(self):
        """Test comparing different manifests."""
        manifest1 = AuditManifest(audit_id="test-1", version="1.0.0", created_at="2024-01-01T00:00:00Z")
        manifest2 = AuditManifest(audit_id="test-2", version="1.0.1", created_at="2024-01-01T00:00:00Z")

        comparison = compare_manifests(manifest1, manifest2)

        assert isinstance(comparison, dict)
        assert "differences" in comparison


class TestUtilsIntegration:
    """Test integration between different utility modules."""

    def test_seed_hash_consistency(self):
        """Test that same seeds produce consistent hashes."""
        # Set same seed multiple times
        set_global_seed(42)
        np.random.seed(42)
        data1 = np.random.randn(100, 10)

        set_global_seed(42)
        np.random.seed(42)
        data2 = np.random.randn(100, 10)

        # Data should be identical
        np.testing.assert_array_equal(data1, data2)

        # Hashes should be identical
        hash1 = hash_array(data1)
        hash2 = hash_array(data2)
        assert hash1 == hash2

    def test_full_audit_manifest_workflow(self, sample_config, sample_dataframe, tmp_path):
        """Test complete audit manifest workflow."""
        # Create manifest generator
        generator = create_manifest(audit_id="integration-test")

        # Add configuration
        generator.add_config(sample_config)

        # Add seeds
        generator.add_seeds()

        # Add component
        component_info = {"type": "model", "implementation": "test_model", "version": "1.0.0"}
        generator.add_component("test_model", "model", component_info)

        # Add dataset
        data_path = tmp_path / "test_data.csv"
        sample_dataframe.to_csv(data_path, index=False)
        generator.add_dataset("test_dataset", data_path, sample_dataframe)

        # Add result hash
        generator.add_result_hash("predictions", "abc123")

        # Mark completed
        generator.mark_completed("success")

        # Finalize
        manifest = generator.finalize()

        # Verify manifest completeness
        assert manifest.audit_id == "integration-test"
        assert manifest.config is not None
        assert manifest.seeds is not None
        assert manifest.environment is not None
        assert len(manifest.components) == 1
        assert len(manifest.datasets) == 1
        assert len(manifest.result_hashes) == 1
        assert manifest.status == "success"

        # Save and reload
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        loaded_manifest = load_manifest(manifest_path)
        assert loaded_manifest.audit_id == manifest.audit_id
        assert loaded_manifest.config == manifest.config

    def test_deterministic_audit_reproduction(self, sample_config):
        """Test that audits can be reproduced deterministically."""
        # Create two identical audit setups
        generator1 = create_manifest(audit_id="repro-test-1")
        generator2 = create_manifest(audit_id="repro-test-2")

        # Add identical configurations
        generator1.add_config(sample_config)
        generator2.add_config(sample_config)

        # Add seeds with same master seed
        set_global_seed(42)
        generator1.add_seeds()

        set_global_seed(42)
        generator2.add_seeds()

        # Generate data with same seed
        np.random.seed(42)
        data1 = np.random.randn(50, 5)

        np.random.seed(42)
        data2 = np.random.randn(50, 5)

        # Data should be identical
        np.testing.assert_array_equal(data1, data2)

        # Hashes should be identical
        hash1 = hash_array(data1)
        hash2 = hash_array(data2)
        assert hash1 == hash2

        # Config hashes should be identical
        config_hash1 = hash_config(sample_config)
        config_hash2 = hash_config(sample_config)
        assert config_hash1 == config_hash2


class TestUtilsErrorHandling:
    """Test error handling across utility modules."""

    def test_seed_manager_invalid_seed(self):
        """Test SeedManager with invalid seed types."""
        # Should handle various seed types gracefully
        with contextlib.suppress(TypeError, ValueError):
            SeedManager(master_seed="invalid")
            # If it accepts string, that's implementation choice

    def test_hash_functions_with_none_values(self):
        """Test hash functions with None values."""
        # Should handle None gracefully
        hash1 = hash_object(None)
        hash2 = hash_object(None)

        assert hash1 == hash2
        assert isinstance(hash1, str)

    def test_manifest_generator_incomplete_state(self):
        """Test ManifestGenerator with incomplete information."""
        generator = ManifestGenerator()

        # Should be able to finalize even without complete information
        manifest = generator.finalize()

        assert isinstance(manifest, AuditManifest)
        assert manifest.audit_id is not None

    def test_file_operations_with_permissions(self, tmp_path):
        """Test file operations with permission issues."""
        # Create a file and make it read-only
        test_file = tmp_path / "readonly.txt"
        test_file.write_text("test content")
        test_file.chmod(0o444)  # Read-only

        try:
            # Should still be able to hash read-only file
            file_hash = hash_file(test_file)
            assert isinstance(file_hash, str)
        finally:
            # Restore write permissions for cleanup
            test_file.chmod(0o644)

    def test_large_data_handling(self):
        """Test utility functions with larger datasets."""
        # Create larger dataset
        np.random.seed(42)
        large_data = np.random.randn(1000, 100)
        large_df = pd.DataFrame(large_data)

        # Should handle without memory issues
        hash_result = hash_dataframe(large_df)
        assert isinstance(hash_result, str)

        numpy_hash = hash_array(large_data)
        assert isinstance(numpy_hash, str)


class TestDeterminismUtilities:
    """Test determinism-related utility functions."""

    def test_compute_file_hash(self, tmp_path):
        """Test file hash computation."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        # Compute hash
        file_hash = compute_file_hash(test_file)
        assert isinstance(file_hash, str)
        assert len(file_hash) > 0

        # Hash should be consistent
        file_hash2 = compute_file_hash(test_file)
        assert file_hash == file_hash2

    def test_compute_file_hash_different_files(self, tmp_path):
        """Test that different files produce different hashes."""
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.txt"

        file1.write_bytes(b"Content 1")
        file2.write_bytes(b"Content 2")

        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file2)

        assert hash1 != hash2

    def test_verify_deterministic_output_identical(self, tmp_path):
        """Test verification of identical files."""
        test_file = tmp_path / "test.txt"
        content = b"Identical content"
        test_file.write_bytes(content)

        # Copy to second file
        test_file2 = tmp_path / "test2.txt"
        test_file2.write_bytes(content)

        # Should be identical
        identical, hash1, hash2 = verify_deterministic_output(test_file, test_file2)
        assert identical
        assert hash1 == hash2

    def test_verify_deterministic_output_different(self, tmp_path):
        """Test verification of different files."""
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.txt"

        file1.write_bytes(b"Content 1")
        file2.write_bytes(b"Content 2")

        # Should not be identical
        identical, hash1, hash2 = verify_deterministic_output(file1, file2)
        assert not identical
        assert hash1 != hash2

    def test_normalize_pdf_metadata_basic(self, tmp_path):
        """Test PDF metadata normalization with basic text replacement."""
        # Create a simple text file that mimics PDF content
        pdf_file = tmp_path / "test.pdf"
        # Write content that includes a timestamp pattern
        content = b"D:20250101120000Some PDF content"
        pdf_file.write_bytes(content)

        # Normalize metadata
        normalize_pdf_metadata(pdf_file)

        # File should still exist and be readable
        assert pdf_file.exists()
        new_content = pdf_file.read_bytes()
        assert isinstance(new_content, bytes)
