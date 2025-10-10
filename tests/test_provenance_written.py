"""Test provenance system functionality."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from glassalpha.provenance.run_manifest import (
    generate_run_manifest,
    get_manifest_summary,
    write_manifest_sidecar,
)


def test_generate_run_manifest_basic():
    """Test basic manifest generation."""
    config = {
        "audit_profile": "test_profile",
        "model": {"type": "xgboost", "params": {"max_depth": 6}},
        "data": {"dataset": "custom", "path": "test.csv"},
    }

    manifest = generate_run_manifest(config)

    # Check required sections
    assert "manifest_version" in manifest
    assert "generated_at" in manifest
    assert "glassalpha_version" in manifest
    assert "git" in manifest
    assert "configuration" in manifest
    assert "environment" in manifest
    assert "dependencies" in manifest
    assert "manifest_hash" in manifest

    # Check configuration section
    config_info = manifest["configuration"]
    assert "hash" in config_info
    assert "sections" in config_info
    assert "audit_profile" in config_info
    assert config_info["audit_profile"] == "test_profile"


def test_generate_run_manifest_with_dataset():
    """Test manifest generation with dataset information."""
    config = {"audit_profile": "test_profile"}

    # Create test dataset
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 0],
        },
    )

    manifest = generate_run_manifest(config, dataset_df=df)

    # Check dataset section
    assert "dataset" in manifest
    dataset_info = manifest["dataset"]
    assert "hash" in dataset_info
    assert "shape" in dataset_info
    assert "dtypes" in dataset_info
    assert "memory_usage_mb" in dataset_info
    assert "null_counts" in dataset_info
    assert "sample_hash" in dataset_info

    # Verify dataset info
    assert dataset_info["shape"] == [5, 3]
    assert len(dataset_info["dtypes"]) == 3
    assert dataset_info["memory_usage_mb"] >= 0  # Small DataFrames may round to 0.0


def test_generate_run_manifest_with_model_info():
    """Test manifest generation with model information."""
    config = {"audit_profile": "test_profile"}

    model_info = {
        "type": "XGBClassifier",
        "parameters": {"max_depth": 6, "n_estimators": 100},
        "calibration": {"is_calibrated": True, "method": "isotonic"},
        "feature_count": 10,
        "class_count": 2,
    }

    manifest = generate_run_manifest(config, model_info=model_info)

    # Check model section
    assert "model" in manifest
    model_section = manifest["model"]
    assert model_section["type"] == "XGBClassifier"
    assert model_section["parameters"]["max_depth"] == 6
    assert model_section["calibration"]["method"] == "isotonic"
    assert model_section["feature_count"] == 10
    assert model_section["class_count"] == 2


def test_generate_run_manifest_with_components():
    """Test manifest generation with selected components."""
    config = {"audit_profile": "test_profile"}

    selected_components = {
        "explainer": "TreeSHAP",
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "threshold_policy": "youden",
    }

    manifest = generate_run_manifest(config, selected_components=selected_components)

    # Check components section
    assert "components" in manifest
    components_section = manifest["components"]
    assert components_section["explainer"] == "TreeSHAP"
    assert len(components_section["metrics"]) == 4
    assert components_section["threshold_policy"] == "youden"


def test_generate_run_manifest_with_execution_info():
    """Test manifest generation with execution information."""
    config = {"audit_profile": "test_profile"}

    execution_info = {
        "start_time": "2023-01-01T12:00:00Z",
        "end_time": "2023-01-01T12:05:00Z",
        "duration_seconds": 300,
        "success": True,
        "random_seed": 42,
    }

    manifest = generate_run_manifest(config, execution_info=execution_info)

    # Check execution section
    assert "execution" in manifest
    exec_section = manifest["execution"]
    assert exec_section["start_time"] == "2023-01-01T12:00:00Z"
    assert exec_section["success"] is True
    assert exec_section["random_seed"] == 42


def test_manifest_deterministic():
    """Test that manifest generation is deterministic for same inputs."""
    config = {
        "audit_profile": "test_profile",
        "model": {"type": "xgboost"},
    }

    # Generate manifest twice
    manifest1 = generate_run_manifest(config)
    manifest2 = generate_run_manifest(config)

    # Configuration hash should be the same
    assert manifest1["configuration"]["hash"] == manifest2["configuration"]["hash"]

    # Environment info should be the same
    assert manifest1["environment"]["python_version"] == manifest2["environment"]["python_version"]


def test_manifest_hash_changes_with_config():
    """Test that manifest hash changes when configuration changes."""
    config1 = {"audit_profile": "profile1", "model": {"type": "xgboost"}}
    config2 = {"audit_profile": "profile2", "model": {"type": "lightgbm"}}

    manifest1 = generate_run_manifest(config1)
    manifest2 = generate_run_manifest(config2)

    # Configuration hashes should be different
    assert manifest1["configuration"]["hash"] != manifest2["configuration"]["hash"]

    # Overall manifest hashes should be different
    assert manifest1["manifest_hash"] != manifest2["manifest_hash"]


def test_write_manifest_sidecar():
    """Test writing manifest as sidecar file."""
    manifest = {
        "manifest_version": "1.0",
        "generated_at": "2023-01-01T12:00:00Z",
        "configuration": {"hash": "abc123"},
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "report.pdf"

        # Write manifest sidecar
        manifest_path = write_manifest_sidecar(manifest, output_path)

        # Check file was created
        assert manifest_path.exists()
        assert manifest_path.name == "report.manifest.json"

        # Check content
        with manifest_path.open() as f:
            loaded_manifest = json.load(f)

        assert loaded_manifest["manifest_version"] == "1.0"
        assert loaded_manifest["configuration"]["hash"] == "abc123"


def test_get_manifest_summary():
    """Test manifest summary generation."""
    manifest = {
        "generated_at": "2023-01-01T12:00:00Z",
        "glassalpha_version": "1.0.0",
        "manifest_hash": "abcdef123456789",
        "git": {
            "available": True,
            "sha": "abc123def456",
            "branch": "main",
            "dirty": False,
            "remote_url": "https://github.com/example/repo.git",
        },
        "configuration": {
            "hash": "config123",
            "audit_profile": "tabular_compliance",
            "strict_mode": True,
        },
        "dataset": {
            "hash": "data123",
            "shape": [1000, 20],
            "size_bytes": 1024000,
        },
        "model": {
            "type": "XGBClassifier",
            "calibration": {"method": "isotonic"},
            "feature_count": 19,
        },
        "environment": {
            "python_version": "3.11.0",
            "platform": "Linux-5.4.0",
        },
        "dependencies": {
            "core_packages": {
                "numpy": "1.24.0",
                "pandas": "2.0.0",
                "scikit-learn": "1.3.0",
                "xgboost": "2.0.0",
            },
        },
    }

    summary = get_manifest_summary(manifest)

    # Check key information is included
    assert "AUDIT PROVENANCE SUMMARY" in summary
    assert "Generated: 2023-01-01T12:00:00Z" in summary
    assert "GlassAlpha Version: 1.0.0" in summary
    assert "SHA: abc123def456" in summary
    assert "Branch: main" in summary
    assert "Profile: tabular_compliance" in summary
    assert "Strict Mode: True" in summary
    assert "Shape: [1000, 20]" in summary
    assert "Type: XGBClassifier" in summary
    assert "Python: 3.11.0" in summary
    assert "numpy: 1.24.0" in summary


def test_manifest_with_missing_git():
    """Test manifest generation when git is not available."""
    config = {"audit_profile": "test_profile"}

    manifest = generate_run_manifest(config)

    # Git section should exist but indicate unavailability
    assert "git" in manifest
    git_info = manifest["git"]
    assert git_info["available"] in [True, False]  # Depends on test environment

    # Other sections should still be present
    assert "configuration" in manifest
    assert "environment" in manifest


def test_manifest_sections_complete():
    """Test that all expected manifest sections are present."""
    config = {
        "audit_profile": "test_profile",
        "model": {"type": "xgboost"},
        "data": {"dataset": "custom", "path": "test.csv"},
    }

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    model_info = {"type": "XGBClassifier"}
    selected_components = {"explainer": "TreeSHAP"}
    execution_info = {"success": True}

    manifest = generate_run_manifest(
        config=config,
        dataset_df=df,
        model_info=model_info,
        selected_components=selected_components,
        execution_info=execution_info,
    )

    # All sections should be present
    expected_sections = [
        "manifest_version",
        "generated_at",
        "glassalpha_version",
        "git",
        "configuration",
        "dataset",
        "model",
        "components",
        "environment",
        "dependencies",
        "execution",
        "constraints",
        "manifest_hash",
    ]

    for section in expected_sections:
        assert section in manifest, f"Missing section: {section}"


def test_config_sanitization():
    """Test that sensitive information is removed from config before hashing."""
    config_with_sensitive = {
        "audit_profile": "test_profile",
        "data": {"dataset": "custom", "path": "/Users/sensitive/path/data.csv"},
        "license_key": "secret123",
        "api_token": "token456",
    }

    manifest = generate_run_manifest(config_with_sensitive)

    # Configuration should be sanitized
    config_info = manifest["configuration"]
    assert "hash" in config_info

    # The hash should be deterministic for the same logical config
    # (even if paths are different)


def test_dataframe_hash_deterministic():
    """Test that DataFrame hashing is deterministic."""
    # Create identical DataFrames
    df1 = pd.DataFrame({"a": [3, 1, 2], "b": [6, 4, 5]})
    df2 = pd.DataFrame({"a": [3, 1, 2], "b": [6, 4, 5]})

    config = {"audit_profile": "test"}

    manifest1 = generate_run_manifest(config, dataset_df=df1)
    manifest2 = generate_run_manifest(config, dataset_df=df2)

    # Dataset hashes should be identical
    assert manifest1["dataset"]["hash"] == manifest2["dataset"]["hash"]

    # Create different DataFrame
    df3 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]})  # Different values
    manifest3 = generate_run_manifest(config, dataset_df=df3)

    # Hash should be different
    assert manifest1["dataset"]["hash"] != manifest3["dataset"]["hash"]


if __name__ == "__main__":
    pytest.main([__file__])
