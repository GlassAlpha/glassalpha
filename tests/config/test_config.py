"""Consolidated configuration tests.

This module consolidates all configuration-related tests from:
- tests/test_config_loading.py
- tests/test_config_validation.py
- tests/unit/test_config_strict.py

Tests the complete configuration system including loading, validation,
and strict mode requirements.
"""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from glassalpha.config import load_config_from_file, load_yaml
from glassalpha.config import AuditConfig, DataConfig, ExplainerConfig, ModelConfig

# ============================================================================
# YAML Loading Tests (from test_config_loading.py)
# ============================================================================


def test_load_yaml_valid():
    """Test loading valid YAML file."""
    data = {"key": "value", "number": 42}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        yaml_path = f.name

    try:
        result = load_yaml(yaml_path)
        assert result == data
    finally:
        Path(yaml_path).unlink(missing_ok=True)


def test_load_yaml_file_not_found():
    """Test loading non-existent YAML file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_yaml("nonexistent_file.yaml")


# ============================================================================
# Schema Validation Tests (from test_config_loading.py)
# ============================================================================


def test_model_config_validation():
    """Test ModelConfig validation."""
    # Valid config
    config = ModelConfig(type="xgboost", path=Path("test.pkl"))
    assert config.type == "xgboost"
    assert config.path == Path("test.pkl")

    # Type gets lowercased
    config = ModelConfig(type="XGBOOST")
    assert config.type == "xgboost"


def test_data_config_validation():
    """Test DataConfig validation."""
    config = DataConfig(dataset="custom", path="test.csv")
    assert config.path == "test.csv"
    assert config.protected_attributes == []  # default


def test_explainer_config_validation():
    """Test ExplainerConfig validation."""
    config = ExplainerConfig(strategy="first_compatible", priority=["treeshap", "kernelshap"])
    assert config.strategy == "first_compatible"
    assert config.priority == ["treeshap", "kernelshap"]


def test_audit_config_full():
    """Test full AuditConfig validation with all required fields."""
    config_data = {
        "audit_profile": "tabular_compliance",
        "model": {"type": "xgboost", "path": "model.pkl"},
        "data": {"dataset": "custom", "path": "data.csv"},
        "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"]},
        "reproducibility": {"random_seed": 42},
    }

    config = AuditConfig(**config_data)
    assert config.audit_profile == "tabular_compliance"
    assert config.model.type == "xgboost"
    assert config.data.path == "data.csv"


def test_load_config_from_file():
    """Test loading config from YAML file."""
    config_data = {
        "audit_profile": "tabular_compliance",
        "model": {"type": "xgboost"},
        "data": {"dataset": "custom", "path": "test.csv"},
        "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"]},
        "reproducibility": {"random_seed": 42},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        # Test that we can at least load the YAML and validate the schema
        yaml_data = load_yaml(config_path)
        config = AuditConfig(**yaml_data)
        assert isinstance(config, AuditConfig)
        assert config.audit_profile == "tabular_compliance"
        assert config.model.type == "xgboost"
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_config_validation_basic():
    """Test basic config validation."""
    config_data = {
        "audit_profile": "tabular_compliance",
        "model": {"type": "xgboost"},
        "data": {"dataset": "custom", "path": "test.csv"},
        "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"]},
        "reproducibility": {"random_seed": 42},
    }
    config = AuditConfig(**config_data)

    # Basic validation - object created successfully
    assert config.audit_profile == "tabular_compliance"


def test_config_with_missing_required_fields():
    """Test config validation fails with missing required fields."""
    with pytest.raises(ValidationError):
        AuditConfig(audit_profile="tabular_compliance")  # Missing model, data, etc.


def test_config_extra_fields_forbidden():
    """Test that extra fields are rejected."""
    config_data = {
        "audit_profile": "tabular_compliance",
        "model": {"type": "xgboost"},
        "data": {"dataset": "custom", "path": "test.csv"},
        "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"]},
        "reproducibility": {"random_seed": 42},
        "extra_field": "not_allowed",  # This should cause validation error
    }

    with pytest.raises(ValidationError):
        AuditConfig(**config_data)


# ============================================================================
# Example Config Validation Tests (from test_config_validation.py)
# ============================================================================


def get_all_config_files():
    """Get all YAML config files from configs directory."""
    configs_dir = Path(__file__).parent.parent.parent / "src" / "glassalpha" / "data" / "configs"
    # Exclude test configs (they're for testing specific features, not full validation)
    return [f for f in configs_dir.glob("*.yaml") if not f.name.startswith("test_")]


@pytest.mark.parametrize("config_file", get_all_config_files(), ids=lambda p: p.name)
def test_config_file_validates(config_file: Path) -> None:
    """Ensure each example config file is valid.

    Args:
        config_file: Path to configuration file to validate

    This test ensures that:
    1. Config file can be loaded without errors
    2. Schema validation passes
    3. No critical validation errors exist

    Note: Warnings about missing data files are acceptable since
    these are example configs pointing to user-specific paths.

    """
    try:
        config = load_config_from_file(config_file)
        assert config is not None, f"Config {config_file.name} failed to load"

        # Verify basic required fields exist
        assert hasattr(config, "audit_profile"), f"{config_file.name} missing audit_profile"
        assert hasattr(config, "data"), f"{config_file.name} missing data section"
        assert hasattr(config, "model"), f"{config_file.name} missing model section"

        # Verify data section has required fields
        assert hasattr(config.data, "dataset"), f"{config_file.name} data missing dataset field"
        assert hasattr(config.data, "target_column"), f"{config_file.name} data missing target_column"

    except Exception as e:
        pytest.fail(f"Config {config_file.name} validation failed: {e}")


def test_all_configs_found() -> None:
    """Verify that config files were found for testing."""
    config_files = get_all_config_files()
    assert len(config_files) > 0, "No config files found in src/glassalpha/data/configs/ directory"
    print(f"\nFound {len(config_files)} config files to validate:")
    for config_file in config_files:
        print(f"  - {config_file.name}")


def test_specific_config_examples_exist() -> None:
    """Verify that key example configs exist."""
    configs_dir = Path(__file__).parent.parent.parent / "src" / "glassalpha" / "data" / "configs"

    required_examples = [
        "quickstart.yaml",
        "german_credit_simple.yaml",
        "adult_income_simple.yaml",
    ]

    for example in required_examples:
        config_path = configs_dir / example
        assert config_path.exists(), f"Required example config missing: {example}"


def test_quickstart_config_works() -> None:
    """Verify the quickstart config specifically (most important for new users)."""
    configs_dir = Path(__file__).parent.parent.parent / "src" / "glassalpha" / "data" / "configs"
    quickstart_path = configs_dir / "quickstart.yaml"

    config = load_config_from_file(quickstart_path)

    # Verify quickstart uses safe defaults
    assert config.audit_profile == "tabular_compliance"
    assert config.model.type == "logistic_regression"  # Always available
    assert config.data.dataset == "german_credit"  # Built-in dataset
    assert config.reproducibility.random_seed is not None  # Deterministic


# ============================================================================
# Strict Mode Tests (from test_config_strict.py)
# ============================================================================


def _create_valid_strict_config() -> dict:
    """Create a fully valid strict mode configuration."""
    return {
        "audit_profile": "tabular_compliance",
        "strict_mode": True,
        "model": {"type": "logistic_regression", "path": "/tmp/model.pkl"},
        "data": {
            "dataset": "custom",
            "path": "/tmp/data.csv",
            "target_column": "target",
            "protected_attributes": ["gender"],
            "schema_path": "/tmp/schema.yaml",
        },
        "reproducibility": {
            "random_seed": 42,
            "deterministic": True,
            "capture_environment": True,
        },
        "explainers": {"priority": ["treeshap"], "strategy": "first_compatible"},
        "manifest": {
            "enabled": True,
            "include_git_sha": True,
            "include_config_hash": True,
            "include_data_hash": True,
        },
        "metrics": {
            "performance": ["accuracy"],
            "fairness": ["demographic_parity"],
        },
        "preprocessing": {
            "mode": "artifact",
            "artifact_path": "/tmp/preprocessor.joblib",
            "expected_file_hash": "sha256:test_hash",
            "expected_params_hash": "sha256:test_params_hash",
        },
    }


def test_strict_mode_missing_seed_raises():
    """Test that strict mode raises error when random seed is missing."""
    config_dict = _create_valid_strict_config()
    config_dict["reproducibility"]["random_seed"] = None

    with pytest.raises(ValueError, match="Explicit random seed is required"):
        AuditConfig(**config_dict)


def test_strict_mode_valid_config_passes():
    """Test that a fully valid strict mode configuration passes validation."""
    config_dict = _create_valid_strict_config()
    # Should not raise
    config = AuditConfig(**config_dict)
    assert config.strict_mode is True


def test_strict_mode_requires_deterministic():
    """Test that strict mode requires deterministic mode enabled."""
    config_dict = _create_valid_strict_config()
    config_dict["reproducibility"]["deterministic"] = False

    with pytest.raises(ValueError, match="Deterministic mode must be enabled"):
        AuditConfig(**config_dict)


def test_strict_mode_requires_capture_environment():
    """Test that strict mode requires environment capture."""
    config_dict = _create_valid_strict_config()
    config_dict["reproducibility"]["capture_environment"] = False

    with pytest.raises(ValueError, match="Environment capture must be enabled"):
        AuditConfig(**config_dict)


def test_strict_mode_requires_data_schema():
    """Test that strict mode requires data schema specification."""
    config_dict = _create_valid_strict_config()
    # Remove both schema_path and data_schema
    config_dict["data"].pop("schema_path", None)
    config_dict["data"].pop("data_schema", None)

    with pytest.raises(ValueError, match="Data schema must be specified"):
        AuditConfig(**config_dict)


def test_strict_mode_requires_protected_attributes():
    """Test that strict mode requires protected attributes for fairness analysis."""
    config_dict = _create_valid_strict_config()
    config_dict["data"]["protected_attributes"] = []

    with pytest.raises(ValueError, match="Protected attributes must be specified"):
        AuditConfig(**config_dict)


def test_strict_mode_requires_target_column():
    """Test that strict mode requires explicit target column."""
    config_dict = _create_valid_strict_config()
    config_dict["data"]["target_column"] = None

    with pytest.raises(ValueError, match="Target column must be explicitly specified"):
        AuditConfig(**config_dict)


def test_strict_mode_requires_explainer_priority():
    """Test that strict mode requires explainer priority list."""
    config_dict = _create_valid_strict_config()
    config_dict["explainers"]["priority"] = []

    with pytest.raises(ValueError, match="Explainer priority list must be specified"):
        AuditConfig(**config_dict)


def test_strict_mode_requires_first_compatible_strategy():
    """Test that strict mode enforces first_compatible explainer strategy for determinism."""
    config_dict = _create_valid_strict_config()
    config_dict["explainers"]["strategy"] = "best_available"

    with pytest.raises(ValueError, match="Explainer strategy must be 'first_compatible'"):
        AuditConfig(**config_dict)


def test_strict_mode_requires_manifest_enabled():
    """Test that strict mode requires manifest generation."""
    config_dict = _create_valid_strict_config()
    config_dict["manifest"]["enabled"] = False

    with pytest.raises(ValueError, match="Manifest generation must be enabled"):
        AuditConfig(**config_dict)


def test_strict_mode_requires_git_sha():
    """Test that strict mode requires git SHA in manifest."""
    config_dict = _create_valid_strict_config()
    config_dict["manifest"]["include_git_sha"] = False

    with pytest.raises(ValueError, match="Git SHA must be included in manifest"):
        AuditConfig(**config_dict)


def test_strict_mode_requires_config_hash():
    """Test that strict mode requires config hash in manifest."""
    config_dict = _create_valid_strict_config()
    config_dict["manifest"]["include_config_hash"] = False

    with pytest.raises(ValueError, match="Config hash must be included in manifest"):
        AuditConfig(**config_dict)


def test_strict_mode_requires_data_hash():
    """Test that strict mode requires data hash in manifest."""
    config_dict = _create_valid_strict_config()
    config_dict["manifest"]["include_data_hash"] = False

    with pytest.raises(ValueError, match="Data hash must be included in manifest"):
        AuditConfig(**config_dict)


def test_strict_mode_requires_audit_profile():
    """Test that strict mode requires audit profile specification."""
    config_dict = _create_valid_strict_config()
    # Use empty string instead of None (None would fail Pydantic validation)
    config_dict["audit_profile"] = ""

    with pytest.raises(ValueError, match="Audit profile must be specified"):
        AuditConfig(**config_dict)


def test_strict_mode_requires_performance_metrics():
    """Test that strict mode requires performance metrics."""
    config_dict = _create_valid_strict_config()
    # Use empty list which will pass Pydantic but fail strict mode check
    config_dict["metrics"]["performance"] = []

    with pytest.raises(ValueError, match="Performance metrics must be specified"):
        AuditConfig(**config_dict)


def test_strict_mode_requires_fairness_metrics():
    """Test that strict mode requires fairness metrics."""
    config_dict = _create_valid_strict_config()
    # Use empty list which will pass Pydantic but fail strict mode check
    config_dict["metrics"]["fairness"] = []

    with pytest.raises(ValueError, match="Fairness metrics must be specified"):
        AuditConfig(**config_dict)


def test_strict_mode_recourse_requires_immutables():
    """Test that strict mode requires immutable features when recourse is enabled."""
    config_dict = _create_valid_strict_config()
    config_dict["recourse"] = {
        "enabled": True,
        "immutable_features": [],  # Empty immutables with recourse enabled
    }

    with pytest.raises(ValueError, match="Immutable features must be specified when recourse is enabled"):
        AuditConfig(**config_dict)


def test_strict_mode_multiple_errors_reported():
    """Test that strict mode reports all validation errors at once."""
    config_dict = _create_valid_strict_config()
    # Introduce multiple errors (use empty string for audit_profile, not None)
    config_dict["reproducibility"]["random_seed"] = None
    config_dict["manifest"]["enabled"] = False
    config_dict["audit_profile"] = ""  # Empty string instead of None

    with pytest.raises(ValueError) as exc_info:
        AuditConfig(**config_dict)

    # Check that multiple errors are in the message
    error_msg = str(exc_info.value)
    assert "Explicit random seed is required" in error_msg
    assert "Manifest generation must be enabled" in error_msg
    assert "Audit profile must be specified" in error_msg


def test_strict_mode_requires_preprocessing_artifact():
    """Test that strict mode requires artifact preprocessing mode."""
    config_dict = _create_valid_strict_config()
    config_dict["preprocessing"]["mode"] = "auto"

    with pytest.raises(ValueError, match="Preprocessing mode must be 'artifact' in strict mode"):
        AuditConfig(**config_dict)


def test_strict_mode_requires_artifact_hashes():
    """Test that strict mode requires preprocessing artifact hashes."""
    config_dict = _create_valid_strict_config()
    config_dict["preprocessing"]["expected_file_hash"] = None

    with pytest.raises(ValueError, match="expected_file_hash must be specified"):
        AuditConfig(**config_dict)


# ============================================================================
# Integration Tests (New - comprehensive config scenarios)
# ============================================================================


def test_config_with_runtime_options():
    """Test config with runtime configuration options."""
    config_data = {
        "audit_profile": "tabular_compliance",
        "runtime": {
            "fast_mode": True,
            "compact_report": False,
            "no_fallback": True,
        },
        "model": {"type": "xgboost", "path": "model.pkl"},
        "data": {"dataset": "custom", "path": "data.csv"},
        "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"]},
        "reproducibility": {"random_seed": 42},
    }

    config = AuditConfig(**config_data)
    assert config.runtime.fast_mode is True
    assert config.runtime.compact_report is False
    assert config.runtime.no_fallback is True


def test_config_strict_mode_with_runtime():
    """Test strict mode with runtime configuration."""
    config_data = {
        "audit_profile": "tabular_compliance",
        "strict_mode": True,
        "runtime": {
            "fast_mode": False,
            "compact_report": True,
        },
        "model": {"type": "xgboost", "path": "model.pkl"},
        "data": {
            "dataset": "custom",
            "path": "data.csv",
            "target_column": "target",
            "protected_attributes": ["gender"],
            "schema_path": "schema.yaml",
        },
        "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"], "fairness": ["demographic_parity"]},
        "reproducibility": {"random_seed": 42, "deterministic": True, "capture_environment": True},
        "manifest": {"enabled": True, "include_git_sha": True, "include_config_hash": True, "include_data_hash": True},
        "preprocessing": {
            "mode": "artifact",
            "artifact_path": "preprocessor.joblib",
            "expected_file_hash": "sha256:test_hash",
            "expected_params_hash": "sha256:test_params_hash",
        },
    }

    config = AuditConfig(**config_data)
    assert config.strict_mode is True
    assert config.runtime.fast_mode is False
    assert config.runtime.compact_report is True


def test_config_defaults_applied():
    """Test that config defaults are properly applied."""
    config_data = {
        "audit_profile": "tabular_compliance",
        "model": {"type": "xgboost"},
        "data": {"dataset": "custom", "path": "data.csv"},
        "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"]},
        "reproducibility": {"random_seed": 42},
    }

    config = AuditConfig(**config_data)

    # Check that defaults were applied
    assert config.runtime.fast_mode is False  # default
    assert config.runtime.compact_report is True  # default
    assert config.runtime.no_fallback is False  # default
    assert config.strict_mode is False  # default
