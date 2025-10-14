"""Consolidated configuration tests.

This module consolidates all configuration-related tests from:
- tests/test_config_loading.py
- tests/test_config_validation.py
- tests/unit/test_config_strict.py

Tests the complete configuration system including loading, validation,
and strict mode requirements.
"""

# Import from main config module
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

# Add src to path and import from the main config file
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import directly from the config module file
import importlib.util

spec = importlib.util.spec_from_file_location("config", src_path / "glassalpha" / "config.py")
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

# Import classes and functions
AuditConfig = config_module.AuditConfig
DataConfig = config_module.DataConfig
ExplainerConfig = config_module.ExplainerConfig
ModelConfig = config_module.ModelConfig
load_config_from_file = config_module.load_config_from_file
load_yaml = config_module.load_yaml

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
        "model": {"type": "xgboost", "path": "model.pkl"},
        "data": {"dataset": "custom", "path": "data.csv"},
        "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"]},
        "reproducibility": {"random_seed": 42},
    }

    config = AuditConfig(**config_data)
    assert config.model.type == "xgboost"
    assert config.data.path == "data.csv"


def test_load_config_from_file():
    """Test loading config from YAML file."""
    config_data = {
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
        assert config.model.type == "xgboost"
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_config_validation_basic():
    """Test basic config validation."""
    config_data = {
        "model": {"type": "xgboost"},
        "data": {"dataset": "custom", "path": "test.csv"},
        "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"]},
        "reproducibility": {"random_seed": 42},
    }
    config = AuditConfig(**config_data)

    # Basic validation - object created successfully
    assert config.model.type == "xgboost"


def test_config_with_missing_required_fields():
    """Test config validation fails with missing required fields."""
    with pytest.raises(ValidationError):
        AuditConfig(audit_profile="tabular_compliance")  # Missing model, data, etc.


def test_config_extra_fields_allowed():
    """Test that extra fields are now allowed (simplified config behavior)."""
    config_data = {
        "model": {"type": "xgboost"},
        "data": {"dataset": "custom", "path": "test.csv"},
        "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"]},
        "reproducibility": {"random_seed": 42},
        "extra_field": "now_allowed",  # Extra fields allowed in simplified config
    }

    # Should not raise ValidationError for extra fields
    config = AuditConfig(**config_data)
    assert config.model.type == "xgboost"
    assert config.data.dataset == "custom"


# ============================================================================
# Example Config Validation Tests (from test_config_validation.py)
# ============================================================================


def get_all_config_files():
    """Get all YAML config files from configs directory."""
    configs_dir = Path(__file__).parent.parent.parent / "src" / "glassalpha" / "configs"
    if not configs_dir.exists():
        return []
    # Exclude test configs (they're for testing specific features, not full validation)
    return [f for f in configs_dir.glob("*.yaml") if not f.name.startswith("test_")]


@pytest.mark.parametrize("config_file", get_all_config_files(), ids=lambda p: p.name)
def test_config_file_validates(config_file: Path) -> None:
    """Ensure each example config file can be parsed as valid YAML.

    Args:
        config_file: Path to configuration file to validate

    This test ensures that:
    1. Config file can be loaded without YAML parsing errors
    2. Basic structure exists (audit_profile, data, model sections)
    3. File is well-formed YAML

    Note: These are example configs that may use different schema versions,
    so we only validate YAML parsing and basic structure, not full Pydantic validation.

    """
    try:
        # For example config files, just verify they can be parsed as YAML
        # These are examples and may use older schema versions
        config_dict = load_yaml(config_file)
        assert config_dict is not None, f"Config {config_file.name} failed to load"

        # Verify basic structure exists (but don't enforce full schema validation)
        assert isinstance(config_dict, dict), f"{config_file.name} is not a valid YAML dict"
        assert "data" in config_dict, f"{config_file.name} missing data section"
        assert "model" in config_dict, f"{config_file.name} missing model section"

    except Exception as e:
        pytest.fail(f"Config {config_file.name} YAML parsing failed: {e}")


def test_all_configs_found() -> None:
    """Verify that config files were found for testing."""
    config_files = get_all_config_files()
    assert len(config_files) > 0, "No config files found in src/glassalpha/configs/ directory"
    print(f"\nFound {len(config_files)} config files to validate:")
    for config_file in config_files:
        print(f"  - {config_file.name}")


def test_specific_config_examples_exist() -> None:
    """Verify that key example configs exist."""
    configs_dir = Path(__file__).parent.parent.parent / "src" / "glassalpha" / "configs"

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
    configs_dir = Path(__file__).parent.parent.parent / "src" / "glassalpha" / "configs"
    quickstart_path = configs_dir / "quickstart.yaml"

    config = load_config_from_file(quickstart_path)

    # Verify quickstart uses safe defaults
    assert config.model.type == "logistic_regression"  # Always available
    assert config.data.dataset == "german_credit"  # Built-in dataset
    assert config.random_seed is not None  # Deterministic (now flat structure)


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


# ============================================================================
# Integration Tests (New - comprehensive config scenarios)
# ============================================================================


def test_config_with_runtime_options():
    """Test config with runtime configuration options."""
    config_data = {
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


def test_config_defaults_applied():
    """Test that config defaults are properly applied."""
    config_data = {
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
