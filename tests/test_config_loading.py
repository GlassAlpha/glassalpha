"""Configuration loading and validation tests.

Tests that config loading works correctly and covers loader.py and schema.py.
These tests focus on exercising the config validation paths.
"""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from glassalpha.config.loader import load_yaml
from glassalpha.config.schema import AuditConfig, DataConfig, ExplainerConfig, ModelConfig


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
