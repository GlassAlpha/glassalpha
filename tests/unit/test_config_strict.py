"""Unit tests for config strict mode validation."""

import warnings

import pytest

from glassalpha.config.schema import AuditConfig
from glassalpha.config.strict import (
    StrictModeError,
    enforce_strict_defaults,
    validate_deterministic_config,
    validate_reproducible_environment,
    validate_strict_mode,
)


def _create_valid_strict_config() -> dict:
    """Create a fully valid strict mode configuration."""
    return {
        "audit_profile": "tabular_compliance",
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

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Explicit random seed is required"):
        validate_strict_mode(config)


def test_strict_mode_valid_config_passes():
    """Test that a fully valid strict mode configuration passes validation."""
    config_dict = _create_valid_strict_config()
    config = AuditConfig(**config_dict)

    # Should not raise
    validate_strict_mode(config)


def test_strict_mode_requires_deterministic():
    """Test that strict mode requires deterministic mode enabled."""
    config_dict = _create_valid_strict_config()
    config_dict["reproducibility"]["deterministic"] = False

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Deterministic mode must be enabled"):
        validate_strict_mode(config)


def test_strict_mode_requires_capture_environment():
    """Test that strict mode requires environment capture."""
    config_dict = _create_valid_strict_config()
    config_dict["reproducibility"]["capture_environment"] = False

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Environment capture must be enabled"):
        validate_strict_mode(config)


def test_strict_mode_requires_data_path():
    """Test that strict mode requires explicit data path."""
    # Skip this test - Pydantic already validates data path is required
    # when dataset="custom", so this check is redundant in strict mode validation
    pytest.skip("Data path validation is already handled by Pydantic schema")


def test_strict_mode_requires_data_schema():
    """Test that strict mode requires data schema specification."""
    config_dict = _create_valid_strict_config()
    # Remove both schema_path and data_schema
    config_dict["data"].pop("schema_path", None)
    config_dict["data"].pop("data_schema", None)

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Data schema must be specified"):
        validate_strict_mode(config)


def test_strict_mode_requires_protected_attributes():
    """Test that strict mode requires protected attributes for fairness analysis."""
    config_dict = _create_valid_strict_config()
    config_dict["data"]["protected_attributes"] = []

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Protected attributes must be specified"):
        validate_strict_mode(config)


def test_strict_mode_requires_target_column():
    """Test that strict mode requires explicit target column."""
    config_dict = _create_valid_strict_config()
    config_dict["data"]["target_column"] = None

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Target column must be explicitly specified"):
        validate_strict_mode(config)


def test_strict_mode_requires_model_path():
    """Test that strict mode requires model path specification."""
    config_dict = _create_valid_strict_config()
    config_dict["model"]["path"] = None

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Model path must be specified"):
        validate_strict_mode(config)


def test_strict_mode_requires_explainer_priority():
    """Test that strict mode requires explainer priority list."""
    config_dict = _create_valid_strict_config()
    config_dict["explainers"]["priority"] = []

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Explainer priority list must be specified"):
        validate_strict_mode(config)


def test_strict_mode_requires_first_compatible_strategy():
    """Test that strict mode enforces first_compatible explainer strategy for determinism."""
    config_dict = _create_valid_strict_config()
    config_dict["explainers"]["strategy"] = "best_available"

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Explainer strategy must be 'first_compatible'"):
        validate_strict_mode(config)


def test_strict_mode_requires_manifest_enabled():
    """Test that strict mode requires manifest generation."""
    config_dict = _create_valid_strict_config()
    config_dict["manifest"]["enabled"] = False

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Manifest generation must be enabled"):
        validate_strict_mode(config)


def test_strict_mode_requires_git_sha():
    """Test that strict mode requires git SHA in manifest."""
    config_dict = _create_valid_strict_config()
    config_dict["manifest"]["include_git_sha"] = False

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Git SHA must be included in manifest"):
        validate_strict_mode(config)


def test_strict_mode_requires_config_hash():
    """Test that strict mode requires config hash in manifest."""
    config_dict = _create_valid_strict_config()
    config_dict["manifest"]["include_config_hash"] = False

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Config hash must be included in manifest"):
        validate_strict_mode(config)


def test_strict_mode_requires_data_hash():
    """Test that strict mode requires data hash in manifest."""
    config_dict = _create_valid_strict_config()
    config_dict["manifest"]["include_data_hash"] = False

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Data hash must be included in manifest"):
        validate_strict_mode(config)


def test_strict_mode_requires_audit_profile():
    """Test that strict mode requires audit profile specification."""
    config_dict = _create_valid_strict_config()
    # Use empty string instead of None (None would fail Pydantic validation)
    config_dict["audit_profile"] = ""

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Audit profile must be specified"):
        validate_strict_mode(config)


def test_strict_mode_requires_performance_metrics():
    """Test that strict mode requires performance metrics."""
    config_dict = _create_valid_strict_config()
    # Use empty list which will pass Pydantic but fail strict mode check
    config_dict["metrics"]["performance"] = []

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Performance metrics must be specified"):
        validate_strict_mode(config)


def test_strict_mode_requires_fairness_metrics():
    """Test that strict mode requires fairness metrics."""
    config_dict = _create_valid_strict_config()
    # Use empty list which will pass Pydantic but fail strict mode check
    config_dict["metrics"]["fairness"] = []

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Fairness metrics must be specified"):
        validate_strict_mode(config)


def test_strict_mode_recourse_requires_immutables():
    """Test that strict mode requires immutable features when recourse is enabled."""
    config_dict = _create_valid_strict_config()
    config_dict["recourse"] = {
        "enabled": True,
        "immutable_features": [],  # Empty immutables with recourse enabled
    }

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError, match="Immutable features must be specified when recourse is enabled"):
        validate_strict_mode(config)


def test_strict_mode_multiple_errors_reported():
    """Test that strict mode reports all validation errors at once."""
    config_dict = _create_valid_strict_config()
    # Introduce multiple errors (use empty string for audit_profile, not None)
    config_dict["reproducibility"]["random_seed"] = None
    config_dict["manifest"]["enabled"] = False
    config_dict["audit_profile"] = ""  # Empty string instead of None

    config = AuditConfig(**config_dict)

    with pytest.raises(StrictModeError) as exc_info:
        validate_strict_mode(config)

    # Check that multiple errors are in the message
    error_msg = str(exc_info.value)
    assert "Explicit random seed is required" in error_msg
    assert "Manifest generation must be enabled" in error_msg
    assert "Audit profile must be specified" in error_msg


def test_strict_mode_converts_warnings_to_errors():
    """Test that strict mode converts warnings to errors."""
    config_dict = _create_valid_strict_config()
    config = AuditConfig(**config_dict)

    # This should succeed but set warnings to error mode
    validate_strict_mode(config)

    # After validation, warnings should be set to error mode
    # Verify by triggering a warning (should raise)
    with pytest.raises(UserWarning):
        warnings.warn("Test warning", UserWarning)

    # Reset warnings for other tests
    warnings.resetwarnings()


def test_validate_deterministic_config_no_seed():
    """Test that deterministic validation catches missing seed."""
    config = {"reproducibility": {"random_seed": None}}

    result = validate_deterministic_config(config)

    assert result is False


def test_validate_deterministic_config_no_explainer_priority():
    """Test that deterministic validation catches missing explainer priority."""
    config = {
        "reproducibility": {"random_seed": 42},
        "explainers": {"priority": []},
    }

    result = validate_deterministic_config(config)

    assert result is False


def test_validate_deterministic_config_set_metrics():
    """Test that deterministic validation catches set-based metrics."""
    config = {
        "reproducibility": {"random_seed": 42},
        "explainers": {"priority": ["treeshap"]},
        "metrics": {"performance": {"accuracy", "f1"}},  # Using set instead of list
    }

    result = validate_deterministic_config(config)

    assert result is False


def test_validate_deterministic_config_valid():
    """Test that deterministic validation passes for valid config."""
    config = {
        "reproducibility": {"random_seed": 42},
        "explainers": {"priority": ["treeshap"]},
        "metrics": {
            "performance": ["accuracy"],  # List, not set
            "fairness": ["demographic_parity"],
        },
    }

    result = validate_deterministic_config(config)

    assert result is True


def test_validate_reproducible_environment():
    """Test that environment validation runs without error."""
    # Should run successfully (may return True or False depending on environment)
    result = validate_reproducible_environment()
    assert isinstance(result, bool)


def test_enforce_strict_defaults():
    """Test that enforce_strict_defaults applies all necessary settings."""
    config_dict = {
        "audit_profile": "tabular_compliance",
        "model": {"type": "logistic_regression", "path": "/tmp/model.pkl"},
        "data": {
            "dataset": "custom",
            "path": "/tmp/data.csv",
            "target_column": "target",
            "protected_attributes": ["gender"],
        },
        "reproducibility": {"random_seed": 42},
        "explainers": {"priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"], "fairness": ["demographic_parity"]},
    }

    config = AuditConfig(**config_dict)
    config = enforce_strict_defaults(config)

    # Verify strict defaults are applied
    assert config.reproducibility.deterministic is True
    assert config.reproducibility.capture_environment is True
    assert config.manifest.enabled is True
    assert config.manifest.include_git_sha is True
    assert config.manifest.include_config_hash is True
    assert config.manifest.include_data_hash is True
    assert config.manifest.include_model_hash is True
    assert config.explainers.strategy == "first_compatible"
    assert "lineage" in config.report.include_sections
    assert "fairness" in config.report.include_sections


def test_enforce_strict_defaults_preserves_seed():
    """Test that enforce_strict_defaults doesn't modify the random seed."""
    config_dict = {
        "audit_profile": "tabular_compliance",
        "model": {"type": "logistic_regression", "path": "/tmp/model.pkl"},
        "data": {
            "dataset": "custom",
            "path": "/tmp/data.csv",
            "target_column": "target",
            "protected_attributes": ["gender"],
        },
        "reproducibility": {"random_seed": 12345},
        "explainers": {"priority": ["treeshap"]},
        "metrics": {"performance": ["accuracy"], "fairness": ["demographic_parity"]},
    }

    config = AuditConfig(**config_dict)
    config = enforce_strict_defaults(config)

    # Seed should be unchanged
    assert config.reproducibility.random_seed == 12345
