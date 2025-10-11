"""Test strict mode enforcement for preprocessing."""

from pathlib import Path

import pytest

# Import required modules
from glassalpha.config import AuditConfig, PreprocessingConfig
from glassalpha.preprocessing.validation import assert_runtime_versions


def test_strict_mode_fails_on_version_mismatch(mismatched_version_manifest: dict):
    """Patch manifest to simulate sklearn 1.3.2 vs runtime 1.5.0.
    In strict profile, expect RuntimeError. In non-strict, UserWarning.
    """
    # Strict mode: should raise
    with pytest.raises(RuntimeError) as exc_info:
        assert_runtime_versions(
            mismatched_version_manifest,
            strict=True,
            allow_minor=False,
        )

    error_msg = str(exc_info.value)
    assert "version" in error_msg.lower() or "mismatch" in error_msg.lower()
    assert "1.3.2" in error_msg or "1.5.0" in error_msg

    # Non-strict mode: should warn but not raise
    with pytest.warns(UserWarning):
        assert_runtime_versions(
            mismatched_version_manifest,
            strict=False,
            allow_minor=True,
        )


def test_strict_profile_blocks_auto_mode():
    """With profile=tabular_compliance and strict_mode=True, preprocessing.mode='auto' must error."""
    # Pydantic validation should fail when creating the config
    with pytest.raises(ValueError) as exc_info:
        AuditConfig(
            audit_profile="tabular_compliance",
            strict_mode=True,
            preprocessing=PreprocessingConfig(mode="auto"),
            model={"type": "xgboost", "path": "dummy.pkl"},
            data={"dataset": "custom", "path": "dummy.csv", "target_column": "target"},
        )

    error_msg = str(exc_info.value)
    assert "auto" in error_msg.lower() or "artifact" in error_msg.lower()


def test_strict_mode_requires_hashes():
    """Strict mode must require both file_hash and params_hash."""
    # Missing file_hash - Pydantic validation should fail
    with pytest.raises(ValueError, match="expected_file_hash must be specified"):
        AuditConfig(
            audit_profile="tabular_compliance",
            strict_mode=True,
            preprocessing=PreprocessingConfig(
                mode="artifact",
                artifact_path=Path("dummy.pkl"),
                expected_params_hash="sha256:abc123",
                # expected_file_hash missing
            ),
            model={"type": "xgboost", "path": "dummy.pkl"},
            data={"dataset": "custom", "path": "dummy.csv", "target_column": "target"},
        )

    # Missing params_hash - Pydantic validation should fail
    with pytest.raises(ValueError, match="expected_params_hash must be specified"):
        AuditConfig(
            audit_profile="tabular_compliance",
            strict_mode=True,
            preprocessing=PreprocessingConfig(
                mode="artifact",
                artifact_path=Path("dummy.pkl"),
                expected_file_hash="sha256:def456",
                # expected_params_hash missing
            ),
            model={"type": "xgboost", "path": "dummy.pkl"},
            data={"dataset": "custom", "path": "dummy.csv", "target_column": "target"},
        )
