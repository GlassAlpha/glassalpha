"""Test preprocessing artifact loading and hashing."""

from pathlib import Path

import pytest

# Import stubs - these will fail until we implement
try:
    from glassalpha.preprocessing.loader import (
        compute_file_hash,
        compute_params_hash,
        load_artifact,
    )
except ImportError:
    pytest.skip("preprocessing module not implemented yet", allow_module_level=True)


def test_hashes_present_and_stable(sklearn_artifact: Path, tmp_path: Path):
    """Load saved artifact twice. File hash may differ if repickled, but params hash must be identical."""
    import joblib

    # Load original artifact
    artifact1 = load_artifact(sklearn_artifact)
    file_hash1 = compute_file_hash(sklearn_artifact)
    params_hash1 = compute_params_hash(artifact1)

    # Repickle the artifact (simulates different environment)
    repickled_path = tmp_path / "repickled.pkl"
    joblib.dump(artifact1, repickled_path)

    # Load repickled artifact
    artifact2 = load_artifact(repickled_path)
    file_hash2 = compute_file_hash(repickled_path)
    params_hash2 = compute_params_hash(artifact2)

    # File hashes may differ (pickle metadata)
    # But params hashes MUST be identical (logical equivalence)
    assert params_hash1 == params_hash2, "Params hash must be stable across repickling"

    # Both hashes must be present and formatted correctly
    assert file_hash1.startswith("sha256:") or file_hash1.startswith("blake2b:")
    assert params_hash1.startswith("sha256:")


def test_missing_artifact_is_actionable_error():
    """Clear message including path when artifact not found."""
    fake_path = Path("/nonexistent/artifact.pkl")

    with pytest.raises((FileNotFoundError, ValueError)) as exc_info:
        load_artifact(fake_path)

    # Error message must include the path
    assert str(fake_path) in str(exc_info.value) or "not found" in str(exc_info.value).lower()


def test_corrupted_artifact_is_actionable_error(corrupted_artifact: Path):
    """Clear 'unloadable' error, not a stack dump."""
    with pytest.raises((ValueError, RuntimeError, Exception)) as exc_info:
        load_artifact(corrupted_artifact)

    # Should not be a raw pickle error
    error_msg = str(exc_info.value).lower()
    assert "corrupted" in error_msg or "unloadable" in error_msg or "invalid" in error_msg
