"""Load and hash preprocessing artifacts."""

import hashlib
from pathlib import Path
from typing import Any

import joblib


def load_artifact(path: Path) -> Any:
    """Load preprocessing artifact from path.

    Args:
        path: Path to joblib-serialized artifact

    Returns:
        Loaded artifact (typically sklearn Pipeline or ColumnTransformer)

    Raises:
        FileNotFoundError: If artifact doesn't exist
        ValueError: If artifact is corrupted or unloadable

    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Preprocessing artifact not found: {path}")

    try:
        artifact = joblib.load(path)
        return artifact
    except Exception as e:
        raise ValueError(f"Corrupted or unloadable preprocessing artifact: {path}. Error: {e}")


def compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute file hash of artifact.

    Args:
        path: Path to artifact file
        algorithm: Hash algorithm (sha256 or blake2b)

    Returns:
        Hash string with algorithm prefix (e.g., "sha256:abc123...")

    """
    path = Path(path)

    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "blake2b":
        hasher = hashlib.blake2b()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return f"{algorithm}:{hasher.hexdigest()}"


def compute_params_hash(artifact: Any) -> str:
    """Compute canonical hash over artifact's learned parameters.

    This hash is stable across repickling (unlike file hash).

    Args:
        artifact: Loaded sklearn preprocessing artifact

    Returns:
        Hash string (e.g., "sha256:def456...")

    """
    from glassalpha.preprocessing.introspection import extract_sklearn_manifest
    from glassalpha.preprocessing.manifest import params_hash

    manifest = extract_sklearn_manifest(artifact)
    return params_hash(manifest)
