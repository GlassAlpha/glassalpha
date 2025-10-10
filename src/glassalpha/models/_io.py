"""Serialization utilities for GlassAlpha model wrappers.

This module provides consistent save/load helpers to prevent
serialization regressions across different wrapper implementations.
"""

import json
from pathlib import Path
from typing import Any


def write_wrapper_state(
    path: Path | str,
    model_str: str,
    feature_names: list[str] | None,
    n_classes: int | None,
    **extra_fields: Any,  # noqa: ANN401
) -> None:
    """Write wrapper state to JSON file with consistent structure.

    Args:
        path: File path to write to
        model_str: Serialized model string (JSON or other format)
        feature_names: List of feature names from training
        n_classes: Number of classes for classification
        **extra_fields: Additional fields to include in the state

    Note:
        Creates parent directories if they don't exist.
        Always includes the three core fields: model, feature_names_, n_classes.

    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model": model_str,
        "feature_names_": feature_names,
        "n_classes": n_classes,
        **extra_fields,
    }

    path_obj.write_text(json.dumps(state, indent=2), encoding="utf-8")


def read_wrapper_state(path: Path | str) -> tuple[str, list[str] | None, int | None]:
    """Read wrapper state from JSON file.

    Args:
        path: File path to read from

    Returns:
        Tuple of (model_str, feature_names, n_classes)

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        KeyError: If required fields are missing

    """
    path_obj = Path(path)
    data = json.loads(path_obj.read_text(encoding="utf-8"))

    model_str = data["model"]
    feature_names = data["feature_names_"]
    n_classes = data["n_classes"]

    return model_str, feature_names, n_classes


def read_wrapper_state_with_extras(path: Path | str) -> dict[str, Any]:
    """Read complete wrapper state including extra fields.

    Args:
        path: File path to read from

    Returns:
        Complete state dictionary with all saved fields

    """
    path_obj = Path(path)
    return json.loads(path_obj.read_text(encoding="utf-8"))


def validate_wrapper_state(state: dict[str, Any]) -> bool:
    """Validate that wrapper state contains required fields.

    Args:
        state: State dictionary to validate

    Returns:
        True if state contains all required fields

    """
    required_fields = {"model", "feature_names_", "n_classes"}
    return all(field in state for field in required_fields)


def create_temp_model_file(model_content: str, suffix: str = ".json") -> Path:
    """Create temporary file with model content for loading.

    Args:
        model_content: Model content to write to temp file
        suffix: File suffix (e.g., ".json", ".model")

    Returns:
        Path to temporary file (caller responsible for cleanup)

    """
    import tempfile  # noqa: PLC0415

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w", encoding="utf-8") as temp_file:
        temp_file.write(model_content)
        temp_file.flush()
    temp_file.close()

    return Path(temp_file.name)
