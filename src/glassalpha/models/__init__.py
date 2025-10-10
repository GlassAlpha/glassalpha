"""Model wrappers for GlassAlpha."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from glassalpha.core.registry import ModelRegistry
from glassalpha.exceptions import GlassAlphaError


class ModelLoadError(GlassAlphaError):
    """Raised when model loading fails."""


# Discover models from entry points
ModelRegistry.discover()


def load_model_from_config(model_config) -> Any:
    """Load model from configuration.

    Args:
        model_config: Model configuration object with type and optional path

    Returns:
        Loaded or trained model instance

    Raises:
        ValueError: If model type is unknown or loading fails

    """
    model_type = model_config.type
    model_path = getattr(model_config, "path", None)

    # Get model class from registry (auto-imports if needed)
    model_class = ModelRegistry.get(model_type)
    if not model_class:
        msg = f"Unknown model type: {model_type}"
        raise ModelLoadError(msg)

    if model_path and Path(model_path).exists():
        # Load existing model
        try:
            model = model_class.from_file(Path(model_path))
        except Exception as e:
            raise ModelLoadError(f"Failed to load model from {model_path}: {e}") from e
    else:
        # Return untrained model wrapper - caller is responsible for training
        model = model_class()

    return model


__all__ = ["ModelLoadError", "ModelRegistry", "load_model_from_config"]
