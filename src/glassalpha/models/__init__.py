"""Model wrappers for GlassAlpha with explicit dispatch.

Replaces registry pattern with simple if/elif dispatch for AI maintainability.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from glassalpha.config import ModelConfig

from glassalpha.exceptions import ModelLoadError


def load_model(model_type: str, model_path: Path | None = None, **kwargs: Any) -> Any:
    """Load model by type with clear error messages.

    Args:
        model_type: Model type (logistic_regression, xgboost, lightgbm)
        model_path: Optional path to saved model file
        **kwargs: Additional model parameters

    Returns:
        Model wrapper instance

    Raises:
        ValueError: If model_type is unknown
        ImportError: If optional dependencies missing (with install instructions)
        ModelLoadError: If model loading fails

    Examples:
        >>> model = load_model("xgboost", model_path="model.pkl")
        >>> model = load_model("logistic_regression")

    """
    model_type = model_type.lower()

    # LogisticRegression (always available - no optional deps)
    if model_type in ("logistic_regression", "logistic", "sklearn_logistic"):
        from glassalpha.models.sklearn import LogisticRegressionWrapper

        wrapper = LogisticRegressionWrapper()
        if model_path and Path(model_path).exists():
            try:
                wrapper = LogisticRegressionWrapper.from_file(Path(model_path))
            except Exception as e:
                raise ModelLoadError(f"Failed to load LogisticRegression from {model_path}: {e}") from e
        return wrapper

    # XGBoost (optional dependency)
    if model_type in ("xgboost", "xgb"):
        try:
            from glassalpha.models.xgboost import XGBoostWrapper
        except ImportError as e:
            raise ImportError(
                "XGBoost not installed. Install with: pip install 'glassalpha[xgboost]' or pip install xgboost",
            ) from e

        wrapper = XGBoostWrapper()
        if model_path and Path(model_path).exists():
            try:
                wrapper = XGBoostWrapper.from_file(Path(model_path))
            except Exception as e:
                raise ModelLoadError(f"Failed to load XGBoost from {model_path}: {e}") from e
        return wrapper

    # LightGBM (optional dependency)
    if model_type in ("lightgbm", "lgb", "lgbm"):
        try:
            from glassalpha.models.lightgbm import LightGBMWrapper
        except ImportError as e:
            raise ImportError(
                "LightGBM not installed. Install with: pip install 'glassalpha[lightgbm]' or pip install lightgbm",
            ) from e

        wrapper = LightGBMWrapper()
        if model_path and Path(model_path).exists():
            try:
                wrapper = LightGBMWrapper.from_file(Path(model_path))
            except Exception as e:
                raise ModelLoadError(f"Failed to load LightGBM from {model_path}: {e}") from e
        return wrapper

    # Unknown model type
    raise ValueError(
        f"Unknown model_type: '{model_type}'. Supported: logistic_regression, xgboost, lightgbm",
    )


def load_model_from_config(model_config: ModelConfig) -> Any:
    """Load model from configuration object.

    Args:
        model_config: Configuration with model type and optional path

    Returns:
        Loaded model instance

    Raises:
        ModelLoadError: If loading fails

    Examples:
        >>> from glassalpha.config import load_config
        >>> config = load_config("audit.yaml")
        >>> model = load_model_from_config(config.model)

    """
    return load_model(
        model_type=model_config.type,
        model_path=model_config.path,
        **(model_config.params or {}),
    )


__all__ = [
    "ModelLoadError",
    "load_model",
    "load_model_from_config",
]
