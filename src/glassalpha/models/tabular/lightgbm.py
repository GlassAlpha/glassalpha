"""LightGBM model wrapper for GlassAlpha.

This wrapper enables LightGBM models to work within the GlassAlpha pipeline.
It supports SHAP explanations and provides feature importance capabilities.
"""

import logging
from pathlib import Path
from typing import Any, ClassVar

# Lazy import - lightgbm is optional
# import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import BaseTabularWrapper

logger = logging.getLogger(__name__)


def _import_lightgbm():
    """Lazy import LightGBM with proper error handling."""
    try:
        import lightgbm as lgb

        return lgb
    except ImportError as e:
        msg = (
            "LightGBM is required but not installed. "
            "Install with: pip install 'glassalpha[lightgbm]' or pip install lightgbm"
        )
        raise ImportError(msg) from e


def register_lightgbm():
    """Register LightGBM model plugin."""
    try:
        _import_lightgbm()  # Check if LightGBM is available
        return LightGBMWrapper
    except ImportError:
        # Return a dummy class that raises an error when instantiated
        class UnavailableLightGBM:
            def __init__(self, *args, **kwargs):
                msg = (
                    "LightGBM is not installed. "
                    "Install with: pip install 'glassalpha[lightgbm]' or pip install lightgbm"
                )
                raise ImportError(msg)

        return UnavailableLightGBM


class LightGBMWrapper(BaseTabularWrapper):
    """Wrapper for LightGBM models with GlassAlpha compatibility.

    This class wraps LightGBM models to make them compatible with the GlassAlpha
    audit pipeline. It supports loading pre-trained models, predictions, and
    capability declaration for plugin selection.
    """

    # Model capabilities
    capabilities: ClassVar[dict[str, Any]] = {
        "supports_shap": True,
        "supports_feature_importance": True,
        "supports_proba": True,
        "data_modality": "tabular",
    }
    version = "1.0.0"
    # Constants for magic values
    BINARY_CLASS_COUNT = 2
    BINARY_THRESHOLD = 0.5
    NDIM_MULTICLASS = 2

    # Friend's spec: Mark as trainable (has fit method)
    trainable_in_pipeline = True

    def __init__(self, model_path: str | Path | None = None, model: Any | None = None) -> None:
        """Initialize LightGBM wrapper.

        Args:
            model_path: Path to pre-trained LightGBM model file
            model: Pre-loaded LightGBM Booster object

        """
        super().__init__()
        self.model: Any | None = model
        self.feature_names: list | None = None
        self.feature_names_: list | None = None  # For sklearn compatibility
        self.n_classes: int = 2  # Default to binary classification
        self._feature_name_mapping: dict[str, str] | None = None  # Maps original -> sanitized names

        if model_path:
            self.load(model_path)
        elif model:
            self.model = model
            self._is_fitted = True  # Pre-loaded model is considered fitted
            self._extract_model_info()

        if self.model:
            logger.info("LightGBMWrapper initialized with model")
        else:
            logger.info("LightGBMWrapper initialized without model")

    @staticmethod
    def _sanitize_feature_names(feature_names: list[str]) -> tuple[list[str], dict[str, str]]:
        """Sanitize feature names for LightGBM compatibility.

        LightGBM doesn't accept special JSON characters in feature names.
        This method replaces problematic characters with safe alternatives.

        Args:
            feature_names: Original feature names

        Returns:
            Tuple of (sanitized_names, mapping from original to sanitized)

        """
        # Characters that need to be replaced
        replacements = {
            "=": "_eq_",
            "<": "_lt_",
            ">": "_gt_",
            ".": "_",
            ":": "_",
            '"': "_",
            "[": "_",
            "]": "_",
            "{": "_",
            "}": "_",
            ",": "_",
            " ": "_",
        }

        sanitized = []
        mapping = {}

        for name in feature_names:
            sanitized_name = name
            for char, replacement in replacements.items():
                sanitized_name = sanitized_name.replace(char, replacement)

            # Remove duplicate underscores
            while "__" in sanitized_name:
                sanitized_name = sanitized_name.replace("__", "_")

            # Remove leading/trailing underscores
            sanitized_name = sanitized_name.strip("_")

            sanitized.append(sanitized_name)
            mapping[name] = sanitized_name

        return sanitized, mapping

    def load(self, path: str | Path) -> "LightGBMWrapper":
        """Load trained LightGBM model from saved file.

        Contract compliance: Supports both old text format and new JSON format
        with complete wrapper state {"model", "feature_names_", "n_classes"}.

        Args:
            path: Path to model file

        Returns:
            Self for method chaining

        Raises:
            FileNotFoundError: If the specified model file does not exist

        """
        path = Path(path)
        if not path.exists():
            msg = f"Model file not found: {path}"
            raise FileNotFoundError(msg)

        logger.info(f"Loading LightGBM model from {path}")

        # Try to load as JSON format first (new format)
        try:
            import json  # noqa: PLC0415
            import tempfile  # noqa: PLC0415

            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Contract compliance: Load wrapper state
            model_str = data["model"]
            self.feature_names_ = data["feature_names_"]
            self.n_classes = data["n_classes"]

            # Write model string to temp file and load
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
                tmp.write(model_str)
                tmp.flush()
                lgb = _import_lightgbm()
                self.model = lgb.Booster(model_file=tmp.name)

            # Clean up temp file
            Path(tmp.name).unlink(missing_ok=True)

        except (json.JSONDecodeError, KeyError):
            # Fall back to old text format
            lgb = _import_lightgbm()
            self.model = lgb.Booster(model_file=str(path))
            self._extract_model_info()

        self._is_fitted = True  # Loaded model is fitted
        return self

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "LightGBMWrapper":  # noqa: ANN401
        """Train LightGBM model on provided data.

        Args:
            X: Training features (DataFrame preferred for feature names)
            y: Training targets
            **kwargs: Additional parameters including random_state

        """
        # Handle random_state - LightGBM doesn't have set_params but we handle it via params dict
        num_classes = len(np.unique(y))
        params = {
            "objective": "binary" if num_classes == self.BINARY_CLASS_COUNT else "multiclass",
            "verbose": -1,
            "force_row_wise": True,
        }

        # For multiclass, we must specify num_class parameter
        if num_classes > self.BINARY_CLASS_COUNT:
            params["num_class"] = num_classes

        if "random_state" in kwargs:
            params["random_seed"] = kwargs["random_state"]

        # Capture feature names from DataFrame
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
            self.feature_names = list(X.columns)  # Also set for compatibility

            # Sanitize feature names for LightGBM (it doesn't accept special JSON characters)
            sanitized_names, self._feature_name_mapping = self._sanitize_feature_names(list(X.columns))
            X_processed = self._prepare_x(X)
            # Rename columns with sanitized names
            if hasattr(X_processed, "columns"):
                X_processed.columns = sanitized_names
        else:
            X_processed = self._prepare_x(X)
            self._feature_name_mapping = None

        lgb = _import_lightgbm()
        train_data = lgb.Dataset(X_processed, label=y)

        # Train model
        self.model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(0)],  # Silent training
        )

        # Set fitted state - use len(set(y)) as fallback since LightGBM doesn't have classes_
        self._is_fitted = True
        self.n_classes = len(getattr(self.model, "classes_", [])) or len(set(y))
        n_features = len(X_processed.columns) if hasattr(X_processed, "columns") else X_processed.shape[1]
        logger.info(f"Trained LightGBM model with {n_features} features")
        return self

    def _extract_model_info(self) -> None:
        """Extract metadata from loaded model."""
        if self.model:
            # Try to get feature names
            try:
                self.feature_names = self.model.feature_name()
            except Exception:  # noqa: BLE001
                # Graceful fallback - some models don't have feature names
                logger.debug("Could not extract feature names from model")

            # Try to determine number of classes
            try:
                # LightGBM stores the number of classes in the model
                num_class = self.model.num_model_per_iteration()
                if num_class > 1:
                    # Multi-class model
                    self.n_classes = num_class
                else:
                    # Binary classification typically has 1 model per iteration
                    self.n_classes = self.BINARY_CLASS_COUNT
            except Exception:  # noqa: BLE001
                # Graceful fallback - determine from prediction shape later
                logger.debug("Could not determine number of classes, defaulting to binary")
                self.n_classes = self.BINARY_CLASS_COUNT

    # Removed custom _prepare_x - now uses base class version with centralized align_features

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for input data.

        Args:
            X: Input features as DataFrame

        Returns:
            Array of predictions

        """
        self._ensure_fitted()

        # Use _prepare_X for robust feature handling
        X_processed = self._prepare_x(X)

        # Get raw predictions
        predictions = self.model.predict(X_processed, num_iteration=self.model.best_iteration)

        # Handle different prediction formats
        if predictions.ndim == 1:
            # Binary classification - convert probabilities to class predictions
            if np.all((predictions >= 0) & (predictions <= 1)):
                predictions = (predictions > self.BINARY_THRESHOLD).astype(int)
        elif predictions.ndim == self.NDIM_MULTICLASS:
            # Multi-class - take argmax
            predictions = np.argmax(predictions, axis=1)

        logger.debug(f"Generated predictions for {len(X)} samples")
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions for input data.

        Args:
            X: Input features as DataFrame

        Returns:
            Array of prediction probabilities (n_samples, n_classes)

        """
        self._ensure_fitted()

        # Use _prepare_X for robust feature handling
        X_processed = self._prepare_x(X)

        # Get raw predictions (probabilities)
        predictions = self.model.predict(X_processed, num_iteration=self.model.best_iteration)

        # Handle binary vs multiclass
        proba = np.column_stack([1 - predictions, predictions]) if predictions.ndim == 1 else predictions

        logger.debug(f"Generated probability predictions for {len(X)} samples")
        return proba

    def get_model_type(self) -> str:
        """Return the model type identifier.

        Returns:
            String identifier for LightGBM models

        """
        return "lightgbm"

    def get_capabilities(self) -> dict[str, Any]:
        """Return model capabilities for plugin selection.

        Returns:
            Dictionary of capability flags

        """
        return self.capabilities

    def get_model_info(self) -> dict[str, Any]:
        """Return model information including n_classes.

        Returns:
            Dictionary with model information

        """
        return {
            "model_type": "lightgbm",
            "n_classes": self.n_classes,
            "status": "loaded" if self.model else "not_loaded",
            "n_features": len(self.feature_names_) if self.feature_names_ else None,
        }

    def get_feature_importance(self, importance_type: str = "split") -> dict[str, float]:
        """Get feature importance scores from the model.

        Args:
            importance_type: Type of importance to return
                - "split": Number of times feature is used to split
                - "gain": Total gain when feature is used to split

        Returns:
            Dictionary mapping feature names to importance scores

        """
        self._ensure_fitted()

        # Get importance scores
        importance = self.model.feature_importance(importance_type=importance_type)

        # Get feature names
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importance))]

        # Create dictionary
        importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance, strict=False)}

        logger.debug(f"Extracted {importance_type} feature importance for {len(importance_dict)} features")
        return importance_dict

    def save(self, path: str | Path) -> None:
        """Save trained LightGBM model to disk with complete wrapper state.

        Contract compliance: Saves {"model", "feature_names_", "n_classes"}
        for proper round-trip serialization.

        Args:
            path: Target file path for model storage (JSON format)

        Raises:
            ValueError: If no trained model exists to save

        """
        from glassalpha.constants import NO_MODEL_MSG  # noqa: PLC0415

        if self.model is None:
            raise ValueError(NO_MODEL_MSG)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model to temp file then read as string
        import json  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            self.model.save_model(tmp.name)
            model_str = Path(tmp.name).read_text(encoding="utf-8")

        # Clean up temp file
        Path(tmp.name).unlink(missing_ok=True)

        # Contract compliance: Save wrapper state with required fields
        data = {
            "model": model_str,
            "feature_names_": self.feature_names_,
            "n_classes": self.n_classes,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved LightGBM model to {path}")

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        status = "loaded" if self.model else "not loaded"
        return f"LightGBMWrapper(status={status}, n_classes={self.n_classes}, version={self.version})"


# Register LightGBM model with ModelRegistry at module import
try:
    from glassalpha.core.registry import ModelRegistry

    # Check if LightGBM is actually available before registering
    _import_lightgbm()
    ModelRegistry.register(
        "lightgbm",
        LightGBMWrapper,
        import_check="lightgbm",
        extra_hint="lightgbm",
    )
except ImportError:
    # LightGBM not available - model will show as unavailable in registry
    pass
