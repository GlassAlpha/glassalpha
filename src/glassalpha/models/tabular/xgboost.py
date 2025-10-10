"""XGBoost model wrapper for GlassAlpha.

This wrapper enables XGBoost models to work within the GlassAlpha pipeline.
It supports SHAP explanations and provides feature importance capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

# Lazy import - xgboost is optional
# import xgboost as xgb
from .base import BaseTabularWrapper

logger = logging.getLogger(__name__)


def _import_xgboost():
    """Lazy import XGBoost with proper error handling."""
    try:
        import xgboost as xgb

        return xgb
    except ImportError as e:
        msg = (
            "XGBoost is required but not installed. "
            "Install with: pip install 'glassalpha[xgboost]' or pip install xgboost"
        )
        raise ImportError(msg) from e


def register_xgboost():
    """Register XGBoost model plugin."""
    try:
        _import_xgboost()  # Check if XGBoost is available
        return XGBoostWrapper
    except ImportError:
        # Return a dummy class that raises an error when instantiated
        class UnavailableXGBoost:
            def __init__(self, *args, **kwargs):
                msg = "XGBoost is not installed. Install with: pip install 'glassalpha[xgboost]' or pip install xgboost"
                raise ImportError(msg)

        return UnavailableXGBoost


class XGBoostWrapper(BaseTabularWrapper):
    """Wrapper for XGBoost models with GlassAlpha compatibility.

    This class wraps XGBoost models to make them compatible with the GlassAlpha
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

    # Contract compliance: Binary classification constants
    BINARY_CLASSES = 2
    BINARY_THRESHOLD = 0.5

    def __init__(self, model_path: str | Path | None = None, model: Any | None = None) -> None:
        """Initialize XGBoost wrapper.

        Args:
            model_path: Path to pre-trained XGBoost model file
            model: Pre-loaded XGBoost Booster object

        """
        super().__init__()
        self.model: Any | None = model
        self.feature_names: list | None = None
        self.feature_names_: list | None = None  # For sklearn compatibility
        self.n_classes_: int = 2  # Default to binary classification
        self.classes_: np.ndarray | None = None  # Original class labels

        if model_path:
            self.load(model_path)
        elif model:
            self.model = model
            self._is_fitted = True
            self._extract_model_info()

        if self.model:
            logger.info("XGBoostWrapper initialized with model")
        else:
            logger.info("XGBoostWrapper initialized without model")

    def _to_dmatrix(self, X, y=None):
        """Convert input to XGBoost DMatrix with proper feature alignment.

        Args:
            X: Input features (DataFrame, array, or existing DMatrix)
            y: Target values (optional)

        Returns:
            xgb.DMatrix: Properly formatted DMatrix with aligned features

        """
        xgb = _import_xgboost()
        if isinstance(X, xgb.DMatrix):
            return X
        X = pd.DataFrame(X)
        if getattr(self, "feature_names_", None):
            from ...utils.features import align_features  # lazy import  # noqa: PLC0415

            X = align_features(X, self.feature_names_)

        # Sanitize feature names for XGBoost (remove special characters [, ], <, >)
        feature_names = [
            str(col).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt") for col in X.columns
        ]
        return xgb.DMatrix(X, label=y, feature_names=feature_names)

    def _canonicalize_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize XGBoost parameters to handle aliases and ensure consistency.

        Args:
            kwargs: Raw parameters from configuration

        Returns:
            Canonicalized parameters

        """
        params = dict(kwargs)

        # Handle seed aliases
        if "random_state" not in params and "seed" in params:
            params["random_state"] = params["seed"]

        # Handle boosting rounds aliases
        if "n_estimators" not in params and "num_boost_round" in params:
            params["n_estimators"] = params["num_boost_round"]
            # Remove the original to avoid XGBoost warnings
            del params["num_boost_round"]

        # Keep num_class - do not filter it out, it's needed for validation
        # XGBClassifier will handle it appropriately

        return params

    def _validate_objective_compatibility(self, params: dict[str, Any], n_classes: int) -> None:
        """Validate that the objective is compatible with the number of classes.

        Args:
            params: XGBoost parameters including objective
            n_classes: Number of unique classes in target

        Raises:
            ValueError: If objective is incompatible with number of classes

        """
        objective = params.get("objective", "")

        # Binary objectives
        binary_objectives = ["binary:logistic", "binary:hinge", "binary:logitraw"]
        if objective in binary_objectives and n_classes != 2:
            raise ValueError(f"Binary objective '{objective}' incompatible with {n_classes} classes")

        # Multi-class objectives
        multiclass_objectives = ["multi:softmax", "multi:softprob"]
        if objective in multiclass_objectives and n_classes == 2:
            raise ValueError(f"Multi-class objective '{objective}' incompatible with {n_classes} classes")

    def _validate_num_class_parameter(self, params: dict[str, Any], n_classes: int) -> None:
        """Validate that num_class parameter matches observed classes.

        Args:
            params: XGBoost parameters including num_class
            n_classes: Number of unique classes in target

        Raises:
            ValueError: If num_class doesn't match observed classes

        """
        num_class = params.get("num_class")
        if num_class is not None:
            # Validate that num_class is reasonable
            if num_class <= 0:
                raise ValueError(f"num_class={num_class} must be > 0")
            if num_class != n_classes:
                raise ValueError(f"num_class={num_class} does not match observed classes={n_classes}")

    def fit(
        self,
        X: Any,
        y: Any,
        random_state: int | None = None,
        require_proba: bool = True,
        **kwargs: Any,
    ) -> "XGBoostWrapper":
        """Fit XGBoost model with training data using native XGBoost API.

        Args:
            X: Training features (DataFrame or array)
            y: Target values
            random_state: Random seed for reproducibility
            **kwargs: Additional XGBoost parameters

        Returns:
            Self for method chaining

        Raises:
            ValueError: If objective and num_class are inconsistent with data

        """
        # Convert inputs to proper format
        X = pd.DataFrame(X)
        self.feature_names_ = list(X.columns)
        self.classes_ = np.unique(y)  # Store original class labels
        self.n_classes = int(self.classes_.size)
        if self.n_classes <= 0:
            raise ValueError(f"Invalid number of classes: {self.n_classes}")
        self.n_classes_ = self.n_classes  # sklearn compatibility

        # Create DMatrix for training
        dtrain = self._to_dmatrix(X, y)

        # Set up training parameters
        params = {"objective": "binary:logistic" if self.n_classes == 2 else "multi:softprob"}
        if random_state is not None:
            params["seed"] = random_state

        # For multi-class objectives, num_class is required
        if self.n_classes > 2:
            params["num_class"] = self.n_classes
        elif self.n_classes == 2:
            # For binary, ensure num_class is not set (XGBoost handles binary automatically)
            params.pop("num_class", None)

        # Canonicalize user parameters before applying them
        kwargs = self._canonicalize_params(kwargs)

        # Add any additional parameters
        params.update(kwargs)

        # Handle softmax to softprob coercion for audit compatibility
        user_objective = params.get("objective")
        if user_objective == "multi:softmax" and self.n_classes > 2 and require_proba:
            logger.warning(
                "Coercing multi:softmax to multi:softprob for audit compatibility (predict_proba required)",
            )
            params["objective"] = "multi:softprob"

        # Validate objective compatibility with number of classes
        self._validate_objective_compatibility(params, self.n_classes)

        # Validate num_class parameter if provided by user and objective is multiclass
        user_num_class = kwargs.get("num_class")
        user_objective = kwargs.get("objective")
        if user_num_class is not None and (user_objective is None or "multi" in str(user_objective)):
            self._validate_num_class_parameter({"num_class": user_num_class}, self.n_classes)

        # Extract n_estimators from params (default to 50)
        num_boost_round = params.pop("n_estimators", 50)

        # Train the model using xgb.train (not sklearn wrapper)
        xgb = _import_xgboost()
        self.model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        self._is_fitted = True

        logger.info(f"Fitted XGBoost model: {self.n_classes} classes using native API")
        return self

    def load(self, path: str | Path) -> "XGBoostWrapper":
        """Load trained XGBoost model from saved file for inference and analysis.

        Loads a previously saved XGBoost booster model from disk, supporting both
        JSON and binary formats. The complete model structure including boosted
        trees, learned parameters, and training configuration is restored for
        immediate use in predictions and explainability analysis.

        Args:
            path: Path to XGBoost model file (supports .json, .model, .ubj extensions)

        Raises:
            FileNotFoundError: If the specified model file does not exist
            XGBoostError: If the model file is corrupted, incompatible, or malformed
            OSError: If file system permissions prevent reading the model file
            ValueError: If the model file format is unrecognized by XGBoost
            ImportError: If XGBoost library is not properly installed or configured
            JSONDecodeError: If JSON model file contains malformed JSON data

        Note:
            Model loading restores full training context including objective function,
            evaluation metrics, and feature importance calculations. Loaded models
            maintain compatibility with SHAP explainers and audit requirements.

        """
        path = Path(path)
        if not path.exists():
            msg = f"Model file not found: {path}"
            raise FileNotFoundError(msg)

        logger.info(f"Loading XGBoost model from {path}")

        path = Path(path)

        # For JSON files, use the test-specific format
        if path.suffix.lower() == ".json":
            # Load using the test-specific format
            with path.open() as f:
                saved_data = json.load(f)

            # Check that it has the expected structure
            if "model" in saved_data and isinstance(saved_data["model"], str):
                # Load the model from the JSON string by creating a temporary file
                import os
                import tempfile

                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
                    tmp_file.write(saved_data["model"])
                    tmp_file_path = tmp_file.name

                xgb = _import_xgboost()
                try:
                    self.model = xgb.Booster()
                    self.model.load_model(tmp_file_path)
                    self.feature_names_ = saved_data.get("feature_names_", [])
                    self.n_classes = saved_data.get("n_classes", None)
                    self._is_fitted = True
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_file_path)

                return self
            # Fall back to XGBoost's native format
            try:
                xgb = _import_xgboost()
                self.model = xgb.Booster()
                self.model.load_model(str(path))
                meta = json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))
                self.feature_names_ = meta.get("feature_names_", [])
                self.n_classes = meta.get("n_classes", None)
                self._is_fitted = True
            except (FileNotFoundError, json.JSONDecodeError):
                # Final fallback to direct XGBoost model loading without metadata
                xgb = _import_xgboost()
                self.model = xgb.Booster()
                self.model.load_model(str(path))
                self._is_fitted = True
                self._extract_model_info()

            return self

        # For other formats, try base wrapper's format first (joblib format)
        try:
            super().load(path)
            # Ensure the loaded model is an XGBoost Booster
            xgb = _import_xgboost()
            if not isinstance(self.model, xgb.Booster):
                raise ValueError("Loaded model is not an XGBoost Booster")
            logger.info(f"Loaded XGBoost model using base wrapper format from {path}")
            return self
        except Exception:
            # Fall back to XGBoost's native format
            logger.debug(f"Falling back to XGBoost native format for {path}")

        # Load using XGBoost's native format
        try:
            # Try to load with metadata
            xgb = _import_xgboost()
            self.model = xgb.Booster()
            self.model.load_model(str(path))
            meta = json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))
            self.feature_names_ = meta.get("feature_names_", [])
            self.n_classes = meta.get("n_classes", None)
            self._is_fitted = True
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback to direct XGBoost model loading without metadata
            xgb = _import_xgboost()
            self.model = xgb.Booster()
            self.model.load_model(str(path))
            self._is_fitted = True
            self._extract_model_info()

        return self

    def _extract_model_info(self) -> None:
        """Extract metadata from loaded model."""
        if self.model:
            # Try to get feature names
            try:
                self.feature_names = self.model.feature_names
            except Exception:  # noqa: BLE001
                # Graceful fallback - some models don't have feature names
                logger.debug("Could not extract feature names from model")

            # Try to determine number of classes
            try:
                # For binary classification, XGBoost often outputs single value
                # For multiclass, it outputs multiple values
                config = self.model.save_config()
                if '"num_class"' in config:
                    config_dict = json.loads(config)
                    learner = config_dict.get("learner", {})
                    learner_params = learner.get("learner_model_param", {})
                    num_class = learner_params.get("num_class", "0")
                    self.n_classes_ = max(2, int(num_class))
                else:
                    self.n_classes_ = 2  # Default to binary
            except Exception:  # noqa: BLE001
                # Graceful fallback - config parsing can fail for various reasons
                logger.debug("Could not determine number of classes, defaulting to binary")
                self.n_classes_ = 2

    def predict(self, X):
        """Generate predictions for input data.

        Args:
            X: Input features as DataFrame

        Returns:
            Array of predictions

        """
        self._ensure_fitted()
        dm = self._to_dmatrix(X)
        xgb = _import_xgboost()
        raw = self.model.predict(dm)
        if raw.ndim == 1:  # binary
            return (raw >= 0.5).astype(int)
        return np.argmax(raw, axis=1)

    def predict_proba(self, X):
        """Generate probability predictions for input data.

        Args:
            X: Input features as DataFrame

        Returns:
            Array of prediction probabilities (n_samples, n_classes)

        """
        self._ensure_fitted()
        dm = self._to_dmatrix(X)
        xgb = _import_xgboost()
        raw = self.model.predict(dm)
        if raw.ndim == 1:  # binary -> (n,2)
            return np.column_stack([1.0 - raw, raw])
        return raw

    def get_model_type(self) -> str:
        """Return the model type identifier.

        Returns:
            String identifier for XGBoost models

        """
        return "xgboost"

    def get_capabilities(self) -> dict[str, Any]:
        """Return model capabilities for plugin selection.

        Returns:
            Dictionary of capability flags

        """
        return self.capabilities

    def get_model_info(self) -> dict[str, Any]:
        """Get model information including n_classes.

        Returns:
            Dictionary with model information

        """
        return {
            "model_type": "xgboost",
            "n_classes": self.n_classes_,
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
            "status": "loaded" if self.model else "not_loaded",
            "n_features": len(self.feature_names_) if self.feature_names_ else None,
        }

    def get_feature_importance(self, importance_type: str = "weight") -> dict[str, float]:
        """Get feature importance scores from the model.

        Args:
            importance_type: Type of importance to return
                - "weight": Number of times feature is used to split
                - "gain": Average gain when feature is used
                - "cover": Average coverage when feature is used

        Returns:
            Dictionary mapping feature names to importance scores

        """
        self._ensure_fitted()

        # Get importance scores
        xgb = _import_xgboost()
        importance = self.model.get_score(importance_type=importance_type)

        logger.debug(f"Extracted {importance_type} feature importance for {len(importance)} features")
        return importance

    def save(self, path: str | Path) -> None:
        """Save trained XGBoost model with metadata for round-trip compatibility.

        Args:
            path: Target file path for model storage

        """
        self._ensure_fitted()

        path = Path(path)

        # For JSON files, save in the expected test format
        if path.suffix.lower() == ".json":
            # Save XGBoost model to file first
            xgb = _import_xgboost()
            self.model.save_model(str(path))

            # Read the saved model as JSON string
            model_json = path.read_text(encoding="utf-8")

            # Create the expected structure
            save_data = {
                "model": model_json,  # XGBoost model as JSON string
                "feature_names_": self.feature_names_,
                "n_classes": self.n_classes,
            }

            # Write the combined structure (overwrite the XGBoost file)
            with path.open("w") as f:
                json.dump(save_data, f, indent=2)

            logger.debug(
                f"Saved XGBoost model to {path} in test-compatible format",
            )
        else:
            # Use the base wrapper's save method for consistent format
            super().save(path)

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        status = "loaded" if self.model else "not loaded"
        return f"XGBoostWrapper(status={status}, n_classes={self.n_classes}, version={self.version})"


# Register XGBoost model with ModelRegistry at module import
try:
    from glassalpha.core.registry import ModelRegistry

    # Check if XGBoost is actually available before registering
    _import_xgboost()
    ModelRegistry.register(
        "xgboost",
        XGBoostWrapper,
        import_check="xgboost",
        extra_hint="xgboost",
    )
except ImportError:
    # XGBoost not available - model will show as unavailable in registry
    pass
