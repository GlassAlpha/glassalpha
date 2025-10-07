"""Coefficients-based explainer for linear models.

This explainer uses the model's coefficients and intercept to generate
feature attributions for linear and logistic regression models.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from glassalpha.explain.base import ExplainerBase

logger = logging.getLogger(__name__)


class CoefficientsExplainer(ExplainerBase):
    """Explainer that uses model coefficients for linear/logistic regression."""

    name = "coefficients"
    priority = 10  # Low priority, used as fallback
    version = "1.0.0"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize coefficients explainer."""
        super().__init__()
        self.model = None
        self.feature_names = None
        self.is_fitted = False

    @classmethod
    def is_compatible(cls, *, model: Any = None, model_type: str | None = None, config: dict | None = None) -> bool:
        """Check if model is compatible with Coefficients explainer.

        Args:
            model: Model instance (optional)
            model_type: String model type identifier (optional)
            config: Configuration dict (optional, unused)

        Returns:
            True if model is a linear/logistic regression model

        Note:
            All arguments are keyword-only. Coefficients explainer works with linear models.

        """
        linear_models = {
            "logistic_regression",
            "logisticregression",
            "linear_model",
            "linearregression",
            "ridge",
            "lasso",
            "elasticnet",
            "sklearn.linear_model.logisticregression",  # Full sklearn class path
            "sklearn.linear_model.linearregression",  # Full sklearn class path
        }

        # Check model_type string if provided
        if model_type:
            return model_type.lower() in linear_models

        # Check model object if provided
        if model is not None:
            # Handle string model type
            if isinstance(model, str):
                return model.lower() in linear_models

            # For model objects, check model type via get_model_info()
            try:
                model_info = getattr(model, "get_model_info", dict)()
                extracted_type = model_info.get("model_type", "")
                if extracted_type:
                    return extracted_type.lower() in linear_models
            except Exception:  # noqa: BLE001
                pass

            # Fallback: check if model has coef_ attribute (indicator of linear model)
            underlying_model = getattr(model, "model", model)
            return hasattr(underlying_model, "coef_")

        # If neither provided, return False
        return False

    def fit(self, wrapper: Any, background_X: Any, feature_names: list[str] | None = None) -> CoefficientsExplainer:
        """Fit the explainer.

        Args:
            wrapper: Model wrapper with predict/predict_proba methods
            background_X: Background data for baseline (not used for coefficients)
            feature_names: Optional feature names

        Returns:
            self: Returns self for chaining

        """
        self.model = wrapper
        self.feature_names = feature_names
        self.is_fitted = True
        return self

    def explain(self, model: Any, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any) -> dict[str, Any]:
        """Generate explanations using model coefficients.

        Note: model and y parameters are accepted for API compatibility but not used by CoefficientsExplainer.

        Args:
            model: Model instance (unused, but required by interface)
            X: Input data to explain
            y: Target labels (unused, but accepted for compatibility)
            **kwargs: Additional parameters

        Returns:
            Dictionary containing explanation results

        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Explainer must be fitted before explaining")

        # Get model coefficients and intercept from the underlying model
        underlying_model = getattr(self.model, "model", self.model)
        coef_ = getattr(underlying_model, "coef_", None)
        intercept_ = getattr(underlying_model, "intercept_", None)

        if coef_ is None or (hasattr(coef_, "size") and coef_.size == 0):
            raise RuntimeError("Model does not have coefficients (coef_)")

        # Handle 2D coefficient arrays (e.g., from sklearn LogisticRegression with multiclass)
        # For multiclass, coef_ has shape (n_classes, n_features)
        # We'll average the coefficients across classes for global importance
        is_multiclass = coef_.ndim > 1 and coef_.shape[0] > 1
        if is_multiclass:
            # Average across classes for global importance
            coef_ = np.mean(np.abs(coef_), axis=0)
            logger.debug(
                f"Multiclass model detected, using mean absolute coefficients across {underlying_model.coef_.shape[0]} classes",
            )
        elif coef_.ndim > 1:
            # Single-row 2D array (binary classification in some frameworks)
            coef_ = coef_.flatten()

        # Handle intercept (could be scalar or array)
        if intercept_ is None:
            intercept_ = 0.0
        elif hasattr(intercept_, "__len__"):
            # For multiclass, average intercepts
            intercept_ = np.mean(intercept_)
        else:
            intercept_ = float(intercept_)

        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples, n_features = X.shape

        # Check if feature count matches coefficients
        n_coefficients = len(coef_)
        if n_features != n_coefficients:
            raise ValueError(
                f"Feature count mismatch: X has {n_features} features but model has {n_coefficients} coefficients. "
                f"This typically means the model was trained on a subset of features. "
                f"Ensure X contains only the features used during model training.",
            )

        # Compute feature attributions: coef * (X - mean_X)
        # Use mean of background data if available, otherwise use 0
        if hasattr(self.model, "_background_data"):
            background_data = self.model._background_data
        # Use mean of input data as baseline
        elif hasattr(X, "mean"):  # pandas DataFrame
            background_data = X.mean(axis=0).values.reshape(1, -1)
        else:  # numpy array
            background_data = np.mean(X, axis=0, keepdims=True)

        # Compute attributions: coef * (X - baseline)
        if hasattr(X, "values"):  # pandas DataFrame
            X_values = X.values
        else:  # numpy array
            X_values = X
        attributions = X_values - background_data
        attributions = attributions * coef_

        # Add intercept contribution (intercept * coef_0 equivalent)
        intercept_attribution = np.full((n_samples, 1), intercept_)

        # Combine attributions with intercept
        attributions = np.column_stack([intercept_attribution, attributions])

        # Set feature names
        if self.feature_names is not None:
            feature_names = ["intercept"] + list(self.feature_names)
        else:
            feature_names = ["intercept"] + [f"feature_{i}" for i in range(n_features)]

        # For global importance, we want a single scalar per feature
        # Take the mean absolute coefficient for each feature
        global_importance = {}
        for i, feature_name in enumerate(feature_names):
            if i == 0:  # Skip intercept
                continue
            coef = coef_[i - 1]  # Skip intercept in coefficients
            global_importance[feature_name] = float(abs(coef))  # Use absolute value for importance

        # For coefficients, we can compute per-sample attributions
        # Each sample's contribution is X[i] * coef_
        per_sample_attributions = []
        for i in range(n_samples):
            sample_dict = {}
            for j, feature_name in enumerate(feature_names):
                if j == 0:  # Skip intercept
                    continue
                # Attribution for this feature = coefficient * feature_value
                coef = coef_[j - 1]  # Skip intercept in coefficients
                sample_dict[feature_name] = float(coef * X_values[i, j - 1])
            per_sample_attributions.append(sample_dict)

        return {
            "local_explanations": attributions,
            "global_importance": global_importance,
            "feature_names": feature_names,
            "status": "coefficients_explanation",
        }

    def explain_local(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Generate local explanations.

        Args:
            X: Input data to explain
            **kwargs: Additional parameters

        Returns:
            Feature attributions for each sample

        """
        explanation = self.explain(self.model, X, **kwargs)
        return explanation["shap_values"]

    def supports_model(self, model: Any) -> bool:
        """Check if model is supported.

        Args:
            model: Model to check

        Returns:
            True if model has coef_ attribute (linear/logistic regression)

        """
        return hasattr(model, "coef_")
