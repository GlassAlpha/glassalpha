"""Permutation-based explainer for feature importance.

This explainer uses sklearn's permutation_importance to compute feature importance
by measuring how much the model's performance decreases when a feature's values
are randomly shuffled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance

from glassalpha.explain.base import ExplainerBase


@dataclass
class PermutationExplainer(ExplainerBase):
    """Explainer that uses permutation importance for feature importance."""

    name = "permutation"
    priority = 5  # Medium priority, used as fallback
    version = "1.0.0"

    n_repeats: int = 5
    random_state: int | None = 42

    def __init__(self, n_repeats: int = 5, random_state: int | None = 42, **kwargs: Any) -> None:
        """Initialize permutation explainer.

        Args:
            n_repeats: Number of times to permute each feature
            random_state: Random state for reproducible results
            **kwargs: Additional arguments

        """
        super().__init__(**kwargs)
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.model = None
        self.feature_names: Sequence[str] | None = None
        self.is_fitted = False

    @classmethod
    def is_compatible(cls, *, model: Any = None, model_type: str | None = None, config: dict | None = None) -> bool:
        """Check if model is compatible with Permutation explainer.

        Permutation importance works with any model that has predict/predict_proba methods.

        Args:
            model: Model instance (optional)
            model_type: String model type identifier (optional)
            config: Configuration dict (optional, unused)

        Returns:
            True if model has predict/predict_proba methods

        """
        if model is not None:
            # Check if model has the required methods
            return hasattr(model, "predict")

        # If no model provided, assume compatible (will check at fit time)
        return True

    def fit(self, wrapper: Any, background_X: Any, feature_names: Sequence[str] | None = None) -> PermutationExplainer:
        """Fit the explainer.

        Args:
            wrapper: Model wrapper with predict/predict_proba methods
            background_X: Background data for baseline (not used for permutation)
            feature_names: Optional feature names

        Returns:
            self: Returns self for chaining

        """
        self.model = wrapper
        self.feature_names = feature_names
        self.is_fitted = True
        return self

    def explain(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any) -> dict[str, Any]:
        """Generate explanations using permutation importance.

        Args:
            X: Input data to explain
            y: Target values (required for permutation importance)
            **kwargs: Additional parameters (show_progress, strict_mode)

        Returns:
            Dictionary containing explanation results

        Raises:
            RuntimeError: If explainer not fitted
            ValueError: If y is not provided

        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Explainer must be fitted before explaining")

        if y is None:
            raise ValueError(
                "PermutationExplainer requires target values (y) to compute feature importance. "
                "Pass y as: explainer.explain(X, y=y_test)",
            )

        # Use the wrapper model, not the underlying model
        # sklearn's permutation_importance needs an object with predict/predict_proba
        # Our wrappers provide this interface even for XGBoost/LightGBM
        estimator = self.model

        # Determine appropriate scoring method
        # Scorers must have signature: scorer(estimator, X, y) -> float
        if hasattr(estimator, "predict_proba"):
            # For classifiers, use negative log loss
            from sklearn.metrics import log_loss

            def scorer(estimator: Any, X: np.ndarray, y: np.ndarray) -> float:
                y_pred = estimator.predict_proba(X)
                return -log_loss(y, y_pred)
        else:
            # For regressors, use negative mean squared error
            from sklearn.metrics import mean_squared_error

            def scorer(estimator: Any, X: np.ndarray, y: np.ndarray) -> float:
                y_pred = estimator.predict(X)
                return -mean_squared_error(y, y_pred)

        # Get progress settings from kwargs
        show_progress = kwargs.get("show_progress", True)
        strict_mode = kwargs.get("strict_mode", False)

        # Import progress utility
        from glassalpha.utils.progress import get_progress_bar, is_progress_enabled

        # Determine if we should show progress (only for large datasets)
        progress_enabled = is_progress_enabled(strict_mode) and show_progress and len(X) > 1000

        # Compute permutation importance with optional progress
        if progress_enabled:
            # Show progress indicator (permutation_importance doesn't support callbacks)
            with get_progress_bar(total=X.shape[1] * self.n_repeats, desc="Computing Permutation", leave=False) as pbar:
                r = permutation_importance(
                    estimator,
                    X,
                    y,
                    n_repeats=self.n_repeats,
                    random_state=self.random_state,
                    scoring=scorer,
                )
                pbar.update(X.shape[1] * self.n_repeats)
        else:
            r = permutation_importance(
                estimator,
                X,
                y,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
                scoring=scorer,
            )

        # Set feature names
        if self.feature_names is not None:
            feature_names = list(self.feature_names)
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # For permutation importance, we need to create a baseline prediction
        # We'll use the mean prediction as baseline for feature importance
        baseline_pred = estimator.predict(X)
        if baseline_pred.ndim == 1:
            baseline_pred = baseline_pred.reshape(-1, 1)

        # Create attributions by computing the decrease in score for each feature
        importances = r.importances_mean
        std = r.importances_std

        # Normalize importances (optional, but often helpful)
        # We'll use absolute values for weighting
        abs_importances = np.abs(importances)

        # Create feature attributions for each sample
        # For permutation importance, we compute per-sample attributions
        # by measuring how much each feature contributes to the prediction

        # Convert X to numpy array if it's a DataFrame
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        n_samples, n_features = X_array.shape

        # For each sample, compute attribution as: feature_value * importance_weight
        # This is a simplified approach - in practice you'd want more sophisticated methods
        attributions = []
        for i in range(n_samples):
            sample_attributions = {}
            for j, feature_name in enumerate(feature_names):
                # Attribution = feature_value * normalized_importance
                importance_weight = importances[j] / max(abs_importances) if max(abs_importances) > 0 else 0
                sample_attributions[feature_name] = float(X_array[i, j] * importance_weight)
            attributions.append(sample_attributions)

        return {
            "local_explanations": attributions,
            "global_importance": {feature_names[i]: float(abs_importances[i]) for i in range(n_features)},
            "feature_names": feature_names,
            "status": "permutation_explanation",
            "importances": [float(importances[i]) for i in range(n_features)],
            "importances_std": [float(std[i]) for i in range(n_features)],
        }

    def explain_local(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
        """Generate local explanations.

        Args:
            X: Input data to explain
            y: Target values (required for permutation importance)
            **kwargs: Additional parameters

        Returns:
            Feature attributions for each sample

        """
        explanation = self.explain(X, y=y, **kwargs)
        return explanation["local_explanations"]

    def supports_model(self, model: Any) -> bool:
        """Check if model is supported.

        Args:
            model: Model to check

        Returns:
            True if model has predict method

        """
        return hasattr(model, "predict")
