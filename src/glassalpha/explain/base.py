"""Base classes and interfaces for explainers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class ExplainerBase:
    """Base class for all explainers with expected interface contract.

    This defines the interface that tests expect all explainers to implement.
    Tests specifically check for 'priority' as a class attribute and 'explainer'
    instance attribute that starts as None before fit() is called.
    """

    # Tests expect these as class attributes
    priority: int = 0
    version: str = "1.0.0"

    @classmethod
    def is_compatible(cls, *, model: Any = None, model_type: str | None = None, config: dict | None = None) -> bool:
        """Check if this explainer is compatible with the given model.

        Args:
            model: The model instance (optional)
            model_type: String model type identifier (optional)
            config: Configuration dict (optional)

        Returns:
            True if compatible, False otherwise

        Note:
            All arguments are keyword-only to prevent signature drift.
            Subclasses must implement this method with the same signature.

        """
        raise NotImplementedError(f"{cls.__name__} must implement is_compatible classmethod")

    def fit(self, wrapper: Any, background_X, feature_names: Sequence[str] | None = None):
        """Fit the explainer with a model wrapper and background data.

        Args:
            wrapper: Model wrapper with predict/predict_proba methods
            background_X: Background data for explainer baseline
            feature_names: Optional feature names for interpretation

        Returns:
            self: Returns self for chaining

        """
        raise NotImplementedError

    def explain(self, X, y=None, **kwargs):
        """Generate explanations for input data.

        Args:
            X: Input data to explain
            y: Target values (optional, required by some explainers like PermutationExplainer)
            **kwargs: Additional parameters for explanation generation

        Returns:
            Dictionary containing explanation results

        """
        raise NotImplementedError
