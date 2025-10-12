"""PassThroughModel for testing (stub for test compatibility).

This module provides a simple pass-through model for testing purposes.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class PassThroughModel(BaseEstimator, ClassifierMixin):
    """A simple pass-through model for testing.

    Always predicts the positive class for demonstration.
    """

    def __init__(self):
        """Initialize PassThroughModel."""

    def fit(self, X, y):
        """Fit the model (no-op)."""
        return self

    def predict(self, X):
        """Predict positive class for all samples."""
        if hasattr(X, "shape"):
            return np.ones(X.shape[0])
        return np.array([1])

    def predict_proba(self, X):
        """Predict high probability for positive class."""
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X) if hasattr(X, "__len__") else 1
        return np.array([[0.1, 0.9]] * n_samples)
