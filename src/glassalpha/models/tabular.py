"""Tabular model registry for sklearn-compatible models.

This module provides access to sklearn-based models for tabular data.
"""

from __future__ import annotations

# Re-export sklearn models for backward compatibility
from glassalpha.models.sklearn import LogisticRegressionWrapper

__all__ = [
    "LogisticRegressionWrapper",
    "sklearn",
]

# For backward compatibility with CLI imports
sklearn = LogisticRegressionWrapper
