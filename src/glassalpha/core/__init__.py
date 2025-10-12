"""Core architecture components for GlassAlpha.

Simplified for AI maintainability - registries removed in favor of explicit dispatch.

This module provides backwards compatibility stubs for tests.
"""

from ..data.base import DataInterface


# Backwards compatibility stubs for tests
class ModelRegistry:
    """Stub for test compatibility - registry system removed."""

    @staticmethod
    def get(model_type: str):
        from glassalpha.models import load_model

        return load_model(model_type)

    @staticmethod
    def discover():
        pass  # No-op for compatibility

    @staticmethod
    def available_plugins():
        return {
            "logistic_regression": True,
            "xgboost": True,
            "lightgbm": True,
        }


class ExplainerRegistry:
    """Stub for test compatibility - registry system removed."""

    @staticmethod
    def get(explainer_name: str):
        from glassalpha.explain import select_explainer

        # Map old names to model types
        if "tree" in explainer_name.lower():
            return select_explainer("xgboost")
        if "kernel" in explainer_name.lower():
            return select_explainer("logistic_regression")
        if "coeff" in explainer_name.lower():
            return select_explainer("logistic_regression")
        return select_explainer("logistic_regression")

    @staticmethod
    def discover():
        pass  # No-op for compatibility

    @staticmethod
    def find_compatible(model_type: str, config=None):
        from glassalpha.explain import select_explainer

        return select_explainer(model_type, config)

    @staticmethod
    def available_plugins():
        return {
            "treeshap": True,
            "kernelshap": True,
            "coefficients": True,
        }


class MetricRegistry:
    """Stub for test compatibility - registry system removed."""

    @staticmethod
    def discover():
        pass  # No-op for compatibility


class ProfileRegistry:
    """Stub for test compatibility - registry system removed."""

    @staticmethod
    def discover():
        pass  # No-op for compatibility


# NoOp components for test compatibility
class PassThroughModel:
    """NoOp model for testing."""

    def __init__(self, default_value=0.5):
        self.default_value = default_value

    def predict(self, X):
        import numpy as np

        return np.full(len(X), self.default_value)

    def predict_proba(self, X):
        import numpy as np

        proba = self.default_value
        return np.column_stack([1 - proba, proba] * len(X))


class NoOpMetric:
    """NoOp metric for testing."""

    def compute(self, **kwargs):
        return {"noop": 0.0}


def list_components(component_type=None, include_enterprise=False):
    """List available components.

    Args:
        component_type: Filter by type (models, explainers, metrics, profiles) or None for all
        include_enterprise: Include enterprise components (not implemented yet)

    Returns:
        Dictionary of component types and their available options

    """
    all_components = {
        "models": ["logistic_regression", "xgboost", "lightgbm"],
        "explainers": ["treeshap", "kernelshap", "coefficients"],
        "metrics": [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc_roc",
            "demographic_parity",
            "equal_opportunity",
            "equalized_odds",
        ],
        "profiles": ["tabular_compliance"],
    }

    if component_type:
        return {component_type: all_components.get(component_type, [])}

    return all_components


def select_explainer(model_type: str, config=None):
    """Select explainer (stub for compatibility)."""
    from glassalpha.explain import select_explainer as _select

    return _select(model_type, config)


def instantiate_explainer(explainer_name: str):
    """Instantiate explainer (stub for compatibility)."""
    return ExplainerRegistry.get(explainer_name)


__all__ = [
    "DataInterface",
    "ExplainerRegistry",
    "MetricRegistry",
    "ModelRegistry",
    "NoOpMetric",
    "PassThroughModel",
    "ProfileRegistry",
    "instantiate_explainer",
    "list_components",
    "select_explainer",
]
