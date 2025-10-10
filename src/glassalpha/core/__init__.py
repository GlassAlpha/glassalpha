"""Core architecture components for GlassAlpha.

This module provides the foundational interfaces, registries, and
feature management system that enable the plugin architecture.
"""

# Import models package to trigger registration of available models
# This ensures models are registered when core is imported
import glassalpha.models  # noqa: F401

from ..data.base import DataInterface
from .features import (
    FeatureNotAvailable,
    check_feature,
    is_enterprise,
)

# Import NoOp components to auto-register them
from .noop_components import (
    NoOpMetric,
    PassThroughModel,
)
from .registry import (
    DataRegistry,
    ModelRegistry,
    instantiate_explainer,
    list_components,
    select_explainer,
)


# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    """Lazy import registries from their canonical locations."""
    if name == "ExplainerRegistry":
        from ..explain.registry import ExplainerRegistry

        return ExplainerRegistry
    if name == "MetricRegistry":
        from ..metrics.registry import MetricRegistry

        return MetricRegistry
    if name == "ProfileRegistry":
        from ..profiles.registry import ProfileRegistry

        return ProfileRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Data interface
    "DataInterface",
    # Registries
    "ModelRegistry",
    "ExplainerRegistry",
    "MetricRegistry",
    "ProfileRegistry",
    "DataRegistry",
    # Registry utilities
    "instantiate_explainer",
    "list_components",
    "select_explainer",
    # Feature management
    "is_enterprise",
    "check_feature",
    "FeatureNotAvailable",
    # NoOp implementations
    "PassThroughModel",
    "NoOpMetric",
]
