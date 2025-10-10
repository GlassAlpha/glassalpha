"""Simple feature flag system for OSS/Enterprise separation.

This module provides a minimal feature gating mechanism to separate
open-source and enterprise features without complex licensing infrastructure.
"""

import os
from collections.abc import Callable
from functools import wraps


class FeatureNotAvailable(Exception):
    """Raised when attempting to use an enterprise feature without license."""


def is_enterprise() -> bool:
    """Check if enterprise features are enabled.

    Returns:
        True if enterprise license key is present

    """
    return bool(os.environ.get("GLASSALPHA_LICENSE_KEY"))


def check_feature(feature_name: str, message: str | None = None):
    """Decorator to gate enterprise features.

    Args:
        feature_name: Name of the feature being gated
        message: Optional custom error message

    Returns:
        Decorator function that checks for enterprise license

    Example:
        @check_feature("advanced_explainers")
        def deep_shap_explain(model, data):
            # Enterprise-only implementation
            ...

    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_enterprise():
                error_msg = message or (
                    f"Feature '{feature_name}' requires an enterprise license. "
                    f"Contact sales@glassalpha.ai for licensing options."
                )
                raise FeatureNotAvailable(error_msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator
