"""Model state guard decorators for GlassAlpha.

This module provides decorators to enforce model state contracts
and prevent silent misuse of wrapper methods.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from glassalpha.constants import ERR_NOT_FITTED

F = TypeVar("F", bound=Callable[..., Any])


def requires_fitted(func: F) -> F:
    """Decorator to ensure model is fitted/loaded before method execution.

    Args:
        func: Method to wrap

    Returns:
        Wrapped method that checks fitted state

    Raises:
        ValueError: If model is not loaded/fitted with exact contract message

    """

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if getattr(self, "model", None) is None:
            raise ValueError(ERR_NOT_FITTED)
        return func(self, *args, **kwargs)

    return wrapper


def requires_fitted_or_has_is_fitted_flag(func: F) -> F:
    """Decorator for models that use _is_fitted flag instead of model presence.

    Args:
        func: Method to wrap

    Returns:
        Wrapped method that checks fitted state via _is_fitted flag

    Raises:
        ValueError: If model is not fitted with exact contract message

    """

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        # Check _is_fitted flag first, then model presence
        if (hasattr(self, "_is_fitted") and not getattr(self, "_is_fitted", False)) or (
            not hasattr(self, "_is_fitted") and getattr(self, "model", None) is None
        ):
            raise ValueError(ERR_NOT_FITTED)
        return func(self, *args, **kwargs)

    return wrapper
