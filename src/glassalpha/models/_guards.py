"""Stub model guards for test compatibility.

This module provides backwards compatibility for tests that expect
model guard functionality that was simplified.
"""

from collections.abc import Callable

# BaseModel import removed - was not used in stub


def guards(*guard_names: str):
    """Decorator that applies model guards (stub implementation).

    This is a simplified stub that just returns the function unchanged.
    The real implementation had complex guard logic that was removed during simplification.
    """

    def decorator(func: Callable) -> Callable:
        return func

    return decorator


def fitted_state_guard(func: Callable) -> Callable:
    """Guard that checks fitted state (stub implementation).

    This is a simplified stub that just returns the function unchanged.
    The real implementation had complex state checking logic that was removed during simplification.
    """
    return func


# Export for compatibility
__all__ = ["fitted_state_guard", "guards"]
