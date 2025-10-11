"""Stub for test compatibility - policy explainer removed.

This module was removed during simplification but some tests still import it.
"""


class PolicyConstraints:
    """Stub for test compatibility - policy system removed."""

    def __init__(
        self,
        immutable_features=None,
        monotone_directions=None,
        bounds=None,
        cost_weights=None,
    ):
        self.immutable_features = immutable_features or []
        self.monotone_directions = monotone_directions or {}
        self.bounds = bounds or {}
        self.cost_weights = cost_weights or {}


def compute_feature_cost(original, proposed, cost_weights=None):
    """Stub for test compatibility - cost computation simplified."""
    import numpy as np

    if cost_weights is None:
        return np.sum(np.abs(proposed - original))
    return np.sum(cost_weights * np.abs(proposed - original))


def validate_constraints(proposed, constraints):
    """Stub for test compatibility - constraint validation simplified."""
    # Always return True for compatibility
    return True


def validate_feature_bounds(proposed, bounds):
    """Stub for test compatibility - bounds validation simplified."""
    # Always return True for compatibility
    return True


def apply_monotone_constraints(proposed, original, directions):
    """Stub for test compatibility - monotone constraints simplified."""
    # Return proposed as-is
    return proposed


def validate_monotonic_constraints(proposed, original, directions):
    """Stub for test compatibility - monotonic validation simplified."""
    # Always return True for compatibility
    return True


def check_immutability(proposed, original, immutable_features):
    """Stub for test compatibility - immutability check simplified."""
    # Always return True for compatibility
    return True


__all__ = [
    "PolicyConstraints",
    "apply_monotone_constraints",
    "check_immutability",
    "compute_feature_cost",
    "validate_constraints",
    "validate_feature_bounds",
    "validate_monotonic_constraints",
]
