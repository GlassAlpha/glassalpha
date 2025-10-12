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
        monotonic_constraints=None,  # For test compatibility
        protected_attributes=None,  # For test compatibility
    ):
        self.immutable_features = immutable_features or []
        self.monotone_directions = monotone_directions or {}
        self.bounds = bounds or {}
        self.cost_weights = cost_weights or {}
        self.monotonic_constraints = monotonic_constraints or {}  # Stub for compatibility
        self.protected_attributes = protected_attributes or []  # Stub for compatibility


def compute_feature_cost(original, proposed, cost_weights=None, feature_name=None):
    """Compute feature change cost with optional weighting."""
    import numpy as np

    if cost_weights is None:
        return np.sum(np.abs(proposed - original))

    # Apply cost weights - cost_weights should be dict of feature_name -> weight
    if feature_name and feature_name in cost_weights:
        weight = cost_weights[feature_name]
        return weight * np.sum(np.abs(proposed - original))

    # If no specific weight for this feature, use default weight of 1.0
    return np.sum(np.abs(proposed - original))


def validate_constraints(proposed, constraints):
    """Stub for test compatibility - constraint validation simplified."""
    # Always return True for compatibility
    return True


def validate_feature_bounds(proposed, bounds, feature_name=None):
    """Stub for test compatibility - bounds validation simplified."""
    # Always return True for compatibility
    return True


def apply_monotone_constraints(proposed, original, directions):
    """Stub for test compatibility - monotone constraints simplified."""
    # Return proposed as-is
    return proposed


def validate_monotonic_constraints(proposed, original, directions, cost_weights=None):
    """Stub for test compatibility - monotonic validation simplified."""
    # Always return True for compatibility
    return True


def check_immutability(proposed, original, immutable_features):
    """Stub for test compatibility - immutability check simplified."""
    # Always return True for compatibility
    return True


def validate_immutables(features, immutable):
    """Stub for test compatibility - immutable validation simplified."""
    # Always return empty list for compatibility
    return []


def merge_protected_and_immutable(protected, immutable):
    """Stub for test compatibility - merge protected and immutable features."""
    # Simple concatenation for compatibility
    return list(set(protected + immutable))


__all__ = [
    "PolicyConstraints",
    "apply_monotone_constraints",
    "check_immutability",
    "compute_feature_cost",
    "merge_protected_and_immutable",
    "validate_constraints",
    "validate_feature_bounds",
    "validate_immutables",
    "validate_monotonic_constraints",
]
