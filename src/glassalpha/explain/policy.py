"""Shared policy constraints for reason codes and recourse.

This module provides common policy constraint validation used by both
E2 (Reason Codes) and E2.5 (Recourse) features.

Policy types:
- Immutable features: Cannot be changed (age, gender, race, etc.)
- Monotonic constraints: Can only increase or decrease (income, debt)
- Cost weights: Relative difficulty of changing features
- Bounds: Valid ranges for feature values

Architecture note:
This module is intentionally kept separate from reason_codes.py to support
future recourse implementation without circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PolicyConstraints:
    """Policy constraints for feature modifications.

    Used by both reason codes (to validate exclusions) and recourse
    (to enforce feasible counterfactuals).

    Attributes:
        immutable_features: Features that cannot be changed
        monotonic_constraints: Features with directional constraints
        feature_costs: Relative difficulty of changing each feature
        feature_bounds: Valid ranges for feature values

    """

    immutable_features: list[str]
    monotonic_constraints: dict[str, Literal["increase_only", "decrease_only"]]
    feature_costs: dict[str, float]
    feature_bounds: dict[str, tuple[float, float]]


def validate_immutables(
    features: list[str],
    immutable_features: list[str],
) -> list[str]:
    """Validate that immutable features are not in the feature list.

    Args:
        features: List of feature names to check
        immutable_features: List of immutable feature names

    Returns:
        List of violations (immutable features found in features list)

    Examples:
        >>> validate_immutables(["age", "debt"], ["age", "gender"])
        ['age']
        >>> validate_immutables(["debt", "income"], ["age", "gender"])
        []

    """
    immutable_set = {f.lower() for f in immutable_features}
    violations = []

    for feature in features:
        if feature.lower() in immutable_set:
            violations.append(feature)

    return violations


def validate_monotonic_constraints(
    feature: str,
    old_value: float,
    new_value: float,
    monotonic_constraints: dict[str, Literal["increase_only", "decrease_only"]],
) -> bool:
    """Validate that a feature change respects monotonic constraints.

    Args:
        feature: Feature name
        old_value: Current feature value
        new_value: Proposed new feature value
        monotonic_constraints: Monotonic constraint dict

    Returns:
        True if change is valid, False otherwise

    Examples:
        >>> constraints = {"income": "increase_only", "debt": "decrease_only"}
        >>> validate_monotonic_constraints("income", 30000, 35000, constraints)
        True
        >>> validate_monotonic_constraints("income", 35000, 30000, constraints)
        False
        >>> validate_monotonic_constraints("debt", 5000, 3000, constraints)
        True
        >>> validate_monotonic_constraints("age", 25, 30, constraints)
        True  # No constraint on age

    """
    if feature not in monotonic_constraints:
        return True  # No constraint

    constraint = monotonic_constraints[feature]

    if constraint == "increase_only":
        return new_value >= old_value
    if constraint == "decrease_only":
        return new_value <= old_value

    return True


def compute_feature_cost(
    feature: str,
    old_value: float,
    new_value: float,
    feature_costs: dict[str, float],
    default_cost: float = 1.0,
) -> float:
    """Compute the cost of changing a feature value.

    Cost is weighted by the relative difficulty of changing the feature
    and the magnitude of the change (L1 distance).

    Args:
        feature: Feature name
        old_value: Current feature value
        new_value: Proposed new feature value
        feature_costs: Cost weights per feature (0.0 to 1.0+)
        default_cost: Default cost for features without explicit weights

    Returns:
        Cost of the change (weighted L1 distance)

    Examples:
        >>> costs = {"debt": 0.5, "income": 0.8}
        >>> compute_feature_cost("debt", 5000, 4000, costs)
        500.0  # |5000 - 4000| * 0.5
        >>> compute_feature_cost("income", 30000, 35000, costs)
        4000.0  # |30000 - 35000| * 0.8
        >>> compute_feature_cost("age", 25, 26, costs)
        1.0  # |25 - 26| * 1.0 (default)

    """
    cost_weight = feature_costs.get(feature, default_cost)
    change_magnitude = abs(new_value - old_value)
    return change_magnitude * cost_weight


def validate_feature_bounds(
    feature: str,
    value: float,
    feature_bounds: dict[str, tuple[float, float]],
) -> bool:
    """Validate that a feature value is within acceptable bounds.

    Args:
        feature: Feature name
        value: Feature value to check
        feature_bounds: Dict of (min, max) bounds per feature

    Returns:
        True if value is within bounds (or no bounds defined)

    Examples:
        >>> bounds = {"age": (18, 100), "income": (0, 1000000)}
        >>> validate_feature_bounds("age", 25, bounds)
        True
        >>> validate_feature_bounds("age", 15, bounds)
        False
        >>> validate_feature_bounds("debt", 5000, bounds)
        True  # No bounds defined

    """
    if feature not in feature_bounds:
        return True  # No bounds defined

    min_val, max_val = feature_bounds[feature]
    return min_val <= value <= max_val


def merge_protected_and_immutable(
    protected_attributes: list[str],
    immutable_features: list[str],
) -> list[str]:
    """Merge protected attributes and immutable features into single list.

    Protected attributes (ECOA) and immutable features (recourse) are
    conceptually similar but serve different purposes. This helper merges
    them for unified policy enforcement.

    Args:
        protected_attributes: ECOA protected attributes (age, gender, etc.)
        immutable_features: Policy-defined immutable features

    Returns:
        Deduplicated list of features that cannot be changed

    Examples:
        >>> merge_protected_and_immutable(["age", "gender"], ["age", "nationality"])
        ['age', 'gender', 'nationality']

    """
    # Use case-insensitive matching for deduplication
    combined = {}
    for attr in protected_attributes + immutable_features:
        key = attr.lower()
        if key not in combined:
            combined[key] = attr  # Keep first occurrence (preserve case)

    return list(combined.values())
