"""Counterfactual recourse generation for ECOA compliance (E2.5).

This module generates feasible counterfactual recommendations with policy constraints
for individuals receiving adverse decisions. Supports ECOA requirements for actionable
recourse and aligns with SR 11-7 guidance.

Algorithm:
- Greedy search with policy constraints (gradient-free)
- Identifies negative contributors using E2 (reason codes)
- Generates candidate changes respecting immutables and monotonic constraints
- Scores candidates by cost (weighted L1 distance)
- Returns top-N recommendations sorted by cost

Policy Constraints:
- Immutables: Cannot change (age, gender, race, etc.)
- Monotonic: Can only increase or decrease (income, debt)
- Bounds: Valid ranges for feature values
- Costs: Relative difficulty of changing features

Architecture:
- Reuses E2 reason codes for identifying features to change
- Reuses shared policy validation from policy.py
- Pure functions for deterministic output
- Seeded randomness for reproducibility
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .policy import (
    PolicyConstraints,
    compute_feature_cost,
    validate_feature_bounds,
    validate_monotonic_constraints,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Protocol for Model Interface
# ============================================================================


class ModelProtocol(Protocol):
    """Protocol for model with predict_proba method."""

    def predict_proba(self, X: pd.DataFrame | pd.Series) -> np.ndarray:
        """Predict probability for instances.

        Args:
            X: Feature DataFrame or Series

        Returns:
            Array of shape (n_samples, n_classes) with predicted probabilities

        """


# ============================================================================
# Data Classes
# ============================================================================


@dataclass(frozen=True)
class RecourseRecommendation:
    """Single counterfactual recommendation with cost and feasibility.

    Attributes:
        feature_changes: Dict of feature -> (old_value, new_value) tuples
        total_cost: Total cost of all feature changes (weighted L1)
        predicted_probability: Model prediction for counterfactual
        feasible: Whether recommendation passes threshold
        rank: Importance rank (1 = lowest cost)

    """

    feature_changes: dict[str, tuple[float, float]]
    total_cost: float
    predicted_probability: float
    feasible: bool
    rank: int


@dataclass(frozen=True)
class RecourseResult:
    """Complete recourse generation result with audit trail.

    Attributes:
        instance_id: Unique instance identifier
        original_prediction: Original model prediction
        threshold: Decision threshold
        recommendations: Top-N counterfactual recommendations
        policy_constraints: Policy constraints used
        seed: Random seed for reproducibility
        total_candidates: Total candidates generated
        feasible_candidates: Number of candidates passing threshold

    """

    instance_id: str | int
    original_prediction: float
    threshold: float
    recommendations: list[RecourseRecommendation]
    policy_constraints: PolicyConstraints
    seed: int
    total_candidates: int
    feasible_candidates: int


# ============================================================================
# Core Recourse Generation
# ============================================================================


def generate_recourse(
    model: ModelProtocol,
    feature_values: pd.Series,
    shap_values: np.ndarray,
    feature_names: list[str],
    instance_id: str | int,
    original_prediction: float,
    threshold: float = 0.5,
    policy_constraints: PolicyConstraints | None = None,
    top_n: int = 5,
    max_changes_per_feature: int = 10,
    seed: int = 42,
) -> RecourseResult:
    """Generate counterfactual recommendations with policy constraints.

    This function implements greedy search to find feasible counterfactuals
    that would change the model's decision while respecting policy constraints.

    Args:
        model: Model with predict_proba method
        feature_values: Feature values for instance (Series)
        shap_values: SHAP values for instance (1D array)
        feature_names: List of feature names matching SHAP values
        instance_id: Unique instance identifier
        original_prediction: Original model prediction
        threshold: Decision threshold (default 0.5)
        policy_constraints: Policy constraints (immutables, monotonic, costs, bounds)
        top_n: Number of recommendations to return
        max_changes_per_feature: Maximum candidate changes per feature
        seed: Random seed for deterministic tie-breaking

    Returns:
        RecourseResult with top-N recommendations and audit trail

    Raises:
        ValueError: If SHAP values don't match feature names
        ValueError: If feature_values don't match feature names

    Examples:
        >>> policy = PolicyConstraints(
        ...     immutable_features=["age", "gender"],
        ...     monotonic_constraints={"income": "increase_only", "debt": "decrease_only"},
        ...     feature_costs={"income": 0.8, "debt": 0.5},
        ...     feature_bounds={},
        ... )
        >>> result = generate_recourse(
        ...     model=model,
        ...     feature_values=pd.Series({"income": 30000, "debt": 5000, "age": 25}),
        ...     shap_values=np.array([-0.3, -0.2, 0.1]),
        ...     feature_names=["income", "debt", "age"],
        ...     instance_id=42,
        ...     original_prediction=0.35,
        ...     threshold=0.5,
        ...     policy_constraints=policy,
        ...     top_n=5,
        ...     seed=42,
        ... )
        >>> len(result.recommendations) <= 5
        True
        >>> all(rec.feasible for rec in result.recommendations)
        True

    """
    # Validate inputs
    if len(shap_values.shape) != 1:
        msg = f"Expected 1D SHAP values, got shape {shap_values.shape}"
        raise ValueError(msg)

    if len(shap_values) != len(feature_names):
        msg = f"SHAP values ({len(shap_values)}) don't match feature names ({len(feature_names)})"
        raise ValueError(msg)

    if len(feature_values) != len(feature_names):
        msg = f"Feature values ({len(feature_values)}) don't match feature names ({len(feature_names)})"
        raise ValueError(msg)

    # Use default policy if none provided
    if policy_constraints is None:
        policy_constraints = PolicyConstraints(
            immutable_features=[],
            monotonic_constraints={},
            feature_costs={},
            feature_bounds={},
        )

    # Early exit: already approved
    if original_prediction >= threshold:
        logger.info(
            f"Instance {instance_id} already approved (pred={original_prediction:.2f} >= threshold={threshold:.2f})",
        )
        return RecourseResult(
            instance_id=instance_id,
            original_prediction=original_prediction,
            threshold=threshold,
            recommendations=[],
            policy_constraints=policy_constraints,
            seed=seed,
            total_candidates=0,
            feasible_candidates=0,
        )

    # Identify mutable negative contributors
    mutable_features = _identify_mutable_negative_features(
        shap_values=shap_values,
        feature_names=feature_names,
        immutable_features=policy_constraints.immutable_features,
    )

    if not mutable_features:
        logger.warning(f"Instance {instance_id}: no mutable features to change")
        return RecourseResult(
            instance_id=instance_id,
            original_prediction=original_prediction,
            threshold=threshold,
            recommendations=[],
            policy_constraints=policy_constraints,
            seed=seed,
            total_candidates=0,
            feasible_candidates=0,
        )

    # Generate candidate counterfactuals
    candidates = _generate_candidates(
        model=model,
        feature_values=feature_values,
        mutable_features=mutable_features,
        policy_constraints=policy_constraints,
        threshold=threshold,
        max_changes_per_feature=max_changes_per_feature,
        seed=seed,
    )

    # Filter feasible candidates
    feasible = [c for c in candidates if c.feasible]

    # Sort by cost (lowest first) and rank
    sorted_candidates = sorted(feasible, key=lambda c: c.total_cost)
    for rank, candidate in enumerate(sorted_candidates, start=1):
        # Replace rank (immutable dataclass, so recreate)
        sorted_candidates[rank - 1] = RecourseRecommendation(
            feature_changes=candidate.feature_changes,
            total_cost=candidate.total_cost,
            predicted_probability=candidate.predicted_probability,
            feasible=candidate.feasible,
            rank=rank,
        )

    # Return top-N
    top_recommendations = sorted_candidates[:top_n]

    return RecourseResult(
        instance_id=instance_id,
        original_prediction=original_prediction,
        threshold=threshold,
        recommendations=top_recommendations,
        policy_constraints=policy_constraints,
        seed=seed,
        total_candidates=len(candidates),
        feasible_candidates=len(feasible),
    )


# ============================================================================
# Helper Functions
# ============================================================================


def _identify_mutable_negative_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    immutable_features: list[str],
) -> list[tuple[str, float]]:
    """Identify mutable features with negative SHAP contributions.

    Args:
        shap_values: SHAP values (1D array)
        feature_names: Feature names
        immutable_features: Immutable feature names

    Returns:
        List of (feature_name, shap_value) tuples, sorted by SHAP magnitude

    """
    immutable_set = {f.lower() for f in immutable_features}

    mutable_negative = []
    for name, shap_val in zip(feature_names, shap_values, strict=True):
        # Skip immutables
        if name.lower() in immutable_set:
            continue

        # Only negative contributors
        if shap_val < 0:
            mutable_negative.append((name, float(shap_val)))

    # Sort by SHAP magnitude (most negative first)
    mutable_negative.sort(key=lambda x: x[1])

    return mutable_negative


def _generate_candidates(
    model: ModelProtocol,
    feature_values: pd.Series,
    mutable_features: list[tuple[str, float]],
    policy_constraints: PolicyConstraints,
    threshold: float,
    max_changes_per_feature: int,
    seed: int,
) -> list[RecourseRecommendation]:
    """Generate candidate counterfactuals using greedy search.

    Args:
        model: Model with predict_proba
        feature_values: Original feature values
        mutable_features: List of (feature, shap_value) tuples
        policy_constraints: Policy constraints
        threshold: Decision threshold
        max_changes_per_feature: Maximum changes per feature
        seed: Random seed

    Returns:
        List of candidate recommendations

    """
    rng = np.random.default_rng(seed)
    candidates = []

    # Try changing each mutable feature
    for feature, shap_val in mutable_features:
        old_value = feature_values[feature]

        # Generate candidate changes based on monotonic constraints
        change_deltas = _generate_change_deltas(
            feature=feature,
            old_value=old_value,
            shap_value=shap_val,
            monotonic_constraints=policy_constraints.monotonic_constraints,
            feature_bounds=policy_constraints.feature_bounds,
            max_changes=max_changes_per_feature,
            rng=rng,
        )

        # Try each change
        for delta in change_deltas:
            new_value = old_value + delta

            # Validate constraints
            if not _validate_change(
                feature=feature,
                old_value=old_value,
                new_value=new_value,
                policy_constraints=policy_constraints,
            ):
                continue

            # Create counterfactual
            counterfactual = feature_values.copy()
            # Preserve dtype to avoid pandas FutureWarning
            original_dtype = feature_values[feature].dtype
            if original_dtype.kind in ("i", "u"):  # Integer types
                counterfactual[feature] = int(round(new_value))
            else:
                counterfactual[feature] = new_value

            # Predict
            try:
                pred_proba = model.predict_proba(counterfactual)
                if len(pred_proba.shape) == 2:
                    prediction = float(pred_proba[0, 1])  # Binary classification
                else:
                    prediction = float(pred_proba[1])
            except (ValueError, TypeError, IndexError) as e:
                logger.warning(f"Prediction failed for counterfactual: {e}")
                continue

            # Compute cost
            cost = compute_feature_cost(
                feature=feature,
                old_value=old_value,
                new_value=new_value,
                feature_costs=policy_constraints.feature_costs,
            )

            # Create recommendation
            feasible = prediction >= threshold
            rec = RecourseRecommendation(
                feature_changes={feature: (float(old_value), float(new_value))},
                total_cost=cost,
                predicted_probability=prediction,
                feasible=feasible,
                rank=0,  # Will be assigned later
            )
            candidates.append(rec)

    return candidates


def _generate_change_deltas(
    feature: str,
    old_value: float,
    shap_value: float,
    monotonic_constraints: dict[str, str] | None,
    feature_bounds: dict[str, tuple[float, float]],
    max_changes: int,
    rng: np.random.Generator,
) -> list[float]:
    """Generate candidate change deltas for a feature.

    Args:
        feature: Feature name
        old_value: Current feature value
        shap_value: SHAP value (negative = pushing toward denial)
        monotonic_constraints: Monotonic constraints
        feature_bounds: Feature bounds
        max_changes: Maximum number of changes to generate
        rng: Random number generator

    Returns:
        List of delta values to try

    """
    # Determine direction based on SHAP value and constraints
    direction = _determine_change_direction(feature, shap_value, monotonic_constraints)

    if direction == "none":
        return []

    # Determine magnitude range
    base_magnitude = abs(old_value * 0.1)  # 10% of current value
    if base_magnitude < 1:
        base_magnitude = 1.0  # Minimum change

    # Generate candidate deltas
    deltas = []
    for i in range(max_changes):
        # Exponential spacing: small changes first, then larger
        magnitude = base_magnitude * (1.5**i)

        if direction == "increase":
            delta = magnitude
        elif direction == "decrease":
            delta = -magnitude
        else:  # "both"
            # Alternate between increase and decrease
            delta = magnitude if i % 2 == 0 else -magnitude

        # Check bounds
        new_value = old_value + delta
        if not _check_bounds(feature, new_value, feature_bounds):
            continue

        deltas.append(delta)

    # Add small random perturbations for diversity
    extra_deltas = []
    for _ in range(max_changes // 2):
        if direction == "increase":
            delta = abs(rng.normal(base_magnitude, base_magnitude * 0.3))
        elif direction == "decrease":
            delta = -abs(rng.normal(base_magnitude, base_magnitude * 0.3))
        else:
            delta = rng.normal(0, base_magnitude * 0.5)

        new_value = old_value + delta
        if _check_bounds(feature, new_value, feature_bounds):
            extra_deltas.append(delta)

    return deltas + extra_deltas


def _determine_change_direction(
    feature: str,
    shap_value: float,
    monotonic_constraints: dict[str, str] | None,
) -> str:
    """Determine which direction feature should change.

    Args:
        feature: Feature name
        shap_value: SHAP value (negative = pushing toward denial)
        monotonic_constraints: Monotonic constraints

    Returns:
        "increase", "decrease", "both", or "none"

    """
    if feature in monotonic_constraints:
        constraint = monotonic_constraints[feature]
        if constraint == "increase_only":
            return "increase"
        if constraint == "decrease_only":
            return "decrease"
        if constraint == "fixed":
            return "none"

    # No constraint: change in direction that improves prediction
    # Negative SHAP means feature is pushing toward denial
    # So we want to change in opposite direction
    if shap_value < 0:
        # Try both directions (we don't know the feature's relationship with target)
        return "both"

    return "both"


def _check_bounds(
    feature: str,
    value: float,
    feature_bounds: dict[str, tuple[float, float]],
) -> bool:
    """Check if value is within bounds."""
    return validate_feature_bounds(feature, value, feature_bounds)


def _validate_change(
    feature: str,
    old_value: float,
    new_value: float,
    policy_constraints: PolicyConstraints,
) -> bool:
    """Validate that a feature change respects all policy constraints.

    Args:
        feature: Feature name
        old_value: Current value
        new_value: Proposed value
        policy_constraints: Policy constraints

    Returns:
        True if change is valid, False otherwise

    """
    # Check monotonic constraints
    if not validate_monotonic_constraints(
        feature=feature,
        old_value=old_value,
        new_value=new_value,
        monotonic_constraints=policy_constraints.monotonic_constraints,
    ):
        return False

    # Check bounds
    if not validate_feature_bounds(
        feature=feature,
        value=new_value,
        feature_bounds=policy_constraints.feature_bounds,
    ):
        return False

    return True
