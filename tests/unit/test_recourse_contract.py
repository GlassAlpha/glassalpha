"""Contract tests for recourse generation (E2.5).

These tests define the API and behavior for counterfactual recommendation
generation with policy constraints.

Tests follow TDD workflow:
1. Define expected API and behavior
2. Implement to make tests pass
3. Refactor for clarity

Coverage:
- Data class contracts
- Greedy search algorithm
- Policy constraint enforcement
- Determinism (seeded randomness)
- Top-N sorting by cost
- Integration with E2 (reason codes)
- Edge cases (no feasible recourse, all immutable)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glassalpha.explain.policy import PolicyConstraints
from glassalpha.explain.recourse import (
    RecourseRecommendation,
    RecourseResult,
    generate_recourse,
)

# ============================================================================
# Data Class Contracts
# ============================================================================


def test_recourse_recommendation_frozen():
    """RecourseRecommendation should be immutable."""
    rec = RecourseRecommendation(
        feature_changes={"income": (30000, 35000)},
        total_cost=4000.0,
        predicted_probability=0.65,
        feasible=True,
        rank=1,
    )

    with pytest.raises(AttributeError):
        rec.total_cost = 5000.0


def test_recourse_recommendation_requires_all_fields():
    """RecourseRecommendation should require all fields."""
    with pytest.raises(TypeError, match="missing"):
        # Missing required fields: total_cost, predicted_probability, feasible, rank
        RecourseRecommendation(feature_changes={"income": (30000, 35000)})  # type: ignore[call-arg]


def test_recourse_result_frozen():
    """RecourseResult should be immutable."""
    result = RecourseResult(
        instance_id=42,
        original_prediction=0.35,
        threshold=0.5,
        recommendations=[],
        policy_constraints=PolicyConstraints(
            immutable_features=[],
            monotonic_constraints={},
            feature_costs={},
            feature_bounds={},
        ),
        seed=42,
        total_candidates=10,
        feasible_candidates=5,
    )

    with pytest.raises(AttributeError):
        result.threshold = 0.6


def test_recourse_result_requires_all_fields():
    """RecourseResult should require all fields."""
    with pytest.raises(TypeError, match="missing"):
        # Missing required fields
        RecourseResult(instance_id=42, original_prediction=0.35)  # type: ignore[call-arg]


# ============================================================================
# Core Algorithm Contracts
# ============================================================================


def test_generate_recourse_returns_recourse_result():
    """generate_recourse() should return RecourseResult."""
    # Simple model: always returns 0.5 (threshold)
    model = MockModel(constant_pred=0.5)

    feature_values = pd.Series({"income": 30000, "debt": 5000, "age": 25})
    shap_values = np.array([-0.3, -0.2, 0.1])  # income and debt are negative

    policy = PolicyConstraints(
        immutable_features=["age"],
        monotonic_constraints={"income": "increase_only", "debt": "decrease_only"},
        feature_costs={"income": 0.8, "debt": 0.5},
        feature_bounds={"income": (0, 1000000), "debt": (0, 100000)},
    )

    result = generate_recourse(
        model=model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=["income", "debt", "age"],
        instance_id=42,
        original_prediction=0.35,
        threshold=0.5,
        policy_constraints=policy,
        top_n=5,
        seed=42,
    )

    assert isinstance(result, RecourseResult)
    assert result.instance_id == 42
    assert result.original_prediction == 0.35
    assert result.threshold == 0.5
    assert result.seed == 42


def test_generate_recourse_respects_immutables():
    """Recourse should never change immutable features."""
    model = MockModel(constant_pred=0.6)  # Always returns above threshold

    feature_values = pd.Series({"income": 30000, "debt": 5000, "age": 25, "gender": 1})
    shap_values = np.array([-0.3, -0.2, -0.1, -0.05])

    policy = PolicyConstraints(
        immutable_features=["age", "gender"],  # Cannot change
        monotonic_constraints={"income": "increase_only", "debt": "decrease_only"},
        feature_costs={"income": 0.8, "debt": 0.5},
        feature_bounds={},
    )

    result = generate_recourse(
        model=model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=["income", "debt", "age", "gender"],
        instance_id=42,
        original_prediction=0.35,
        threshold=0.5,
        policy_constraints=policy,
        top_n=5,
        seed=42,
    )

    # Check that no recommendation changes immutable features
    for rec in result.recommendations:
        assert "age" not in rec.feature_changes
        assert "gender" not in rec.feature_changes


def test_generate_recourse_respects_monotonic_constraints():
    """Recourse should respect monotonic constraints."""
    model = MockModelWithThreshold(threshold_feature="income", threshold_value=35000)

    feature_values = pd.Series({"income": 30000, "debt": 5000})
    shap_values = np.array([-0.3, -0.2])

    policy = PolicyConstraints(
        immutable_features=[],
        monotonic_constraints={
            "income": "increase_only",  # Can only increase
            "debt": "decrease_only",  # Can only decrease
        },
        feature_costs={"income": 0.8, "debt": 0.5},
        feature_bounds={},
    )

    result = generate_recourse(
        model=model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=["income", "debt"],
        instance_id=42,
        original_prediction=0.35,
        threshold=0.5,
        policy_constraints=policy,
        top_n=5,
        seed=42,
    )

    # Check monotonic constraints
    for rec in result.recommendations:
        if "income" in rec.feature_changes:
            _, new_val = rec.feature_changes["income"]
            old_income = feature_values["income"]
            assert new_val >= old_income, "Income should only increase"

        if "debt" in rec.feature_changes:
            _, new_val = rec.feature_changes["debt"]
            old_debt = feature_values["debt"]
            assert new_val <= old_debt, "Debt should only decrease"


def test_generate_recourse_respects_bounds():
    """Recourse should respect feature bounds."""
    model = MockModel(constant_pred=0.6)

    feature_values = pd.Series({"income": 30000, "debt": 5000})
    shap_values = np.array([-0.3, -0.2])

    policy = PolicyConstraints(
        immutable_features=[],
        monotonic_constraints={},
        feature_costs={"income": 0.8, "debt": 0.5},
        feature_bounds={
            "income": (0, 100000),  # Income capped at 100k
            "debt": (0, 50000),  # Debt capped at 50k
        },
    )

    result = generate_recourse(
        model=model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=["income", "debt"],
        instance_id=42,
        original_prediction=0.35,
        threshold=0.5,
        policy_constraints=policy,
        top_n=5,
        seed=42,
    )

    # Check bounds
    for rec in result.recommendations:
        for feature, (old_val, new_val) in rec.feature_changes.items():
            if feature in policy.feature_bounds:
                min_val, max_val = policy.feature_bounds[feature]
                assert min_val <= new_val <= max_val, f"Feature {feature} out of bounds"


def test_generate_recourse_sorts_by_cost():
    """Recommendations should be sorted by cost (lowest first)."""
    model = MockModel(constant_pred=0.6)

    feature_values = pd.Series({"income": 30000, "debt": 5000, "savings": 1000})
    shap_values = np.array([-0.3, -0.2, -0.1])

    policy = PolicyConstraints(
        immutable_features=[],
        monotonic_constraints={},
        feature_costs={
            "income": 0.8,  # Expensive to change
            "debt": 0.5,  # Medium cost
            "savings": 0.2,  # Cheap to change
        },
        feature_bounds={},
    )

    result = generate_recourse(
        model=model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=["income", "debt", "savings"],
        instance_id=42,
        original_prediction=0.35,
        threshold=0.5,
        policy_constraints=policy,
        top_n=5,
        seed=42,
    )

    # Check sorting
    if len(result.recommendations) >= 2:
        costs = [rec.total_cost for rec in result.recommendations]
        assert costs == sorted(costs), "Recommendations not sorted by cost"


def test_generate_recourse_returns_top_n():
    """generate_recourse() should return at most top_n recommendations."""
    model = MockModel(constant_pred=0.6)

    feature_values = pd.Series({"income": 30000, "debt": 5000, "savings": 1000})
    shap_values = np.array([-0.3, -0.2, -0.1])

    policy = PolicyConstraints(
        immutable_features=[],
        monotonic_constraints={},
        feature_costs={"income": 0.8, "debt": 0.5, "savings": 0.2},
        feature_bounds={},
    )

    result = generate_recourse(
        model=model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=["income", "debt", "savings"],
        instance_id=42,
        original_prediction=0.35,
        threshold=0.5,
        policy_constraints=policy,
        top_n=2,  # Only return top 2
        seed=42,
    )

    assert len(result.recommendations) <= 2


def test_generate_recourse_is_deterministic():
    """Same seed should produce same recommendations."""
    model = MockModel(constant_pred=0.6)

    feature_values = pd.Series({"income": 30000, "debt": 5000})
    shap_values = np.array([-0.3, -0.2])

    policy = PolicyConstraints(
        immutable_features=[],
        monotonic_constraints={},
        feature_costs={"income": 0.8, "debt": 0.5},
        feature_bounds={},
    )

    result1 = generate_recourse(
        model=model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=["income", "debt"],
        instance_id=42,
        original_prediction=0.35,
        threshold=0.5,
        policy_constraints=policy,
        top_n=5,
        seed=42,
    )

    result2 = generate_recourse(
        model=model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=["income", "debt"],
        instance_id=42,
        original_prediction=0.35,
        threshold=0.5,
        policy_constraints=policy,
        top_n=5,
        seed=42,
    )

    # Check determinism
    assert len(result1.recommendations) == len(result2.recommendations)

    for rec1, rec2 in zip(result1.recommendations, result2.recommendations, strict=False):
        assert rec1.feature_changes == rec2.feature_changes
        assert rec1.total_cost == rec2.total_cost
        assert rec1.predicted_probability == rec2.predicted_probability


# ============================================================================
# Edge Cases
# ============================================================================


def test_generate_recourse_no_feasible_solution():
    """generate_recourse() should handle no feasible solutions gracefully."""
    # Model requires income > 100k, but feature is immutable
    model = MockModelWithThreshold(threshold_feature="income", threshold_value=100000)

    feature_values = pd.Series({"income": 30000, "debt": 5000})
    shap_values = np.array([-0.3, -0.2])

    policy = PolicyConstraints(
        immutable_features=["income"],  # Cannot change income!
        monotonic_constraints={},
        feature_costs={},
        feature_bounds={},
    )

    result = generate_recourse(
        model=model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=["income", "debt"],
        instance_id=42,
        original_prediction=0.35,
        threshold=0.5,
        policy_constraints=policy,
        top_n=5,
        seed=42,
    )

    # Should return empty recommendations
    assert len(result.recommendations) == 0
    assert result.feasible_candidates == 0


def test_generate_recourse_all_features_immutable():
    """generate_recourse() should handle all immutable features."""
    model = MockModel(constant_pred=0.6)

    feature_values = pd.Series({"income": 30000, "debt": 5000, "age": 25})
    shap_values = np.array([-0.3, -0.2, -0.1])

    policy = PolicyConstraints(
        immutable_features=["income", "debt", "age"],  # All immutable
        monotonic_constraints={},
        feature_costs={},
        feature_bounds={},
    )

    result = generate_recourse(
        model=model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=["income", "debt", "age"],
        instance_id=42,
        original_prediction=0.35,
        threshold=0.5,
        policy_constraints=policy,
        top_n=5,
        seed=42,
    )

    # Should return empty recommendations
    assert len(result.recommendations) == 0
    assert result.total_candidates == 0


def test_generate_recourse_already_approved():
    """generate_recourse() should handle already approved instances."""
    model = MockModel(constant_pred=0.6)

    feature_values = pd.Series({"income": 30000, "debt": 5000})
    shap_values = np.array([-0.3, -0.2])

    policy = PolicyConstraints(
        immutable_features=[],
        monotonic_constraints={},
        feature_costs={},
        feature_bounds={},
    )

    result = generate_recourse(
        model=model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=["income", "debt"],
        instance_id=42,
        original_prediction=0.65,  # Already approved
        threshold=0.5,
        policy_constraints=policy,
        top_n=5,
        seed=42,
    )

    # Should return empty recommendations (no recourse needed)
    assert len(result.recommendations) == 0


# ============================================================================
# Input Validation Contracts
# ============================================================================


def test_generate_recourse_validates_shap_shape():
    """generate_recourse() should validate SHAP values shape."""
    model = MockModel(constant_pred=0.6)

    feature_values = pd.Series({"income": 30000, "debt": 5000})
    shap_values = np.array([[-0.3, -0.2]])  # 2D instead of 1D

    policy = PolicyConstraints(
        immutable_features=[],
        monotonic_constraints={},
        feature_costs={},
        feature_bounds={},
    )

    with pytest.raises(ValueError, match="Expected 1D SHAP values"):
        generate_recourse(
            model=model,
            feature_values=feature_values,
            shap_values=shap_values,
            feature_names=["income", "debt"],
            instance_id=42,
            original_prediction=0.35,
            threshold=0.5,
            policy_constraints=policy,
            top_n=5,
            seed=42,
        )


def test_generate_recourse_validates_feature_count():
    """generate_recourse() should validate feature count consistency."""
    model = MockModel(constant_pred=0.6)

    feature_values = pd.Series({"income": 30000, "debt": 5000})
    shap_values = np.array([-0.3, -0.2, -0.1])  # 3 values but 2 features

    policy = PolicyConstraints(
        immutable_features=[],
        monotonic_constraints={},
        feature_costs={},
        feature_bounds={},
    )

    with pytest.raises(ValueError, match="don't match feature names"):
        generate_recourse(
            model=model,
            feature_values=feature_values,
            shap_values=shap_values,
            feature_names=["income", "debt"],
            instance_id=42,
            original_prediction=0.35,
            threshold=0.5,
            policy_constraints=policy,
            top_n=5,
            seed=42,
        )


# ============================================================================
# Mock Models for Testing
# ============================================================================


class MockModel:
    """Mock model that always returns constant prediction."""

    def __init__(self, constant_pred: float):
        self.constant_pred = constant_pred

    def predict_proba(self, X: pd.DataFrame | pd.Series) -> np.ndarray:
        """Return constant prediction."""
        if isinstance(X, pd.Series):
            return np.array([[1 - self.constant_pred, self.constant_pred]])
        return np.array([[1 - self.constant_pred, self.constant_pred]] * len(X))


class MockModelWithThreshold:
    """Mock model that returns 0.6 if feature > threshold, else 0.4."""

    def __init__(self, threshold_feature: str, threshold_value: float):
        self.threshold_feature = threshold_feature
        self.threshold_value = threshold_value

    def predict_proba(self, X: pd.DataFrame | pd.Series) -> np.ndarray:
        """Return prediction based on threshold."""
        if isinstance(X, pd.Series):
            value = X[self.threshold_feature]
            pred = 0.6 if value >= self.threshold_value else 0.4
            return np.array([[1 - pred, pred]])

        preds = []
        for _, row in X.iterrows():
            value = row[self.threshold_feature]
            pred = 0.6 if value >= self.threshold_value else 0.4
            preds.append([1 - pred, pred])

        return np.array(preds)
