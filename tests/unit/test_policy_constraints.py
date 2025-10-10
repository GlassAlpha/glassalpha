"""Contract tests for policy constraint validation.

These tests verify the shared policy logic used by both E2 (Reason Codes)
and E2.5 (Recourse) features.
"""

import pytest

from glassalpha.explain.policy import (
    PolicyConstraints,
    compute_feature_cost,
    merge_protected_and_immutable,
    validate_feature_bounds,
    validate_immutables,
    validate_monotonic_constraints,
)


class TestImmutableValidation:
    """Test immutable feature validation."""

    def test_validate_immutables_detects_violations(self):
        """Detect immutable features in feature list."""
        features = ["age", "debt", "income"]
        immutable = ["age", "gender"]

        violations = validate_immutables(features, immutable)

        assert violations == ["age"]

    def test_validate_immutables_case_insensitive(self):
        """Matching is case-insensitive."""
        features = ["AGE", "debt"]
        immutable = ["age", "gender"]

        violations = validate_immutables(features, immutable)

        assert violations == ["AGE"]

    def test_validate_immutables_no_violations(self):
        """No violations when no overlap."""
        features = ["debt", "income", "duration"]
        immutable = ["age", "gender"]

        violations = validate_immutables(features, immutable)

        assert violations == []


class TestMonotonicConstraints:
    """Test monotonic constraint validation."""

    def test_increase_only_allows_increase(self):
        """increase_only allows value to increase."""
        constraints = {"income": "increase_only"}

        assert validate_monotonic_constraints("income", 30000, 35000, constraints) is True

    def test_increase_only_rejects_decrease(self):
        """increase_only rejects decrease."""
        constraints = {"income": "increase_only"}

        assert validate_monotonic_constraints("income", 35000, 30000, constraints) is False

    def test_increase_only_allows_same(self):
        """increase_only allows same value."""
        constraints = {"income": "increase_only"}

        assert validate_monotonic_constraints("income", 30000, 30000, constraints) is True

    def test_decrease_only_allows_decrease(self):
        """decrease_only allows value to decrease."""
        constraints = {"debt": "decrease_only"}

        assert validate_monotonic_constraints("debt", 5000, 3000, constraints) is True

    def test_decrease_only_rejects_increase(self):
        """decrease_only rejects increase."""
        constraints = {"debt": "decrease_only"}

        assert validate_monotonic_constraints("debt", 3000, 5000, constraints) is False

    def test_no_constraint_allows_any_change(self):
        """Features without constraints can change freely."""
        constraints = {"income": "increase_only"}

        assert validate_monotonic_constraints("age", 25, 30, constraints) is True
        assert validate_monotonic_constraints("age", 30, 25, constraints) is True


class TestFeatureCosts:
    """Test feature cost computation."""

    def test_compute_cost_with_weight(self):
        """Cost is magnitude * weight."""
        costs = {"debt": 0.5}

        cost = compute_feature_cost("debt", 5000, 4000, costs)

        assert cost == 500.0  # |5000 - 4000| * 0.5

    def test_compute_cost_default_weight(self):
        """Use default weight when not specified."""
        costs = {}

        cost = compute_feature_cost("unknown", 100, 110, costs, default_cost=1.0)

        assert cost == 10.0  # |100 - 110| * 1.0

    def test_compute_cost_handles_negative_change(self):
        """Absolute value handles negative changes."""
        costs = {"income": 0.8}

        cost = compute_feature_cost("income", 30000, 35000, costs)

        assert cost == 4000.0  # |30000 - 35000| * 0.8


class TestFeatureBounds:
    """Test feature bounds validation."""

    def test_value_within_bounds(self):
        """Valid value passes."""
        bounds = {"age": (18, 100)}

        assert validate_feature_bounds("age", 25, bounds) is True

    def test_value_at_lower_bound(self):
        """Value at lower bound passes."""
        bounds = {"age": (18, 100)}

        assert validate_feature_bounds("age", 18, bounds) is True

    def test_value_at_upper_bound(self):
        """Value at upper bound passes."""
        bounds = {"age": (18, 100)}

        assert validate_feature_bounds("age", 100, bounds) is True

    def test_value_below_lower_bound(self):
        """Value below lower bound fails."""
        bounds = {"age": (18, 100)}

        assert validate_feature_bounds("age", 15, bounds) is False

    def test_value_above_upper_bound(self):
        """Value above upper bound fails."""
        bounds = {"age": (18, 100)}

        assert validate_feature_bounds("age", 105, bounds) is False

    def test_no_bounds_allows_any_value(self):
        """Features without bounds accept any value."""
        bounds = {"age": (18, 100)}

        assert validate_feature_bounds("income", 999999, bounds) is True


class TestMergeProtectedAndImmutable:
    """Test merging protected and immutable lists."""

    def test_merge_deduplicates(self):
        """Merge removes duplicates."""
        protected = ["age", "gender"]
        immutable = ["age", "nationality"]

        merged = merge_protected_and_immutable(protected, immutable)

        assert set(merged) == {"age", "gender", "nationality"}
        assert len(merged) == 3

    def test_merge_case_insensitive_deduplication(self):
        """Case-insensitive deduplication."""
        protected = ["age", "gender"]
        immutable = ["AGE", "nationality"]

        merged = merge_protected_and_immutable(protected, immutable)

        # Should deduplicate age/AGE
        assert len(merged) == 3
        assert any(f.lower() == "age" for f in merged)

    def test_merge_empty_lists(self):
        """Handle empty lists."""
        assert merge_protected_and_immutable([], []) == []
        assert merge_protected_and_immutable(["age"], []) == ["age"]
        assert merge_protected_and_immutable([], ["age"]) == ["age"]


class TestPolicyConstraintsDataclass:
    """Test PolicyConstraints dataclass."""

    def test_create_policy_constraints(self):
        """Can create PolicyConstraints instance."""
        policy = PolicyConstraints(
            immutable_features=["age", "gender"],
            monotonic_constraints={"income": "increase_only"},
            feature_costs={"debt": 0.5},
            feature_bounds={"age": (18, 100)},
        )

        assert policy.immutable_features == ["age", "gender"]
        assert policy.monotonic_constraints == {"income": "increase_only"}
        assert policy.feature_costs == {"debt": 0.5}
        assert policy.feature_bounds == {"age": (18, 100)}

    def test_policy_constraints_immutable(self):
        """PolicyConstraints is immutable (frozen)."""
        policy = PolicyConstraints(
            immutable_features=["age"],
            monotonic_constraints={},
            feature_costs={},
            feature_bounds={},
        )

        with pytest.raises(AttributeError):
            policy.immutable_features = ["gender"]
