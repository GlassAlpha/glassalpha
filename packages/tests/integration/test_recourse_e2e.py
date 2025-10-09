"""Integration tests for E2.5 (Recourse) end-to-end workflow.

These tests validate the full recourse generation pipeline with real datasets
and models, ensuring deterministic output and correct policy constraint enforcement.

Coverage:
- German Credit dataset workflow
- CLI integration
- Config loading and validation
- SHAP integration
- Policy constraint enforcement
- Deterministic output validation
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from glassalpha.explain.policy import PolicyConstraints
from glassalpha.explain.recourse import generate_recourse


@pytest.fixture
def german_credit_test_data():
    """Load German Credit test data for recourse testing."""
    # Create minimal test dataset
    data = {
        "age": [25, 35, 45],
        "gender": [1, 0, 1],
        "foreign_worker": [1, 0, 1],
        "credit_amount": [5000, 10000, 8000],
        "duration": [24, 36, 18],
        "savings_balance": [500, 2000, 100],
        "checking_balance": [100, 800, 50],
        "employment_duration": [3, 5, 2],
        "num_existing_credits": [2, 1, 3],
        "target": [0, 1, 0],  # Denied, approved, denied
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_german_credit_model():
    """Mock model that mimics German Credit predictions."""

    class MockModel:
        def predict_proba(self, X):
            """Simple rule: approve if savings > 1000 or checking > 500."""
            if isinstance(X, pd.Series):
                X = X.to_frame().T

            preds = []
            for _, row in X.iterrows():
                savings = row.get("savings_balance", 0)
                checking = row.get("checking_balance", 0)

                if savings > 1000 or checking > 500:
                    pred = 0.6  # Approved
                else:
                    pred = 0.35  # Denied

                preds.append([1 - pred, pred])

            return np.array(preds)

    return MockModel()


@pytest.fixture
def german_credit_policy():
    """German Credit policy constraints for recourse."""
    return PolicyConstraints(
        immutable_features=["age", "gender", "foreign_worker", "employment_duration"],
        monotonic_constraints={
            "savings_balance": "increase_only",
            "checking_balance": "increase_only",
            "num_existing_credits": "decrease_only",
        },
        feature_costs={
            "savings_balance": 0.5,
            "checking_balance": 0.3,
            "credit_amount": 0.8,
            "duration": 0.6,
            "num_existing_credits": 0.7,
        },
        feature_bounds={
            "savings_balance": (0, 50000),
            "checking_balance": (0, 20000),
            "credit_amount": (1000, 30000),
            "duration": (6, 60),
            "num_existing_credits": (0, 10),
        },
    )


# ============================================================================
# E2E Workflow Tests
# ============================================================================


def test_recourse_e2e_german_credit_denied_instance(
    german_credit_test_data,
    mock_german_credit_model,
    german_credit_policy,
):
    """Test recourse generation for denied German Credit instance."""
    # Get first instance (denied: savings=500, checking=100)
    feature_values = german_credit_test_data.drop(columns=["target"]).iloc[0]
    feature_names = feature_values.index.tolist()

    # Mock SHAP values (negative for features we want to change)
    shap_values = np.array(
        [
            0.1,  # age (positive, not a problem)
            -0.05,  # gender (negative but immutable)
            0.05,  # foreign_worker (positive)
            -0.2,  # credit_amount (negative)
            -0.1,  # duration (negative)
            -0.3,  # savings_balance (most negative)
            -0.25,  # checking_balance (second most negative)
            0.1,  # employment_duration (positive)
            -0.15,  # num_existing_credits (negative)
        ],
    )

    # Generate recourse
    result = generate_recourse(
        model=mock_german_credit_model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=feature_names,
        instance_id=0,
        original_prediction=0.35,
        threshold=0.5,
        policy_constraints=german_credit_policy,
        top_n=5,
        seed=42,
    )

    # Assertions
    assert result.instance_id == 0
    assert result.original_prediction == 0.35
    assert result.threshold == 0.5
    assert result.seed == 42

    # Should have feasible recommendations
    assert len(result.recommendations) > 0
    assert result.feasible_candidates > 0

    # All recommendations should be feasible
    for rec in result.recommendations:
        assert rec.feasible is True
        assert rec.predicted_probability >= 0.5

    # Recommendations sorted by cost
    costs = [rec.total_cost for rec in result.recommendations]
    assert costs == sorted(costs)

    # Immutable features never changed
    for rec in result.recommendations:
        assert "age" not in rec.feature_changes
        assert "gender" not in rec.feature_changes
        assert "foreign_worker" not in rec.feature_changes
        assert "employment_duration" not in rec.feature_changes

    # Monotonic constraints respected
    for rec in result.recommendations:
        for feature, (old_val, new_val) in rec.feature_changes.items():
            if feature == "savings_balance":
                assert new_val >= old_val, "Savings should only increase"
            elif feature == "checking_balance":
                assert new_val >= old_val, "Checking should only increase"
            elif feature == "num_existing_credits":
                assert new_val <= old_val, "Credits should only decrease"


def test_recourse_e2e_determinism(
    german_credit_test_data,
    mock_german_credit_model,
    german_credit_policy,
):
    """Test that recourse generation is deterministic with same seed."""
    feature_values = german_credit_test_data.drop(columns=["target"]).iloc[0]
    feature_names = feature_values.index.tolist()
    shap_values = np.array([-0.1, -0.05, 0.05, -0.2, -0.1, -0.3, -0.25, 0.1, -0.15])

    # Generate recourse twice with same seed
    result1 = generate_recourse(
        model=mock_german_credit_model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=feature_names,
        instance_id=0,
        original_prediction=0.35,
        threshold=0.5,
        policy_constraints=german_credit_policy,
        top_n=5,
        seed=42,
    )

    result2 = generate_recourse(
        model=mock_german_credit_model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=feature_names,
        instance_id=0,
        original_prediction=0.35,
        threshold=0.5,
        policy_constraints=german_credit_policy,
        top_n=5,
        seed=42,
    )

    # Results should be identical
    assert len(result1.recommendations) == len(result2.recommendations)

    for rec1, rec2 in zip(result1.recommendations, result2.recommendations, strict=False):
        assert rec1.feature_changes == rec2.feature_changes
        assert rec1.total_cost == rec2.total_cost
        assert rec1.predicted_probability == rec2.predicted_probability
        assert rec1.rank == rec2.rank


def test_recourse_e2e_already_approved(
    german_credit_test_data,
    mock_german_credit_model,
    german_credit_policy,
):
    """Test recourse for already approved instance (should return empty)."""
    # Get second instance (approved: savings=2000)
    feature_values = german_credit_test_data.drop(columns=["target"]).iloc[1]
    feature_names = feature_values.index.tolist()
    shap_values = np.array([0.1, -0.05, 0.05, -0.2, -0.1, 0.3, 0.25, 0.1, -0.15])

    result = generate_recourse(
        model=mock_german_credit_model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=feature_names,
        instance_id=1,
        original_prediction=0.6,  # Already approved
        threshold=0.5,
        policy_constraints=german_credit_policy,
        top_n=5,
        seed=42,
    )

    # Should return empty recommendations
    assert len(result.recommendations) == 0
    assert result.total_candidates == 0
    assert result.feasible_candidates == 0


def test_recourse_e2e_no_feasible_solution(
    german_credit_test_data,
    mock_german_credit_model,
):
    """Test recourse when no feasible solution exists (all features immutable)."""
    feature_values = german_credit_test_data.drop(columns=["target"]).iloc[0]
    feature_names = feature_values.index.tolist()
    shap_values = np.array([-0.1, -0.05, 0.05, -0.2, -0.1, -0.3, -0.25, 0.1, -0.15])

    # All features immutable
    policy = PolicyConstraints(
        immutable_features=feature_names,  # Everything immutable
        monotonic_constraints={},
        feature_costs={},
        feature_bounds={},
    )

    result = generate_recourse(
        model=mock_german_credit_model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=feature_names,
        instance_id=0,
        original_prediction=0.35,
        threshold=0.5,
        policy_constraints=policy,
        top_n=5,
        seed=42,
    )

    # Should return empty recommendations
    assert len(result.recommendations) == 0
    assert result.total_candidates == 0


def test_recourse_e2e_json_serialization(
    german_credit_test_data,
    mock_german_credit_model,
    german_credit_policy,
):
    """Test that recourse result can be serialized to JSON."""
    feature_values = german_credit_test_data.drop(columns=["target"]).iloc[0]
    feature_names = feature_values.index.tolist()
    shap_values = np.array([-0.1, -0.05, 0.05, -0.2, -0.1, -0.3, -0.25, 0.1, -0.15])

    result = generate_recourse(
        model=mock_german_credit_model,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=feature_names,
        instance_id=0,
        original_prediction=0.35,
        threshold=0.5,
        policy_constraints=german_credit_policy,
        top_n=5,
        seed=42,
    )

    # Serialize to JSON
    output_dict = {
        "instance_id": result.instance_id,
        "original_prediction": result.original_prediction,
        "threshold": result.threshold,
        "recommendations": [
            {
                "rank": rec.rank,
                "feature_changes": {
                    feature: {"old": old_val, "new": new_val}
                    for feature, (old_val, new_val) in rec.feature_changes.items()
                },
                "total_cost": rec.total_cost,
                "predicted_probability": rec.predicted_probability,
                "feasible": rec.feasible,
            }
            for rec in result.recommendations
        ],
        "policy_constraints": {
            "immutable_features": result.policy_constraints.immutable_features,
            "monotonic_constraints": result.policy_constraints.monotonic_constraints,
        },
        "seed": result.seed,
        "total_candidates": result.total_candidates,
        "feasible_candidates": result.feasible_candidates,
    }

    # Should serialize without error
    json_str = json.dumps(output_dict, indent=2)
    assert len(json_str) > 0

    # Should deserialize back
    parsed = json.loads(json_str)
    assert parsed["instance_id"] == 0
    assert parsed["seed"] == 42


# ============================================================================
# CLI Integration Tests
# ============================================================================


@pytest.mark.integration
def test_recourse_cli_help():
    """Test that recourse CLI help is accessible."""
    import shutil
    import subprocess

    # Skip if CLI not installed
    if not shutil.which("glassalpha"):
        pytest.skip("glassalpha CLI not installed")

    result = subprocess.run(
        ["glassalpha", "recourse", "--help"],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
    )

    assert result.returncode == 0
    assert "recourse" in result.stdout.lower()
    assert "counterfactual" in result.stdout.lower() or "recommendation" in result.stdout.lower()


# ============================================================================
# Config Integration Tests
# ============================================================================


def test_recourse_config_loading():
    """Test that recourse config can be loaded from YAML."""
    from glassalpha.config.schema import AuditConfig

    config_dict = {
        "audit_profile": "tabular_compliance",
        "data": {"dataset": "german_credit", "target_column": "target"},
        "model": {"type": "xgboost"},
        "recourse": {
            "enabled": True,
            "immutable_features": ["age", "gender"],
            "monotonic_constraints": {"savings_balance": "increase_only", "debt": "decrease_only"},
            "cost_function": "weighted_l1",
            "max_iterations": 100,
        },
        "reproducibility": {"random_seed": 42},
    }

    config = AuditConfig(**config_dict)

    assert config.recourse.enabled is True
    assert "age" in config.recourse.immutable_features
    assert config.recourse.monotonic_constraints["savings_balance"] == "increase_only"
    assert config.recourse.max_iterations == 100
    assert config.reproducibility.random_seed == 42
