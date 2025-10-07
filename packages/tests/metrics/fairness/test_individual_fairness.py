"""Contract tests for individual fairness metrics (E11).

Tests:
1. Consistency score: determinism, distance metrics, threshold sensitivity
2. Matched pairs: edge cases (no pairs, many pairs, exact matches)
3. Flip test: binary/multiclass protected attributes, integration with recourse
4. Integration: fairness runner, audit pipeline
"""

import numpy as np
import pandas as pd
import pytest

from glassalpha.metrics.fairness.individual import (
    compute_consistency_score,
    find_matched_pairs,
    counterfactual_flip_test,
    IndividualFairnessMetrics,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_data():
    """Simple dataset for testing."""
    np.random.seed(42)
    n = 100

    # Non-protected features
    income = np.random.normal(50000, 15000, n)
    age = np.random.randint(18, 70, n)

    # Protected attribute (binary)
    gender = np.random.randint(0, 2, n)

    # Create DataFrame
    df = pd.DataFrame({
        "income": income,
        "age": age,
        "gender": gender,
    })

    # Simple predictions (higher income → higher score)
    predictions = 1 / (1 + np.exp(-(income - 45000) / 10000))

    return df, predictions


@pytest.fixture
def disparate_treatment_data():
    """Data with clear disparate treatment (same features, different predictions by gender)."""
    # Two pairs of individuals with identical non-protected features
    df = pd.DataFrame({
        "income": [50000, 50000, 60000, 60000],
        "age": [30, 30, 40, 40],
        "gender": [0, 1, 0, 1],  # Only difference is gender
    })

    # Predictions differ by gender (disparate treatment)
    predictions = np.array([0.3, 0.7, 0.4, 0.8])  # Same features, different outcomes

    return df, predictions


@pytest.fixture
def fair_data():
    """Data with no disparate treatment (same features → same predictions)."""
    df = pd.DataFrame({
        "income": [50000, 50000, 60000, 60000],
        "age": [30, 30, 40, 40],
        "gender": [0, 1, 0, 1],
    })

    # Predictions same regardless of gender (fair treatment)
    predictions = np.array([0.5, 0.5, 0.6, 0.6])

    return df, predictions


# ============================================================================
# Test: Consistency Score
# ============================================================================


def test_consistency_score_deterministic(simple_data):
    """Consistency score should be deterministic with same seed."""
    df, predictions = simple_data

    score1 = compute_consistency_score(
        features=df[["income", "age"]],
        predictions=predictions,
        protected_attributes=["gender"],
        distance_metric="euclidean",
        similarity_percentile=90,
        seed=42,
    )

    score2 = compute_consistency_score(
        features=df[["income", "age"]],
        predictions=predictions,
        protected_attributes=["gender"],
        distance_metric="euclidean",
        similarity_percentile=90,
        seed=42,
    )

    assert score1["consistency_score"] == score2["consistency_score"]
    assert score1["similarity_threshold"] == score2["similarity_threshold"]


def test_consistency_score_euclidean_vs_mahalanobis(simple_data):
    """Different distance metrics should give different results."""
    df, predictions = simple_data

    score_euclidean = compute_consistency_score(
        features=df[["income", "age"]],
        predictions=predictions,
        protected_attributes=["gender"],
        distance_metric="euclidean",
        similarity_percentile=90,
        seed=42,
    )

    score_mahalanobis = compute_consistency_score(
        features=df[["income", "age"]],
        predictions=predictions,
        protected_attributes=["gender"],
        distance_metric="mahalanobis",
        similarity_percentile=90,
        seed=42,
    )

    # Scores should differ (different distance metrics)
    assert score_euclidean["consistency_score"] != score_mahalanobis["consistency_score"]
    assert score_euclidean["distance_metric"] == "euclidean"
    assert score_mahalanobis["distance_metric"] == "mahalanobis"


def test_consistency_score_perfect_consistency():
    """Perfect consistency: identical features → identical predictions."""
    df = pd.DataFrame({
        "income": [50000, 50000, 60000, 60000],
        "age": [30, 30, 40, 40],
        "gender": [0, 1, 0, 1],
    })

    # Same features → same predictions (perfect consistency)
    predictions = np.array([0.5, 0.5, 0.6, 0.6])

    score = compute_consistency_score(
        features=df,  # Include all features
        predictions=predictions,
        protected_attributes=["gender"],
        distance_metric="euclidean",
        similarity_percentile=90,
        seed=42,
    )

    # Lipschitz constant should be ~0 (minimal prediction difference for similar pairs)
    assert score["consistency_score"] < 0.1  # Allow small numerical tolerance
    assert score["n_similar_pairs"] >= 2  # At least 2 pairs with identical features


def test_consistency_score_threshold_sensitivity(simple_data):
    """Higher percentile threshold → more similar pairs (more lenient)."""
    df, predictions = simple_data

    score_50th = compute_consistency_score(
        features=df[["income", "age"]],
        predictions=predictions,
        protected_attributes=["gender"],
        distance_metric="euclidean",
        similarity_percentile=50,
        seed=42,
    )

    score_90th = compute_consistency_score(
        features=df[["income", "age"]],
        predictions=predictions,
        protected_attributes=["gender"],
        distance_metric="euclidean",
        similarity_percentile=90,
        seed=42,
    )

    # 90th percentile → MORE similar pairs (higher distance threshold = more lenient)
    assert score_90th["n_similar_pairs"] >= score_50th["n_similar_pairs"]
    assert score_90th["similarity_threshold"] >= score_50th["similarity_threshold"]


def test_consistency_score_excludes_protected_features(simple_data):
    """Protected features should be excluded from distance calculation."""
    df, predictions = simple_data

    # Distance should be computed only on income and age (not gender)
    score = compute_consistency_score(
        features=df,  # Include all features
        predictions=predictions,
        protected_attributes=["gender"],
        distance_metric="euclidean",
        similarity_percentile=90,
        seed=42,
    )

    # Verify protected features were excluded (check metadata)
    assert "gender" in score["protected_attributes_excluded"]
    assert "income" in score["features_used"]
    assert "age" in score["features_used"]
    assert "gender" not in score["features_used"]


# ============================================================================
# Test: Matched Pairs
# ============================================================================


def test_matched_pairs_finds_disparate_treatment(disparate_treatment_data):
    """Should identify pairs with similar features but different predictions."""
    df, predictions = disparate_treatment_data

    pairs = find_matched_pairs(
        features=df[["income", "age"]],
        predictions=predictions,
        protected_features=df[["gender"]],
        distance_threshold=0.1,  # Very low threshold (exact matches)
        prediction_diff_threshold=0.1,  # Flag if prediction differs by >10%
        seed=42,
    )

    # Should find 2 matched pairs: (0,1) and (2,3)
    assert len(pairs["matched_pairs"]) >= 2

    # Verify pairs have similar features but different predictions
    for pair in pairs["matched_pairs"]:
        assert pair["feature_distance"] < 0.1
        assert pair["prediction_diff"] >= 0.1
        # Should flag disparate treatment
        assert pair["protected_attributes_differ"] is True


def test_matched_pairs_no_pairs_when_fair(fair_data):
    """Should find no matched pairs when treatment is fair."""
    df, predictions = fair_data

    pairs = find_matched_pairs(
        features=df[["income", "age"]],
        predictions=predictions,
        protected_features=df[["gender"]],
        distance_threshold=0.1,
        prediction_diff_threshold=0.1,
        seed=42,
    )

    # No pairs with similar features AND different predictions
    assert len(pairs["matched_pairs"]) == 0


def test_matched_pairs_handles_no_similar_features():
    """Should handle case where no individuals have similar features."""
    df = pd.DataFrame({
        "income": [30000, 50000, 70000, 90000],  # All very different
        "age": [20, 40, 60, 80],
        "gender": [0, 1, 0, 1],
    })
    predictions = np.array([0.3, 0.5, 0.7, 0.9])

    pairs = find_matched_pairs(
        features=df[["income", "age"]],
        predictions=predictions,
        protected_features=df[["gender"]],
        distance_threshold=0.1,  # Very strict threshold
        prediction_diff_threshold=0.1,
        seed=42,
    )

    assert len(pairs["matched_pairs"]) == 0
    assert pairs["n_comparisons_made"] > 0


def test_matched_pairs_deterministic():
    """Matched pairs should be deterministic with same seed."""
    df = pd.DataFrame({
        "income": [50000, 50100, 60000, 60100],
        "age": [30, 31, 40, 41],
        "gender": [0, 1, 0, 1],
    })
    predictions = np.array([0.3, 0.7, 0.4, 0.8])

    pairs1 = find_matched_pairs(
        features=df[["income", "age"]],
        predictions=predictions,
        protected_features=df[["gender"]],
        distance_threshold=1.0,  # Reasonable threshold for scaled features
        prediction_diff_threshold=0.1,
        seed=42,
    )

    pairs2 = find_matched_pairs(
        features=df[["income", "age"]],
        predictions=predictions,
        protected_features=df[["gender"]],
        distance_threshold=1.0,
        prediction_diff_threshold=0.1,
        seed=42,
    )

    # Results should be identical
    assert pairs1["matched_pairs"] == pairs2["matched_pairs"]


# ============================================================================
# Test: Counterfactual Flip Test
# ============================================================================


def test_flip_test_detects_disparate_treatment(disparate_treatment_data):
    """Flip test should detect when protected attribute change affects prediction."""
    df, predictions = disparate_treatment_data

    # Mock model that treats genders differently
    class DisparateModel:
        def predict_proba(self, X):
            # Higher score for gender=1
            if isinstance(X, pd.DataFrame):
                gender_boost = X["gender"].values * 0.4
            else:
                gender_boost = X[:, 2] * 0.4  # Assume gender is 3rd column
            base = 0.3
            return np.column_stack([1 - (base + gender_boost), base + gender_boost])

    model = DisparateModel()

    flip_results = counterfactual_flip_test(
        model=model,
        features=df,
        protected_attributes=["gender"],
        threshold=0.5,
        seed=42,
    )

    # Should detect disparate treatment
    assert flip_results["disparate_treatment_rate"] > 0
    assert len(flip_results["flip_changes_prediction"]) > 0

    # Verify flipped individuals identified
    for instance_id in flip_results["flip_changes_prediction"]:
        assert instance_id in [0, 1, 2, 3]


def test_flip_test_no_treatment_when_fair(fair_data):
    """Flip test should find no disparate treatment when model is fair."""
    df, _ = fair_data

    # Mock fair model (ignores gender)
    class FairModel:
        def predict_proba(self, X):
            if isinstance(X, pd.DataFrame):
                income = X["income"].values
            else:
                income = X[:, 0]
            prob = (income - 40000) / 40000  # Only uses income
            prob = np.clip(prob, 0.1, 0.9)
            return np.column_stack([1 - prob, prob])

    model = FairModel()

    flip_results = counterfactual_flip_test(
        model=model,
        features=df,
        protected_attributes=["gender"],
        threshold=0.5,
        seed=42,
    )

    # Should find no disparate treatment
    assert flip_results["disparate_treatment_rate"] == 0.0
    assert len(flip_results["flip_changes_prediction"]) == 0


def test_flip_test_multiclass_protected_attribute():
    """Flip test should handle protected attributes with >2 categories."""
    df = pd.DataFrame({
        "income": [50000, 50000, 50000],
        "age": [30, 30, 30],
        "race": [0, 1, 2],  # 3 categories
    })

    # Model that treats races differently
    class RaceDisparateModel:
        def predict_proba(self, X):
            if isinstance(X, pd.DataFrame):
                race_boost = X["race"].values * 0.2
            else:
                race_boost = X[:, 2] * 0.2
            prob = 0.3 + race_boost
            return np.column_stack([1 - prob, prob])

    model = RaceDisparateModel()

    flip_results = counterfactual_flip_test(
        model=model,
        features=df,
        protected_attributes=["race"],
        threshold=0.5,
        seed=42,
    )

    # Should detect disparate treatment across race categories
    assert flip_results["disparate_treatment_rate"] > 0
    assert "race" in flip_results["protected_attributes_tested"]


def test_flip_test_deterministic():
    """Flip test should be deterministic with same seed."""
    df = pd.DataFrame({
        "income": [50000, 60000],
        "gender": [0, 1],
    })

    class SimpleModel:
        def predict_proba(self, X):
            if isinstance(X, pd.DataFrame):
                vals = X["gender"].values * 0.3 + 0.4
            else:
                vals = X[:, 1] * 0.3 + 0.4
            return np.column_stack([1 - vals, vals])

    model = SimpleModel()

    result1 = counterfactual_flip_test(
        model=model,
        features=df,
        protected_attributes=["gender"],
        threshold=0.5,
        seed=42,
    )

    result2 = counterfactual_flip_test(
        model=model,
        features=df,
        protected_attributes=["gender"],
        threshold=0.5,
        seed=42,
    )

    assert result1 == result2


# ============================================================================
# Test: Integration with Metrics Class
# ============================================================================


def test_individual_fairness_metrics_complete_analysis(disparate_treatment_data):
    """End-to-end test of IndividualFairnessMetrics."""
    df, predictions = disparate_treatment_data

    # Mock model
    class Model:
        def predict_proba(self, X):
            if isinstance(X, pd.DataFrame):
                gender_boost = X["gender"].values * 0.4
            else:
                gender_boost = X[:, 2] * 0.4
            prob = 0.3 + gender_boost
            return np.column_stack([1 - prob, prob])

    metrics = IndividualFairnessMetrics(
        model=Model(),
        features=df,
        predictions=predictions,
        protected_attributes=["gender"],
        seed=42,
    )

    results = metrics.compute()

    # Check all components present
    assert "consistency_score" in results
    assert "matched_pairs" in results
    assert "flip_test" in results

    # Verify disparate treatment detected
    assert results["consistency_score"]["consistency_score"] > 0
    assert len(results["matched_pairs"]["matched_pairs"]) > 0
    assert results["flip_test"]["disparate_treatment_rate"] > 0


def test_individual_fairness_metrics_json_serializable(simple_data):
    """Results should be JSON serializable."""
    import json

    df, predictions = simple_data

    class FairModel:
        def predict_proba(self, X):
            if isinstance(X, pd.DataFrame):
                income = X["income"].values
            else:
                income = X[:, 0]
            prob = (income - 40000) / 40000
            prob = np.clip(prob, 0.1, 0.9)
            return np.column_stack([1 - prob, prob])

    metrics = IndividualFairnessMetrics(
        model=FairModel(),
        features=df,
        predictions=predictions,
        protected_attributes=["gender"],
        seed=42,
    )

    results = metrics.compute()

    # Should be JSON serializable
    json_str = json.dumps(results)
    assert len(json_str) > 0

    # Should round-trip
    parsed = json.loads(json_str)
    assert "consistency_score" in parsed


# ============================================================================
# Test: Edge Cases
# ============================================================================


def test_handles_small_sample_size():
    """Should handle small datasets gracefully."""
    df = pd.DataFrame({
        "income": [50000, 60000],
        "gender": [0, 1],
    })
    predictions = np.array([0.4, 0.6])

    # Should not crash
    score = compute_consistency_score(
        features=df[["income"]],
        predictions=predictions,
        protected_attributes=["gender"],
        distance_metric="euclidean",
        similarity_percentile=90,
        seed=42,
    )

    assert score["n_samples"] == 2


def test_handles_all_identical_predictions():
    """Should handle case where all predictions are identical."""
    df = pd.DataFrame({
        "income": [50000, 60000, 70000],
        "gender": [0, 1, 0],
    })
    predictions = np.array([0.5, 0.5, 0.5])  # All same

    score = compute_consistency_score(
        features=df[["income"]],
        predictions=predictions,
        protected_attributes=["gender"],
        distance_metric="euclidean",
        similarity_percentile=90,
        seed=42,
    )

    # Perfect consistency (no prediction differences)
    assert score["consistency_score"] == 0.0


def test_handles_missing_values():
    """Should handle missing values gracefully."""
    df = pd.DataFrame({
        "income": [50000, np.nan, 70000],
        "age": [30, 40, np.nan],
        "gender": [0, 1, 0],
    })
    predictions = np.array([0.4, 0.5, 0.6])

    # Should either handle NaN or raise clear error
    with pytest.raises((ValueError, TypeError)):
        compute_consistency_score(
            features=df[["income", "age"]],
            predictions=predictions,
            protected_attributes=["gender"],
            distance_metric="euclidean",
            similarity_percentile=90,
            seed=42,
        )
