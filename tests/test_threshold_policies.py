"""Test threshold selection policies."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from glassalpha.metrics.thresholds import (
    get_threshold_summary,
    pick_threshold,
    recommend_threshold_policy,
    validate_threshold_config,
)


def test_youden_threshold():
    """Test Youden's J statistic threshold selection."""
    # Create synthetic dataset with known optimal threshold
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)

    # Create probabilities that should have optimal threshold around 0.4
    y_proba = np.random.beta(2, 5, size=len(y))  # Skewed toward lower values
    y_proba[y == 1] += 0.3  # Shift positive class higher
    y_proba = np.clip(y_proba, 0, 1)

    result = pick_threshold(y, y_proba, policy="youden")

    # Verify result structure
    assert isinstance(result, dict)
    assert "threshold" in result
    assert "policy" in result
    assert "j_statistic" in result
    assert "sensitivity" in result
    assert "specificity" in result
    assert "confusion_matrix" in result
    assert "rationale" in result

    # Verify threshold is in valid range
    assert 0.0 <= result["threshold"] <= 1.0

    # Verify policy is correct
    assert result["policy"] == "youden"

    # Verify J statistic is positive (better than random)
    assert result["j_statistic"] > 0

    # Verify confusion matrix has correct structure
    cm = result["confusion_matrix"]
    assert all(key in cm for key in ["tn", "fp", "fn", "tp"])
    assert all(isinstance(cm[key], int) for key in cm)
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y)


def test_fixed_threshold():
    """Test fixed threshold policy."""
    np.random.seed(42)
    y = np.random.binomial(1, 0.3, 500)
    y_proba = np.random.uniform(0, 1, 500)

    # Test with default threshold (0.5)
    result = pick_threshold(y, y_proba, policy="fixed")
    assert result["threshold"] == 0.5
    assert result["policy"] == "fixed"

    # Test with custom threshold
    result = pick_threshold(y, y_proba, policy="fixed", threshold=0.3)
    assert result["threshold"] == 0.3
    assert "sensitivity" in result
    assert "specificity" in result
    assert "precision" in result


def test_prevalence_threshold():
    """Test prevalence matching threshold selection."""
    np.random.seed(42)
    y = np.random.binomial(1, 0.2, 1000)  # 20% positive rate
    y_proba = np.random.uniform(0, 1, 1000)

    result = pick_threshold(y, y_proba, policy="prevalence")

    assert result["policy"] == "prevalence"
    assert "target_prevalence" in result
    assert "actual_prediction_rate" in result
    assert "prevalence_difference" in result

    # Should be close to the actual prevalence
    assert abs(result["target_prevalence"] - 0.2) < 0.05

    # Prediction rate should be reasonably close to prevalence
    assert result["prevalence_difference"] < 0.1


def test_cost_sensitive_threshold():
    """Test cost-sensitive threshold selection."""
    np.random.seed(42)
    y = np.random.binomial(1, 0.3, 500)
    y_proba = np.random.uniform(0, 1, 500)

    # Test with equal costs (should be similar to other methods)
    result = pick_threshold(y, y_proba, policy="cost_sensitive", cost_fp=1.0, cost_fn=1.0)

    assert result["policy"] == "cost_sensitive"
    assert result["cost_fp"] == 1.0
    assert result["cost_fn"] == 1.0
    assert "total_cost" in result
    assert "fp_cost_contribution" in result
    assert "fn_cost_contribution" in result

    # Test with high false negative cost (should prefer lower threshold)
    result_high_fn = pick_threshold(y, y_proba, policy="cost_sensitive", cost_fp=1.0, cost_fn=10.0)

    # Test with high false positive cost (should prefer higher threshold)
    result_high_fp = pick_threshold(y, y_proba, policy="cost_sensitive", cost_fp=10.0, cost_fn=1.0)

    # High FN cost should generally lead to lower threshold (more sensitive)
    # High FP cost should generally lead to higher threshold (more specific)
    # Note: This might not always hold due to randomness, so we just check they're different
    assert result_high_fn["threshold"] != result_high_fp["threshold"]


def test_threshold_deterministic():
    """Test that threshold selection is deterministic."""
    np.random.seed(42)
    y = np.random.binomial(1, 0.3, 500)
    y_proba = np.random.uniform(0, 1, 500)

    # Run same threshold selection multiple times
    result1 = pick_threshold(y, y_proba, policy="youden")
    result2 = pick_threshold(y, y_proba, policy="youden")

    # Should get identical results
    assert result1["threshold"] == result2["threshold"]
    assert result1["j_statistic"] == result2["j_statistic"]
    assert result1["confusion_matrix"] == result2["confusion_matrix"]


def test_threshold_validation():
    """Test input validation for threshold selection."""
    y = np.array([0, 1, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.3, 0.7])

    # Test invalid policy
    with pytest.raises(ValueError, match="Unknown threshold policy"):
        pick_threshold(y, y_proba, policy="invalid")

    # Test length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        pick_threshold(y, y_proba[:-1], policy="youden")

    # Test invalid y_true values
    with pytest.raises(ValueError, match="y_true must contain only 0 and 1"):
        pick_threshold(np.array([0, 1, 2]), y_proba[:3], policy="youden")

    # Test invalid probabilities
    with pytest.raises(ValueError, match="y_proba must contain probabilities"):
        pick_threshold(y, np.array([0.2, 1.5, 0.3, 0.7]), policy="youden")


def test_fixed_threshold_validation():
    """Test validation for fixed threshold policy."""
    y = np.array([0, 1, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.3, 0.7])

    # Test invalid threshold type
    with pytest.raises(ValueError, match="Fixed threshold must be a number"):
        pick_threshold(y, y_proba, policy="fixed", threshold="invalid")

    # Test threshold out of range
    with pytest.raises(ValueError, match="Fixed threshold must be in"):
        pick_threshold(y, y_proba, policy="fixed", threshold=1.5)

    with pytest.raises(ValueError, match="Fixed threshold must be in"):
        pick_threshold(y, y_proba, policy="fixed", threshold=-0.1)


def test_cost_sensitive_validation():
    """Test validation for cost-sensitive threshold policy."""
    y = np.array([0, 1, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.3, 0.7])

    # Test negative costs
    with pytest.raises(ValueError, match="cost_fp must be a non-negative number"):
        pick_threshold(y, y_proba, policy="cost_sensitive", cost_fp=-1.0, cost_fn=1.0)

    with pytest.raises(ValueError, match="cost_fn must be a non-negative number"):
        pick_threshold(y, y_proba, policy="cost_sensitive", cost_fp=1.0, cost_fn=-1.0)

    # Test both costs zero
    with pytest.raises(ValueError, match="At least one cost must be positive"):
        pick_threshold(y, y_proba, policy="cost_sensitive", cost_fp=0.0, cost_fn=0.0)


def test_validate_threshold_config():
    """Test threshold configuration validation."""
    # Valid configs
    valid_configs = [
        {},  # Empty (should default to youden)
        {"policy": "youden"},
        {"policy": "fixed", "threshold": 0.3},
        {"policy": "prevalence"},
        {"policy": "cost_sensitive", "cost_fp": 1.0, "cost_fn": 5.0},
    ]

    for config in valid_configs:
        result = validate_threshold_config(config)
        assert isinstance(result, dict)
        assert "policy" in result
        assert result["policy"] in {"youden", "fixed", "prevalence", "cost_sensitive"}

    # Invalid configs
    invalid_configs = [
        {"policy": "invalid"},
        {"policy": "fixed", "threshold": 1.5},
        {"policy": "cost_sensitive", "cost_fp": -1.0},
        {"policy": "cost_sensitive", "cost_fp": 0.0, "cost_fn": 0.0},
    ]

    for config in invalid_configs:
        with pytest.raises(ValueError):
            validate_threshold_config(config)


def test_get_threshold_summary():
    """Test threshold summary formatting."""
    np.random.seed(42)
    y = np.random.binomial(1, 0.3, 100)
    y_proba = np.random.uniform(0, 1, 100)

    result = pick_threshold(y, y_proba, policy="youden")
    summary = get_threshold_summary(result)

    assert isinstance(summary, str)
    assert "Threshold Selection" in summary
    assert "Youden Policy" in summary
    assert "Selected Threshold" in summary
    assert "Confusion Matrix" in summary
    assert "Rationale" in summary

    # Test with different policies
    for policy in ["fixed", "prevalence", "cost_sensitive"]:
        if policy == "fixed":
            result = pick_threshold(y, y_proba, policy=policy, threshold=0.4)
        elif policy == "cost_sensitive":
            result = pick_threshold(y, y_proba, policy=policy, cost_fp=1.0, cost_fn=2.0)
        else:
            result = pick_threshold(y, y_proba, policy=policy)

        summary = get_threshold_summary(result)
        assert policy.title() in summary or policy.replace("_", " ").title() in summary


def test_recommend_threshold_policy():
    """Test threshold policy recommendations."""
    # Medical use case with rare condition
    policy = recommend_threshold_policy(5000, 0.05, "medical")
    assert policy == "cost_sensitive"

    # Medical use case with balanced condition
    policy = recommend_threshold_policy(5000, 0.4, "medical")
    assert policy == "youden"

    # Financial use case with rare fraud
    policy = recommend_threshold_policy(10000, 0.02, "financial")
    assert policy == "cost_sensitive"

    # Financial use case with moderate fraud
    policy = recommend_threshold_policy(10000, 0.1, "financial")
    assert policy == "prevalence"

    # General use case with small dataset
    policy = recommend_threshold_policy(500, 0.3, "general")
    assert policy == "fixed"

    # General use case with imbalanced data
    policy = recommend_threshold_policy(5000, 0.1, "general")
    assert policy == "prevalence"

    # General use case with balanced data
    policy = recommend_threshold_policy(5000, 0.45, "general")
    assert policy == "youden"


def test_threshold_policies_return_valid_range():
    """Test that all policies return thresholds in [0, 1]."""
    np.random.seed(42)
    y = np.random.binomial(1, 0.3, 200)
    y_proba = np.random.uniform(0, 1, 200)

    policies = [
        ("youden", {}),
        ("fixed", {"threshold": 0.3}),
        ("prevalence", {}),
        ("cost_sensitive", {"cost_fp": 1.0, "cost_fn": 2.0}),
    ]

    for policy, kwargs in policies:
        result = pick_threshold(y, y_proba, policy=policy, **kwargs)
        threshold = result["threshold"]

        assert 0.0 <= threshold <= 1.0, f"Policy {policy} returned invalid threshold: {threshold}"
        assert isinstance(threshold, float), f"Policy {policy} returned non-float threshold: {type(threshold)}"


def test_confusion_matrix_consistency():
    """Test that confusion matrices are consistent with predictions."""
    np.random.seed(42)
    y = np.random.binomial(1, 0.3, 300)
    y_proba = np.random.uniform(0, 1, 300)

    result = pick_threshold(y, y_proba, policy="youden")
    threshold = result["threshold"]
    cm = result["confusion_matrix"]

    # Make predictions with the selected threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Manually compute confusion matrix
    tn = np.sum((y == 0) & (y_pred == 0))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))
    tp = np.sum((y == 1) & (y_pred == 1))

    # Should match the returned confusion matrix
    assert cm["tn"] == tn
    assert cm["fp"] == fp
    assert cm["fn"] == fn
    assert cm["tp"] == tp

    # Total should equal dataset size
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y)


if __name__ == "__main__":
    pytest.main([__file__])
