"""
Integration test for dataset bias metrics with German Credit dataset.

Tests E12 functionality end-to-end with real data.
"""

import numpy as np
import pandas as pd
import pytest

from glassalpha.datasets.german_credit import load_german_credit
from glassalpha.metrics.fairness.dataset import compute_dataset_bias_metrics


def test_german_credit_dataset_bias_full_analysis():
    """Test full dataset bias analysis on German Credit."""
    # Load dataset
    data = load_german_credit()

    # Add split column (70/30 train/test)
    rng = np.random.RandomState(42)
    n = len(data)
    split = rng.choice(["train", "test"], size=n, p=[0.7, 0.3])
    data["split"] = split

    # Define protected attributes and features
    protected_attrs = ["gender", "foreign_worker"]
    feature_cols = [col for col in data.columns
                   if col not in protected_attrs + ["credit_risk", "split"]]

    # Compute dataset bias metrics
    result = compute_dataset_bias_metrics(
        data=data,
        protected_attrs=protected_attrs,
        feature_cols=feature_cols,
        split_col="split",
        seed=42,
    )

    # Verify all metrics computed
    assert result.proxy_correlations is not None
    assert result.distribution_drift is not None
    assert result.sampling_bias_power is not None
    assert result.split_imbalance is not None

    # Check proxy correlations exist for both protected attributes
    assert "gender" in result.proxy_correlations.correlations
    assert "foreign_worker" in result.proxy_correlations.correlations

    # Each protected attribute should have correlations with features
    assert len(result.proxy_correlations.correlations["gender"]) > 0
    assert len(result.proxy_correlations.correlations["foreign_worker"]) > 0

    # Check that some proxy correlations are flagged
    # (German Credit likely has some proxy features)
    gender_correlations = result.proxy_correlations.correlations["gender"]
    severities = [v["severity"] for v in gender_correlations.values()]
    # Should have at least one WARNING or ERROR
    assert any(s in ["WARNING", "ERROR"] for s in severities), \
        "Expected to find some concerning proxy correlations in German Credit"

    # Check distribution drift results
    assert len(result.distribution_drift.drift_tests) > 0
    for feature, drift_test in result.distribution_drift.drift_tests.items():
        assert "test_type" in drift_test
        assert "p_value" in drift_test
        assert "drifted" in drift_test
        assert drift_test["test_type"] in ["ks_test", "chi2_test"]

    # Check sampling bias power
    assert "gender" in result.sampling_bias_power.power_by_group
    assert "foreign_worker" in result.sampling_bias_power.power_by_group

    # Check that power is computed for all groups
    for attr in protected_attrs:
        groups = result.sampling_bias_power.power_by_group[attr]
        assert len(groups) > 0
        for group, power_result in groups.items():
            assert 0.0 <= power_result["power"] <= 1.0
            assert power_result["severity"] in ["OK", "WARNING", "ERROR"]

    # Check split imbalance
    assert "gender" in result.split_imbalance.imbalance_tests
    assert "foreign_worker" in result.split_imbalance.imbalance_tests

    for attr in protected_attrs:
        imbalance = result.split_imbalance.imbalance_tests[attr]
        assert "p_value" in imbalance
        assert "imbalanced" in imbalance
        assert "train_distribution" in imbalance
        assert "test_distribution" in imbalance


def test_german_credit_dataset_bias_deterministic():
    """Test that German Credit dataset bias analysis is deterministic."""
    # Load dataset
    data = load_german_credit()

    # Add split column
    rng = np.random.RandomState(42)
    split = rng.choice(["train", "test"], size=len(data), p=[0.7, 0.3])
    data["split"] = split

    protected_attrs = ["gender"]
    feature_cols = ["age_years", "credit_amount"]

    # Run twice with same seed
    result1 = compute_dataset_bias_metrics(
        data=data,
        protected_attrs=protected_attrs,
        feature_cols=feature_cols,
        split_col="split",
        seed=42,
    )

    result2 = compute_dataset_bias_metrics(
        data=data,
        protected_attrs=protected_attrs,
        feature_cols=feature_cols,
        split_col="split",
        seed=42,
    )

    # Proxy correlations should be identical
    for feature in feature_cols:
        r1 = result1.proxy_correlations.correlations["gender"][feature]["correlation"]
        r2 = result2.proxy_correlations.correlations["gender"][feature]["correlation"]
        assert abs(r1 - r2) < 1e-10, f"Proxy correlation for {feature} not deterministic"

    # Drift tests should be identical
    for feature in feature_cols:
        p1 = result1.distribution_drift.drift_tests[feature]["p_value"]
        p2 = result2.distribution_drift.drift_tests[feature]["p_value"]
        assert p1 == p2, f"Drift test for {feature} not deterministic"

    # Power should be identical
    for group in result1.sampling_bias_power.power_by_group["gender"]:
        pow1 = result1.sampling_bias_power.power_by_group["gender"][group]["power"]
        pow2 = result2.sampling_bias_power.power_by_group["gender"][group]["power"]
        assert abs(pow1 - pow2) < 1e-10, f"Power for {group} not deterministic"


def test_german_credit_dataset_bias_json_serializable():
    """Test that German Credit dataset bias results are JSON serializable."""
    import json

    # Load dataset
    data = load_german_credit()

    # Add split column
    rng = np.random.RandomState(42)
    split = rng.choice(["train", "test"], size=len(data), p=[0.7, 0.3])
    data["split"] = split

    # Compute metrics
    result = compute_dataset_bias_metrics(
        data=data,
        protected_attrs=["gender"],
        feature_cols=["age_years", "credit_amount"],
        split_col="split",
        seed=42,
    )

    # Should be JSON serializable
    result_dict = result.to_dict()
    json_str = json.dumps(result_dict)

    # Should round-trip
    loaded = json.loads(json_str)
    assert "proxy_correlations" in loaded
    assert "distribution_drift" in loaded
    assert "sampling_bias_power" in loaded
    assert "split_imbalance" in loaded

    # Check structure
    assert "gender" in loaded["proxy_correlations"]["correlations"]
    assert "age_years" in loaded["proxy_correlations"]["correlations"]["gender"]


def test_german_credit_proxy_correlation_specific_features():
    """Test that specific known proxy correlations are detected."""
    # Load dataset
    data = load_german_credit()
    data["split"] = "train"  # Single split for this test

    protected_attrs = ["gender"]

    # Test specific features that might correlate with gender
    # Use only columns that exist in the dataset
    feature_cols = ["age_years", "credit_amount", "duration_months"]

    from glassalpha.metrics.fairness.dataset import compute_proxy_correlations

    result = compute_proxy_correlations(
        data=data,
        protected_attrs=protected_attrs,
        feature_cols=feature_cols,
        seed=42,
    )

    # Check that correlations are computed
    assert "gender" in result.correlations
    for feature in feature_cols:
        assert feature in result.correlations["gender"]
        corr_result = result.correlations["gender"][feature]

        # Check structure
        assert "correlation" in corr_result
        assert "p_value" in corr_result
        assert "severity" in corr_result

        # Correlation should be bounded
        assert -1.0 <= corr_result["correlation"] <= 1.0

        # P-value should be valid
        assert 0.0 <= corr_result["p_value"] <= 1.0

        # Severity should be valid
        assert corr_result["severity"] in ["INFO", "WARNING", "ERROR"]


def test_german_credit_sampling_bias_power_warnings():
    """Test that small groups trigger power warnings."""
    # Load dataset
    data = load_german_credit()

    # Create imbalanced gender distribution by sampling
    rng = np.random.RandomState(42)

    # Get gender distribution
    gender_counts = data["gender"].value_counts()

    # Subsample to create small minority group
    # Keep only 15 samples from minority group
    minority_group = gender_counts.idxmin()
    majority_group = gender_counts.idxmax()

    minority_mask = data["gender"] == minority_group
    majority_mask = data["gender"] == majority_group

    # Sample 15 from minority, 100 from majority
    minority_indices = data[minority_mask].sample(n=15, random_state=42).index
    majority_indices = data[majority_mask].sample(n=100, random_state=42).index

    data_subset = data.loc[minority_indices.union(majority_indices)].copy()
    data_subset["split"] = "train"

    from glassalpha.metrics.fairness.dataset import compute_sampling_bias_power

    result = compute_sampling_bias_power(
        data=data_subset,
        protected_attrs=["gender"],
        target_effect_size=0.1,
        seed=42,
    )

    # Minority group should have low power (WARNING or ERROR)
    minority_power = result.power_by_group["gender"][minority_group]
    assert minority_power["severity"] in ["WARNING", "ERROR"], \
        f"Expected low power for minority group with n=15, got {minority_power['severity']}"

    # Majority group should have better power
    majority_power = result.power_by_group["gender"][majority_group]
    assert majority_power["power"] > minority_power["power"], \
        "Expected majority group to have higher power"


def test_german_credit_continuous_binning():
    """Test continuous attribute binning on age_years."""
    # Load dataset
    data = load_german_credit()

    from glassalpha.metrics.fairness.dataset import bin_continuous_attribute

    # Test domain strategy (should use default age bins)
    binned = bin_continuous_attribute(
        values=data["age_years"].values,
        attr_name="age_years",
        strategy="domain",
    )

    # Should use default age bins
    expected_bins = [18, 25, 35, 50, 65, 100]
    assert binned.bins == expected_bins

    # Check that all values are binned
    assert len(binned.binned_values) == len(data)

    # Check that categories match bins
    assert len(binned.categories) == len(expected_bins) - 1

    # Test equal-frequency binning
    binned_freq = bin_continuous_attribute(
        values=data["age_years"].values,
        attr_name="age_years",
        strategy="equal_frequency",
        n_bins=4,
    )

    # Should create 4 bins
    assert len(binned_freq.categories) == 4

    # Bins should have approximately equal counts
    counts = binned_freq.binned_values.value_counts()
    assert counts.std() < 50  # Low variance in counts
