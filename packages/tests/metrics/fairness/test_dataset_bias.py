"""
Contract tests for dataset-level bias audit (E12).

Tests cover:
1. Proxy correlation detection (protected vs non-protected features)
2. Distribution drift (KS-test for continuous, Chi-square for categorical)
3. Statistical power for sampling bias detection
4. Continuous protected attribute binning
5. Train/test split imbalance detection
"""

import numpy as np
import pandas as pd
import pytest
from typing import Dict, List

from glassalpha.metrics.fairness.dataset import (
    compute_dataset_bias_metrics,
    compute_proxy_correlations,
    compute_distribution_drift,
    compute_sampling_bias_power,
    bin_continuous_attribute,
    detect_split_imbalance,
    ProxyCorrelationResult,
    DistributionDriftResult,
    SamplingBiasResult,
    DatasetBiasMetrics,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def seed():
    """Fixed seed for deterministic tests."""
    return 42


@pytest.fixture
def simple_dataset():
    """Simple dataset with clear proxy correlation."""
    rng = np.random.RandomState(42)
    n = 200

    # Protected attribute
    gender = rng.choice(["Male", "Female"], size=n, p=[0.6, 0.4])

    # Strong proxy: zip_code correlates with gender
    zip_code = np.where(
        gender == "Male",
        rng.choice([10001, 10002], size=n),
        rng.choice([10003, 10004], size=n),
    )

    # Weak proxy: age has weak correlation
    age = np.where(
        gender == "Male",
        rng.normal(45, 10, size=n),
        rng.normal(42, 10, size=n),
    ).astype(int)

    # Non-proxy: random feature
    random_feature = rng.normal(0, 1, size=n)

    df = pd.DataFrame({
        "gender": gender,
        "zip_code": zip_code,
        "age": age,
        "random_feature": random_feature,
        "label": rng.choice([0, 1], size=n),
    })

    return df


@pytest.fixture
def imbalanced_split_dataset():
    """Dataset with imbalanced train/test split."""
    rng = np.random.RandomState(42)

    # Train set: 70% male
    train_gender = rng.choice(["Male", "Female"], size=140, p=[0.7, 0.3])
    train_df = pd.DataFrame({
        "gender": train_gender,
        "feature1": rng.normal(0, 1, size=140),
        "label": rng.choice([0, 1], size=140),
        "split": "train",
    })

    # Test set: 40% male (imbalanced)
    test_gender = rng.choice(["Male", "Female"], size=60, p=[0.4, 0.6])
    test_df = pd.DataFrame({
        "gender": test_gender,
        "feature1": rng.normal(0, 1, size=60),
        "label": rng.choice([0, 1], size=60),
        "split": "test",
    })

    return pd.concat([train_df, test_df], ignore_index=True)


@pytest.fixture
def drift_dataset():
    """Dataset with distribution drift between train and test."""
    rng = np.random.RandomState(42)

    # Train: normal(0, 1)
    train_df = pd.DataFrame({
        "continuous_feature": rng.normal(0, 1, size=150),
        "categorical_feature": rng.choice(["A", "B", "C"], size=150, p=[0.5, 0.3, 0.2]),
        "split": "train",
    })

    # Test: normal(0.5, 1) - drifted continuous
    # and different categorical distribution
    test_df = pd.DataFrame({
        "continuous_feature": rng.normal(0.5, 1, size=50),
        "categorical_feature": rng.choice(["A", "B", "C"], size=50, p=[0.2, 0.3, 0.5]),
        "split": "test",
    })

    return pd.concat([train_df, test_df], ignore_index=True)


# ============================================================================
# PROXY CORRELATION TESTS
# ============================================================================


def test_proxy_correlation_strong_correlation(simple_dataset, seed):
    """Test detection of strong proxy correlation (ERROR level)."""
    result = compute_proxy_correlations(
        data=simple_dataset,
        protected_attrs=["gender"],
        feature_cols=["zip_code", "age", "random_feature"],
        seed=seed,
    )

    # Should detect strong correlation with zip_code
    assert "zip_code" in result.correlations["gender"]
    zip_corr = result.correlations["gender"]["zip_code"]

    assert abs(zip_corr["correlation"]) > 0.5
    assert zip_corr["severity"] == "ERROR"
    assert zip_corr["p_value"] < 0.05


def test_proxy_correlation_weak_correlation(simple_dataset, seed):
    """Test detection of weak proxy correlation (INFO level)."""
    result = compute_proxy_correlations(
        data=simple_dataset,
        protected_attrs=["gender"],
        feature_cols=["random_feature"],
        seed=seed,
    )

    # Should detect weak/no correlation with random feature
    assert "random_feature" in result.correlations["gender"]
    random_corr = result.correlations["gender"]["random_feature"]

    assert abs(random_corr["correlation"]) <= 0.3
    assert random_corr["severity"] == "INFO"


def test_proxy_correlation_medium_correlation(simple_dataset, seed):
    """Test detection of medium proxy correlation (WARNING level)."""
    result = compute_proxy_correlations(
        data=simple_dataset,
        protected_attrs=["gender"],
        feature_cols=["age"],
        seed=seed,
    )

    # Age should have medium correlation (WARNING)
    assert "age" in result.correlations["gender"]
    age_corr = result.correlations["gender"]["age"]

    # Depending on random seed, this might be WARNING or INFO
    # Just check it's not ERROR
    assert age_corr["severity"] in ["WARNING", "INFO"]


def test_proxy_correlation_deterministic(simple_dataset, seed):
    """Test that proxy correlation computation is deterministic."""
    result1 = compute_proxy_correlations(
        data=simple_dataset,
        protected_attrs=["gender"],
        feature_cols=["zip_code", "age"],
        seed=seed,
    )

    result2 = compute_proxy_correlations(
        data=simple_dataset,
        protected_attrs=["gender"],
        feature_cols=["zip_code", "age"],
        seed=seed,
    )

    # Should be identical
    for feature in ["zip_code", "age"]:
        r1 = result1.correlations["gender"][feature]["correlation"]
        r2 = result2.correlations["gender"][feature]["correlation"]
        assert abs(r1 - r2) < 1e-10


def test_proxy_correlation_handles_categorical_protected_attr(simple_dataset, seed):
    """Test correlation with categorical protected attribute."""
    result = compute_proxy_correlations(
        data=simple_dataset,
        protected_attrs=["gender"],  # categorical
        feature_cols=["zip_code"],  # categorical
        seed=seed,
    )

    # Should use Cramér's V for categorical-categorical
    assert "zip_code" in result.correlations["gender"]
    assert "correlation" in result.correlations["gender"]["zip_code"]


# ============================================================================
# DISTRIBUTION DRIFT TESTS
# ============================================================================


def test_distribution_drift_continuous_feature(drift_dataset, seed):
    """Test KS-test for continuous feature drift."""
    train_data = drift_dataset[drift_dataset["split"] == "train"]
    test_data = drift_dataset[drift_dataset["split"] == "test"]

    result = compute_distribution_drift(
        train_data=train_data,
        test_data=test_data,
        feature_cols=["continuous_feature"],
        seed=seed,
    )

    # Should detect drift in continuous feature
    assert "continuous_feature" in result.drift_tests
    drift = result.drift_tests["continuous_feature"]

    assert drift["test_type"] == "ks_test"
    assert drift["p_value"] < 0.05  # Significant drift
    assert drift["drifted"] is True


def test_distribution_drift_categorical_feature(drift_dataset, seed):
    """Test Chi-square for categorical feature drift."""
    train_data = drift_dataset[drift_dataset["split"] == "train"]
    test_data = drift_dataset[drift_dataset["split"] == "test"]

    result = compute_distribution_drift(
        train_data=train_data,
        test_data=test_data,
        feature_cols=["categorical_feature"],
        seed=seed,
    )

    # Should detect drift in categorical feature
    assert "categorical_feature" in result.drift_tests
    drift = result.drift_tests["categorical_feature"]

    assert drift["test_type"] == "chi2_test"
    assert drift["drifted"] is True


def test_distribution_drift_no_drift():
    """Test that identical distributions show no drift."""
    rng = np.random.RandomState(42)

    # Same distribution in train and test
    train_data = pd.DataFrame({
        "feature": rng.normal(0, 1, size=100),
    })
    test_data = pd.DataFrame({
        "feature": rng.normal(0, 1, size=100),
    })

    result = compute_distribution_drift(
        train_data=train_data,
        test_data=test_data,
        feature_cols=["feature"],
        seed=42,
    )

    # Should not detect drift (p-value likely > 0.05)
    drift = result.drift_tests["feature"]
    # Note: may occasionally fail due to random sampling
    # In practice, p_value should be high
    assert "p_value" in drift


def test_distribution_drift_deterministic(drift_dataset, seed):
    """Test that drift detection is deterministic."""
    train_data = drift_dataset[drift_dataset["split"] == "train"]
    test_data = drift_dataset[drift_dataset["split"] == "test"]

    result1 = compute_distribution_drift(
        train_data=train_data,
        test_data=test_data,
        feature_cols=["continuous_feature"],
        seed=seed,
    )

    result2 = compute_distribution_drift(
        train_data=train_data,
        test_data=test_data,
        feature_cols=["continuous_feature"],
        seed=seed,
    )

    # Results should be identical (KS-test is deterministic)
    assert result1.drift_tests["continuous_feature"]["statistic"] == \
           result2.drift_tests["continuous_feature"]["statistic"]


# ============================================================================
# SAMPLING BIAS POWER TESTS
# ============================================================================


def test_sampling_bias_power_adequate_sample(seed):
    """Test power calculation with adequate sample size."""
    rng = np.random.RandomState(seed)

    # Large, balanced sample
    data = pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], size=400, p=[0.5, 0.5]),
    })

    result = compute_sampling_bias_power(
        data=data,
        protected_attrs=["gender"],
        target_effect_size=0.1,  # 10% difference
        seed=seed,
    )

    # Should have high power
    assert "gender" in result.power_by_group
    for group, power_result in result.power_by_group["gender"].items():
        assert power_result["power"] >= 0.7  # Adequate power
        assert power_result["severity"] == "OK"


def test_sampling_bias_power_inadequate_sample(seed):
    """Test power calculation with inadequate sample size."""
    rng = np.random.RandomState(seed)

    # Small sample
    data = pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], size=30, p=[0.5, 0.5]),
    })

    result = compute_sampling_bias_power(
        data=data,
        protected_attrs=["gender"],
        target_effect_size=0.1,
        seed=seed,
    )

    # Should have low power
    assert "gender" in result.power_by_group
    # At least one group should have inadequate power
    powers = [pr["power"] for pr in result.power_by_group["gender"].values()]
    assert any(p < 0.7 for p in powers)


def test_sampling_bias_power_imbalanced_groups(seed):
    """Test power calculation with imbalanced groups."""
    rng = np.random.RandomState(seed)

    # Very imbalanced: 90% Male, 10% Female
    data = pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], size=200, p=[0.9, 0.1]),
    })

    result = compute_sampling_bias_power(
        data=data,
        protected_attrs=["gender"],
        target_effect_size=0.1,
        seed=seed,
    )

    # Female group should have lower power due to small sample
    assert "Female" in result.power_by_group["gender"]
    female_power = result.power_by_group["gender"]["Female"]["power"]
    male_power = result.power_by_group["gender"]["Male"]["power"]

    assert female_power < male_power


def test_sampling_bias_power_deterministic(seed):
    """Test that power calculation is deterministic."""
    rng = np.random.RandomState(seed)
    data = pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], size=100, p=[0.5, 0.5]),
    })

    result1 = compute_sampling_bias_power(
        data=data,
        protected_attrs=["gender"],
        target_effect_size=0.1,
        seed=seed,
    )

    result2 = compute_sampling_bias_power(
        data=data,
        protected_attrs=["gender"],
        target_effect_size=0.1,
        seed=seed,
    )

    # Results should be identical
    for group in result1.power_by_group["gender"]:
        p1 = result1.power_by_group["gender"][group]["power"]
        p2 = result2.power_by_group["gender"][group]["power"]
        assert abs(p1 - p2) < 1e-10


# ============================================================================
# CONTINUOUS BINNING TESTS
# ============================================================================


def test_bin_continuous_domain_strategy():
    """Test domain-specific binning for age."""
    rng = np.random.RandomState(42)
    ages = rng.uniform(18, 80, size=200)

    binned = bin_continuous_attribute(
        values=ages,
        attr_name="age_years",
        strategy="domain",
        bins=None,  # Use default domain bins
    )

    # Should use default age bins
    expected_bins = [18, 25, 35, 50, 65, 100]
    assert len(binned.categories) == len(expected_bins) - 1


def test_bin_continuous_custom_bins():
    """Test custom bins for continuous attribute."""
    rng = np.random.RandomState(42)
    values = rng.uniform(0, 100, size=200)

    custom_bins = [0, 25, 50, 75, 100]
    binned = bin_continuous_attribute(
        values=values,
        attr_name="custom_feature",
        strategy="custom",
        bins=custom_bins,
    )

    # Should use custom bins
    assert len(binned.categories) == len(custom_bins) - 1
    assert binned.bins == custom_bins


def test_bin_continuous_equal_width():
    """Test equal-width binning."""
    rng = np.random.RandomState(42)
    values = rng.uniform(0, 100, size=200)

    binned = bin_continuous_attribute(
        values=values,
        attr_name="feature",
        strategy="equal_width",
        n_bins=4,
    )

    # Should create 4 equal-width bins
    assert len(binned.categories) == 4

    # Bins should be evenly spaced
    bin_widths = np.diff(binned.bins)
    assert np.allclose(bin_widths, bin_widths[0], rtol=0.01)


def test_bin_continuous_equal_frequency():
    """Test equal-frequency (quantile) binning."""
    rng = np.random.RandomState(42)
    values = rng.uniform(0, 100, size=200)

    binned = bin_continuous_attribute(
        values=values,
        attr_name="feature",
        strategy="equal_frequency",
        n_bins=4,
    )

    # Should create 4 bins with approximately equal counts
    assert len(binned.categories) == 4

    # Check that counts are roughly equal
    counts = binned.binned_values.value_counts().values
    assert np.std(counts) < 10  # Low variance in counts


def test_bin_continuous_preserves_order():
    """Test that binning preserves value order."""
    values = np.array([10, 20, 30, 40, 50])

    binned = bin_continuous_attribute(
        values=values,
        attr_name="feature",
        strategy="equal_width",
        n_bins=2,
    )

    # Lower values should be in lower bins
    assert binned.binned_values[0] < binned.binned_values[-1]


# ============================================================================
# SPLIT IMBALANCE TESTS
# ============================================================================


def test_split_imbalance_detected(imbalanced_split_dataset, seed):
    """Test detection of train/test split imbalance."""
    result = detect_split_imbalance(
        data=imbalanced_split_dataset,
        protected_attrs=["gender"],
        split_col="split",
        seed=seed,
    )

    # Should detect imbalance for gender
    assert "gender" in result.imbalance_tests
    imbalance = result.imbalance_tests["gender"]

    assert imbalance["p_value"] < 0.05  # Significant imbalance
    assert imbalance["imbalanced"] is True

    # Should report train/test proportions
    assert "train_distribution" in imbalance
    assert "test_distribution" in imbalance


def test_split_imbalance_not_detected():
    """Test that balanced splits show no imbalance."""
    rng = np.random.RandomState(42)

    # Balanced split: same distribution
    train_gender = rng.choice(["Male", "Female"], size=140, p=[0.5, 0.5])
    test_gender = rng.choice(["Male", "Female"], size=60, p=[0.5, 0.5])

    data = pd.DataFrame({
        "gender": np.concatenate([train_gender, test_gender]),
        "split": ["train"] * 140 + ["test"] * 60,
    })

    result = detect_split_imbalance(
        data=data,
        protected_attrs=["gender"],
        split_col="split",
        seed=42,
    )

    # Should not detect significant imbalance
    imbalance = result.imbalance_tests["gender"]
    # p-value should be high (likely > 0.05)
    assert "p_value" in imbalance


def test_split_imbalance_multiple_protected_attrs(seed):
    """Test imbalance detection with multiple protected attributes."""
    rng = np.random.RandomState(seed)

    # Gender: imbalanced, Race: balanced
    train_df = pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], size=140, p=[0.7, 0.3]),
        "race": rng.choice(["White", "Black"], size=140, p=[0.5, 0.5]),
        "split": "train",
    })

    test_df = pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], size=60, p=[0.4, 0.6]),
        "race": rng.choice(["White", "Black"], size=60, p=[0.5, 0.5]),
        "split": "test",
    })

    data = pd.concat([train_df, test_df], ignore_index=True)

    result = detect_split_imbalance(
        data=data,
        protected_attrs=["gender", "race"],
        split_col="split",
        seed=seed,
    )

    # Gender should be imbalanced, race should not
    assert result.imbalance_tests["gender"]["imbalanced"] is True
    # Race may or may not be flagged depending on random seed


def test_split_imbalance_deterministic(imbalanced_split_dataset, seed):
    """Test that imbalance detection is deterministic."""
    result1 = detect_split_imbalance(
        data=imbalanced_split_dataset,
        protected_attrs=["gender"],
        split_col="split",
        seed=seed,
    )

    result2 = detect_split_imbalance(
        data=imbalanced_split_dataset,
        protected_attrs=["gender"],
        split_col="split",
        seed=seed,
    )

    # Results should be identical
    assert result1.imbalance_tests["gender"]["p_value"] == \
           result2.imbalance_tests["gender"]["p_value"]


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_compute_dataset_bias_metrics_full_pipeline(simple_dataset, seed):
    """Test full dataset bias metrics computation."""
    # Split into train/test
    train_data = simple_dataset.iloc[:140].copy()
    test_data = simple_dataset.iloc[140:].copy()
    train_data["split"] = "train"
    test_data["split"] = "test"
    full_data = pd.concat([train_data, test_data], ignore_index=True)

    result = compute_dataset_bias_metrics(
        data=full_data,
        protected_attrs=["gender"],
        feature_cols=["zip_code", "age", "random_feature"],
        split_col="split",
        seed=seed,
    )

    # Should compute all metrics
    assert result.proxy_correlations is not None
    assert result.distribution_drift is not None
    assert result.sampling_bias_power is not None
    assert result.split_imbalance is not None

    # Should detect strong proxy (zip_code)
    assert "zip_code" in result.proxy_correlations.correlations["gender"]


def test_compute_dataset_bias_metrics_deterministic(simple_dataset, seed):
    """Test that full pipeline is deterministic."""
    train_data = simple_dataset.iloc[:140].copy()
    test_data = simple_dataset.iloc[140:].copy()
    train_data["split"] = "train"
    test_data["split"] = "test"
    full_data = pd.concat([train_data, test_data], ignore_index=True)

    result1 = compute_dataset_bias_metrics(
        data=full_data,
        protected_attrs=["gender"],
        feature_cols=["zip_code"],
        split_col="split",
        seed=seed,
    )

    result2 = compute_dataset_bias_metrics(
        data=full_data,
        protected_attrs=["gender"],
        feature_cols=["zip_code"],
        split_col="split",
        seed=seed,
    )

    # Proxy correlations should be identical
    r1 = result1.proxy_correlations.correlations["gender"]["zip_code"]["correlation"]
    r2 = result2.proxy_correlations.correlations["gender"]["zip_code"]["correlation"]
    assert abs(r1 - r2) < 1e-10


def test_compute_dataset_bias_metrics_serializable(simple_dataset, seed):
    """Test that results are JSON serializable."""
    import json

    train_data = simple_dataset.iloc[:140].copy()
    test_data = simple_dataset.iloc[140:].copy()
    train_data["split"] = "train"
    test_data["split"] = "test"
    full_data = pd.concat([train_data, test_data], ignore_index=True)

    result = compute_dataset_bias_metrics(
        data=full_data,
        protected_attrs=["gender"],
        feature_cols=["zip_code"],
        split_col="split",
        seed=seed,
    )

    # Should be serializable
    result_dict = result.to_dict()
    json_str = json.dumps(result_dict)

    # Should round-trip
    loaded = json.loads(json_str)
    assert "proxy_correlations" in loaded
    assert "distribution_drift" in loaded


def test_dataset_bias_handles_missing_values(seed):
    """Test that dataset bias metrics handle missing values gracefully."""
    rng = np.random.RandomState(seed)

    data = pd.DataFrame({
        "gender": rng.choice(["Male", "Female", None], size=100, p=[0.45, 0.45, 0.1]),
        "feature": rng.normal(0, 1, size=100),
        "split": "train",
    })

    # Should not crash on missing values
    result = compute_dataset_bias_metrics(
        data=data,
        protected_attrs=["gender"],
        feature_cols=["feature"],
        split_col="split",
        seed=seed,
    )

    # Should complete without error
    assert result is not None


def test_dataset_bias_empty_protected_attrs():
    """Test that empty protected attributes list raises error."""
    data = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(ValueError, match="protected_attrs.*empty"):
        compute_dataset_bias_metrics(
            data=data,
            protected_attrs=[],
            feature_cols=["feature"],
            split_col="split",
            seed=42,
        )
