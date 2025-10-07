"""Tests for E5.1: Basic Intersectional Fairness.

This module tests two-way intersectional fairness analysis including:
1. Intersection spec parsing and validation
2. Intersectional group creation (cartesian product)
3. Fairness metrics computation for intersectional groups
4. Sample size warnings (n<30 WARNING, n<10 ERROR)
5. Bootstrap confidence intervals (reusing E10 infrastructure)
6. Deterministic computation (seeded reproducibility)
"""

import numpy as np
import pandas as pd
import pytest

from glassalpha.metrics.fairness.intersectional import (
    compute_intersectional_fairness,
    create_intersectional_groups,
    parse_intersection_spec,
)


class TestIntersectionParsing:
    """Test intersection specification parsing and validation."""

    def test_parse_valid_intersection(self):
        """Test parsing valid intersection spec."""
        attr1, attr2 = parse_intersection_spec("gender*race")
        assert attr1 == "gender"
        assert attr2 == "race"

    def test_parse_with_whitespace(self):
        """Test parsing handles whitespace."""
        attr1, attr2 = parse_intersection_spec(" gender * race ")
        assert attr1 == "gender"
        assert attr2 == "race"

    def test_parse_invalid_format_no_separator(self):
        """Test invalid format without separator."""
        with pytest.raises(ValueError, match="Invalid intersection format"):
            parse_intersection_spec("genderrace")

    def test_parse_invalid_format_too_many_parts(self):
        """Test invalid format with too many parts."""
        with pytest.raises(ValueError, match="Invalid intersection format"):
            parse_intersection_spec("gender*race*age")

    def test_parse_empty_attribute(self):
        """Test invalid format with empty attribute."""
        with pytest.raises(ValueError, match="Both attributes must be non-empty"):
            parse_intersection_spec("gender*")

    def test_parse_both_empty(self):
        """Test invalid format with both empty."""
        with pytest.raises(ValueError, match="Both attributes must be non-empty"):
            parse_intersection_spec("*")


class TestIntersectionalGroupCreation:
    """Test intersectional group creation logic."""

    def test_create_binary_binary_groups(self):
        """Test creating groups from two binary attributes."""
        # Create data with all 4 combinations present
        df = pd.DataFrame({
            "gender": ["M", "M", "F", "F", "M", "F"],
            "race": ["White", "Black", "White", "Black", "White", "Black"],
        })

        groups = create_intersectional_groups(df, "gender", "race")

        # Should create 4 unique groups
        unique_groups = np.unique(groups)
        assert len(unique_groups) == 4
        assert "M*White" in unique_groups
        assert "M*Black" in unique_groups
        assert "F*White" in unique_groups
        assert "F*Black" in unique_groups

    def test_create_groups_with_numeric_values(self):
        """Test creating groups with numeric attribute values."""
        df = pd.DataFrame({
            "age_group": [0, 1, 0, 1, 0, 1],
            "income_group": [0, 0, 1, 1, 0, 1],
        })

        groups = create_intersectional_groups(df, "age_group", "income_group")

        # Should create 4 unique groups
        unique_groups = np.unique(groups)
        assert len(unique_groups) == 4
        assert "0*0" in unique_groups
        assert "0*1" in unique_groups
        assert "1*0" in unique_groups
        assert "1*1" in unique_groups

    def test_create_groups_missing_attribute(self):
        """Test error when attribute missing."""
        df = pd.DataFrame({
            "gender": ["M", "F"],
        })

        with pytest.raises(ValueError, match="Attribute 'race' not found"):
            create_intersectional_groups(df, "gender", "race")

    def test_create_groups_deterministic(self):
        """Test group creation is deterministic."""
        df = pd.DataFrame({
            "gender": ["M", "F", "M", "F"],
            "race": ["White", "Black", "White", "Black"],
        })

        groups1 = create_intersectional_groups(df, "gender", "race")
        groups2 = create_intersectional_groups(df, "gender", "race")

        pd.testing.assert_series_equal(groups1, groups2)


class TestIntersectionalFairnessMetrics:
    """Test intersectional fairness metric computation."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic data with known intersectional bias."""
        np.random.seed(42)
        n_samples = 200

        # Create two binary attributes
        gender = np.random.choice(["M", "F"], size=n_samples)
        race = np.random.choice(["White", "Black"], size=n_samples)

        # Create intersectional groups
        df = pd.DataFrame({"gender": gender, "race": race})

        # Create biased labels and predictions
        # Bias pattern: Model favors White Males
        y_true = np.random.randint(0, 2, size=n_samples)
        y_pred = y_true.copy()

        # Inject bias: Lower TPR for Black Females
        for i in range(n_samples):
            if gender[i] == "F" and race[i] == "Black" and y_true[i] == 1:
                # Reduce TPR by missing positive predictions
                if np.random.rand() < 0.3:  # 30% miss rate
                    y_pred[i] = 0

        return y_true, y_pred, df

    def test_compute_basic_metrics(self, synthetic_data):
        """Test basic intersectional metrics computation."""
        y_true, y_pred, df = synthetic_data

        results = compute_intersectional_fairness(
            y_true,
            y_pred,
            df,
            intersection_spec="gender*race",
            compute_confidence_intervals=False,
            seed=42,
        )

        # Check structure
        assert "intersection_spec" in results
        assert results["intersection_spec"] == "gender*race"
        assert "attr1" in results
        assert results["attr1"] == "gender"
        assert "attr2" in results
        assert results["attr2"] == "race"
        assert "n_groups" in results

        # Check metrics for each group
        for group in ["M*White", "M*Black", "F*White", "F*Black"]:
            group_key = f"gender*race_{group}"
            assert f"{group_key}_tpr" in results
            assert f"{group_key}_fpr" in results
            assert f"{group_key}_precision" in results
            assert f"{group_key}_recall" in results
            assert f"{group_key}_selection_rate" in results
            assert f"{group_key}_n_samples" in results

    def test_sample_size_warnings(self, synthetic_data):
        """Test sample size warnings for intersectional groups."""
        y_true, y_pred, df = synthetic_data

        results = compute_intersectional_fairness(
            y_true,
            y_pred,
            df,
            intersection_spec="gender*race",
            compute_confidence_intervals=False,
            seed=42,
        )

        # Check sample size warnings exist
        assert "sample_size_warnings" in results
        warnings = results["sample_size_warnings"]

        # Each group should have a warning entry
        for group in ["M*White", "M*Black", "F*White", "F*Black"]:
            group_key = f"gender*race_{group}"
            assert group_key in warnings
            assert "status" in warnings[group_key]
            assert warnings[group_key]["status"] in ["OK", "WARNING", "ERROR"]

    def test_statistical_power_analysis(self, synthetic_data):
        """Test statistical power analysis for intersectional groups."""
        y_true, y_pred, df = synthetic_data

        results = compute_intersectional_fairness(
            y_true,
            y_pred,
            df,
            intersection_spec="gender*race",
            compute_confidence_intervals=False,
            seed=42,
        )

        # Check power analysis exists
        assert "statistical_power" in results
        power = results["statistical_power"]

        # Each group should have power analysis
        for group in ["M*White", "M*Black", "F*White", "F*Black"]:
            group_key = f"gender*race_{group}"
            assert group_key in power
            assert "power" in power[group_key]
            assert "interpretation" in power[group_key]
            assert 0.0 <= power[group_key]["power"] <= 1.0

    def test_disparity_metrics(self, synthetic_data):
        """Test disparity metrics across intersectional groups."""
        y_true, y_pred, df = synthetic_data

        results = compute_intersectional_fairness(
            y_true,
            y_pred,
            df,
            intersection_spec="gender*race",
            compute_confidence_intervals=False,
            seed=42,
        )

        # Check disparity metrics exist
        assert "gender*race_tpr_disparity_difference" in results
        assert "gender*race_tpr_disparity_ratio" in results
        assert "gender*race_fpr_disparity_difference" in results
        assert "gender*race_fpr_disparity_ratio" in results

        # Ratios should be in [0, 1]
        assert 0.0 <= results["gender*race_tpr_disparity_ratio"] <= 1.0

    def test_confidence_intervals(self, synthetic_data):
        """Test bootstrap confidence intervals for intersectional metrics."""
        y_true, y_pred, df = synthetic_data

        results = compute_intersectional_fairness(
            y_true,
            y_pred,
            df,
            intersection_spec="gender*race",
            compute_confidence_intervals=True,
            n_bootstrap=100,  # Reduced for speed
            confidence_level=0.95,
            seed=42,
        )

        # Check CIs exist
        assert "confidence_intervals" in results
        cis = results["confidence_intervals"]

        # Should have CIs for each group and metric type
        # Check that at least some CIs were computed successfully
        ci_count = 0
        for group in ["M*White", "M*Black", "F*White", "F*Black"]:
            for metric_type in ["tpr", "fpr", "precision", "recall", "selection_rate"]:
                ci_key = f"gender*race_{group}_{metric_type}"
                if ci_key in cis:
                    ci = cis[ci_key]
                    assert "point_estimate" in ci
                    assert "ci_lower" in ci
                    assert "ci_upper" in ci
                    assert "std_error" in ci

                    # CI bounds should bracket point estimate (with small tolerance for numerical errors)
                    # In rare cases with bootstrap, point estimate might be slightly outside due to sampling
                    if not np.isnan(ci["ci_lower"]) and not np.isnan(ci["ci_upper"]):
                        # Allow small numerical tolerance (1% of CI width)
                        tolerance = 0.01 * (ci["ci_upper"] - ci["ci_lower"])
                        assert ci["ci_lower"] - tolerance <= ci["point_estimate"] <= ci["ci_upper"] + tolerance, \
                            f"CI bounds [{ci['ci_lower']}, {ci['ci_upper']}] do not bracket point estimate {ci['point_estimate']}"

                    ci_count += 1

        # At least some CIs should have been computed
        assert ci_count > 0, "No confidence intervals were computed"

    def test_deterministic_computation(self, synthetic_data):
        """Test computation is deterministic with same seed."""
        y_true, y_pred, df = synthetic_data

        results1 = compute_intersectional_fairness(
            y_true,
            y_pred,
            df,
            intersection_spec="gender*race",
            compute_confidence_intervals=True,
            n_bootstrap=100,
            seed=42,
        )

        results2 = compute_intersectional_fairness(
            y_true,
            y_pred,
            df,
            intersection_spec="gender*race",
            compute_confidence_intervals=True,
            n_bootstrap=100,
            seed=42,
        )

        # Check key metrics match
        assert results1["gender*race_tpr_disparity_difference"] == results2["gender*race_tpr_disparity_difference"]

        # Check CIs match (deterministic bootstrap)
        ci1 = results1["confidence_intervals"]["gender*race_M*White_tpr"]
        ci2 = results2["confidence_intervals"]["gender*race_M*White_tpr"]
        assert ci1["point_estimate"] == ci2["point_estimate"]
        assert ci1["ci_lower"] == ci2["ci_lower"]
        assert ci1["ci_upper"] == ci2["ci_upper"]


class TestSmallSampleHandling:
    """Test handling of small intersectional groups."""

    def test_small_group_warning(self):
        """Test WARNING for groups with n<30."""
        # Create data with small group
        df = pd.DataFrame({
            "gender": ["M"] * 50 + ["F"] * 25,  # F group has n=25 < 30
            "race": ["White"] * 50 + ["Black"] * 25,
        })
        y_true = np.random.randint(0, 2, size=75)
        y_pred = y_true.copy()

        results = compute_intersectional_fairness(
            y_true,
            y_pred,
            df,
            intersection_spec="gender*race",
            compute_confidence_intervals=False,
            seed=42,
        )

        warnings = results["sample_size_warnings"]

        # Check that small groups get WARNING
        for warning in warnings.values():
            n = warning["n_samples"]
            if 10 <= n < 30:
                assert warning["status"] == "WARNING"
            elif n < 10:
                assert warning["status"] == "ERROR"
            else:
                assert warning["status"] == "OK"

    def test_very_small_group_error(self):
        """Test ERROR for groups with n<10."""
        # Create data with very small group
        df = pd.DataFrame({
            "gender": ["M"] * 50 + ["F"] * 5,  # F group has n=5 < 10
            "race": ["White"] * 50 + ["Black"] * 5,
        })
        y_true = np.random.randint(0, 2, size=55)
        y_pred = y_true.copy()

        results = compute_intersectional_fairness(
            y_true,
            y_pred,
            df,
            intersection_spec="gender*race",
            compute_confidence_intervals=False,
            seed=42,
        )

        warnings = results["sample_size_warnings"]

        # Check that very small groups get ERROR
        for warning in warnings.values():
            if warning["n_samples"] < 10:
                assert warning["status"] == "ERROR"

    def test_empty_group_handling(self):
        """Test handling of empty intersectional groups."""
        # Create data where one intersection doesn't exist
        df = pd.DataFrame({
            "gender": ["M"] * 50,  # No F group
            "race": ["White"] * 30 + ["Black"] * 20,
        })
        y_true = np.random.randint(0, 2, size=50)
        y_pred = y_true.copy()

        results = compute_intersectional_fairness(
            y_true,
            y_pred,
            df,
            intersection_spec="gender*race",
            compute_confidence_intervals=False,
            seed=42,
        )

        # Empty groups should have n_samples=0 and NaN metrics
        # (only M*White and M*Black exist)
        assert results["n_groups"] == 2  # Only 2 groups exist


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_value_attribute(self):
        """Test when one attribute has only one value."""
        df = pd.DataFrame({
            "gender": ["M"] * 100,  # Only one value
            "race": np.random.choice(["White", "Black"], size=100),
        })
        y_true = np.random.randint(0, 2, size=100)
        y_pred = y_true.copy()

        results = compute_intersectional_fairness(
            y_true,
            y_pred,
            df,
            intersection_spec="gender*race",
            compute_confidence_intervals=False,
            seed=42,
        )

        # Should have 2 groups: M*White and M*Black
        assert results["n_groups"] == 2

    def test_all_same_prediction(self):
        """Test when all predictions are the same."""
        df = pd.DataFrame({
            "gender": np.random.choice(["M", "F"], size=100),
            "race": np.random.choice(["White", "Black"], size=100),
        })
        y_true = np.random.randint(0, 2, size=100)
        y_pred = np.ones(100, dtype=int)  # All predict 1

        results = compute_intersectional_fairness(
            y_true,
            y_pred,
            df,
            intersection_spec="gender*race",
            compute_confidence_intervals=False,
            seed=42,
        )

        # Should compute without error
        assert "gender*race_tpr_disparity_difference" in results

    def test_perfect_predictions(self):
        """Test when predictions are perfect."""
        df = pd.DataFrame({
            "gender": np.random.choice(["M", "F"], size=100),
            "race": np.random.choice(["White", "Black"], size=100),
        })
        y_true = np.random.randint(0, 2, size=100)
        y_pred = y_true.copy()  # Perfect predictions

        results = compute_intersectional_fairness(
            y_true,
            y_pred,
            df,
            intersection_spec="gender*race",
            compute_confidence_intervals=False,
            seed=42,
        )

        # TPR disparity should be 0 (all groups have TPR=1)
        # FPR disparity should be 0 (all groups have FPR=0)
        assert results["gender*race_tpr_disparity_difference"] == 0.0
        assert results["gender*race_fpr_disparity_difference"] == 0.0


class TestIntegrationWithRunner:
    """Test integration with fairness runner."""

    def test_runner_integration(self):
        """Test intersectional metrics integrate with fairness runner."""
        from glassalpha.metrics.fairness.runner import run_fairness_metrics

        # Create test data
        np.random.seed(42)
        n_samples = 200
        gender = np.random.choice(["M", "F"], size=n_samples)
        race = np.random.choice(["White", "Black"], size=n_samples)
        df = pd.DataFrame({"gender": gender, "race": race})

        y_true = np.random.randint(0, 2, size=n_samples)
        y_pred = y_true.copy()

        # Run with intersections
        results = run_fairness_metrics(
            y_true,
            y_pred,
            df,
            metrics=[],  # No regular metrics for this test
            intersections=["gender*race"],
            compute_confidence_intervals=False,
            seed=42,
        )

        # Check intersectional results are included
        assert "intersectional" in results
        assert "gender*race" in results["intersectional"]
        assert "intersection_spec" in results["intersectional"]["gender*race"]
