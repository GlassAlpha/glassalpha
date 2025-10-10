"""Tests for bootstrap confidence intervals in fairness metrics.

This module tests E10: Statistical Confidence & Uncertainty implementation.
Focus areas:
1. Determinism: same seed → identical CIs
2. Small sample warnings: n<30 → WARNING, n<10 → ERROR
3. Statistical power calculation accuracy
4. Integration with existing fairness metrics
"""

import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:
    pytest = None  # Optional for standalone testing

from glassalpha.metrics.fairness.confidence import (
    BootstrapCI,
    check_sample_size_adequacy,
    compute_bootstrap_ci,
    compute_statistical_power,
)


class TestBootstrapDeterminism:
    """Test that bootstrap CIs are deterministic under same seed."""

    def test_same_seed_identical_cis(self):
        """Same seed should produce byte-identical confidence intervals."""
        # Setup: Simple binary classification data
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        sensitive = np.random.randint(0, 2, 100)

        # Compute TPR for group 0
        def metric_fn(yt, yp, sens):
            mask = sens == 0
            tp = np.sum((yt[mask] == 1) & (yp[mask] == 1))
            fn = np.sum((yt[mask] == 1) & (yp[mask] == 0))
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Run twice with same seed
        ci1 = compute_bootstrap_ci(
            y_true,
            y_pred,
            sensitive,
            metric_fn=metric_fn,
            n_bootstrap=100,
            confidence_level=0.95,
            seed=42,
        )

        ci2 = compute_bootstrap_ci(
            y_true,
            y_pred,
            sensitive,
            metric_fn=metric_fn,
            n_bootstrap=100,
            confidence_level=0.95,
            seed=42,
        )

        # Assert byte-identical results
        assert ci1.lower == ci2.lower, "Lower bound should be identical"
        assert ci1.upper == ci2.upper, "Upper bound should be identical"
        assert ci1.point_estimate == ci2.point_estimate, "Point estimate should be identical"
        assert ci1.std_error == ci2.std_error, "Standard error should be identical"

    def test_different_seed_different_cis(self):
        """Different seeds should produce different CIs (sanity check)."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        sensitive = np.random.randint(0, 2, 100)

        def metric_fn(yt, yp, sens):
            mask = sens == 0
            tp = np.sum((yt[mask] == 1) & (yp[mask] == 1))
            fn = np.sum((yt[mask] == 1) & (yp[mask] == 0))
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        ci1 = compute_bootstrap_ci(
            y_true,
            y_pred,
            sensitive,
            metric_fn,
            n_bootstrap=100,
            seed=42,
        )
        ci2 = compute_bootstrap_ci(
            y_true,
            y_pred,
            sensitive,
            metric_fn,
            n_bootstrap=100,
            seed=123,
        )

        # Should be different (with high probability)
        assert ci1.lower != ci2.lower or ci1.upper != ci2.upper


class TestSmallSampleWarnings:
    """Test sample size adequacy checks and warnings."""

    def test_sample_size_less_than_10_error(self):
        """n<10 should raise ERROR level warning."""
        result = check_sample_size_adequacy(n=8, group_name="group_0")

        assert result["status"] == "ERROR"
        assert result["n_samples"] == 8
        assert result["min_required"] == 10
        assert "insufficient" in result["message"].lower()

    def test_sample_size_10_to_29_warning(self):
        """10 ≤ n < 30 should raise WARNING level."""
        result = check_sample_size_adequacy(n=25, group_name="group_1")

        assert result["status"] == "WARNING"
        assert result["n_samples"] == 25
        assert result["min_recommended"] == 30
        assert "small sample" in result["message"].lower()

    def test_sample_size_30_or_more_ok(self):
        """N ≥ 30 should pass without warning."""
        result = check_sample_size_adequacy(n=50, group_name="group_2")

        assert result["status"] == "OK"
        assert result["n_samples"] == 50
        assert "adequate" in result["message"].lower()

    def test_sample_size_boundary_cases(self):
        """Test exact boundary values."""
        # n=9 → ERROR
        assert check_sample_size_adequacy(n=9)["status"] == "ERROR"

        # n=10 → WARNING
        assert check_sample_size_adequacy(n=10)["status"] == "WARNING"

        # n=29 → WARNING
        assert check_sample_size_adequacy(n=29)["status"] == "WARNING"

        # n=30 → OK
        assert check_sample_size_adequacy(n=30)["status"] == "OK"


class TestStatisticalPower:
    """Test statistical power calculations for detecting disparity."""

    def test_power_calculation_basic(self):
        """Test basic statistical power calculation."""
        # Power to detect 10% disparity with n=100 per group
        power = compute_statistical_power(
            n_samples=100,
            effect_size=0.10,  # 10% disparity
            alpha=0.05,
            test_type="two_proportion",
        )

        assert 0.0 <= power <= 1.0, "Power should be between 0 and 1"
        # n=100 with 10% effect gives low power (~0.30) - this is expected
        assert 0.25 <= power <= 0.35, f"Expected power ~0.30, got {power}"

    def test_larger_sample_higher_power(self):
        """Larger sample size should increase statistical power."""
        power_100 = compute_statistical_power(
            n_samples=100,
            effect_size=0.10,
            alpha=0.05,
        )
        power_500 = compute_statistical_power(
            n_samples=500,
            effect_size=0.10,
            alpha=0.05,
        )

        assert power_500 > power_100, "Larger n should have higher power"

    def test_larger_effect_higher_power(self):
        """Larger effect size should increase statistical power."""
        power_small = compute_statistical_power(
            n_samples=100,
            effect_size=0.05,
            alpha=0.05,
        )
        power_large = compute_statistical_power(
            n_samples=100,
            effect_size=0.20,
            alpha=0.05,
        )

        assert power_large > power_small, "Larger effect should have higher power"

    def test_small_sample_low_power_warning(self):
        """Small samples should have low power (integration check)."""
        power = compute_statistical_power(
            n_samples=15,
            effect_size=0.10,
            alpha=0.05,
        )

        # With n=15, power should be quite low for 10% effect
        assert power < 0.5, "Small sample should have low power"


class TestBootstrapCIComputation:
    """Test bootstrap CI computation mechanics."""

    def test_ci_width_decreases_with_larger_sample(self):
        """CI width should decrease with larger sample size."""

        def metric_fn(yt, yp, sens):
            return np.mean(yp)

        # Small sample
        np.random.seed(42)
        y_true_small = np.random.randint(0, 2, 50)
        y_pred_small = np.random.randint(0, 2, 50)
        sensitive_small = np.random.randint(0, 2, 50)

        ci_small = compute_bootstrap_ci(
            y_true_small,
            y_pred_small,
            sensitive_small,
            metric_fn,
            n_bootstrap=100,
            seed=42,
        )

        # Large sample
        y_true_large = np.random.randint(0, 2, 500)
        y_pred_large = np.random.randint(0, 2, 500)
        sensitive_large = np.random.randint(0, 2, 500)

        ci_large = compute_bootstrap_ci(
            y_true_large,
            y_pred_large,
            sensitive_large,
            metric_fn,
            n_bootstrap=100,
            seed=42,
        )

        width_small = ci_small.upper - ci_small.lower
        width_large = ci_large.upper - ci_large.lower

        assert width_large < width_small, "Larger sample should have narrower CI"

    def test_bootstrap_with_stratification(self):
        """Bootstrap should use stratified sampling for fairness metrics."""
        # Imbalanced groups
        np.random.seed(42)
        y_true = np.concatenate(
            [
                np.ones(80),  # 80% in class 1
                np.zeros(20),  # 20% in class 0
            ]
        )
        y_pred = np.random.randint(0, 2, 100)
        sensitive = np.random.randint(0, 2, 100)

        def metric_fn(yt, yp, sens):
            return np.mean(yp)

        ci = compute_bootstrap_ci(
            y_true,
            y_pred,
            sensitive,
            metric_fn,
            n_bootstrap=100,
            stratify=True,  # Stratified bootstrap
            seed=42,
        )

        # Should complete without error
        assert ci.lower <= ci.point_estimate <= ci.upper

    def test_percentile_vs_bca_methods(self):
        """Test different CI methods (percentile vs BCa)."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        sensitive = np.random.randint(0, 2, 100)

        def metric_fn(yt, yp, sens):
            return np.mean(yp)

        # Percentile method
        ci_percentile = compute_bootstrap_ci(
            y_true,
            y_pred,
            sensitive,
            metric_fn,
            n_bootstrap=100,
            method="percentile",
            seed=42,
        )

        # BCa method (bias-corrected and accelerated)
        ci_bca = compute_bootstrap_ci(
            y_true,
            y_pred,
            sensitive,
            metric_fn,
            n_bootstrap=100,
            method="bca",
            seed=42,
        )

        # Both should be valid
        assert ci_percentile.lower <= ci_percentile.upper
        assert ci_bca.lower <= ci_bca.upper

        # BCa may differ from percentile for skewed distributions
        # (Just check they both run successfully)


class TestIntegrationWithFairnessMetrics:
    """Test integration of CIs with existing fairness metrics."""

    def test_tpr_with_confidence_intervals(self):
        """Test TPR computation with CIs."""
        # Create simple dataset with known TPR
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0] * 10)
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0] * 10)
        sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 10)

        # TPR for group 0: 3/4 = 0.75
        # TPR for group 1: Should be calculable

        def tpr_metric(yt, yp, sens, group):
            mask = sens == group
            tp = np.sum((yt[mask] == 1) & (yp[mask] == 1))
            fn = np.sum((yt[mask] == 1) & (yp[mask] == 0))
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        ci_group0 = compute_bootstrap_ci(
            y_true,
            y_pred,
            sensitive,
            lambda yt, yp, s: tpr_metric(yt, yp, s, 0),
            n_bootstrap=100,
            seed=42,
        )

        # Point estimate should be close to 0.75
        assert 0.6 <= ci_group0.point_estimate <= 0.9
        assert ci_group0.lower <= ci_group0.point_estimate <= ci_group0.upper

    def test_demographic_parity_with_cis(self):
        """Test demographic parity with confidence intervals."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_pred = np.random.randint(0, 2, 200)
        sensitive = np.random.randint(0, 2, 200)

        def selection_rate(yt, yp, sens, group):
            mask = sens == group
            return np.mean(yp[mask])

        ci_group0 = compute_bootstrap_ci(
            y_true,
            y_pred,
            sensitive,
            lambda yt, yp, s: selection_rate(yt, yp, s, 0),
            n_bootstrap=100,
            seed=42,
        )

        ci_group1 = compute_bootstrap_ci(
            y_true,
            y_pred,
            sensitive,
            lambda yt, yp, s: selection_rate(yt, yp, s, 1),
            n_bootstrap=100,
            seed=42,
        )

        # Both should be valid
        assert ci_group0.lower <= ci_group0.upper
        assert ci_group1.lower <= ci_group1.upper


class TestBootstrapCIDataclass:
    """Test BootstrapCI dataclass structure."""

    def test_dataclass_fields(self):
        """Test BootstrapCI has all required fields."""
        ci = BootstrapCI(
            point_estimate=0.75,
            lower=0.65,
            upper=0.85,
            std_error=0.05,
            confidence_level=0.95,
            n_bootstrap=100,
            method="percentile",
        )

        assert ci.point_estimate == 0.75
        assert ci.lower == 0.65
        assert ci.upper == 0.85
        assert ci.std_error == 0.05
        assert ci.confidence_level == 0.95
        assert ci.n_bootstrap == 100
        assert ci.method == "percentile"

    def test_width_property(self):
        """Test CI width calculation."""
        ci = BootstrapCI(
            point_estimate=0.75,
            lower=0.65,
            upper=0.85,
            std_error=0.05,
            confidence_level=0.95,
            n_bootstrap=100,
            method="percentile",
        )

        assert abs(ci.width - 0.20) < 1e-10  # 0.85 - 0.65 (floating point tolerance)

    def test_to_dict(self):
        """Test conversion to dictionary for JSON export."""
        ci = BootstrapCI(
            point_estimate=0.75,
            lower=0.65,
            upper=0.85,
            std_error=0.05,
            confidence_level=0.95,
            n_bootstrap=100,
            method="percentile",
        )

        d = ci.to_dict()

        assert d["point_estimate"] == 0.75
        assert d["ci_lower"] == 0.65
        assert d["ci_upper"] == 0.85
        assert d["std_error"] == 0.05
        assert d["confidence_level"] == 0.95


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_group(self):
        """Test handling of empty groups."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 0, 1, 0])
        sensitive = np.array([0, 0, 0, 0])  # Only group 0

        def metric_fn(yt, yp, sens):
            # Try to compute for non-existent group 1
            mask = sens == 1
            if not np.any(mask):
                return np.nan
            return np.mean(yp[mask])

        # Should handle NaN gracefully
        ci = compute_bootstrap_ci(
            y_true,
            y_pred,
            sensitive,
            metric_fn,
            n_bootstrap=10,
            seed=42,
        )

        # Should return NaN or raise informative error
        assert np.isnan(ci.point_estimate) or ci.point_estimate is None

    def test_perfect_predictions(self):
        """Test with perfect predictions (no variance)."""
        y_true = np.ones(100)
        y_pred = np.ones(100)  # Perfect predictions
        sensitive = np.random.randint(0, 2, 100)

        def metric_fn(yt, yp, sens):
            return np.mean(yp == yt)

        ci = compute_bootstrap_ci(
            y_true,
            y_pred,
            sensitive,
            metric_fn,
            n_bootstrap=100,
            seed=42,
        )

        # Should be 1.0 with very narrow CI
        assert ci.point_estimate == 1.0
        assert ci.width < 0.1  # Very narrow because no variance

    def test_extreme_imbalance(self):
        """Test with extreme class imbalance (1% positive)."""
        np.random.seed(42)
        y_true = np.concatenate([np.ones(5), np.zeros(495)])
        y_pred = np.random.randint(0, 2, 500)
        sensitive = np.random.randint(0, 2, 500)

        def metric_fn(yt, yp, sens):
            return np.mean(yp)

        # Should handle imbalance gracefully
        ci = compute_bootstrap_ci(
            y_true,
            y_pred,
            sensitive,
            metric_fn,
            n_bootstrap=100,
            stratify=True,
            seed=42,
        )

        assert 0.0 <= ci.point_estimate <= 1.0
        assert ci.lower <= ci.upper
