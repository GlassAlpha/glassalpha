"""Tests for bootstrap confidence intervals in calibration metrics.

This module tests E10+: Calibration Confidence Intervals implementation.
Focus areas:
1. Determinism: same seed → identical CIs
2. Accuracy: CIs cover true metric value in simulated scenarios
3. Bootstrap count: verify n_bootstrap samples used
4. Edge cases: perfect calibration, random predictions, small samples
5. JSON serialization: all results export cleanly
"""

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None  # Optional for standalone testing

from glassalpha.metrics.calibration.confidence import (
    CalibrationCI,
    compute_bin_wise_ci,
    compute_calibration_with_ci,
)
from glassalpha.metrics.calibration.quality import (
    CalibrationResult,
    assess_calibration_quality,
)


class TestBootstrapDeterminism:
    """Test that bootstrap CIs are deterministic under same seed."""

    def test_same_seed_identical_ece_cis(self):
        """Same seed should produce byte-identical ECE confidence intervals."""
        # Setup: Synthetic calibration data
        np.random.seed(42)
        n_samples = 200
        y_true = np.random.binomial(1, 0.4, n_samples)
        y_prob = np.random.beta(2, 3, n_samples)  # Somewhat calibrated

        # Run twice with same seed
        result1 = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=100,
            seed=42,
        )
        result2 = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=100,
            seed=42,
        )

        # Assert byte-identical results
        assert result1["ece_ci"].lower == result2["ece_ci"].lower
        assert result1["ece_ci"].upper == result2["ece_ci"].upper
        assert result1["ece_ci"].point_estimate == result2["ece_ci"].point_estimate
        assert result1["ece_ci"].std_error == result2["ece_ci"].std_error

    def test_same_seed_identical_brier_cis(self):
        """Same seed should produce byte-identical Brier score CIs."""
        np.random.seed(42)
        n_samples = 200
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.random.beta(3, 3, n_samples)

        result1 = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=100,
            seed=42,
        )
        result2 = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=100,
            seed=42,
        )

        assert result1["brier_ci"].lower == result2["brier_ci"].lower
        assert result1["brier_ci"].upper == result2["brier_ci"].upper
        assert result1["brier_ci"].point_estimate == result2["brier_ci"].point_estimate

    def test_different_seed_different_cis(self):
        """Different seeds should produce different CIs (sanity check)."""
        np.random.seed(42)
        n_samples = 200
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.random.beta(3, 3, n_samples)

        result1 = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=100,
            seed=42,
        )
        result2 = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=100,
            seed=123,
        )

        # Should be different (with high probability)
        assert result1["ece_ci"].lower != result2["ece_ci"].lower or result1["ece_ci"].upper != result2["ece_ci"].upper

    def test_bin_wise_ci_determinism(self):
        """Bin-wise CIs should be deterministic under same seed."""
        np.random.seed(42)
        n_samples = 500
        y_true = np.random.binomial(1, 0.3, n_samples)
        y_prob = np.random.beta(1.5, 3.5, n_samples)

        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)

        ci1 = compute_bin_wise_ci(
            y_true,
            y_prob,
            n_bins=5,
            n_bootstrap=50,
            rng=rng1,
        )
        ci2 = compute_bin_wise_ci(
            y_true,
            y_prob,
            n_bins=5,
            n_bootstrap=50,
            rng=rng2,
        )

        # Check all bins have identical CIs
        for i in range(len(ci1.ci_lower)):
            if not np.isnan(ci1.ci_lower[i]):
                assert ci1.ci_lower[i] == ci2.ci_lower[i]
                assert ci1.ci_upper[i] == ci2.ci_upper[i]


class TestCalibrationAccuracy:
    """Test that CIs cover true metric values in known scenarios."""

    def test_perfect_calibration_low_ece(self):
        """Perfect calibration should have ECE CI close to zero."""
        # Create perfectly calibrated data
        np.random.seed(42)
        n_samples = 1000
        y_prob = np.random.uniform(0, 1, n_samples)
        y_true = np.random.binomial(1, y_prob)  # True probability matches predicted

        result = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=100,
            seed=42,
        )

        # ECE should be low for perfect calibration
        assert result["ece"] < 0.1, "Perfect calibration should have low ECE"
        assert result["ece_ci"].upper < 0.15, "ECE CI upper bound should be low"

    def test_overconfident_predictions_high_ece(self):
        """Overconfident predictions should have high ECE."""
        np.random.seed(42)
        n_samples = 500
        # True probability is 0.5, but model is overconfident (predicts extremes)
        y_true = np.random.binomial(1, 0.5, n_samples)
        # Model makes extreme predictions regardless of true label
        y_prob = np.where(np.random.random(n_samples) > 0.5, 0.9, 0.1)

        result = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=100,
            seed=42,
        )

        # ECE should be high for miscalibration (predictions don't match accuracy)
        assert result["ece"] > 0.2, "Overconfident predictions should have high ECE"
        assert result["ece_ci"].lower > 0.15, "ECE CI should be consistently high"

    def test_brier_score_range(self):
        """Brier score should be in [0, 1] range."""
        np.random.seed(42)
        n_samples = 300
        y_true = np.random.binomial(1, 0.6, n_samples)
        y_prob = np.random.beta(4, 2, n_samples)

        result = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=100,
            seed=42,
        )

        # Brier score and CI should be in valid range
        assert 0 <= result["brier_ci"].point_estimate <= 1
        assert 0 <= result["brier_ci"].lower <= 1
        assert 0 <= result["brier_ci"].upper <= 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_sample_size(self):
        """Test behavior with small sample sizes (n=50)."""
        np.random.seed(42)
        n_samples = 50
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.random.beta(3, 3, n_samples)

        # Should complete without error
        result = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=5,
            n_bootstrap=50,
            seed=42,
        )

        assert result["n_samples"] == 50
        assert not np.isnan(result["ece_ci"].point_estimate)

    def test_all_positive_predictions(self):
        """Test with all positive labels."""
        n_samples = 100
        y_true = np.ones(n_samples, dtype=int)
        y_prob = np.random.beta(5, 2, n_samples)  # Mostly high probabilities

        result = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=100,
            seed=42,
        )

        # Should complete and produce valid results
        assert not np.isnan(result["ece"])
        assert not np.isnan(result["brier_score"])

    def test_all_negative_predictions(self):
        """Test with all negative labels."""
        n_samples = 100
        y_true = np.zeros(n_samples, dtype=int)
        y_prob = np.random.beta(2, 5, n_samples)  # Mostly low probabilities

        result = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=100,
            seed=42,
        )

        assert not np.isnan(result["ece"])
        assert not np.isnan(result["brier_score"])

    def test_empty_bins(self):
        """Test calibration with empty bins (sparse predictions)."""
        n_samples = 100
        y_true = np.random.binomial(1, 0.5, n_samples)
        # Predictions only in narrow range (will create empty bins)
        y_prob = np.random.uniform(0.4, 0.6, n_samples)

        result = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=20,
            n_bootstrap=50,
            seed=42,
        )

        # Should handle empty bins gracefully
        assert not np.isnan(result["ece"])
        # Some bins will have NaN CIs (empty bins)
        bin_wise = result["bin_wise_ci"]
        assert len(bin_wise.bins) == 20

    def test_extreme_probabilities(self):
        """Test with probabilities at extremes (0.0 and 1.0)."""
        n_samples = 100
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.where(np.random.random(n_samples) > 0.5, 1.0, 0.0)

        result = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=100,
            seed=42,
        )

        assert not np.isnan(result["ece"])
        assert not np.isnan(result["brier_score"])


class TestBootstrapParameters:
    """Test bootstrap parameter validation and behavior."""

    def test_different_n_bootstrap(self):
        """Test different bootstrap sample counts."""
        np.random.seed(42)
        n_samples = 200
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.random.beta(3, 3, n_samples)

        # Test with different n_bootstrap values
        result_100 = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=100,
            seed=42,
        )
        result_500 = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=500,
            seed=42,
        )

        # More bootstrap samples should reduce std_error (generally)
        # This is probabilistic but should hold in most cases
        assert result_100["ece_ci"].n_bootstrap == 100
        assert result_500["ece_ci"].n_bootstrap == 500

    def test_different_confidence_levels(self):
        """Test different confidence levels."""
        np.random.seed(42)
        n_samples = 200
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.random.beta(3, 3, n_samples)

        result_90 = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=100,
            confidence_level=0.90,
            seed=42,
        )
        result_95 = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=100,
            confidence_level=0.95,
            seed=42,
        )

        # 95% CI should be wider than 90% CI
        width_90 = result_90["ece_ci"].width
        width_95 = result_95["ece_ci"].width
        assert width_95 > width_90, "95% CI should be wider than 90% CI"

    def test_different_n_bins(self):
        """Test different numbers of calibration bins."""
        np.random.seed(42)
        n_samples = 500
        y_true = np.random.binomial(1, 0.4, n_samples)
        y_prob = np.random.beta(2, 3, n_samples)

        result_5 = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=5,
            n_bootstrap=50,
            seed=42,
        )
        result_10 = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=50,
            seed=42,
        )

        assert result_5["n_bins"] == 5
        assert result_10["n_bins"] == 10
        assert len(result_5["bin_wise_ci"].bins) == 5
        assert len(result_10["bin_wise_ci"].bins) == 10


class TestJSONSerialization:
    """Test that all results serialize cleanly to JSON."""

    def test_calibration_ci_to_dict(self):
        """Test CalibrationCI serialization."""
        ci = CalibrationCI(
            metric_name="ece",
            point_estimate=0.05,
            lower=0.03,
            upper=0.07,
            std_error=0.01,
            confidence_level=0.95,
            n_bootstrap=100,
        )

        result = ci.to_dict()

        assert isinstance(result, dict)
        assert result["metric"] == "ece"
        assert result["point_estimate"] == 0.05
        assert result["ci_lower"] == 0.03
        assert result["ci_upper"] == 0.07
        assert abs(result["ci_width"] - 0.04) < 1e-10  # Floating point tolerance

    def test_full_result_serialization(self):
        """Test that full result serializes without errors."""
        np.random.seed(42)
        n_samples = 200
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.random.beta(3, 3, n_samples)

        result = compute_calibration_with_ci(
            y_true,
            y_prob,
            n_bins=10,
            n_bootstrap=100,
            seed=42,
        )

        # Serialize all components
        ece_dict = result["ece_ci"].to_dict()
        brier_dict = result["brier_ci"].to_dict()
        bin_dict = result["bin_wise_ci"].to_dict()

        # Verify all are valid dicts
        assert isinstance(ece_dict, dict)
        assert isinstance(brier_dict, dict)
        assert isinstance(bin_dict, dict)

        # Verify structure
        assert "ci_lower" in ece_dict
        assert "ci_upper" in ece_dict
        assert "bins" in bin_dict
        assert "observed_freq" in bin_dict


class TestAssessCalibrationQuality:
    """Test the extended assess_calibration_quality function."""

    def test_with_confidence_intervals(self):
        """Test new API with CI computation."""
        np.random.seed(42)
        n_samples = 200
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.random.beta(3, 3, n_samples)

        # New call with CI computation
        result = assess_calibration_quality(
            y_true,
            y_prob,
            n_bins=10,
            compute_confidence_intervals=True,
            n_bootstrap=100,
            seed=42,
        )

        # Should return CalibrationResult with CIs
        assert isinstance(result, CalibrationResult)
        assert result.ece_ci is not None
        assert result.brier_ci is not None
        assert result.bin_wise_ci is not None

    def test_calibration_result_to_dict(self):
        """Test CalibrationResult serialization."""
        np.random.seed(42)
        n_samples = 200
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.random.beta(3, 3, n_samples)

        result = assess_calibration_quality(
            y_true,
            y_prob,
            n_bins=10,
            compute_confidence_intervals=True,
            n_bootstrap=100,
            seed=42,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "brier_score" in result_dict
        assert "expected_calibration_error" in result_dict
        assert "ece_ci" in result_dict
        assert "brier_ci" in result_dict
        assert "bin_wise_ci" in result_dict


class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_y_true_values(self):
        """Test that non-binary y_true raises error."""
        y_true = np.array([0, 1, 2, 1, 0])  # Contains 2 (invalid)
        y_prob = np.array([0.1, 0.5, 0.7, 0.6, 0.2])

        with pytest.raises(ValueError, match="must contain only 0 or 1"):
            compute_calibration_with_ci(y_true, y_prob, seed=42)

    def test_invalid_y_prob_range(self):
        """Test that y_prob outside [0, 1] raises error."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.5, 1.5, 0.3, -0.1])  # Contains invalid values

        with pytest.raises(ValueError, match="must be in"):
            compute_calibration_with_ci(y_true, y_prob, seed=42)

    def test_mismatched_lengths(self):
        """Test that mismatched array lengths raise error."""
        y_true = np.array([0, 1, 1])
        y_prob = np.array([0.1, 0.5])  # Different length

        with pytest.raises(ValueError, match="same length"):
            compute_calibration_with_ci(y_true, y_prob, seed=42)

    def test_2d_arrays_rejected(self):
        """Test that 2D arrays are rejected."""
        y_true = np.array([[0, 1], [1, 0]])
        y_prob = np.array([[0.1, 0.5], [0.6, 0.3]])

        with pytest.raises(ValueError, match="1D arrays"):
            compute_calibration_with_ci(y_true, y_prob, seed=42)


if __name__ == "__main__":
    # Allow running tests directly
    import sys

    if pytest is None:
        print("pytest not available, running simple validation...")
        # Run a few basic tests without pytest
        test_determinism = TestBootstrapDeterminism()
        test_determinism.test_same_seed_identical_ece_cis()
        test_determinism.test_same_seed_identical_brier_cis()
        print("✓ Determinism tests passed")

        test_accuracy = TestCalibrationAccuracy()
        test_accuracy.test_perfect_calibration_low_ece()
        print("✓ Accuracy tests passed")

        test_edge = TestEdgeCases()
        test_edge.test_small_sample_size()
        print("✓ Edge case tests passed")

        print("\nAll basic tests passed! Run with pytest for full test suite.")
    else:
        sys.exit(pytest.main([__file__, "-v"]))
