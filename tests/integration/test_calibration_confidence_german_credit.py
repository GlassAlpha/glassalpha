"""Integration test for calibration confidence intervals with German Credit dataset.

This test validates E10+ implementation with a real-world dataset:
1. Train LogisticRegression on German Credit
2. Compute calibration with bootstrap CIs
3. Validate CI widths are reasonable
4. Test determinism (same seed = same CIs)
5. Export to JSON and verify structure
"""

import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:
    pytest = None

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from glassalpha.metrics.calibration.quality import assess_calibration_quality


class TestGermanCreditCalibrationCI:
    """Integration tests with German Credit dataset."""

    @pytest.fixture
    def german_credit_predictions(self):
        """Load German Credit, train model, return predictions."""
        # Load German Credit dataset
        from glassalpha.datasets import load_german_credit

        data = load_german_credit()

        # Prepare features and target
        target = "credit_risk"
        feature_cols = [col for col in data.columns if col != target]

        X = data[feature_cols]
        y = data[target]

        # Convert categorical variables to numeric
        X = pd.get_dummies(X, drop_first=True)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )

        # Train LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=2000, solver="liblinear")
        model.fit(X_train, y_train)

        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        return y_test.values, y_pred_proba

    def test_german_credit_calibration_with_ci(self, german_credit_predictions):
        """Test calibration CI computation on German Credit."""
        y_true, y_proba = german_credit_predictions

        # Compute calibration with CIs
        result = assess_calibration_quality(
            y_true,
            y_proba,
            n_bins=10,
            compute_confidence_intervals=True,
            n_bootstrap=1000,
            seed=42,
        )

        # Validate result structure
        assert result.n_samples == len(y_true)
        assert result.n_bins == 10

        # Validate ECE CI
        assert result.ece_ci is not None
        assert 0 <= result.expected_calibration_error <= 1
        assert result.ece_ci["ci_lower"] <= result.expected_calibration_error
        assert result.expected_calibration_error <= result.ece_ci["ci_upper"]

        # ECE CI width should be reasonable (< 0.1 for n~300)
        ece_width = result.ece_ci["ci_upper"] - result.ece_ci["ci_lower"]
        assert ece_width < 0.1, f"ECE CI width too large: {ece_width}"

        # Validate Brier score CI
        assert result.brier_ci is not None
        assert 0 <= result.brier_score <= 1
        assert result.brier_ci["ci_lower"] <= result.brier_score
        assert result.brier_score <= result.brier_ci["ci_upper"]

        # Brier CI width should be reasonable (< 0.07 for n~300)
        brier_width = result.brier_ci["ci_upper"] - result.brier_ci["ci_lower"]
        assert brier_width < 0.07, f"Brier CI width too large: {brier_width}"

        # Validate bin-wise CIs
        assert result.bin_wise_ci is not None
        assert len(result.bin_wise_ci["bins"]) == 10

        # Most bins should have valid CIs (not all NaN)
        non_nan_cis = sum(1 for lower in result.bin_wise_ci["ci_lower"] if not np.isnan(lower))
        assert non_nan_cis >= 5, "At least half of bins should have valid CIs"

    def test_german_credit_determinism(self, german_credit_predictions):
        """Test that same seed produces identical CIs."""
        y_true, y_proba = german_credit_predictions

        # Run twice with same seed
        result1 = assess_calibration_quality(
            y_true,
            y_proba,
            n_bins=10,
            compute_confidence_intervals=True,
            n_bootstrap=100,
            seed=42,
        )
        result2 = assess_calibration_quality(
            y_true,
            y_proba,
            n_bins=10,
            compute_confidence_intervals=True,
            n_bootstrap=100,
            seed=42,
        )

        # Assert byte-identical results
        assert result1.expected_calibration_error == result2.expected_calibration_error
        assert result1.brier_score == result2.brier_score

        assert result1.ece_ci["ci_lower"] == result2.ece_ci["ci_lower"]
        assert result1.ece_ci["ci_upper"] == result2.ece_ci["ci_upper"]

        assert result1.brier_ci["ci_lower"] == result2.brier_ci["ci_lower"]
        assert result1.brier_ci["ci_upper"] == result2.brier_ci["ci_upper"]

        # Check bin-wise CIs
        for i in range(len(result1.bin_wise_ci["ci_lower"])):
            lower1 = result1.bin_wise_ci["ci_lower"][i]
            lower2 = result2.bin_wise_ci["ci_lower"][i]

            # Both NaN or both equal
            if np.isnan(lower1):
                assert np.isnan(lower2)
            else:
                assert lower1 == lower2

    def test_german_credit_json_export(self, german_credit_predictions):
        """Test that results export cleanly to JSON."""
        y_true, y_proba = german_credit_predictions

        result = assess_calibration_quality(
            y_true,
            y_proba,
            n_bins=10,
            compute_confidence_intervals=True,
            n_bootstrap=100,
            seed=42,
        )

        # Export to dict (JSON-serializable)
        result_dict = result.to_dict()

        # Validate structure
        assert isinstance(result_dict, dict)
        assert "brier_score" in result_dict
        assert "expected_calibration_error" in result_dict
        assert "ece_ci" in result_dict
        assert "brier_ci" in result_dict
        assert "bin_wise_ci" in result_dict

        # Validate ECE CI structure
        ece_ci = result_dict["ece_ci"]
        assert "ci_lower" in ece_ci
        assert "ci_upper" in ece_ci
        assert "point_estimate" in ece_ci
        assert "ci_width" in ece_ci

        # All values should be JSON-serializable (float or dict)
        import json

        json_str = json.dumps(result_dict)  # Should not raise
        assert len(json_str) > 0

    def test_german_credit_ci_coverage(self, german_credit_predictions):
        """Test that confidence intervals have proper coverage properties."""
        y_true, y_proba = german_credit_predictions

        # Compute with 90% confidence level
        result_90 = assess_calibration_quality(
            y_true,
            y_proba,
            n_bins=10,
            compute_confidence_intervals=True,
            n_bootstrap=500,
            confidence_level=0.90,
            seed=42,
        )

        # Compute with 95% confidence level
        result_95 = assess_calibration_quality(
            y_true,
            y_proba,
            n_bins=10,
            compute_confidence_intervals=True,
            n_bootstrap=500,
            confidence_level=0.95,
            seed=42,
        )

        # 95% CI should be wider than 90% CI
        width_90_ece = result_90.ece_ci["ci_width"]
        width_95_ece = result_95.ece_ci["ci_width"]
        assert width_95_ece > width_90_ece, "95% CI should be wider than 90% CI"

        width_90_brier = result_90.brier_ci["ci_width"]
        width_95_brier = result_95.brier_ci["ci_width"]
        assert width_95_brier > width_90_brier, "95% Brier CI should be wider than 90% CI"

    def test_german_credit_backward_compatibility(self, german_credit_predictions):
        """Test that legacy API still works (without CIs)."""
        y_true, y_proba = german_credit_predictions

        # Legacy call (no CI computation)
        result = assess_calibration_quality(y_true, y_proba, n_bins=10)

        # Should return dict with basic metrics
        assert isinstance(result, dict)
        assert "brier_score" in result
        assert "expected_calibration_error" in result
        assert "maximum_calibration_error" in result
        assert "ece_ci" not in result  # No CIs in legacy mode


if __name__ == "__main__":
    if pytest is None:
        print("pytest not available - install it to run integration tests")
    else:
        import sys

        sys.exit(pytest.main([__file__, "-v"]))
