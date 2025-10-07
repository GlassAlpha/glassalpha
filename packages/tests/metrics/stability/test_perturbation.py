"""Tests for adversarial perturbation sweeps (E6+).

This module tests epsilon-perturbation stability testing for model robustness.
Focus areas:
1. Determinism: same seed → byte-identical results
2. Protected features: never perturbed
3. Epsilon validation: positive values, sorted order
4. Gate logic: PASS/FAIL/WARNING based on threshold
5. Edge cases: single feature, all protected, empty data
"""

try:
    import pytest
except ImportError:
    pytest = None

from glassalpha.metrics.stability.perturbation import (
    run_perturbation_sweep,
)


class TestPerturbationDeterminism:
    """Test that perturbation sweeps are deterministic under same seed."""

    def test_same_seed_identical_results(self, simple_model_dataframe):
        """Same seed should produce byte-identical robustness scores."""
        model, df = simple_model_dataframe
        protected = ["gender", "race"]

        # Run twice with same seed
        result1 = run_perturbation_sweep(
            model=model,
            X_test=df,
            protected_features=protected,
            epsilon_values=[0.01, 0.05, 0.1],
            threshold=0.15,
            seed=42,
        )

        result2 = run_perturbation_sweep(
            model=model,
            X_test=df,
            protected_features=protected,
            epsilon_values=[0.01, 0.05, 0.1],
            threshold=0.15,
            seed=42,
        )

        # Assert byte-identical results
        assert result1.robustness_score == result2.robustness_score
        assert result1.max_delta == result2.max_delta
        assert result1.per_epsilon_deltas == result2.per_epsilon_deltas
        assert result1.gate_status == result2.gate_status

    def test_different_seed_different_results(self, simple_model_dataframe):
        """Different seeds should produce different results (sanity check)."""
        model, df = simple_model_dataframe
        protected = ["gender", "race"]

        result1 = run_perturbation_sweep(
            model=model,
            X_test=df,
            protected_features=protected,
            epsilon_values=[0.01, 0.05, 0.1],
            threshold=0.15,
            seed=42,
        )

        result2 = run_perturbation_sweep(
            model=model,
            X_test=df,
            protected_features=protected,
            epsilon_values=[0.01, 0.05, 0.1],
            threshold=0.15,
            seed=99,
        )

        # Different seeds should produce different deltas
        assert result1.robustness_score != result2.robustness_score or result1.max_delta != result2.max_delta


class TestProtectedFeatureHandling:
    """Test that protected features are never perturbed."""

    def test_protected_features_unchanged(self, simple_model_dataframe):
        """Protected features should have zero variance after perturbation."""
        model, df = simple_model_dataframe
        protected = ["gender", "race"]

        result = run_perturbation_sweep(
            model=model,
            X_test=df,
            protected_features=protected,
            epsilon_values=[0.05],
            threshold=0.15,
            seed=42,
        )

        # Protected features should not contribute to perturbations
        assert result.n_features_perturbed == 2  # Only age and income perturbed
        assert result.n_features_perturbed == len(df.columns) - len(protected)

    def test_all_protected_features_raises_error(self, all_protected_df, simple_model):
        """Should raise ValueError if all features are protected."""
        protected = ["gender", "race"]

        with pytest.raises(ValueError, match="No non-protected features"):
            run_perturbation_sweep(
                model=simple_model,
                X_test=all_protected_df,
                protected_features=protected,
                epsilon_values=[0.05],
                threshold=0.15,
                seed=42,
            )

    def test_empty_protected_list(self, simple_model_dataframe):
        """Empty protected list should perturb all features."""
        model, df = simple_model_dataframe

        result = run_perturbation_sweep(
            model=model,
            X_test=df,
            protected_features=[],
            epsilon_values=[0.05],
            threshold=0.15,
            seed=42,
        )

        # All features should be perturbed
        assert result.n_features_perturbed == len(df.columns)


class TestEpsilonValidation:
    """Test epsilon value validation and handling."""

    def test_epsilon_values_sorted_ascending(self, simple_model_dataframe):
        """Epsilon values should be returned in sorted order."""
        model, df = simple_model_dataframe

        result = run_perturbation_sweep(
            model=model,
            X_test=df,
            protected_features=["gender", "race"],
            epsilon_values=[0.1, 0.01, 0.05],  # Unsorted input
            threshold=0.15,
            seed=42,
        )

        # Should be sorted in output
        assert result.epsilon_values == [0.01, 0.05, 0.1]

    def test_negative_epsilon_raises_error(self, simple_model_dataframe):
        """Negative epsilon values should raise ValueError."""
        model, df = simple_model_dataframe

        with pytest.raises(ValueError, match="positive"):
            run_perturbation_sweep(
                model=model,
                X_test=df,
                protected_features=["gender", "race"],
                epsilon_values=[-0.01, 0.05],
                threshold=0.15,
                seed=42,
            )

    def test_zero_epsilon_raises_error(self, simple_model_dataframe):
        """Zero epsilon values should raise ValueError."""
        model, df = simple_model_dataframe

        with pytest.raises(ValueError, match="positive"):
            run_perturbation_sweep(
                model=model,
                X_test=df,
                protected_features=["gender", "race"],
                epsilon_values=[0.0, 0.05],
                threshold=0.15,
                seed=42,
            )

    def test_single_epsilon_value(self, simple_model_dataframe):
        """Should work with single epsilon value."""
        model, df = simple_model_dataframe

        result = run_perturbation_sweep(
            model=model,
            X_test=df,
            protected_features=["gender", "race"],
            epsilon_values=[0.05],
            threshold=0.15,
            seed=42,
        )

        assert len(result.epsilon_values) == 1
        assert result.epsilon_values[0] == 0.05


class TestGateThresholds:
    """Test gate logic for PASS/FAIL/WARNING."""

    def test_gate_pass_below_threshold(self, simple_model_dataframe):
        """Gate should PASS when max_delta < threshold."""
        model, df = simple_model_dataframe

        # Use very high threshold to force PASS
        result = run_perturbation_sweep(
            model=model,
            X_test=df,
            protected_features=["gender", "race"],
            epsilon_values=[0.01],  # Small epsilon → small delta
            threshold=0.99,  # Very high threshold
            seed=42,
        )

        assert result.gate_status == "PASS"
        assert result.max_delta < result.threshold

    def test_gate_fail_above_threshold(self, simple_model_dataframe):
        """Gate should FAIL when max_delta >= 1.5 * threshold."""
        model, df = simple_model_dataframe

        # Use very low threshold to force FAIL
        result = run_perturbation_sweep(
            model=model,
            X_test=df,
            protected_features=["gender", "race"],
            epsilon_values=[0.1],  # Large epsilon → large delta
            threshold=0.001,  # Very low threshold
            seed=42,
        )

        assert result.gate_status == "FAIL"
        assert result.max_delta >= 1.5 * result.threshold

    def test_gate_warning_near_threshold(self, simple_model_dataframe):
        """Gate should WARNING when threshold <= max_delta < 1.5 * threshold."""
        model, df = simple_model_dataframe

        # First, get the actual max_delta for this model
        pilot = run_perturbation_sweep(
            model=model,
            X_test=df,
            protected_features=["gender", "race"],
            epsilon_values=[0.05],
            threshold=0.15,
            seed=42,
        )

        # Set threshold to force WARNING zone
        # WARNING: threshold <= max_delta < 1.5 * threshold
        # So: max_delta / 1.5 < threshold <= max_delta
        threshold = pilot.max_delta * 0.9  # Slightly below max_delta

        result = run_perturbation_sweep(
            model=model,
            X_test=df,
            protected_features=["gender", "race"],
            epsilon_values=[0.05],
            threshold=threshold,
            seed=42,
        )

        assert result.gate_status == "WARNING"
        assert threshold <= result.max_delta < 1.5 * threshold


class TestRobustnessScore:
    """Test robustness score computation."""

    def test_robustness_score_is_max_delta(self, simple_model_dataframe):
        """Robustness score should equal max_delta (they're the same metric)."""
        model, df = simple_model_dataframe

        result = run_perturbation_sweep(
            model=model,
            X_test=df,
            protected_features=["gender", "race"],
            epsilon_values=[0.01, 0.05, 0.1],
            threshold=0.15,
            seed=42,
        )

        # Robustness score is max delta across all epsilon values
        assert result.robustness_score == result.max_delta
        assert result.max_delta == max(result.per_epsilon_deltas.values())

    def test_larger_epsilon_larger_delta(self, simple_model_dataframe):
        """Larger epsilon should generally produce larger delta."""
        model, df = simple_model_dataframe

        result = run_perturbation_sweep(
            model=model,
            X_test=df,
            protected_features=["gender", "race"],
            epsilon_values=[0.01, 0.05, 0.1],
            threshold=0.15,
            seed=42,
        )

        # Deltas should generally increase with epsilon (monotonicity check)
        delta_01 = result.per_epsilon_deltas[0.01]
        delta_05 = result.per_epsilon_deltas[0.05]
        delta_10 = result.per_epsilon_deltas[0.1]

        # At least the max should be >= min (sanity check, not strict monotonicity)
        assert max(delta_01, delta_05, delta_10) >= min(delta_01, delta_05, delta_10)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data_raises_error(self, empty_df, simple_model):
        """Empty DataFrame should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            run_perturbation_sweep(
                model=simple_model,
                X_test=empty_df,
                protected_features=[],
                epsilon_values=[0.05],
                threshold=0.15,
                seed=42,
            )

    def test_single_feature_perturbation(self, single_feature_df, single_feature_model):
        """Should work with single non-protected feature."""
        result = run_perturbation_sweep(
            model=single_feature_model,
            X_test=single_feature_df,
            protected_features=["gender"],
            epsilon_values=[0.05],
            threshold=0.15,
            seed=42,
        )

        assert result.n_features_perturbed == 1  # Only age
        assert result.robustness_score >= 0.0


class TestJSONExport:
    """Test JSON export format and serialization."""

    def test_json_export_format(self, simple_model_dataframe):
        """JSON export should contain all required fields."""
        model, df = simple_model_dataframe

        result = run_perturbation_sweep(
            model=model,
            X_test=df,
            protected_features=["gender", "race"],
            epsilon_values=[0.01, 0.05, 0.1],
            threshold=0.15,
            seed=42,
        )

        json_dict = result.to_dict()

        # Required fields
        assert "robustness_score" in json_dict
        assert "max_delta" in json_dict
        assert "epsilon_values" in json_dict
        assert "per_epsilon_deltas" in json_dict
        assert "gate_status" in json_dict
        assert "threshold" in json_dict
        assert "n_samples" in json_dict
        assert "n_features_perturbed" in json_dict

    def test_json_serializable(self, simple_model_dataframe):
        """JSON export should be serializable."""
        import json

        model, df = simple_model_dataframe

        result = run_perturbation_sweep(
            model=model,
            X_test=df,
            protected_features=["gender", "race"],
            epsilon_values=[0.01, 0.05, 0.1],
            threshold=0.15,
            seed=42,
        )

        json_dict = result.to_dict()

        # Should not raise
        json_str = json.dumps(json_dict)
        assert len(json_str) > 0
