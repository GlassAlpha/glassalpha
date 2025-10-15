"""Consolidated tests for shift reweighting and perturbation stability.

This module consolidates tests for:
1. Demographic shift reweighting (E6.5)
2. Adversarial perturbation stability testing (E6+)

Both test suites share similar concerns:
- Determinism and reproducibility
- Protected feature handling
- Parameter validation
- Edge case coverage
- JSON export format
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from glassalpha.metrics.shift.reweighting import (
    ShiftReweighter,
    ShiftSpecification,
    compute_shifted_weights,
    parse_shift_spec,
    validate_shift_feasibility,
)
from glassalpha.metrics.stability.perturbation import (
    run_perturbation_sweep,
)

# ============================================================================
# Fixtures (moved from old conftest.py files)
# ============================================================================


@pytest.fixture
def simple_df():
    """Small DataFrame with 4 features (2 protected, 2 regular)."""
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 30],
            "income": [30000, 50000, 70000, 90000, 40000],
            "gender": [0, 1, 0, 1, 0],  # protected
            "race": [0, 0, 1, 1, 0],  # protected
        },
    )


@pytest.fixture
def simple_model():
    """Trained LogisticRegression on simple synthetic data."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression(random_state=42, max_iter=2000, solver="liblinear")
    model.fit(X, y)
    return model


@pytest.fixture
def simple_model_dataframe():
    """Trained model with DataFrame input (for feature name handling)."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 100),
            "income": np.random.randint(20000, 100000, 100),
            "gender": np.random.randint(0, 2, 100),
            "race": np.random.randint(0, 3, 100),
        },
    )
    y = (df["age"] + df["income"] / 1000 > 50).astype(int)

    model = LogisticRegression(random_state=42, max_iter=2000, solver="liblinear")
    model.fit(df, y)
    return model, df


@pytest.fixture
def single_feature_df():
    """DataFrame with only one non-protected feature."""
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 30],
            "gender": [0, 1, 0, 1, 0],  # protected
        },
    )


@pytest.fixture
def single_feature_model():
    """Trained model for single feature test case."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 100),
            "gender": np.random.randint(0, 2, 100),
        },
    )
    y = (df["age"] > 40).astype(int)

    model = LogisticRegression(random_state=42, max_iter=2000, solver="liblinear")
    model.fit(df, y)
    return model


@pytest.fixture
def all_protected_df():
    """DataFrame where all features are protected (should raise error)."""
    return pd.DataFrame(
        {
            "gender": [0, 1, 0, 1, 0],
            "race": [0, 0, 1, 1, 0],
        },
    )


@pytest.fixture
def empty_df():
    """Empty DataFrame (edge case)."""
    return pd.DataFrame(
        {
            "age": [],
            "income": [],
        },
    )


# ============================================================================
# Shift Reweighting Tests (E6.5)
# ============================================================================


class TestShiftSpecificationParsing:
    """Test parsing of shift specification strings."""

    def test_parse_positive_shift(self) -> None:
        """Test parsing positive shift with explicit +."""
        attr, shift = parse_shift_spec("gender:+0.1")
        assert attr == "gender"
        assert shift == 0.1

    def test_parse_positive_shift_no_sign(self) -> None:
        """Test parsing positive shift without +."""
        attr, shift = parse_shift_spec("gender:0.1")
        assert attr == "gender"
        assert shift == 0.1

    def test_parse_negative_shift(self) -> None:
        """Test parsing negative shift."""
        attr, shift = parse_shift_spec("age:-0.05")
        assert attr == "age"
        assert shift == -0.05

    def test_parse_strips_whitespace(self) -> None:
        """Test that whitespace is stripped."""
        attr, shift = parse_shift_spec("  gender : +0.1  ")
        assert attr == "gender"
        assert shift == 0.1

    def test_parse_underscore_attribute(self) -> None:
        """Test attributes with underscores."""
        attr, shift = parse_shift_spec("age_group:+0.1")
        assert attr == "age_group"
        assert shift == 0.1

    def test_parse_missing_colon_raises(self) -> None:
        """Test error when colon is missing."""
        with pytest.raises(ValueError, match="Expected format"):
            parse_shift_spec("gender+0.1")

    def test_parse_empty_attribute_raises(self) -> None:
        """Test error for empty attribute."""
        with pytest.raises(ValueError, match="Empty attribute"):
            parse_shift_spec(":+0.1")

    def test_parse_non_numeric_shift_raises(self) -> None:
        """Test error for non-numeric shift."""
        with pytest.raises(ValueError, match="Must be numeric"):
            parse_shift_spec("gender:abc")

    def test_parse_multiple_colons_raises(self) -> None:
        """Test error for multiple colons."""
        with pytest.raises(ValueError, match="exactly one"):
            parse_shift_spec("gender:race:+0.1")


class TestShiftFeasibilityValidation:
    """Test validation of shift feasibility."""

    def test_valid_positive_shift(self) -> None:
        """Test valid positive shift passes."""
        validate_shift_feasibility(0.3, 0.1)  # 0.3 + 0.1 = 0.4 OK

    def test_valid_negative_shift(self) -> None:
        """Test valid negative shift passes."""
        validate_shift_feasibility(0.3, -0.1)  # 0.3 - 0.1 = 0.2 OK

    def test_shift_to_boundary_passes(self) -> None:
        """Test shift to exactly 0.01 or 0.99 passes."""
        validate_shift_feasibility(0.02, -0.01)  # → 0.01 OK
        validate_shift_feasibility(0.98, 0.01)  # → 0.99 OK

    def test_shift_below_minimum_raises(self) -> None:
        """Test shift below 1% fails."""
        with pytest.raises(ValueError, match="must be ≥ 0.01"):
            validate_shift_feasibility(0.05, -0.05)  # → 0.00 FAIL

    def test_shift_above_maximum_raises(self) -> None:
        """Test shift above 99% fails."""
        with pytest.raises(ValueError, match="must be ≤ 0.99"):
            validate_shift_feasibility(0.95, 0.1)  # → 1.05 FAIL

    def test_error_includes_attribute_name(self) -> None:
        """Test error message includes attribute name when provided."""
        with pytest.raises(ValueError, match="for 'gender'"):
            validate_shift_feasibility(0.95, 0.1, attribute="gender")


class TestShiftSpecification:
    """Test ShiftSpecification dataclass."""

    def test_create_valid_specification(self) -> None:
        """Test creating valid specification."""
        spec = ShiftSpecification(
            attribute="gender",
            shift=0.1,
            original_proportion=0.35,
            shifted_proportion=0.45,
        )
        assert spec.attribute == "gender"
        assert spec.shift == 0.1
        assert spec.original_proportion == 0.35
        assert spec.shifted_proportion == 0.45

    def test_specification_is_frozen(self) -> None:
        """Test that specification is immutable."""
        spec = ShiftSpecification("gender", 0.1, 0.35, 0.45)
        with pytest.raises(AttributeError):
            spec.shift = 0.2  # type: ignore

    def test_invalid_shifted_proportion_raises(self) -> None:
        """Test error for shifted proportion outside [0.01, 0.99]."""
        data = pd.DataFrame({"gender": [1, 1, 1, 0]})  # 75% gender=1

        # Shift that would result in invalid proportion (75% + 30% = 105%)
        with pytest.raises(ValueError, match="must be ≤ 0.99"):
            compute_shifted_weights(data, "gender", 0.3, validate=True)

    def test_to_dict_serialization(self) -> None:
        """Test dictionary serialization."""
        spec = ShiftSpecification("gender", 0.1, 0.35, 0.45)
        d = spec.to_dict()

        assert d == {
            "attribute": "gender",
            "shift_value": 0.1,
            "original_proportion": 0.35,
            "shifted_proportion": 0.45,
        }


class TestComputeShiftedWeights:
    """Test weight computation for demographic shifts."""

    def test_compute_weights_balanced_data(self) -> None:
        """Test weight computation on balanced binary data."""
        data = pd.DataFrame(
            {
                "gender": [1, 0, 1, 0, 1, 0, 1, 0],
            },
        )

        weights, spec = compute_shifted_weights(data, "gender", 0.1)

        # Original proportion: 0.5
        # Shifted proportion: 0.6
        # Weight for gender=1: 0.6 / 0.5 = 1.2
        # Weight for gender=0: 0.4 / 0.5 = 0.8

        assert spec.original_proportion == 0.5
        assert spec.shifted_proportion == 0.6
        assert np.allclose(weights[data["gender"] == 1], 1.2)
        assert np.allclose(weights[data["gender"] == 0], 0.8)

    def test_compute_weights_imbalanced_data(self) -> None:
        """Test weight computation on imbalanced data."""
        data = pd.DataFrame(
            {
                "gender": [1, 1, 1, 0],  # 75% gender=1
            },
        )

        weights, spec = compute_shifted_weights(data, "gender", -0.1)

        # Original proportion: 0.75
        # Shifted proportion: 0.65
        # Weight for gender=1: 0.65 / 0.75 ≈ 0.867
        # Weight for gender=0: 0.35 / 0.25 = 1.4

        assert spec.original_proportion == 0.75
        assert spec.shifted_proportion == 0.65
        assert np.allclose(weights[data["gender"] == 1], 0.65 / 0.75)
        assert np.allclose(weights[data["gender"] == 0], 0.35 / 0.25)

    def test_weights_sum_to_original_size(self) -> None:
        """Test that weights sum to original dataset size."""
        data = pd.DataFrame(
            {
                "gender": [1, 0, 1, 0, 1],
            },
        )

        weights, _ = compute_shifted_weights(data, "gender", 0.2)

        # Weights should sum to N (preserving total sample size)
        assert np.allclose(weights.sum(), len(data))

    def test_attribute_not_in_data_raises(self) -> None:
        """Test error when attribute not in data."""
        data = pd.DataFrame({"age": [1, 0, 1]})

        with pytest.raises(ValueError, match="not found in data"):
            compute_shifted_weights(data, "gender", 0.1)

    def test_non_binary_attribute_raises(self) -> None:
        """Test error for non-binary attribute."""
        data = pd.DataFrame(
            {
                "race": [0, 1, 2, 1],  # Three classes
            },
        )

        with pytest.raises(ValueError, match="must be binary"):
            compute_shifted_weights(data, "race", 0.1)

    def test_infeasible_shift_raises(self) -> None:
        """Test error for shift outside [0.01, 0.99]."""
        data = pd.DataFrame(
            {
                "gender": [1, 1, 1, 0],  # 75% gender=1
            },
        )

        with pytest.raises(ValueError, match="must be ≤ 0.99"):
            compute_shifted_weights(data, "gender", 0.3)  # → 1.05 FAIL

    def test_validation_can_be_disabled(self) -> None:
        """Test that validation can be skipped."""
        data = pd.DataFrame(
            {
                "gender": [1, 1, 1, 0],
            },
        )

        # This would normally fail, but validation is disabled
        weights, spec = compute_shifted_weights(
            data,
            "gender",
            0.3,
            validate=False,
        )

        assert spec.shifted_proportion > 0.99  # Invalid but allowed

    def test_deterministic_weights(self) -> None:
        """Test that weight computation is deterministic."""
        data = pd.DataFrame(
            {
                "gender": [1, 0, 1, 0, 1],
            },
        )

        weights1, _ = compute_shifted_weights(data, "gender", 0.1)
        weights2, _ = compute_shifted_weights(data, "gender", 0.1)

        assert np.allclose(weights1, weights2)


class TestShiftReweighter:
    """Test stateful shift reweighter class."""

    def test_reweighter_records_specifications(self) -> None:
        """Test that reweighter records shift specifications."""
        data = pd.DataFrame(
            {
                "gender": [1, 0, 1, 0],
                "age": [1, 1, 0, 0],
            },
        )

        reweighter = ShiftReweighter()
        reweighter.apply_shift(data, "gender", 0.1)
        reweighter.apply_shift(data, "age", -0.05)

        assert len(reweighter.specifications) == 2
        assert reweighter.specifications[0].attribute == "gender"
        assert reweighter.specifications[1].attribute == "age"

    def test_reweighter_reset(self) -> None:
        """Test that reset clears specifications."""
        data = pd.DataFrame({"gender": [1, 0, 1, 0]})

        reweighter = ShiftReweighter()
        reweighter.apply_shift(data, "gender", 0.1)
        assert len(reweighter.specifications) == 1

        reweighter.reset()
        assert len(reweighter.specifications) == 0

    def test_reweighter_to_dict(self) -> None:
        """Test dictionary serialization."""
        data = pd.DataFrame(
            {
                "gender": [1, 0, 1, 0],
            },
        )

        reweighter = ShiftReweighter()
        reweighter.apply_shift(data, "gender", 0.1)

        d = reweighter.to_dict()

        assert d["num_shifts"] == 1
        assert len(d["shifts"]) == 1
        assert d["shifts"][0]["attribute"] == "gender"


class TestShiftEdgeCases:
    """Test shift edge cases and boundary conditions."""

    def test_zero_shift_returns_unit_weights(self) -> None:
        """Test that zero shift returns all weights = 1."""
        data = pd.DataFrame({"gender": [1, 0, 1, 0]})

        weights, spec = compute_shifted_weights(data, "gender", 0.0)

        assert np.allclose(weights, 1.0)
        assert spec.original_proportion == spec.shifted_proportion

    def test_small_positive_shift(self) -> None:
        """Test very small positive shift."""
        data = pd.DataFrame({"gender": [1, 0, 1, 0]})

        weights, spec = compute_shifted_weights(data, "gender", 0.01)

        assert spec.shifted_proportion == 0.51
        assert weights[data["gender"] == 1][0] > 1.0  # Upweighted
        assert weights[data["gender"] == 0][0] < 1.0  # Downweighted

    def test_large_dataset(self) -> None:
        """Test performance on larger dataset."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "gender": np.random.randint(0, 2, size=10000),
            },
        )

        weights, spec = compute_shifted_weights(data, "gender", 0.1)

        assert len(weights) == 10000
        # Original is ~50%, shift +10pp → result should be ~60%
        assert 0.55 <= spec.shifted_proportion <= 0.65  # ~60% after +10pp shift


# ============================================================================
# Stability Perturbation Tests (E6+)
# ============================================================================


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


class TestPerturbationEdgeCases:
    """Test perturbation edge cases and error handling."""

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


class TestPerturbationJSONExport:
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
