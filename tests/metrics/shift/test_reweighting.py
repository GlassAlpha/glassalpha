"""Contract tests for demographic shift reweighting.

Tests cover:
- Shift specification parsing
- Weight computation correctness
- Edge case handling (bounds, binary requirement)
- Determinism and reproducibility
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glassalpha.metrics.shift.reweighting import (
    ShiftReweighter,
    ShiftSpecification,
    compute_shifted_weights,
    parse_shift_spec,
    validate_shift_feasibility,
)


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
        """Test error for shifted proportion outside [0.01, 0.99].

        Note: Validation happens at the entry point (compute_shifted_weights),
        not in ShiftSpecification itself. This allows validate=False to work.
        """
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


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

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
