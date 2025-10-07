"""Core reweighting logic for demographic shift testing.

This module implements post-stratification reweighting to simulate demographic
distribution changes. Shifts are specified as absolute percentage point changes.

Mathematical Foundation:
    Given original proportion p_orig and shift δ:
    p_shifted = p_orig + δ

    Weights are computed via post-stratification:
    w[attr=1] = p_shifted / p_orig
    w[attr=0] = (1 - p_shifted) / (1 - p_orig)

Example:
    >>> spec = parse_shift_spec("gender:+0.1")
    >>> weights = compute_shifted_weights(data, spec.attribute, spec.shift)
    >>> # Use weights in metric computation

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShiftSpecification:
    """Specification for a demographic shift.

    Attributes:
        attribute: Protected attribute name (e.g., "gender", "age_group")
        shift: Signed shift value in percentage points (e.g., 0.1 = +10pp, -0.05 = -5pp)
        original_proportion: Original proportion of attribute=1 in data
        shifted_proportion: Target proportion after shift

    Example:
        >>> spec = ShiftSpecification(
        ...     attribute="gender",
        ...     shift=0.1,
        ...     original_proportion=0.35,
        ...     shifted_proportion=0.45
        ... )

    """

    attribute: str
    shift: float
    original_proportion: float
    shifted_proportion: float

    def __post_init__(self) -> None:
        """Validate shift specification.

        Note: Only validates types and basic constraints.
        Range validation (shifted_proportion in [0.01, 0.99]) is done by
        validate_shift_feasibility() which can be optionally skipped.
        """
        if not isinstance(self.attribute, str) or not self.attribute:
            raise ValueError(f"Attribute must be non-empty string, got: {self.attribute}")

        if not isinstance(self.shift, (int, float)):
            raise ValueError(f"Shift must be numeric, got: {type(self.shift)}")

        if not (0.0 <= self.original_proportion <= 1.0):
            raise ValueError(
                f"Original proportion must be in [0, 1], got: {self.original_proportion}",
            )

        # Note: shifted_proportion range validation removed
        # It's done by validate_shift_feasibility() which can be skipped via validate=False

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary for JSON serialization."""
        return {
            "attribute": self.attribute,
            "shift_value": self.shift,
            "original_proportion": self.original_proportion,
            "shifted_proportion": self.shifted_proportion,
        }


def parse_shift_spec(spec_str: str) -> tuple[str, float]:
    """Parse shift specification string.

    Args:
        spec_str: Shift specification in format "attribute:shift" (e.g., "gender:+0.1")

    Returns:
        Tuple of (attribute_name, shift_value)

    Raises:
        ValueError: If format is invalid or shift is not numeric

    Examples:
        >>> parse_shift_spec("gender:+0.1")
        ('gender', 0.1)
        >>> parse_shift_spec("age:-0.05")
        ('age', -0.05)
        >>> parse_shift_spec("race:0.1")  # + is optional for positive
        ('race', 0.1)

    """
    if ":" not in spec_str:
        raise ValueError(
            f"Invalid shift specification '{spec_str}'. Expected format: 'attribute:shift' (e.g., 'gender:+0.1')",
        )

    # Check for exactly one colon (before splitting)
    if spec_str.count(":") != 1:
        raise ValueError(
            f"Invalid shift specification '{spec_str}'. Expected exactly one ':' separator.",
        )

    parts = spec_str.split(":", 1)

    attribute, shift_str = parts
    attribute = attribute.strip()
    shift_str = shift_str.strip()

    if not attribute:
        raise ValueError(
            f"Empty attribute name in shift specification '{spec_str}'",
        )

    # Parse shift value
    try:
        shift = float(shift_str)
    except ValueError as e:
        raise ValueError(
            f"Invalid shift value '{shift_str}' in specification '{spec_str}'. Must be numeric (e.g., '+0.1', '-0.05')",
        ) from e

    return attribute, shift


def validate_shift_feasibility(
    p_orig: float,
    shift: float,
    attribute: str | None = None,
) -> None:
    """Validate that a shift is feasible.

    Args:
        p_orig: Original proportion of attribute=1
        shift: Shift value in percentage points
        attribute: Optional attribute name for error messages

    Raises:
        ValueError: If shift results in proportion outside [0.01, 0.99]

    Examples:
        >>> validate_shift_feasibility(0.3, 0.1)  # OK: 0.3 + 0.1 = 0.4
        >>> validate_shift_feasibility(0.95, 0.1)  # Error: 0.95 + 0.1 = 1.05 > 0.99
        Traceback (most recent call last):
        ...
        ValueError: Shift +0.10 would result in proportion 1.05 (must be in [0.01, 0.99])

    """
    p_shifted = p_orig + shift

    attr_str = f" for '{attribute}'" if attribute else ""

    if p_shifted < 0.01:
        raise ValueError(
            f"Shift {shift:+.2f}{attr_str} would result in proportion {p_shifted:.3f} "
            f"(must be ≥ 0.01). Cannot shift below 1%.",
        )

    if p_shifted > 0.99:
        raise ValueError(
            f"Shift {shift:+.2f}{attr_str} would result in proportion {p_shifted:.3f} "
            f"(must be ≤ 0.99). Cannot shift above 99%.",
        )


def compute_shifted_weights(
    data: pd.DataFrame,
    attribute: str,
    shift: float,
    *,
    validate: bool = True,
) -> tuple[np.ndarray, ShiftSpecification]:
    """Compute post-stratification weights for demographic shift.

    Args:
        data: DataFrame with protected attributes
        attribute: Column name of binary protected attribute
        shift: Shift value in percentage points (e.g., 0.1 = +10pp)
        validate: Whether to validate shift feasibility (default: True)

    Returns:
        Tuple of (weights array, shift specification)

    Raises:
        ValueError: If attribute not in data, not binary, or shift infeasible

    Examples:
        >>> data = pd.DataFrame({'gender': [1, 0, 1, 0]})
        >>> weights, spec = compute_shifted_weights(data, 'gender', 0.1)
        >>> weights.shape
        (4,)
        >>> spec.original_proportion
        0.5
        >>> spec.shifted_proportion
        0.6

    """
    # Validate attribute exists
    if attribute not in data.columns:
        raise ValueError(
            f"Attribute '{attribute}' not found in data. Available columns: {list(data.columns)}",
        )

    # Extract attribute values
    attr_values = data[attribute].values

    # Validate binary attribute
    unique_values = np.unique(attr_values[~pd.isna(attr_values)])
    if not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values, [1]):
        raise ValueError(
            f"Attribute '{attribute}' must be binary (0/1). "
            f"Found values: {unique_values}. "
            "Multi-class shift testing is not supported (reserved for enterprise).",
        )

    # Compute original proportion
    p_orig = float(np.mean(attr_values == 1))
    p_shifted = p_orig + shift

    # Validate feasibility
    if validate:
        validate_shift_feasibility(p_orig, shift, attribute)

    # Compute post-stratification weights
    weights = np.ones(len(data), dtype=np.float64)

    # Weight for attr=1 group
    if p_orig > 0:
        weights[attr_values == 1] = p_shifted / p_orig
    else:
        # Edge case: no examples of attr=1 in original data
        logger.warning(
            f"No examples of {attribute}=1 in data. Cannot apply positive shift.",
        )

    # Weight for attr=0 group
    if p_orig < 1:
        weights[attr_values == 0] = (1 - p_shifted) / (1 - p_orig)
    else:
        # Edge case: all examples are attr=1
        logger.warning(
            f"All examples are {attribute}=1 in data. Cannot apply negative shift.",
        )

    # Create specification object
    spec = ShiftSpecification(
        attribute=attribute,
        shift=shift,
        original_proportion=p_orig,
        shifted_proportion=p_shifted,
    )

    return weights, spec


class ShiftReweighter:
    """Apply demographic shifts via post-stratification reweighting.

    This class provides a stateful interface for applying multiple shifts
    and tracking shift specifications.

    Example:
        >>> reweighter = ShiftReweighter()
        >>> weights1, spec1 = reweighter.apply_shift(data, "gender", 0.1)
        >>> weights2, spec2 = reweighter.apply_shift(data, "age", -0.05)
        >>> reweighter.specifications
        [ShiftSpecification(...), ShiftSpecification(...)]

    """

    def __init__(self) -> None:
        """Initialize reweighter."""
        self.specifications: list[ShiftSpecification] = []

    def apply_shift(
        self,
        data: pd.DataFrame,
        attribute: str,
        shift: float,
    ) -> tuple[np.ndarray, ShiftSpecification]:
        """Apply a demographic shift and record specification.

        Args:
            data: DataFrame with protected attributes
            attribute: Column name of binary protected attribute
            shift: Shift value in percentage points

        Returns:
            Tuple of (weights array, shift specification)

        """
        weights, spec = compute_shifted_weights(data, attribute, shift)
        self.specifications.append(spec)
        return weights, spec

    def reset(self) -> None:
        """Clear all recorded specifications."""
        self.specifications.clear()

    def to_dict(self) -> dict[str, Any]:
        """Export all specifications to dictionary."""
        return {
            "num_shifts": len(self.specifications),
            "shifts": [spec.to_dict() for spec in self.specifications],
        }
