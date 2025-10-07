"""Demographic shift testing for model robustness.

This module provides tools for stress testing ML models under demographic
distribution changes. Key use case: "What if the protected class proportion
increases by 10 percentage points?"

E6.5: Shift Simulator
- Single-factor cohort reweighting
- Before/after metrics comparison
- Degradation gates for CI/CD
- JSON export for programmatic access

Example:
    >>> from glassalpha.metrics.shift import run_shift_analysis
    >>> result = run_shift_analysis(
    ...     model=model,
    ...     X_test=X_test,
    ...     y_test=y_test,
    ...     attribute="gender",
    ...     shift=0.1,  # +10 percentage points
    ... )
    >>> print(result.gate_status)  # "PASS" | "WARNING" | "FAIL"

CLI Usage:
    glassalpha audit --check-shift gender:+0.1 --fail-on-degradation 0.05

"""

from .reweighting import (
    ShiftReweighter,
    ShiftSpecification,
    compute_shifted_weights,
    parse_shift_spec,
    validate_shift_feasibility,
)
from .runner import ShiftAnalysisResult, run_shift_analysis

__all__ = [
    "ShiftAnalysisResult",
    "ShiftReweighter",
    "ShiftSpecification",
    "compute_shifted_weights",
    "parse_shift_spec",
    "run_shift_analysis",
    "validate_shift_feasibility",
]
