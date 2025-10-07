"""Calibration metrics with bootstrap confidence intervals.

This module provides calibration quality assessment with statistical confidence:
1. ECE (Expected Calibration Error) with bootstrap CIs
2. Brier score with bootstrap CIs
3. Bin-wise confidence intervals for calibration curves
4. Deterministic bootstrap (seeded for reproducibility)

Extension of E10 (Statistical Confidence) for calibration metrics.
"""

from __future__ import annotations

from glassalpha.metrics.calibration.confidence import (
    BinWiseCI,
    CalibrationCI,
    compute_bin_wise_ci,
    compute_calibration_with_ci,
)
from glassalpha.metrics.calibration.quality import (
    CalibrationResult,
    assess_calibration_quality,
)

__all__ = [
    "BinWiseCI",
    "CalibrationCI",
    "CalibrationResult",
    "assess_calibration_quality",
    "compute_bin_wise_ci",
    "compute_calibration_with_ci",
]
