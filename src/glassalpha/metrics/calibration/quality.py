"""Calibration quality assessment with confidence intervals.

This module extends the existing models/calibration.py functionality
by adding bootstrap confidence intervals to calibration metrics.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss

from glassalpha.metrics.calibration.confidence import compute_calibration_with_ci

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Container for calibration quality assessment results with CIs."""

    brier_score: float
    expected_calibration_error: float
    maximum_calibration_error: float
    n_samples: int
    n_bins: int

    # Optional CI fields (populated when compute_confidence_intervals=True)
    ece_ci: Any = None
    brier_ci: Any = None
    bin_wise_ci: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        result = {
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "maximum_calibration_error": self.maximum_calibration_error,
            "n_samples": self.n_samples,
            "n_bins": self.n_bins,
        }

        if self.ece_ci is not None:
            result["ece_ci"] = self.ece_ci

        if self.brier_ci is not None:
            result["brier_ci"] = self.brier_ci

        if self.bin_wise_ci is not None:
            result["bin_wise_ci"] = self.bin_wise_ci

        return result


def assess_calibration_quality(
    y_true: Iterable[int],
    y_prob_pos: Iterable[float],
    *,
    n_bins: int = 10,
    compute_confidence_intervals: bool = False,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> CalibrationResult | dict[str, float]:
    """Compute calibration quality diagnostics for binary classification.

    This function extends the original models/calibration.py implementation
    by adding optional bootstrap confidence intervals.

    Args:
        y_true: Ground truth binary labels
        y_prob_pos: Predicted probabilities for positive class
        n_bins: Number of bins for calibration curve (default 10)
        compute_confidence_intervals: Whether to compute bootstrap CIs (default False)
        n_bootstrap: Number of bootstrap samples (default 1000)
        confidence_level: Confidence level for CIs (default 0.95)
        seed: Random seed for determinism (default 42)

    Returns:
        If compute_confidence_intervals=False: dict with brier_score, expected_calibration_error, maximum_calibration_error
        If compute_confidence_intervals=True: CalibrationResult with CIs

    """
    # Convert to numpy arrays
    y_true_arr = np.asarray(y_true).astype(int)
    prob = np.asarray(y_prob_pos, dtype=float)

    if y_true_arr.ndim != 1 or prob.ndim != 1 or y_true_arr.shape[0] != prob.shape[0]:
        raise ValueError("y_true and y_prob_pos must be 1D arrays of the same length")

    # Compute basic metrics (always computed)
    brier = brier_score_loss(y_true_arr, prob)

    # ECE / MCE with equal-width bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(prob, bins) - 1
    ece: float = 0.0
    mce: float = 0.0
    n = len(prob)

    for b in range(n_bins):
        in_bin = bin_ids == b
        m = np.count_nonzero(in_bin)
        if m == 0:
            continue
        avg_conf = float(np.mean(prob[in_bin]))
        avg_acc = float(np.mean(y_true_arr[in_bin]))
        gap = abs(avg_acc - avg_conf)
        ece = float(ece + (m / n) * gap)
        mce = float(max(mce, gap))

    # Legacy return format (for backward compatibility)
    if not compute_confidence_intervals:
        return {
            "brier_score": float(brier),
            "expected_calibration_error": float(ece),
            "maximum_calibration_error": float(mce),
        }

    # Compute bootstrap CIs
    logger.info(f"Computing calibration CIs with {n_bootstrap} bootstrap samples")
    ci_results = compute_calibration_with_ci(
        y_true_arr,
        prob,
        n_bins=n_bins,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )

    return CalibrationResult(
        brier_score=float(brier),
        expected_calibration_error=float(ece),
        maximum_calibration_error=float(mce),
        n_samples=len(y_true_arr),
        n_bins=n_bins,
        ece_ci=ci_results["ece_ci"].to_dict(),
        brier_ci=ci_results["brier_ci"].to_dict(),
        bin_wise_ci=ci_results["bin_wise_ci"].to_dict(),
    )
