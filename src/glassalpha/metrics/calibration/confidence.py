"""Bootstrap confidence intervals for calibration metrics.

This module implements E10+: Calibration Confidence Intervals.
Extends E10's bootstrap infrastructure to calibration metrics (ECE, Brier score).

Key features:
1. Deterministic bootstrap (seeded for reproducibility)
2. Bootstrap CIs for ECE and Brier score
3. Bin-wise CIs for calibration curve error bars
4. Reuses E10's bootstrap infrastructure
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationCI:
    """Container for calibration metric confidence interval results."""

    metric_name: str  # "ece" or "brier_score"
    point_estimate: float
    lower: float
    upper: float
    std_error: float
    confidence_level: float
    n_bootstrap: int

    @property
    def width(self) -> float:
        """Compute CI width."""
        return self.upper - self.lower

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "metric": self.metric_name,
            "point_estimate": self.point_estimate,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "std_error": self.std_error,
            "confidence_level": self.confidence_level,
            "ci_width": self.width,
            "n_bootstrap": self.n_bootstrap,
        }


@dataclass
class BinWiseCI:
    """Container for bin-wise confidence intervals on calibration curve."""

    bins: list[tuple[float, float]]  # List of (lower, upper) bin edges
    observed_freq: list[float]  # Observed frequency (accuracy) in each bin
    expected_freq: list[float]  # Expected frequency (mean confidence) in each bin
    ci_lower: list[float]  # Lower bound of CI for observed frequency
    ci_upper: list[float]  # Upper bound of CI for observed frequency
    sample_counts: list[int]  # Number of samples in each bin
    confidence_level: float
    n_bootstrap: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "bins": [[lower, upper] for lower, upper in self.bins],
            "observed_freq": self.observed_freq,
            "expected_freq": self.expected_freq,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "sample_counts": self.sample_counts,
            "confidence_level": self.confidence_level,
            "n_bootstrap": self.n_bootstrap,
        }


def compute_calibration_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute calibration metrics with bootstrap confidence intervals.

    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_prob: Predicted probabilities for positive class
        n_bins: Number of bins for calibration curve (default 10)
        n_bootstrap: Number of bootstrap samples (default 1000)
        confidence_level: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for determinism (default None)

    Returns:
        Dictionary containing:
        - ece: ECE point estimate
        - ece_ci: CalibrationCI for ECE
        - brier_score: Brier score point estimate
        - brier_ci: CalibrationCI for Brier score
        - bin_wise_ci: BinWiseCI for calibration curve
        - n_samples: Total sample count

    """
    # Validate inputs
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if y_true.ndim != 1 or y_prob.ndim != 1 or y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob must be 1D arrays of the same length")

    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must contain only 0 or 1")

    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError("y_prob must be in [0, 1]")

    n_samples = len(y_true)

    # Set seed for deterministic bootstrap
    rng = np.random.RandomState(seed)

    # Compute point estimates on original data
    ece_point = _compute_ece(y_true, y_prob, n_bins)
    brier_point = _compute_brier(y_true, y_prob)

    # Bootstrap confidence intervals
    ece_ci = _bootstrap_metric(
        y_true,
        y_prob,
        lambda yt, yp: _compute_ece(yt, yp, n_bins),
        ece_point,
        n_bootstrap,
        confidence_level,
        rng,
        "ece",
    )

    brier_ci = _bootstrap_metric(
        y_true,
        y_prob,
        _compute_brier,
        brier_point,
        n_bootstrap,
        confidence_level,
        rng,
        "brier_score",
    )

    # Bin-wise CIs for calibration curve
    bin_wise_ci = compute_bin_wise_ci(
        y_true,
        y_prob,
        n_bins,
        n_bootstrap,
        confidence_level,
        rng,
    )

    return {
        "ece": ece_point,
        "ece_ci": ece_ci,
        "brier_score": brier_point,
        "brier_ci": brier_ci,
        "bin_wise_ci": bin_wise_ci,
        "n_samples": n_samples,
        "n_bins": n_bins,
    }


def compute_bin_wise_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    rng: np.random.RandomState | None = None,
) -> BinWiseCI:
    """Compute bin-wise confidence intervals for calibration curve.

    For each bin, bootstrap the observed frequency (accuracy) to get error bars.

    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        rng: Random state for determinism

    Returns:
        BinWiseCI with confidence intervals for each bin

    """
    if rng is None:
        rng = np.random.RandomState(None)

    # Define bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)  # Clip to valid range

    # Compute point estimates for each bin
    observed_freqs = []
    expected_freqs = []
    ci_lowers = []
    ci_uppers = []
    sample_counts = []
    bins = []

    for b in range(n_bins):
        in_bin = bin_ids == b
        m = np.count_nonzero(in_bin)

        bin_lower = bin_edges[b]
        bin_upper = bin_edges[b + 1]
        bins.append((float(bin_lower), float(bin_upper)))

        if m == 0:
            # Empty bin - no data
            observed_freqs.append(np.nan)
            expected_freqs.append(np.nan)
            ci_lowers.append(np.nan)
            ci_uppers.append(np.nan)
            sample_counts.append(0)
            continue

        # Point estimates
        obs_freq = float(np.mean(y_true[in_bin]))
        exp_freq = float(np.mean(y_prob[in_bin]))

        observed_freqs.append(obs_freq)
        expected_freqs.append(exp_freq)
        sample_counts.append(int(m))

        # Bootstrap CI for observed frequency in this bin
        if m < 10:
            # Too few samples for reliable bootstrap
            ci_lowers.append(np.nan)
            ci_uppers.append(np.nan)
            logger.debug(f"Bin {b} has only {m} samples - CI not reliable")
            continue

        bootstrap_obs_freqs = []
        # Note: Progress bar not added here as this is nested within bin loop
        # and would create too many progress bars. Bin-wise CI is fast anyway.
        for _ in range(n_bootstrap):
            # Resample within this bin
            indices = np.where(in_bin)[0]
            boot_indices = rng.choice(indices, size=len(indices), replace=True)
            boot_obs_freq = float(np.mean(y_true[boot_indices]))
            bootstrap_obs_freqs.append(boot_obs_freq)

        # Compute percentile CI
        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        lower = float(np.percentile(bootstrap_obs_freqs, lower_percentile))
        upper = float(np.percentile(bootstrap_obs_freqs, upper_percentile))

        ci_lowers.append(lower)
        ci_uppers.append(upper)

    return BinWiseCI(
        bins=bins,
        observed_freq=observed_freqs,
        expected_freq=expected_freqs,
        ci_lower=ci_lowers,
        ci_upper=ci_uppers,
        sample_counts=sample_counts,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )


def _bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable,
    point_estimate: float,
    n_bootstrap: int,
    confidence_level: float,
    rng: np.random.RandomState,
    metric_name: str,
) -> CalibrationCI:
    """Bootstrap confidence interval for a calibration metric.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        metric_fn: Function that computes metric(y_true, y_prob)
        point_estimate: Point estimate on original data
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        rng: Random state
        metric_name: Name of the metric ("ece" or "brier_score")

    Returns:
        CalibrationCI with bounds and metadata

    """
    from ...utils.progress import get_progress_bar

    n_samples = len(y_true)
    bootstrap_estimates = []

    # Add progress bar for bootstrap iterations (respects strict mode and env vars)
    with get_progress_bar(
        total=n_bootstrap,
        desc=f"Bootstrap {metric_name}",
        leave=False,
        disable=n_bootstrap < 100,
    ) as pbar:
        for _ in range(n_bootstrap):
            # Simple random resample with replacement
            indices = rng.choice(n_samples, size=n_samples, replace=True)

            y_true_boot = y_true[indices]
            y_prob_boot = y_prob[indices]

            try:
                boot_estimate = metric_fn(y_true_boot, y_prob_boot)
                if not np.isnan(boot_estimate):
                    bootstrap_estimates.append(boot_estimate)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Bootstrap iteration failed: {e}")
                continue

            pbar.update(1)

    if len(bootstrap_estimates) == 0:
        logger.warning(f"All bootstrap iterations failed for {metric_name}")
        return CalibrationCI(
            metric_name=metric_name,
            point_estimate=point_estimate,
            lower=np.nan,
            upper=np.nan,
            std_error=np.nan,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
        )

    bootstrap_estimates_array = np.array(bootstrap_estimates)

    # Compute percentile CI
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower = float(np.percentile(bootstrap_estimates_array, lower_percentile))
    upper = float(np.percentile(bootstrap_estimates_array, upper_percentile))

    # Compute standard error
    std_error = float(np.std(bootstrap_estimates_array, ddof=1))

    return CalibrationCI(
        metric_name=metric_name,
        point_estimate=float(point_estimate),
        lower=lower,
        upper=upper,
        std_error=std_error,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins

    Returns:
        ECE value

    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece: float = 0.0
    n = len(y_prob)

    for b in range(n_bins):
        in_bin = bin_ids == b
        m = np.count_nonzero(in_bin)
        if m == 0:
            continue

        avg_conf = float(np.mean(y_prob[in_bin]))
        avg_acc = float(np.mean(y_true[in_bin]))
        gap = abs(avg_acc - avg_conf)
        ece += (m / n) * gap

    return float(ece)


def _compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score.

    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities

    Returns:
        Brier score

    """
    from sklearn.metrics import brier_score_loss  # noqa: PLC0415

    return float(brier_score_loss(y_true, y_prob))
