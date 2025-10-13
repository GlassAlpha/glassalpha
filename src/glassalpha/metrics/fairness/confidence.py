"""Bootstrap confidence intervals for fairness metrics.

This module implements E10: Statistical Confidence & Uncertainty.

Key features:
1. Deterministic bootstrap (seeded for reproducibility)
2. Small sample size warnings (n<30 WARNING, n<10 ERROR)
3. Statistical power calculations
4. Support for stratified bootstrap
5. Multiple CI methods (percentile, BCa)
"""

import logging
import signal
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class BootstrapTimeout(Exception):
    """Raised when bootstrap computation exceeds timeout."""


def _timeout_handler(signum, frame):
    """Signal handler for bootstrap timeout."""
    raise BootstrapTimeout("Bootstrap computation exceeded 30s timeout")


@dataclass
class BootstrapCI:
    """Container for bootstrap confidence interval results."""

    point_estimate: float
    lower: float
    upper: float
    std_error: float
    confidence_level: float
    n_bootstrap: int
    method: str = "percentile"

    @property
    def width(self) -> float:
        """Compute CI width."""
        return self.upper - self.lower

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "point_estimate": self.point_estimate,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "std_error": self.std_error,
            "confidence_level": self.confidence_level,
            "ci_width": self.width,
            "n_bootstrap": self.n_bootstrap,
            "method": self.method,
        }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    method: str = "percentile",
    stratify: bool = False,
    seed: int | None = None,
) -> BootstrapCI:
    """Compute bootstrap confidence interval for a fairness metric.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        sensitive: Sensitive attribute values
        metric_fn: Function that computes metric given (y_true, y_pred, sensitive)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95 for 95% CI)
        method: CI method ("percentile" or "bca")
        stratify: Whether to use stratified bootstrap
        seed: Random seed for determinism

    Returns:
        BootstrapCI with point estimate, bounds, and metadata

    """
    # Set seed for deterministic bootstrap
    rng = np.random.RandomState(seed)

    # Compute point estimate on original data
    point_estimate = metric_fn(y_true, y_pred, sensitive)

    # Handle NaN point estimate
    if np.isnan(point_estimate) or point_estimate is None:
        return BootstrapCI(
            point_estimate=np.nan,
            lower=np.nan,
            upper=np.nan,
            std_error=np.nan,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            method=method,
        )

    # Generate bootstrap samples
    from ...utils.progress import get_progress_bar

    n_samples = len(y_true)
    bootstrap_estimates = []

    # Set 30-second timeout for bootstrap
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(30)

    try:
        # Add progress bar for bootstrap iterations (respects strict mode and env vars)
        with get_progress_bar(
            total=n_bootstrap,
            desc="Bootstrap fairness CIs",
            leave=False,
            disable=n_bootstrap < 100,
        ) as pbar:
            for i in range(n_bootstrap):
                # Stratified sampling preserves class proportions
                if stratify:
                    # Sample indices maintaining class balance
                    indices = _stratified_resample(y_true, rng)
                else:
                    # Simple random resample with replacement
                    indices = rng.choice(n_samples, size=n_samples, replace=True)

                # Resample data
                y_true_boot = y_true[indices]
                y_pred_boot = y_pred[indices]
                sensitive_boot = sensitive[indices]

                # Compute metric on bootstrap sample
                try:
                    boot_estimate = metric_fn(y_true_boot, y_pred_boot, sensitive_boot)
                    if not np.isnan(boot_estimate):
                        bootstrap_estimates.append(boot_estimate)
                except Exception as e:
                    logger.debug(f"Bootstrap iteration {i} failed: {e}")
                    continue

                pbar.update(1)

    except BootstrapTimeout:
        logger.warning("Bootstrap CI computation timed out after 30s, returning point estimates only")
        return BootstrapCI(
            point_estimate=float(point_estimate),
            lower=np.nan,
            upper=np.nan,
            std_error=np.nan,
            confidence_level=confidence_level,
            n_bootstrap=len(bootstrap_estimates),
            method=method,
        )
    finally:
        signal.alarm(0)  # Cancel alarm
        signal.signal(signal.SIGALRM, old_handler)  # Restore old handler

    if len(bootstrap_estimates) == 0:
        logger.warning("All bootstrap iterations failed")
        return BootstrapCI(
            point_estimate=point_estimate,
            lower=np.nan,
            upper=np.nan,
            std_error=np.nan,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            method=method,
        )

    bootstrap_estimates_array = np.array(bootstrap_estimates)

    # Compute confidence interval
    if method == "percentile":
        lower, upper = _percentile_ci(bootstrap_estimates_array, confidence_level)
    elif method == "bca":
        lower, upper = _bca_ci(
            bootstrap_estimates_array,
            point_estimate,
            y_true,
            y_pred,
            sensitive,
            metric_fn,
            confidence_level,
        )
    else:
        raise ValueError(f"Unknown CI method: {method}")

    # Compute standard error
    std_error = np.std(bootstrap_estimates_array, ddof=1)

    return BootstrapCI(
        point_estimate=float(point_estimate),
        lower=float(lower),
        upper=float(upper),
        std_error=float(std_error),
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        method=method,
    )


def _stratified_resample(y: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Stratified resampling that preserves class proportions."""
    indices: list[int] = []
    unique_classes = np.unique(y)

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        n_cls = len(cls_indices)
        # Resample within class
        resampled = rng.choice(cls_indices, size=n_cls, replace=True)
        indices.extend(resampled.tolist())

    return np.array(indices)


def _percentile_ci(
    bootstrap_estimates: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    """Compute percentile confidence interval."""
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower = np.percentile(bootstrap_estimates, lower_percentile)
    upper = np.percentile(bootstrap_estimates, upper_percentile)

    return lower, upper


def _bca_ci(
    bootstrap_estimates: np.ndarray,
    point_estimate: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    metric_fn: Callable,
    confidence_level: float,
) -> tuple[float, float]:
    """Compute bias-corrected and accelerated (BCa) confidence interval.

    BCa adjusts for bias and skewness in the bootstrap distribution.
    More accurate than percentile method for skewed distributions.
    """
    # Compute bias correction
    n_below = np.sum(bootstrap_estimates < point_estimate)
    z0 = stats.norm.ppf(n_below / len(bootstrap_estimates))

    # Compute acceleration using jackknife
    n_samples = len(y_true)
    jackknife_estimates: list[float] = []

    for i in range(n_samples):
        # Leave-one-out sample
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False

        try:
            jack_est = metric_fn(y_true[mask], y_pred[mask], sensitive[mask])
            jackknife_estimates.append(jack_est)
        except Exception:
            continue

    if len(jackknife_estimates) < n_samples // 2:
        # Fall back to percentile if jackknife fails
        logger.warning("BCa jackknife failed, falling back to percentile")
        return _percentile_ci(bootstrap_estimates, confidence_level)

    jackknife_estimates = np.array(jackknife_estimates)
    jack_mean = np.mean(jackknife_estimates)

    # Acceleration factor
    numerator = np.sum((jack_mean - jackknife_estimates) ** 3)
    denominator = 6 * (np.sum((jack_mean - jackknife_estimates) ** 2) ** 1.5)

    if denominator == 0:
        # Fall back to percentile if acceleration undefined
        return _percentile_ci(bootstrap_estimates, confidence_level)

    acceleration = numerator / denominator

    # Adjusted percentiles
    alpha = 1 - confidence_level
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

    # BCa-adjusted percentiles
    lower_percentile = 100 * stats.norm.cdf(
        z0 + (z0 + z_alpha_lower) / (1 - acceleration * (z0 + z_alpha_lower)),
    )
    upper_percentile = 100 * stats.norm.cdf(
        z0 + (z0 + z_alpha_upper) / (1 - acceleration * (z0 + z_alpha_upper)),
    )

    # Clip percentiles to valid range
    lower_percentile = np.clip(lower_percentile, 0.1, 99.9)
    upper_percentile = np.clip(upper_percentile, 0.1, 99.9)

    lower = np.percentile(bootstrap_estimates, lower_percentile)
    upper = np.percentile(bootstrap_estimates, upper_percentile)

    return lower, upper


def check_sample_size_adequacy(
    n: int,
    group_name: str = "group",
    min_error: int = 10,
    min_warning: int = 30,
) -> dict[str, Any]:
    """Check if sample size is adequate for statistical inference.

    Args:
        n: Sample size to check
        group_name: Name of the group (for message)
        min_error: Minimum samples for ERROR threshold (default 10)
        min_warning: Minimum samples for WARNING threshold (default 30)

    Returns:
        Dictionary with status, message, and metadata

    """
    if n < min_error:
        return {
            "status": "ERROR",
            "n_samples": n,
            "min_required": min_error,
            "message": f"Insufficient sample size for {group_name} (n={n}). "
            f"Minimum {min_error} samples required for reliable inference.",
            "recommendation": f"Increase sample size to at least {min_error} "
            f"(recommended: {min_warning}+) or exclude group from analysis.",
        }
    if n < min_warning:
        return {
            "status": "WARNING",
            "n_samples": n,
            "min_recommended": min_warning,
            "message": f"Small sample size for {group_name} (n={n}). "
            f"Results may have high variance. Recommended: {min_warning}+ samples.",
            "recommendation": "Interpret results with caution. "
            "Consider collecting more data or using wider confidence intervals.",
        }
    return {
        "status": "OK",
        "n_samples": n,
        "message": f"Adequate sample size for {group_name} (n={n}).",
    }


def compute_statistical_power(
    n_samples: int,
    effect_size: float,
    alpha: float = 0.05,
    test_type: str = "two_proportion",
) -> float:
    """Compute statistical power for detecting disparity.

    Args:
        n_samples: Sample size per group
        effect_size: Expected effect size (e.g., 0.10 for 10% disparity)
        alpha: Significance level (default 0.05)
        test_type: Type of test ("two_proportion" for fairness metrics)

    Returns:
        Statistical power (probability of detecting effect if it exists)

    """
    if test_type == "two_proportion":
        # Two-proportion z-test power calculation
        # Assumes equal sample sizes in both groups

        # Base rates (conservative: assume 0.5)
        p1 = 0.5
        p2 = p1 + effect_size

        # Pooled proportion
        p_pooled = (p1 + p2) / 2

        # Standard error under alternative
        se_alt = np.sqrt(
            (p1 * (1 - p1) / n_samples) + (p2 * (1 - p2) / n_samples),
        )

        # Critical value for two-tailed test
        z_critical = stats.norm.ppf(1 - alpha / 2)

        # Non-centrality parameter
        ncp = effect_size / se_alt

        # Power = P(|Z| > z_critical | H1 true)
        # For two-tailed test
        power = 1 - stats.norm.cdf(z_critical - ncp) + stats.norm.cdf(-z_critical - ncp)

        return float(np.clip(power, 0.0, 1.0))

    raise ValueError(f"Unknown test type: {test_type}")


def compute_min_detectable_effect(
    n_samples: int,
    power: float = 0.80,
    alpha: float = 0.05,
    test_type: str = "two_proportion",
) -> float:
    """Compute minimum detectable effect size given sample size and power.

    Args:
        n_samples: Sample size per group
        power: Desired statistical power (default 0.80)
        alpha: Significance level (default 0.05)
        test_type: Type of test

    Returns:
        Minimum detectable effect size

    """
    if test_type == "two_proportion":
        # Approximate formula for minimum detectable effect
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        # Conservative estimate assuming p=0.5 (maximum variance)
        p = 0.5
        se = np.sqrt(2 * p * (1 - p) / n_samples)

        mde = (z_alpha + z_beta) * se

        return float(mde)

    raise ValueError(f"Unknown test type: {test_type}")
