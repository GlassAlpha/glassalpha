"""Drift detection metrics for GlassAlpha.

This module implements drift detection metrics for monitoring changes in data
distributions and model performance over time. These metrics are essential
for production model monitoring and automated retraining decisions.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from ..base import BaseMetric
from ..registry import MetricRegistry

logger = logging.getLogger(__name__)


@MetricRegistry.register("population_stability_index", priority=100)
class PopulationStabilityIndexMetric(BaseMetric):
    """Population Stability Index (PSI) metric.

    Measures the shift in population distribution between a reference dataset
    (e.g., training data) and a current dataset (e.g., production data).
    PSI is widely used in credit risk modeling for monitoring model stability.

    PSI Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 ≤ PSI < 0.25: Some change, monitor closely
    - PSI ≥ 0.25: Significant change, likely needs investigation/retraining
    """

    metric_type = "drift"

    def __init__(self, n_bins: int = 10, min_bin_pct: float = 0.05):
        """Initialize PSI metric.

        Args:
            n_bins: Number of bins for discretizing continuous variables
            min_bin_pct: Minimum percentage per bin to avoid division by zero

        """
        super().__init__("population_stability_index", "drift", "1.0.0")
        self.n_bins = n_bins
        self.min_bin_pct = min_bin_pct
        logger.info(f"PSIMetric initialized (n_bins={n_bins}, min_bin_pct={min_bin_pct})")

    def _calculate_psi_numeric(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate PSI for numeric features."""
        # Create bins based on reference data quantiles
        try:
            # Handle edge case where all values are the same
            if np.var(reference) == 0:
                logger.warning("Reference data has zero variance, PSI = 0")
                return 0.0

            # Create quantile-based bins
            bin_edges = np.quantile(reference, np.linspace(0, 1, self.n_bins + 1))

            # Ensure unique bin edges
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) <= 1:
                logger.warning("Cannot create bins for PSI calculation, returning 0")
                return 0.0

            # Calculate frequencies for both datasets
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            cur_counts, _ = np.histogram(current, bins=bin_edges)

            # Convert to percentages
            ref_pct = ref_counts / len(reference)
            cur_pct = cur_counts / len(current)

            # Apply minimum percentage to avoid division by zero
            ref_pct = np.maximum(ref_pct, self.min_bin_pct / 100)
            cur_pct = np.maximum(cur_pct, self.min_bin_pct / 100)

            # Calculate PSI
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

            return float(psi)

        except Exception as e:
            logger.error(f"Error calculating numeric PSI: {e}")
            return 0.0

    def _calculate_psi_categorical(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate PSI for categorical features."""
        try:
            # Get unique categories from both datasets
            all_categories = np.unique(np.concatenate([reference, current]))

            # Calculate frequencies
            ref_counts = pd.Series(reference).value_counts()
            cur_counts = pd.Series(current).value_counts()

            # Ensure all categories are present in both
            ref_pct = ref_counts.reindex(all_categories, fill_value=0).values / len(reference)
            cur_pct = cur_counts.reindex(all_categories, fill_value=0).values / len(current)

            # Apply minimum percentage
            ref_pct = np.maximum(ref_pct, self.min_bin_pct / 100)
            cur_pct = np.maximum(cur_pct, self.min_bin_pct / 100)

            # Calculate PSI
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

            return float(psi)

        except Exception as e:
            logger.error(f"Error calculating categorical PSI: {e}")
            return 0.0

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute Population Stability Index.

        Args:
            y_true: Reference dataset (e.g., training data predictions)
            y_pred: Current dataset (e.g., production data predictions)
            sensitive_features: Not used for PSI but accepted for interface compatibility

        Returns:
            Dictionary with PSI metrics

        """
        validation = self.validate_inputs(y_true, y_pred, sensitive_features)

        try:
            # Calculate PSI for predictions
            psi_predictions = self._calculate_psi_numeric(y_true, y_pred)

            # Interpret PSI level
            if psi_predictions < 0.1:
                drift_level = "stable"
                action_needed = False
            elif psi_predictions < 0.25:
                drift_level = "minor_drift"
                action_needed = False
            else:
                drift_level = "significant_drift"
                action_needed = True

            results = {
                "population_stability_index": float(psi_predictions),
                "drift_level": drift_level,
                "action_needed": float(action_needed),
                "n_bins": self.n_bins,
                "reference_samples": len(y_true),
                "current_samples": len(y_pred),
                "n_samples": float(validation["n_samples"]),
            }

            logger.debug(f"PSI calculated: {psi_predictions:.4f} ({drift_level})")

            return results

        except Exception as e:
            logger.error(f"Error computing PSI: {e}")
            return {"population_stability_index": 0.0, "error": str(e)}

    def get_metric_names(self) -> list[str]:
        """Return metric names."""
        return [
            "population_stability_index",
            "drift_level",
            "action_needed",
            "n_bins",
            "reference_samples",
            "current_samples",
            "n_samples",
        ]


@MetricRegistry.register("kl_divergence", priority=99)
class KLDivergenceMetric(BaseMetric):
    """Kullback-Leibler Divergence metric for drift detection.

    Measures the difference between two probability distributions.
    KL divergence is asymmetric and measures how much information is lost
    when using one distribution to approximate another.
    """

    metric_type = "drift"

    def __init__(self, n_bins: int = 10, epsilon: float = 1e-10):
        """Initialize KL divergence metric.

        Args:
            n_bins: Number of bins for discretizing continuous variables
            epsilon: Small value to avoid log(0) issues

        """
        super().__init__("kl_divergence", "drift", "1.0.0")
        self.n_bins = n_bins
        self.epsilon = epsilon
        logger.info(f"KLDivergenceMetric initialized (n_bins={n_bins}, epsilon={epsilon})")

    def _calculate_kl_divergence(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate KL divergence between two distributions."""
        try:
            # Handle edge cases
            if len(reference) == 0 or len(current) == 0:
                return 0.0

            # For continuous data, create histograms
            if len(np.unique(reference)) > self.n_bins:
                # Create bins based on combined data range
                data_min = min(np.min(reference), np.min(current))
                data_max = max(np.max(reference), np.max(current))

                if data_max == data_min:
                    return 0.0

                bins = np.linspace(data_min, data_max, self.n_bins + 1)

                ref_counts, _ = np.histogram(reference, bins=bins)
                cur_counts, _ = np.histogram(current, bins=bins)
            else:
                # Categorical data
                all_values = np.unique(np.concatenate([reference, current]))
                ref_counts = np.array([np.sum(reference == val) for val in all_values])
                cur_counts = np.array([np.sum(current == val) for val in all_values])

            # Convert to probabilities
            ref_probs = ref_counts / np.sum(ref_counts)
            cur_probs = cur_counts / np.sum(cur_counts)

            # Add epsilon to avoid log(0)
            ref_probs = ref_probs + self.epsilon
            cur_probs = cur_probs + self.epsilon

            # Normalize after adding epsilon
            ref_probs = ref_probs / np.sum(ref_probs)
            cur_probs = cur_probs / np.sum(cur_probs)

            # Calculate KL divergence: KL(P||Q) = sum(P * log(P/Q))
            kl_div = np.sum(ref_probs * np.log(ref_probs / cur_probs))

            return float(kl_div)

        except Exception as e:
            logger.error(f"Error calculating KL divergence: {e}")
            return 0.0

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute KL divergence.

        Args:
            y_true: Reference distribution
            y_pred: Current distribution
            sensitive_features: Not used for KL divergence

        Returns:
            Dictionary with KL divergence metrics

        """
        validation = self.validate_inputs(y_true, y_pred, sensitive_features)

        try:
            # Calculate KL divergence (reference || current)
            kl_forward = self._calculate_kl_divergence(y_true, y_pred)

            # Calculate reverse KL divergence (current || reference)
            kl_reverse = self._calculate_kl_divergence(y_pred, y_true)

            # Average for symmetry (Jensen-Shannon style)
            kl_symmetric = (kl_forward + kl_reverse) / 2

            # Drift interpretation (thresholds are domain-specific)
            if kl_symmetric < 0.1:
                drift_level = "stable"
            elif kl_symmetric < 0.5:
                drift_level = "minor_drift"
            else:
                drift_level = "significant_drift"

            results = {
                "kl_divergence": float(kl_symmetric),
                "kl_forward": float(kl_forward),
                "kl_reverse": float(kl_reverse),
                "drift_level": drift_level,
                "n_bins": self.n_bins,
                "n_samples": float(validation["n_samples"]),
            }

            logger.debug(f"KL divergence calculated: {kl_symmetric:.4f} ({drift_level})")

            return results

        except Exception as e:
            logger.error(f"Error computing KL divergence: {e}")
            return {"kl_divergence": 0.0, "error": str(e)}

    def get_metric_names(self) -> list[str]:
        """Return metric names."""
        return ["kl_divergence", "kl_forward", "kl_reverse", "drift_level", "n_bins", "n_samples"]


@MetricRegistry.register("kolmogorov_smirnov", priority=98)
class KolmogorovSmirnovMetric(BaseMetric):
    """Kolmogorov-Smirnov test for drift detection.

    Non-parametric test that compares the cumulative distribution functions
    of two datasets. Particularly useful for continuous distributions.
    """

    metric_type = "drift"

    def __init__(self, alpha: float = 0.05):
        """Initialize KS test metric.

        Args:
            alpha: Significance level for the test

        """
        super().__init__("kolmogorov_smirnov", "drift", "1.0.0")
        self.alpha = alpha
        logger.info(f"KSMetric initialized (alpha={alpha})")

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute Kolmogorov-Smirnov test.

        Args:
            y_true: Reference dataset
            y_pred: Current dataset
            sensitive_features: Not used for KS test

        Returns:
            Dictionary with KS test results

        """
        validation = self.validate_inputs(y_true, y_pred, sensitive_features)

        try:
            # Perform two-sample KS test
            ks_statistic, p_value = stats.ks_2samp(y_true, y_pred)

            # Determine if distributions are significantly different
            is_drift = p_value < self.alpha

            # Drift level based on p-value
            if p_value >= self.alpha:
                drift_level = "stable"
            elif p_value >= self.alpha / 10:  # p < 0.05 but >= 0.005
                drift_level = "minor_drift"
            else:
                drift_level = "significant_drift"

            results = {
                "kolmogorov_smirnov": float(ks_statistic),
                "ks_statistic": float(ks_statistic),
                "p_value": float(p_value),
                "is_drift": float(is_drift),
                "drift_level": drift_level,
                "alpha": self.alpha,
                "n_samples": float(validation["n_samples"]),
            }

            logger.debug(f"KS test: statistic={ks_statistic:.4f}, p-value={p_value:.6f}, drift={is_drift}")

            return results

        except Exception as e:
            logger.error(f"Error computing KS test: {e}")
            return {"kolmogorov_smirnov": 0.0, "error": str(e)}

    def get_metric_names(self) -> list[str]:
        """Return metric names."""
        return ["kolmogorov_smirnov", "ks_statistic", "p_value", "is_drift", "drift_level", "alpha", "n_samples"]


@MetricRegistry.register("jensen_shannon_divergence", priority=97)
class JensenShannonDivergenceMetric(BaseMetric):
    """Jensen-Shannon Divergence metric for drift detection.

    Symmetric version of KL divergence. Always finite and bounded between 0 and 1.
    More stable than KL divergence and easier to interpret.
    """

    metric_type = "drift"

    def __init__(self, n_bins: int = 10):
        """Initialize JS divergence metric.

        Args:
            n_bins: Number of bins for discretizing continuous variables

        """
        super().__init__("jensen_shannon_divergence", "drift", "1.0.0")
        self.n_bins = n_bins
        logger.info(f"JSDivergenceMetric initialized (n_bins={n_bins})")

    def _calculate_js_divergence(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence between two distributions."""
        try:
            # Handle edge cases
            if len(reference) == 0 or len(current) == 0:
                return 0.0

            # Create probability distributions
            if len(np.unique(reference)) > self.n_bins:
                # Continuous data - create histograms
                data_min = min(np.min(reference), np.min(current))
                data_max = max(np.max(reference), np.max(current))

                if data_max == data_min:
                    return 0.0

                bins = np.linspace(data_min, data_max, self.n_bins + 1)

                ref_counts, _ = np.histogram(reference, bins=bins)
                cur_counts, _ = np.histogram(current, bins=bins)
            else:
                # Categorical data
                all_values = np.unique(np.concatenate([reference, current]))
                ref_counts = np.array([np.sum(reference == val) for val in all_values])
                cur_counts = np.array([np.sum(current == val) for val in all_values])

            # Convert to probabilities
            ref_probs = ref_counts / np.sum(ref_counts)
            cur_probs = cur_counts / np.sum(cur_counts)

            # Use scipy's Jensen-Shannon distance (which is the square root of JS divergence)
            js_distance = jensenshannon(ref_probs, cur_probs)

            # Square it to get JS divergence
            js_divergence = js_distance**2

            return float(js_divergence)

        except Exception as e:
            logger.error(f"Error calculating JS divergence: {e}")
            return 0.0

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute Jensen-Shannon divergence.

        Args:
            y_true: Reference distribution
            y_pred: Current distribution
            sensitive_features: Not used for JS divergence

        Returns:
            Dictionary with JS divergence metrics

        """
        validation = self.validate_inputs(y_true, y_pred, sensitive_features)

        try:
            js_divergence = self._calculate_js_divergence(y_true, y_pred)

            # Drift interpretation (JS divergence is bounded [0,1])
            if js_divergence < 0.1:
                drift_level = "stable"
            elif js_divergence < 0.3:
                drift_level = "minor_drift"
            else:
                drift_level = "significant_drift"

            results = {
                "jensen_shannon_divergence": float(js_divergence),
                "js_distance": float(np.sqrt(js_divergence)),  # Square root for distance
                "drift_level": drift_level,
                "n_bins": self.n_bins,
                "n_samples": float(validation["n_samples"]),
            }

            logger.debug(f"JS divergence calculated: {js_divergence:.4f} ({drift_level})")

            return results

        except Exception as e:
            logger.error(f"Error computing JS divergence: {e}")
            return {"jensen_shannon_divergence": 0.0, "error": str(e)}

    def get_metric_names(self) -> list[str]:
        """Return metric names."""
        return ["jensen_shannon_divergence", "js_distance", "drift_level", "n_bins", "n_samples"]


@MetricRegistry.register("prediction_drift", priority=96)
class PredictionDriftMetric(BaseMetric):
    """Comprehensive prediction drift metric.

    Combines multiple drift detection methods for robust prediction drift monitoring.
    Computes PSI, KL divergence, and KS test for prediction distributions.
    """

    metric_type = "drift"

    def __init__(self, psi_bins: int = 10, kl_bins: int = 10, ks_alpha: float = 0.05):
        """Initialize prediction drift metric.

        Args:
            psi_bins: Number of bins for PSI calculation
            kl_bins: Number of bins for KL divergence
            ks_alpha: Significance level for KS test

        """
        super().__init__("prediction_drift", "drift", "1.0.0")
        self.psi_metric = PopulationStabilityIndexMetric(n_bins=psi_bins)
        self.kl_metric = KLDivergenceMetric(n_bins=kl_bins)
        self.ks_metric = KolmogorovSmirnovMetric(alpha=ks_alpha)
        logger.info("PredictionDriftMetric initialized")

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute comprehensive prediction drift analysis.

        Args:
            y_true: Reference predictions (e.g., training set predictions)
            y_pred: Current predictions (e.g., production predictions)
            sensitive_features: Not used for prediction drift

        Returns:
            Dictionary with comprehensive drift metrics

        """
        validation = self.validate_inputs(y_true, y_pred, sensitive_features)

        try:
            # Compute individual drift metrics
            psi_results = self.psi_metric.compute(y_true, y_pred, sensitive_features)
            kl_results = self.kl_metric.compute(y_true, y_pred, sensitive_features)
            ks_results = self.ks_metric.compute(y_true, y_pred, sensitive_features)

            # Extract key metrics
            psi_value = psi_results.get("population_stability_index", 0.0)
            kl_value = kl_results.get("kl_divergence", 0.0)
            ks_statistic = ks_results.get("ks_statistic", 0.0)
            ks_p_value = ks_results.get("p_value", 1.0)

            # Combine drift indicators
            drift_indicators = []

            # PSI-based drift
            if psi_value >= 0.25:
                drift_indicators.append("psi_significant")
            elif psi_value >= 0.1:
                drift_indicators.append("psi_minor")

            # KL-based drift
            if kl_value >= 0.5:
                drift_indicators.append("kl_significant")
            elif kl_value >= 0.1:
                drift_indicators.append("kl_minor")

            # KS-based drift
            if ks_p_value < 0.001:
                drift_indicators.append("ks_significant")
            elif ks_p_value < 0.05:
                drift_indicators.append("ks_minor")

            # Overall drift assessment
            significant_indicators = len([d for d in drift_indicators if "significant" in d])
            minor_indicators = len([d for d in drift_indicators if "minor" in d])

            if significant_indicators >= 2:
                overall_drift = "significant_drift"
                action_needed = True
            elif significant_indicators >= 1 or minor_indicators >= 2:
                overall_drift = "minor_drift"
                action_needed = False
            else:
                overall_drift = "stable"
                action_needed = False

            # Create comprehensive drift score (0-1, higher = more drift)
            drift_score = min(1.0, (psi_value / 0.5) * 0.4 + (kl_value / 1.0) * 0.3 + (1 - ks_p_value) * 0.3)

            results = {
                "prediction_drift": float(drift_score),
                "drift_level": overall_drift,
                "action_needed": float(action_needed),
                # Individual metric results
                "psi": float(psi_value),
                "kl_divergence": float(kl_value),
                "ks_statistic": float(ks_statistic),
                "ks_p_value": float(ks_p_value),
                # Indicators
                "drift_indicators": drift_indicators,
                "n_indicators": len(drift_indicators),
                "n_significant": significant_indicators,
                "n_samples": float(validation["n_samples"]),
            }

            logger.debug(
                f"Prediction drift: score={drift_score:.3f}, level={overall_drift}, indicators={drift_indicators}",
            )

            return results

        except Exception as e:
            logger.error(f"Error computing prediction drift: {e}")
            return {"prediction_drift": 0.0, "error": str(e)}

    def get_metric_names(self) -> list[str]:
        """Return metric names."""
        return [
            "prediction_drift",
            "drift_level",
            "action_needed",
            "psi",
            "kl_divergence",
            "ks_statistic",
            "ks_p_value",
            "drift_indicators",
            "n_indicators",
            "n_significant",
            "n_samples",
        ]


# Auto-register drift metrics
logger.debug("Drift metrics registered")
