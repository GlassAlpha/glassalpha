"""Fairness and bias detection metrics for GlassAlpha.

This module implements fairness metrics for detecting and measuring bias
in machine learning models across different demographic groups. These metrics
are essential for regulatory compliance and ethical AI practices.
"""

import logging

import numpy as np
import pandas as pd

from ..base import BaseMetric
from ..registry import MetricRegistry

logger = logging.getLogger(__name__)


@MetricRegistry.register("demographic_parity", priority=100)
class DemographicParityMetric(BaseMetric):
    """Demographic Parity (Statistical Parity) metric.

    Measures whether the selection rate (positive prediction rate) is equal
    across different demographic groups. A fair model should have similar
    positive prediction rates for all groups.

    Formula: P(Ŷ=1|A=a) should be equal for all groups a
    """

    metric_type = "fairness"
    version = "1.0.0"

    def __init__(self, tolerance: float = 0.1):
        """Initialize demographic parity metric.

        Args:
            tolerance: Acceptable difference between group rates (0.1 = 10%)

        """
        super().__init__("demographic_parity", "fairness", "1.0.0")
        self.tolerance = tolerance
        logger.info(f"DemographicParityMetric initialized (tolerance={tolerance})")

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute demographic parity metric.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sensitive_features: DataFrame with sensitive attribute columns or numpy array

        Returns:
            Dictionary with demographic parity metrics

        """
        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if sensitive_features is None:
            logger.warning("Demographic parity requires sensitive features")
            return {"demographic_parity": 0.0, "error": "No sensitive features provided"}

        # Handle sensitive features as numpy array or DataFrame
        if isinstance(sensitive_features, pd.DataFrame):
            # Multiple sensitive features
            results = {}
            for attr_name in sensitive_features.columns:
                attr_values = sensitive_features[attr_name].values
                attr_results = self._compute_single_attribute(y_true, y_pred, attr_values, attr_name)
                results.update(attr_results)
        else:
            # Single sensitive feature as numpy array
            sensitive_features = np.asarray(sensitive_features)
            results = self._compute_single_attribute(y_true, y_pred, sensitive_features, "sensitive")

        return results

    def _compute_single_attribute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_values: np.ndarray,
        attr_name: str,
    ) -> dict[str, float]:
        """Compute demographic parity for a single sensitive attribute."""
        validation = self.validate_inputs(y_true, y_pred, sensitive_values)

        try:
            results = {}

            unique_groups = np.unique(sensitive_values)

            # Compute selection rates for each group
            group_rates = {}
            for group in unique_groups:
                group_mask = sensitive_values == group
                group_y_pred = y_pred[group_mask]

                if len(group_y_pred) > 0:
                    selection_rate = np.mean(group_y_pred)
                    group_rates[f"{attr_name}_{group}"] = float(selection_rate)
                else:
                    group_rates[f"{attr_name}_{group}"] = 0.0

            # Calculate parity metrics
            rates = list(group_rates.values())
            if len(rates) >= 2:
                max_rate = max(rates)
                min_rate = min(rates)

                # Demographic parity difference
                parity_diff = max_rate - min_rate
                results[f"{attr_name}_parity_difference"] = float(parity_diff)

                # Demographic parity ratio (min/max)
                parity_ratio = min_rate / max_rate if max_rate > 0 else 1.0
                results[f"{attr_name}_parity_ratio"] = float(parity_ratio)

                # Fairness indicator (within tolerance)
                is_fair = parity_diff <= self.tolerance
                results[f"{attr_name}_is_fair"] = float(is_fair)

                # Add individual group rates
                results.update(group_rates)

            logger.debug(f"Demographic parity for {attr_name}: {group_rates}")

            # Overall fairness score (average across attributes)
            fair_indicators = [v for k, v in results.items() if k.endswith("_is_fair")]
            if fair_indicators and len(fair_indicators) > 0:
                overall_fairness = float(np.mean(fair_indicators))
            else:
                overall_fairness = 0.0
            results["demographic_parity"] = overall_fairness

            results.update(
                {
                    "n_samples": float(validation["n_samples"]),
                    "tolerance": self.tolerance,
                    "n_sensitive_attributes": 1,
                },
            )

            return results

        except Exception as e:
            logger.error(f"Error computing demographic parity: {e}")
            return {"demographic_parity": 0.0, "error": str(e)}

    def requires_sensitive_features(self) -> bool:
        """Demographic parity requires sensitive features."""
        return True

    def get_metric_names(self) -> list[str]:
        """Return metric names (dynamic based on sensitive features)."""
        return ["demographic_parity", "n_samples", "tolerance", "n_sensitive_attributes"]


@MetricRegistry.register("equal_opportunity", priority=99)
class EqualOpportunityMetric(BaseMetric):
    """Equal Opportunity metric.

    Measures whether the true positive rate (sensitivity/recall) is equal
    across different demographic groups. This ensures equal opportunity
    for positive outcomes across groups.

    Formula: P(Ŷ=1|Y=1,A=a) should be equal for all groups a
    """

    metric_type = "fairness"
    version = "1.0.0"

    def __init__(self, tolerance: float = 0.1):
        """Initialize equal opportunity metric.

        Args:
            tolerance: Acceptable difference between group TPRs (0.1 = 10%)

        """
        super().__init__("equal_opportunity", "fairness", "1.0.0")
        self.tolerance = tolerance
        logger.info(f"EqualOpportunityMetric initialized (tolerance={tolerance})")

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute equal opportunity metric.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sensitive_features: DataFrame with sensitive attribute columns or numpy array

        Returns:
            Dictionary with equal opportunity metrics

        """
        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if sensitive_features is None:
            logger.warning("Equal opportunity requires sensitive features")
            return {"equal_opportunity": 0.0, "error": "No sensitive features provided"}

        # Handle sensitive features as numpy array or DataFrame
        if isinstance(sensitive_features, pd.DataFrame):
            # Multiple sensitive features
            results = {}
            for attr_name in sensitive_features.columns:
                attr_values = sensitive_features[attr_name].values
                attr_results = self._compute_single_attribute(y_true, y_pred, attr_values, attr_name)
                results.update(attr_results)
        else:
            # Single sensitive feature as numpy array
            sensitive_features = np.asarray(sensitive_features)
            results = self._compute_single_attribute(y_true, y_pred, sensitive_features, "sensitive")

        return results

    def _compute_single_attribute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_values: np.ndarray,
        attr_name: str,
    ) -> dict[str, float]:
        """Compute equal opportunity for a single sensitive attribute."""
        validation = self.validate_inputs(y_true, y_pred, sensitive_values)

        try:
            results = {}

            unique_groups = np.unique(sensitive_values)

            # Compute TPR for each group
            group_tprs = {}
            for group in unique_groups:
                group_mask = sensitive_values == group
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]

                if len(group_y_true) > 0:
                    # Calculate TPR = TP / (TP + FN)
                    positive_mask = group_y_true == 1
                    if np.sum(positive_mask) > 0:
                        tp = np.sum((group_y_true == 1) & (group_y_pred == 1))
                        fn = np.sum((group_y_true == 1) & (group_y_pred == 0))
                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    else:
                        tpr = 0.0  # No positive cases in this group

                    group_tprs[f"{attr_name}_{group}_tpr"] = float(tpr)
                else:
                    group_tprs[f"{attr_name}_{group}_tpr"] = 0.0

            # Calculate opportunity metrics
            tprs = list(group_tprs.values())
            if len(tprs) >= 2:
                max_tpr = max(tprs)
                min_tpr = min(tprs)

                # Equal opportunity difference
                opportunity_diff = max_tpr - min_tpr
                results[f"{attr_name}_opportunity_difference"] = float(opportunity_diff)

                # Equal opportunity ratio (min/max)
                opportunity_ratio = min_tpr / max_tpr if max_tpr > 0 else 1.0
                results[f"{attr_name}_opportunity_ratio"] = float(opportunity_ratio)

                # Fairness indicator (within tolerance)
                is_fair = opportunity_diff <= self.tolerance
                results[f"{attr_name}_is_fair"] = float(is_fair)

                # Add individual group TPRs
                results.update(group_tprs)

            logger.debug(f"Equal opportunity for {attr_name}: {group_tprs}")

            # Overall fairness score
            fair_indicators = [v for k, v in results.items() if k.endswith("_is_fair")]
            if fair_indicators and len(fair_indicators) > 0:
                overall_fairness = float(np.mean(fair_indicators))
            else:
                overall_fairness = 0.0
            results["equal_opportunity"] = overall_fairness

            results.update(
                {
                    "n_samples": float(validation["n_samples"]),
                    "tolerance": self.tolerance,
                    "n_sensitive_attributes": 1,
                },
            )

            return results

        except Exception as e:
            logger.error(f"Error computing equal opportunity: {e}")
            return {"equal_opportunity": 0.0, "error": str(e)}

    def requires_sensitive_features(self) -> bool:
        """Equal opportunity requires sensitive features."""
        return True

    def get_metric_names(self) -> list[str]:
        """Return metric names (dynamic based on sensitive features)."""
        return ["equal_opportunity", "n_samples", "tolerance", "n_sensitive_attributes"]


@MetricRegistry.register("equalized_odds", priority=98)
class EqualizedOddsMetric(BaseMetric):
    """Equalized Odds metric.

    Measures whether both the true positive rate (TPR) and false positive rate (FPR)
    are equal across different demographic groups. This is a stronger fairness
    condition than equal opportunity.

    Formula: P(Ŷ=1|Y=y,A=a) should be equal for all groups a and outcomes y∈{0,1}
    """

    metric_type = "fairness"
    version = "1.0.0"

    def __init__(self, tolerance: float = 0.1):
        """Initialize equalized odds metric.

        Args:
            tolerance: Acceptable difference between group rates (0.1 = 10%)

        """
        super().__init__("equalized_odds", "fairness", "1.0.0")
        self.tolerance = tolerance
        logger.info(f"EqualizedOddsMetric initialized (tolerance={tolerance})")

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute equalized odds metric.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sensitive_features: DataFrame with sensitive attribute columns

        Returns:
            Dictionary with equalized odds metrics

        """
        if sensitive_features is None:
            logger.warning("Equalized odds requires sensitive features")
            return {"equalized_odds": 0.0, "error": "No sensitive features provided"}

        validation = self.validate_inputs(y_true, y_pred, sensitive_features)

        try:
            results = {}

            # Compute for each sensitive attribute
            for attr_name in sensitive_features.columns:
                attr_values = sensitive_features[attr_name]
                unique_groups = np.unique(attr_values)

                # Compute TPR and FPR for each group
                group_metrics = {}
                for group in unique_groups:
                    group_mask = attr_values == group
                    group_y_true = y_true[group_mask]
                    group_y_pred = y_pred[group_mask]

                    if len(group_y_true) > 0:
                        # Calculate confusion matrix elements
                        tn = np.sum((group_y_true == 0) & (group_y_pred == 0))
                        fp = np.sum((group_y_true == 0) & (group_y_pred == 1))
                        fn = np.sum((group_y_true == 1) & (group_y_pred == 0))
                        tp = np.sum((group_y_true == 1) & (group_y_pred == 1))

                        # Calculate TPR and FPR
                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                        group_metrics[f"{attr_name}_{group}_tpr"] = float(tpr)
                        group_metrics[f"{attr_name}_{group}_fpr"] = float(fpr)
                    else:
                        group_metrics[f"{attr_name}_{group}_tpr"] = 0.0
                        group_metrics[f"{attr_name}_{group}_fpr"] = 0.0

                # Calculate equalized odds metrics
                tprs = [v for k, v in group_metrics.items() if k.endswith("_tpr")]
                fprs = [v for k, v in group_metrics.items() if k.endswith("_fpr")]

                if len(tprs) >= 2 and len(fprs) >= 2:
                    # TPR equalization
                    max_tpr = max(tprs)
                    min_tpr = min(tprs)
                    tpr_diff = max_tpr - min_tpr

                    # FPR equalization
                    max_fpr = max(fprs)
                    min_fpr = min(fprs)
                    fpr_diff = max_fpr - min_fpr

                    results[f"{attr_name}_tpr_difference"] = float(tpr_diff)
                    results[f"{attr_name}_fpr_difference"] = float(fpr_diff)

                    # Overall equalized odds violation (max of TPR and FPR differences)
                    odds_violation = max(tpr_diff, fpr_diff)
                    results[f"{attr_name}_odds_violation"] = float(odds_violation)

                    # Fairness indicator (both TPR and FPR within tolerance)
                    is_fair = (tpr_diff <= self.tolerance) and (fpr_diff <= self.tolerance)
                    results[f"{attr_name}_is_fair"] = float(is_fair)

                    # Add individual group metrics
                    results.update(group_metrics)

                logger.debug(f"Equalized odds for {attr_name}: TPR diff={tpr_diff:.3f}, FPR diff={fpr_diff:.3f}")

            # Overall fairness score
            fair_indicators = [v for k, v in results.items() if k.endswith("_is_fair")]
            if fair_indicators and len(fair_indicators) > 0:
                overall_fairness = float(np.mean(fair_indicators))
            else:
                overall_fairness = 0.0
            results["equalized_odds"] = overall_fairness

            results.update(
                {
                    "n_samples": float(validation["n_samples"]),
                    "tolerance": self.tolerance,
                    "n_sensitive_attributes": len(sensitive_features.columns),
                },
            )

            return results

        except Exception as e:
            logger.error(f"Error computing equalized odds: {e}")
            return {"equalized_odds": 0.0, "error": str(e)}

    def requires_sensitive_features(self) -> bool:
        """Equalized odds requires sensitive features."""
        return True

    def get_metric_names(self) -> list[str]:
        """Return metric names (dynamic based on sensitive features)."""
        return ["equalized_odds", "n_samples", "tolerance", "n_sensitive_attributes"]


@MetricRegistry.register("predictive_parity", priority=97)
class PredictiveParityMetric(BaseMetric):
    """Predictive Parity (Calibration) metric.

    Measures whether the positive predictive value (precision) is equal
    across different demographic groups. This ensures that a positive
    prediction means the same thing across groups.

    Formula: P(Y=1|Ŷ=1,A=a) should be equal for all groups a
    """

    metric_type = "fairness"
    version = "1.0.0"

    def __init__(self, tolerance: float = 0.1):
        """Initialize predictive parity metric.

        Args:
            tolerance: Acceptable difference between group PPVs (0.1 = 10%)

        """
        super().__init__("predictive_parity", "fairness", "1.0.0")
        self.tolerance = tolerance
        logger.info(f"PredictiveParityMetric initialized (tolerance={tolerance})")

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute predictive parity metric.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sensitive_features: DataFrame with sensitive attribute columns

        Returns:
            Dictionary with predictive parity metrics

        """
        if sensitive_features is None:
            logger.warning("Predictive parity requires sensitive features")
            return {"predictive_parity": 0.0, "error": "No sensitive features provided"}

        validation = self.validate_inputs(y_true, y_pred, sensitive_features)

        try:
            results = {}

            # Compute for each sensitive attribute
            for attr_name in sensitive_features.columns:
                attr_values = sensitive_features[attr_name]
                unique_groups = np.unique(attr_values)

                # Compute PPV (precision) for each group
                group_ppvs = {}
                for group in unique_groups:
                    group_mask = attr_values == group
                    group_y_true = y_true[group_mask]
                    group_y_pred = y_pred[group_mask]

                    if len(group_y_true) > 0:
                        # Calculate PPV = TP / (TP + FP)
                        positive_pred_mask = group_y_pred == 1
                        if np.sum(positive_pred_mask) > 0:
                            tp = np.sum((group_y_true == 1) & (group_y_pred == 1))
                            fp = np.sum((group_y_true == 0) & (group_y_pred == 1))
                            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        else:
                            ppv = 0.0  # No positive predictions in this group

                        group_ppvs[f"{attr_name}_{group}_ppv"] = float(ppv)
                    else:
                        group_ppvs[f"{attr_name}_{group}_ppv"] = 0.0

                # Calculate parity metrics
                ppvs = list(group_ppvs.values())
                if len(ppvs) >= 2:
                    max_ppv = max(ppvs)
                    min_ppv = min(ppvs)

                    # Predictive parity difference
                    parity_diff = max_ppv - min_ppv
                    results[f"{attr_name}_parity_difference"] = float(parity_diff)

                    # Predictive parity ratio (min/max)
                    parity_ratio = min_ppv / max_ppv if max_ppv > 0 else 1.0
                    results[f"{attr_name}_parity_ratio"] = float(parity_ratio)

                    # Fairness indicator (within tolerance)
                    is_fair = parity_diff <= self.tolerance
                    results[f"{attr_name}_is_fair"] = float(is_fair)

                    # Add individual group PPVs
                    results.update(group_ppvs)

                logger.debug(f"Predictive parity for {attr_name}: {group_ppvs}")

            # Overall fairness score
            fair_indicators = [v for k, v in results.items() if k.endswith("_is_fair")]
            if fair_indicators and len(fair_indicators) > 0:
                overall_fairness = float(np.mean(fair_indicators))
            else:
                overall_fairness = 0.0
            results["predictive_parity"] = overall_fairness

            results.update(
                {
                    "n_samples": float(validation["n_samples"]),
                    "tolerance": self.tolerance,
                    "n_sensitive_attributes": len(sensitive_features.columns),
                },
            )

            return results

        except Exception as e:
            logger.error(f"Error computing predictive parity: {e}")
            return {"predictive_parity": 0.0, "error": str(e)}

    def requires_sensitive_features(self) -> bool:
        """Predictive parity requires sensitive features."""
        return True

    def get_metric_names(self) -> list[str]:
        """Return metric names (dynamic based on sensitive features)."""
        return ["predictive_parity", "n_samples", "tolerance", "n_sensitive_attributes"]


# Auto-register fairness metrics
logger.debug("Fairness metrics registered")
