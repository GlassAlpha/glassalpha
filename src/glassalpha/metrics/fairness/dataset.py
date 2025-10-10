"""Dataset-level bias audit metrics (E12).

Detects bias at the data source level before model training:
- Proxy correlation: Protected attributes correlating with non-protected features
- Distribution drift: Train/test feature distribution differences
- Sampling bias power: Statistical power to detect undersampling
- Continuous binning: Configurable binning for continuous protected attributes
- Split imbalance: Train/test protected group distribution differences
"""

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ConstantInputWarning, chi2_contingency, ks_2samp

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class ProxyCorrelationResult:
    """Results from proxy correlation analysis."""

    correlations: dict[str, dict[str, dict[str, Any]]]
    """
    Nested dict: {protected_attr: {feature: {correlation, p_value, severity}}}
    Severity levels: ERROR (|r|>0.5), WARNING (0.3<|r|≤0.5), INFO (|r|≤0.3)
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {"correlations": self.correlations}


@dataclass
class DistributionDriftResult:
    """Results from distribution drift analysis."""

    drift_tests: dict[str, dict[str, Any]]
    """
    Dict: {feature: {test_type, statistic, p_value, drifted}}
    test_type: "ks_test" for continuous, "chi2_test" for categorical
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {"drift_tests": self.drift_tests}


@dataclass
class SamplingBiasResult:
    """Results from sampling bias power analysis."""

    power_by_group: dict[str, dict[str, dict[str, Any]]]
    """
    Nested dict: {protected_attr: {group: {n, power, severity}}}
    Severity: ERROR (power<0.5), WARNING (0.5≤power<0.7), OK (power≥0.7)
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {"power_by_group": self.power_by_group}


@dataclass
class SplitImbalanceResult:
    """Results from train/test split imbalance analysis."""

    imbalance_tests: dict[str, dict[str, Any]]
    """
    Dict: {protected_attr: {p_value, imbalanced, train_distribution, test_distribution}}
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {"imbalance_tests": self.imbalance_tests}


@dataclass
class BinnedAttribute:
    """Result of binning a continuous attribute."""

    attr_name: str
    strategy: str
    bins: list[float]
    categories: list[str]
    binned_values: pd.Series

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "attr_name": self.attr_name,
            "strategy": self.strategy,
            "bins": self.bins,
            "categories": self.categories,
        }


@dataclass
class DatasetBiasMetrics:
    """Complete dataset-level bias audit results."""

    proxy_correlations: ProxyCorrelationResult | None = None
    distribution_drift: DistributionDriftResult | None = None
    sampling_bias_power: SamplingBiasResult | None = None
    split_imbalance: SplitImbalanceResult | None = None
    protected_attr_binning: dict[str, BinnedAttribute] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {}

        if self.proxy_correlations:
            result["proxy_correlations"] = self.proxy_correlations.to_dict()
        if self.distribution_drift:
            result["distribution_drift"] = self.distribution_drift.to_dict()
        if self.sampling_bias_power:
            result["sampling_bias_power"] = self.sampling_bias_power.to_dict()
        if self.split_imbalance:
            result["split_imbalance"] = self.split_imbalance.to_dict()
        if self.protected_attr_binning:
            result["protected_attr_binning"] = {k: v.to_dict() for k, v in self.protected_attr_binning.items()}

        return result


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _cramers_v(contingency_table: np.ndarray) -> float:
    """Compute Cramér's V statistic for categorical association.

    Args:
        contingency_table: 2D array of category counts

    Returns:
        Cramér's V (0 = no association, 1 = perfect association)

    """
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum()
    min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)

    if n == 0 or min_dim == 0:
        return 0.0

    return np.sqrt(chi2 / (n * min_dim))


# ============================================================================
# PROXY CORRELATION DETECTION
# ============================================================================


def compute_proxy_correlations(
    data: pd.DataFrame,
    protected_attrs: list[str],
    feature_cols: list[str],
    seed: int = 42,
) -> ProxyCorrelationResult:
    """Detect proxy features that correlate with protected attributes.

    Uses:
    - Pearson correlation for continuous-continuous
    - Point-biserial for continuous-binary
    - Cramér's V for categorical-categorical

    Severity levels:
    - ERROR: |r| > 0.5 (strong correlation, likely indirect discrimination)
    - WARNING: 0.3 < |r| ≤ 0.5 (medium correlation, investigate)
    - INFO: |r| ≤ 0.3 (weak correlation, document)

    Args:
        data: Full dataset
        protected_attrs: List of protected attribute column names
        feature_cols: List of non-protected feature column names
        seed: Random seed for reproducibility

    Returns:
        ProxyCorrelationResult with correlations and severity levels

    """
    np.random.seed(seed)

    correlations = {}

    for protected_attr in protected_attrs:
        correlations[protected_attr] = {}

        for feature in feature_cols:
            # Skip if either column has all NaN
            if data[protected_attr].isna().all() or data[feature].isna().all():
                continue

            # Determine correlation method based on types
            protected_is_numeric = pd.api.types.is_numeric_dtype(data[protected_attr])
            feature_is_numeric = pd.api.types.is_numeric_dtype(data[feature])

            if protected_is_numeric and feature_is_numeric:
                # Pearson correlation for continuous-continuous
                # Suppress ConstantInputWarning - we handle constant inputs gracefully
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConstantInputWarning)
                    corr, p_value = stats.pearsonr(
                        data[protected_attr].dropna(),
                        data[feature].dropna(),
                    )
            elif not protected_is_numeric and not feature_is_numeric:
                # Cramér's V for categorical-categorical
                contingency_table = pd.crosstab(data[protected_attr], data[feature])
                chi2, p_value, _, _ = chi2_contingency(contingency_table)

                # Cramér's V
                corr = _cramers_v(contingency_table.values)
            else:
                # Point-biserial for mixed types
                # Convert categorical to numeric codes
                if not protected_is_numeric:
                    protected_numeric = pd.Categorical(data[protected_attr]).codes
                    feature_numeric = data[feature]
                else:
                    protected_numeric = data[protected_attr]
                    feature_numeric = pd.Categorical(data[feature]).codes

                corr, p_value = stats.pearsonr(
                    protected_numeric[~np.isnan(protected_numeric) & ~np.isnan(feature_numeric)],
                    feature_numeric[~np.isnan(protected_numeric) & ~np.isnan(feature_numeric)],
                )

            # Determine severity
            abs_corr = abs(corr)
            if abs_corr > 0.5:
                severity = "ERROR"
            elif abs_corr > 0.3:
                severity = "WARNING"
            else:
                severity = "INFO"

            correlations[protected_attr][feature] = {
                "correlation": float(corr),
                "p_value": float(p_value),
                "severity": severity,
            }

    return ProxyCorrelationResult(correlations=correlations)


# ============================================================================
# DISTRIBUTION DRIFT DETECTION
# ============================================================================


def compute_distribution_drift(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: list[str],
    seed: int = 42,
    alpha: float = 0.05,
) -> DistributionDriftResult:
    """Detect distribution drift between train and test sets.

    Uses:
    - Kolmogorov-Smirnov test for continuous features
    - Chi-square test for categorical features

    Args:
        train_data: Training dataset
        test_data: Test dataset
        feature_cols: List of feature column names to check
        seed: Random seed for reproducibility
        alpha: Significance level (default 0.05)

    Returns:
        DistributionDriftResult with drift tests

    """
    np.random.seed(seed)

    drift_tests = {}

    for feature in feature_cols:
        if feature not in train_data.columns or feature not in test_data.columns:
            continue

        train_values = train_data[feature].dropna()
        test_values = test_data[feature].dropna()

        if len(train_values) == 0 or len(test_values) == 0:
            continue

        # Determine test type based on dtype
        is_numeric = pd.api.types.is_numeric_dtype(train_data[feature])

        if is_numeric:
            # KS test for continuous features
            statistic, p_value = ks_2samp(train_values, test_values)
            test_type = "ks_test"
        else:
            # Chi-square test for categorical features
            # Create contingency table
            train_counts = train_values.value_counts()
            test_counts = test_values.value_counts()

            # Align categories
            all_categories = sorted(set(train_counts.index) | set(test_counts.index))
            train_aligned = [train_counts.get(cat, 0) for cat in all_categories]
            test_aligned = [test_counts.get(cat, 0) for cat in all_categories]

            contingency_table = np.array([train_aligned, test_aligned])

            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            statistic = chi2
            test_type = "chi2_test"

        drift_tests[feature] = {
            "test_type": test_type,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drifted": bool(p_value < alpha),
        }

    return DistributionDriftResult(drift_tests=drift_tests)


# ============================================================================
# SAMPLING BIAS POWER ANALYSIS
# ============================================================================


def compute_sampling_bias_power(
    data: pd.DataFrame,
    protected_attrs: list[str],
    target_effect_size: float = 0.1,
    seed: int = 42,
    alpha: float = 0.05,
) -> SamplingBiasResult:
    """Compute statistical power to detect sampling bias.

    For each protected group, computes power to detect a target_effect_size
    difference in representation (e.g., 10% undersampling).

    Severity levels:
    - ERROR: power < 0.5 (severely underpowered)
    - WARNING: 0.5 ≤ power < 0.7 (inadequate power)
    - OK: power ≥ 0.7 (adequate power)

    Args:
        data: Full dataset
        protected_attrs: List of protected attribute column names
        target_effect_size: Minimum effect size to detect (default 0.1 = 10%)
        seed: Random seed for reproducibility
        alpha: Significance level (default 0.05)

    Returns:
        SamplingBiasResult with power by group

    """
    np.random.seed(seed)

    power_by_group = {}

    for protected_attr in protected_attrs:
        power_by_group[protected_attr] = {}

        # Get group counts
        value_counts = data[protected_attr].value_counts()
        total_n = len(data[protected_attr].dropna())

        for group, n in value_counts.items():
            # Compute expected proportion
            expected_prop = n / total_n

            # Alternative hypothesis: proportion differs by target_effect_size
            alternative_prop = max(0, min(1, expected_prop - target_effect_size))

            # Approximate power using normal approximation for proportion test
            # Standard error under null hypothesis
            se_null = np.sqrt(expected_prop * (1 - expected_prop) / n)

            # Standard error under alternative hypothesis
            se_alt = np.sqrt(alternative_prop * (1 - alternative_prop) / n)

            # Critical value (two-tailed test)
            z_crit = stats.norm.ppf(1 - alpha / 2)

            # Effect size
            effect = abs(expected_prop - alternative_prop)

            # Power calculation: probability of detecting effect when it exists
            # Under alternative, test statistic is shifted by effect/se_alt
            if se_alt > 0 and effect > 0:
                # Non-centrality parameter
                ncp = effect / se_alt
                # Power for two-tailed test
                power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
                # Clamp to [0, 1]
                power = max(0.0, min(1.0, power))
            else:
                power = 0.0

            # Determine severity
            if power < 0.5:
                severity = "ERROR"
            elif power < 0.7:
                severity = "WARNING"
            else:
                severity = "OK"

            power_by_group[protected_attr][str(group)] = {
                "n": int(n),
                "expected_proportion": float(expected_prop),
                "power": float(power),
                "severity": severity,
            }

    return SamplingBiasResult(power_by_group=power_by_group)


# ============================================================================
# CONTINUOUS ATTRIBUTE BINNING
# ============================================================================


# Default domain-specific bins for common protected attributes
DEFAULT_BINS = {
    "age": [18, 25, 35, 50, 65, 100],
    "age_years": [18, 25, 35, 50, 65, 100],
    "income": None,  # Use quartiles
}


def bin_continuous_attribute(
    values: np.ndarray | pd.Series,
    attr_name: str,
    strategy: str = "domain",
    bins: list[float] | None = None,
    n_bins: int = 4,
) -> BinnedAttribute:
    """Bin continuous protected attribute for group analysis.

    Strategies:
    - "domain": Use domain-specific bins (e.g., age brackets)
    - "custom": Use user-provided bins
    - "equal_width": Equal-width bins
    - "equal_frequency": Equal-frequency (quantile) bins

    Args:
        values: Continuous values to bin
        attr_name: Attribute name (used for domain defaults)
        strategy: Binning strategy
        bins: Custom bin edges (required for "custom" strategy)
        n_bins: Number of bins for equal_width/equal_frequency

    Returns:
        BinnedAttribute with binning info and binned values

    """
    if isinstance(values, pd.Series):
        values = values.values

    # Remove NaN
    valid_values = values[~np.isnan(values)]

    if strategy == "domain":
        # Use default domain bins if available
        if attr_name in DEFAULT_BINS:
            bins = DEFAULT_BINS[attr_name]
            if bins is None:
                # Use quartiles
                bins = np.percentile(valid_values, [0, 25, 50, 75, 100]).tolist()
        else:
            # Fall back to equal-width
            bins = np.linspace(valid_values.min(), valid_values.max(), n_bins + 1).tolist()

    elif strategy == "custom":
        if bins is None:
            raise ValueError("Custom strategy requires bins parameter")

    elif strategy == "equal_width":
        bins = np.linspace(valid_values.min(), valid_values.max(), n_bins + 1).tolist()

    elif strategy == "equal_frequency":
        bins = np.percentile(valid_values, np.linspace(0, 100, n_bins + 1)).tolist()

    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    # Create categories
    categories = []
    for i in range(len(bins) - 1):
        categories.append(f"[{bins[i]:.1f}, {bins[i + 1]:.1f})")

    # Bin the values
    binned = pd.cut(values, bins=bins, labels=categories, include_lowest=True)

    return BinnedAttribute(
        attr_name=attr_name,
        strategy=strategy,
        bins=bins,
        categories=categories,
        binned_values=binned,
    )


# ============================================================================
# TRAIN/TEST SPLIT IMBALANCE DETECTION
# ============================================================================


def detect_split_imbalance(
    data: pd.DataFrame,
    protected_attrs: list[str],
    split_col: str = "split",
    seed: int = 42,
    alpha: float = 0.05,
) -> SplitImbalanceResult:
    """Detect imbalanced train/test splits by protected group.

    Uses Chi-square test to detect if protected group distributions
    differ significantly between train and test sets.

    Args:
        data: Full dataset with split indicator
        protected_attrs: List of protected attribute column names
        split_col: Column indicating train/test split
        seed: Random seed for reproducibility
        alpha: Significance level (default 0.05)

    Returns:
        SplitImbalanceResult with imbalance tests

    """
    np.random.seed(seed)

    imbalance_tests = {}

    for protected_attr in protected_attrs:
        # Create contingency table: split x protected_attr
        contingency_table = pd.crosstab(data[split_col], data[protected_attr])

        # Skip if missing train or test split
        if "train" not in contingency_table.index or "test" not in contingency_table.index:
            continue

        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # Compute train/test distributions
        train_dist = contingency_table.loc["train"].to_dict()
        test_dist = contingency_table.loc["test"].to_dict()

        # Normalize to proportions
        train_total = sum(train_dist.values())
        test_total = sum(test_dist.values())

        train_dist = {k: v / train_total for k, v in train_dist.items()}
        test_dist = {k: v / test_total for k, v in test_dist.items()}

        imbalance_tests[protected_attr] = {
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "imbalanced": bool(p_value < alpha),
            "train_distribution": {str(k): float(v) for k, v in train_dist.items()},
            "test_distribution": {str(k): float(v) for k, v in test_dist.items()},
        }

    return SplitImbalanceResult(imbalance_tests=imbalance_tests)


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def compute_dataset_bias_metrics(
    data: pd.DataFrame,
    protected_attrs: list[str],
    feature_cols: list[str],
    split_col: str = "split",
    seed: int = 42,
    compute_proxy: bool = True,
    compute_drift: bool = True,
    compute_power: bool = True,
    compute_imbalance: bool = True,
) -> DatasetBiasMetrics:
    """Compute all dataset-level bias metrics.

    Args:
        data: Full dataset with split indicator
        protected_attrs: List of protected attribute column names
        feature_cols: List of non-protected feature column names
        split_col: Column indicating train/test split
        seed: Random seed for reproducibility
        compute_proxy: Compute proxy correlation analysis
        compute_drift: Compute distribution drift analysis
        compute_power: Compute sampling bias power analysis
        compute_imbalance: Compute split imbalance analysis

    Returns:
        DatasetBiasMetrics with all computed metrics

    Raises:
        ValueError: If protected_attrs is empty

    """
    if not protected_attrs:
        raise ValueError("protected_attrs cannot be empty")

    result = DatasetBiasMetrics()

    # Proxy correlations
    if compute_proxy and feature_cols:
        result.proxy_correlations = compute_proxy_correlations(
            data=data,
            protected_attrs=protected_attrs,
            feature_cols=feature_cols,
            seed=seed,
        )

    # Distribution drift (requires split column)
    if compute_drift and split_col in data.columns and feature_cols:
        train_data = data[data[split_col] == "train"]
        test_data = data[data[split_col] == "test"]

        if len(train_data) > 0 and len(test_data) > 0:
            result.distribution_drift = compute_distribution_drift(
                train_data=train_data,
                test_data=test_data,
                feature_cols=feature_cols,
                seed=seed,
            )

    # Sampling bias power
    if compute_power:
        result.sampling_bias_power = compute_sampling_bias_power(
            data=data,
            protected_attrs=protected_attrs,
            target_effect_size=0.1,
            seed=seed,
        )

    # Split imbalance
    if compute_imbalance and split_col in data.columns:
        result.split_imbalance = detect_split_imbalance(
            data=data,
            protected_attrs=protected_attrs,
            split_col=split_col,
            seed=seed,
        )

    return result
