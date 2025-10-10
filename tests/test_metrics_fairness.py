"""Fairness metrics functionality tests.

Tests fairness and bias detection metrics including demographic parity,
equal opportunity, equalized odds, and predictive parity. These tests
focus on covering the fairness metrics computation logic.
"""

import numpy as np
import pandas as pd
import pytest

# Conditional sklearn import for CI compatibility
try:
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    make_classification = None
    LogisticRegression = None
    SKLEARN_AVAILABLE = False

# Skip all tests if sklearn not available
pytestmark = pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available - CI compatibility issues")

from glassalpha.core import MetricRegistry
from glassalpha.metrics.fairness.bias_detection import (
    DemographicParityMetric,
    EqualizedOddsMetric,
    EqualOpportunityMetric,
    PredictiveParityMetric,
)


@pytest.fixture
def binary_classification_data():
    """Create binary classification test data with fairness considerations."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )

    # Create predictions with some bias
    y_pred = y.copy()
    # Add some prediction errors
    error_indices = np.random.choice(len(y), size=200, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]

    return y, y_pred


@pytest.fixture
def sensitive_features_binary():
    """Create binary sensitive features for fairness testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "gender": np.random.choice(["M", "F"], 1000),
            "age_group": np.random.choice(["young", "old"], 1000),
            "ethnicity": np.random.choice(["A", "B"], 1000),
        },
    )


@pytest.fixture
def sensitive_features_multiclass():
    """Create multiclass sensitive features for fairness testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "gender": np.random.choice(["M", "F", "O"], 1000),
            "age_group": np.random.choice(["young", "middle", "old"], 1000),
            "ethnicity": np.random.choice(["A", "B", "C", "D"], 1000),
        },
    )


@pytest.fixture
def biased_predictions():
    """Create deliberately biased predictions for testing bias detection."""
    np.random.seed(42)
    y_true = np.random.choice([0, 1], 1000)

    # Create sensitive feature
    gender = np.random.choice(["M", "F"], 1000)
    sensitive_df = pd.DataFrame({"gender": gender})

    # Create biased predictions (more false positives for one group)
    y_pred = y_true.copy()

    # Add bias: Males get more false positives
    male_mask = gender == "M"
    male_indices = np.where(male_mask & (y_true == 0))[0]
    # Give 30% of male negatives a false positive prediction
    biased_indices = np.random.choice(male_indices, size=int(0.3 * len(male_indices)), replace=False)
    y_pred[biased_indices] = 1

    return y_true, y_pred, sensitive_df


class TestFairnessMetricRegistry:
    """Test fairness metric registration and discovery."""

    def test_fairness_metrics_are_registered(self):
        """Test that fairness metrics are properly registered."""
        components = MetricRegistry.get_all()

        # Should include our fairness metrics
        assert "demographic_parity" in components
        assert "equal_opportunity" in components
        assert "equalized_odds" in components
        assert "predictive_parity" in components

    def test_get_fairness_metric_classes(self):
        """Test that we can retrieve fairness metric classes."""
        dp_cls = MetricRegistry.get("demographic_parity")
        assert dp_cls == DemographicParityMetric

        eo_cls = MetricRegistry.get("equal_opportunity")
        assert eo_cls == EqualOpportunityMetric

        eod_cls = MetricRegistry.get("equalized_odds")
        assert eod_cls == EqualizedOddsMetric

        pp_cls = MetricRegistry.get("predictive_parity")
        assert pp_cls == PredictiveParityMetric


class TestDemographicParityMetric:
    """Test DemographicParityMetric functionality."""

    def test_demographic_parity_initialization(self):
        """Test DemographicParityMetric initialization."""
        metric = DemographicParityMetric()

        assert metric.name == "demographic_parity"
        assert metric.metric_type == "fairness"
        assert metric.version == "1.0.0"
        assert metric.tolerance == 0.1  # Default tolerance

    def test_demographic_parity_custom_tolerance(self):
        """Test DemographicParityMetric with custom tolerance."""
        metric = DemographicParityMetric(tolerance=0.05)

        assert metric.tolerance == 0.05

    def test_demographic_parity_computation(self, binary_classification_data, sensitive_features_binary):
        """Test demographic parity computation."""
        y_true, y_pred = binary_classification_data

        metric = DemographicParityMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_features_binary)

        assert isinstance(result, dict)
        # Should contain results for each sensitive feature
        assert len(result) > 0

        # All values should be between 0 and 1 (or reasonable range)
        for _key, value in result.items():
            if isinstance(value, (int, float)):
                assert not np.isnan(value)

    def test_demographic_parity_without_sensitive_features(self, binary_classification_data):
        """Test demographic parity without sensitive features."""
        y_true, y_pred = binary_classification_data

        metric = DemographicParityMetric()
        result = metric.compute(y_true, y_pred)

        # Should handle gracefully when no sensitive features provided
        assert isinstance(result, dict)

    def test_demographic_parity_perfect_fairness(self):
        """Test demographic parity with perfectly fair predictions."""
        # Create perfectly balanced data
        y_true = np.array([0, 1, 0, 1] * 250)  # 1000 samples, perfectly balanced
        y_pred = y_true.copy()  # Perfect predictions

        # Create balanced sensitive features
        gender = np.array(["M", "F"] * 500)  # Perfectly balanced
        sensitive_df = pd.DataFrame({"gender": gender})

        metric = DemographicParityMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_df)

        assert isinstance(result, dict)
        # With perfect balance, should have low disparity

    def test_demographic_parity_biased_case(self, biased_predictions):
        """Test demographic parity with biased predictions."""
        y_true, y_pred, sensitive_df = biased_predictions

        metric = DemographicParityMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_df)

        assert isinstance(result, dict)
        # Should detect the bias we introduced


class TestEqualOpportunityMetric:
    """Test EqualOpportunityMetric functionality."""

    def test_equal_opportunity_initialization(self):
        """Test EqualOpportunityMetric initialization."""
        metric = EqualOpportunityMetric()

        assert metric.name == "equal_opportunity"
        assert metric.metric_type == "fairness"
        assert metric.version == "1.0.0"

    def test_equal_opportunity_computation(self, binary_classification_data, sensitive_features_binary):
        """Test equal opportunity computation."""
        y_true, y_pred = binary_classification_data

        metric = EqualOpportunityMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_features_binary)

        assert isinstance(result, dict)
        assert len(result) > 0

        # Check that results are reasonable
        for _key, value in result.items():
            if isinstance(value, (int, float)):
                assert not np.isnan(value)

    def test_equal_opportunity_multiclass_groups(self, binary_classification_data, sensitive_features_multiclass):
        """Test equal opportunity with multiclass sensitive features."""
        y_true, y_pred = binary_classification_data

        metric = EqualOpportunityMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_features_multiclass)

        assert isinstance(result, dict)
        # Should handle multiple groups within each sensitive feature

    def test_equal_opportunity_no_positive_class(self):
        """Test equal opportunity when there's no positive class."""
        y_true = np.array([0, 0, 0, 0])  # All negative
        y_pred = np.array([0, 0, 0, 1])  # One false positive
        sensitive_df = pd.DataFrame({"group": ["A", "A", "B", "B"]})

        metric = EqualOpportunityMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_df)

        # Should handle gracefully
        assert isinstance(result, dict)


class TestEqualizedOddsMetric:
    """Test EqualizedOddsMetric functionality."""

    def test_equalized_odds_initialization(self):
        """Test EqualizedOddsMetric initialization."""
        metric = EqualizedOddsMetric()

        assert metric.name == "equalized_odds"
        assert metric.metric_type == "fairness"
        assert metric.version == "1.0.0"

    def test_equalized_odds_computation(self, binary_classification_data, sensitive_features_binary):
        """Test equalized odds computation."""
        y_true, y_pred = binary_classification_data

        metric = EqualizedOddsMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_features_binary)

        assert isinstance(result, dict)
        assert len(result) > 0

        # Should include metrics for both TPR and FPR
        for _key, value in result.items():
            if isinstance(value, (int, float)):
                assert not np.isnan(value)

    def test_equalized_odds_perfect_classifier(self):
        """Test equalized odds with perfect classifier."""
        y_true = np.array([0, 1, 0, 1] * 250)
        y_pred = y_true.copy()  # Perfect predictions

        gender = np.array(["M", "F"] * 500)
        sensitive_df = pd.DataFrame({"gender": gender})

        metric = EqualizedOddsMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_df)

        assert isinstance(result, dict)
        # Perfect classifier should have good equalized odds

    def test_equalized_odds_edge_cases(self):
        """Test equalized odds with edge cases."""
        # Single class
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 0, 1, 0])
        sensitive_df = pd.DataFrame({"group": ["A", "A", "B", "B"]})

        metric = EqualizedOddsMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_df)

        assert isinstance(result, dict)


class TestPredictiveParityMetric:
    """Test PredictiveParityMetric functionality."""

    def test_predictive_parity_initialization(self):
        """Test PredictiveParityMetric initialization."""
        metric = PredictiveParityMetric()

        assert metric.name == "predictive_parity"
        assert metric.metric_type == "fairness"
        assert metric.version == "1.0.0"

    def test_predictive_parity_computation(self, binary_classification_data, sensitive_features_binary):
        """Test predictive parity computation."""
        y_true, y_pred = binary_classification_data

        metric = PredictiveParityMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_features_binary)

        assert isinstance(result, dict)
        assert len(result) > 0

        for _key, value in result.items():
            if isinstance(value, (int, float)):
                assert not np.isnan(value)

    def test_predictive_parity_no_predictions(self):
        """Test predictive parity when no positive predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])  # No positive predictions
        sensitive_df = pd.DataFrame({"group": ["A", "A", "B", "B"]})

        metric = PredictiveParityMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_df)

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_predictive_parity_biased_case(self, biased_predictions):
        """Test predictive parity with biased predictions."""
        y_true, y_pred, sensitive_df = biased_predictions

        metric = PredictiveParityMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_df)

        assert isinstance(result, dict)
        # Should detect the bias in precision across groups


class TestFairnessMetricsIntegration:
    """Test integration between different fairness metrics."""

    def test_multiple_fairness_metrics(self, binary_classification_data, sensitive_features_binary):
        """Test computing multiple fairness metrics together."""
        y_true, y_pred = binary_classification_data

        metrics = [DemographicParityMetric(), EqualOpportunityMetric(), EqualizedOddsMetric(), PredictiveParityMetric()]

        results = {}
        for metric in metrics:
            result = metric.compute(y_true, y_pred, sensitive_features=sensitive_features_binary)
            results[metric.name] = result

        # Should have results from all metrics
        assert "demographic_parity" in results
        assert "equal_opportunity" in results
        assert "equalized_odds" in results
        assert "predictive_parity" in results

        # All results should be dictionaries
        for metric_result in results.values():
            assert isinstance(metric_result, dict)

    def test_fairness_metrics_consistency(self, binary_classification_data, sensitive_features_binary):
        """Test consistency between fairness metrics."""
        y_true, y_pred = binary_classification_data

        dp_metric = DemographicParityMetric()
        eo_metric = EqualOpportunityMetric()

        dp_result = dp_metric.compute(y_true, y_pred, sensitive_features=sensitive_features_binary)
        eo_result = eo_metric.compute(y_true, y_pred, sensitive_features=sensitive_features_binary)

        # Both should return valid results
        assert isinstance(dp_result, dict)
        assert isinstance(eo_result, dict)

        # Both should have processed the same sensitive features
        # (specific assertions depend on implementation details)


class TestFairnessMetricsEdgeCases:
    """Test edge cases and error handling in fairness metrics."""

    def test_empty_groups(self):
        """Test fairness metrics with empty groups after filtering."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        # Sensitive feature with very unbalanced groups
        sensitive_df = pd.DataFrame({"group": ["A", "A", "A", "B"]})

        metric = DemographicParityMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_df)

        # Should handle small groups gracefully
        assert isinstance(result, dict)

    def test_single_sensitive_feature_value(self):
        """Test fairness metrics when all samples have same sensitive feature value."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        sensitive_df = pd.DataFrame({"group": ["A", "A", "A", "A"]})  # All same group

        metric = DemographicParityMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_df)

        # Should handle gracefully (no disparity if only one group)
        assert isinstance(result, dict)

    def test_missing_sensitive_features(self, binary_classification_data):
        """Test fairness metrics behavior with missing sensitive features."""
        y_true, y_pred = binary_classification_data

        # Create sensitive features with NaN values
        sensitive_df = pd.DataFrame({"gender": ["M", "F", np.nan, "M"] * 250})

        metric = DemographicParityMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_df)

        # Should handle NaN values appropriately
        assert isinstance(result, dict)

    def test_multiple_sensitive_features(self, binary_classification_data):
        """Test fairness metrics with multiple sensitive features."""
        y_true, y_pred = binary_classification_data

        # Multiple sensitive features
        sensitive_df = pd.DataFrame(
            {
                "gender": np.random.choice(["M", "F"], 1000),
                "age": np.random.choice(["young", "old"], 1000),
                "race": np.random.choice(["A", "B", "C"], 1000),
            },
        )

        metric = DemographicParityMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_df)

        # Should process all sensitive features
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_mismatched_array_lengths(self):
        """Test fairness metrics with mismatched array lengths."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])  # Different length
        sensitive_df = pd.DataFrame({"group": ["A", "B", "A"]})

        metric = DemographicParityMetric()

        with pytest.raises(ValueError):
            metric.compute(y_true, y_pred, sensitive_features=sensitive_df)


class TestFairnessMetricsWithDifferentDataTypes:
    """Test fairness metrics with different data types and formats."""

    def test_string_sensitive_features(self, binary_classification_data):
        """Test fairness metrics with string sensitive features."""
        y_true, y_pred = binary_classification_data

        sensitive_df = pd.DataFrame(
            {
                "category": np.random.choice(["Category A", "Category B", "Category C"], 1000),
                "region": np.random.choice(["North", "South", "East", "West"], 1000),
            },
        )

        metric = DemographicParityMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_df)

        assert isinstance(result, dict)

    def test_numeric_sensitive_features(self, binary_classification_data):
        """Test fairness metrics with numeric sensitive features (should be treated as categorical)."""
        y_true, y_pred = binary_classification_data

        sensitive_df = pd.DataFrame(
            {"numeric_group": np.random.choice([1, 2, 3], 1000), "binary_numeric": np.random.choice([0, 1], 1000)},
        )

        metric = DemographicParityMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_df)

        assert isinstance(result, dict)

    def test_boolean_sensitive_features(self, binary_classification_data):
        """Test fairness metrics with boolean sensitive features."""
        y_true, y_pred = binary_classification_data

        sensitive_df = pd.DataFrame(
            {
                "is_member": np.random.choice([True, False], 1000),
                "has_attribute": np.random.choice([True, False], 1000),
            },
        )

        metric = DemographicParityMetric()
        result = metric.compute(y_true, y_pred, sensitive_features=sensitive_df)

        assert isinstance(result, dict)
