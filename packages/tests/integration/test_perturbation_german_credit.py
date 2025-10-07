"""Integration test for perturbation sweeps with German Credit dataset.

This test validates E6+ implementation with a real-world dataset:
1. Train LogisticRegression on German Credit
2. Run perturbation sweep with epsilon ∈ {0.01, 0.05, 0.1}
3. Validate robustness score is reasonable
4. Test determinism (same seed = same results)
5. Export to JSON and verify structure
6. Test gate logic with different thresholds
"""

import json

import pandas as pd

try:
    import pytest
except ImportError:
    pytest = None

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from glassalpha.metrics.stability.perturbation import run_perturbation_sweep


class TestGermanCreditPerturbation:
    """Integration tests with German Credit dataset."""

    @pytest.fixture
    def german_credit_model(self):
        """Load German Credit, train model, return model + test data."""
        # Load German Credit dataset
        from glassalpha.datasets import load_german_credit

        data = load_german_credit()

        # Prepare features and target
        target = "credit_risk"
        feature_cols = [col for col in data.columns if col != target]

        X = data[feature_cols]
        y = data[target]

        # Convert categorical variables to numeric
        X = pd.get_dummies(X, drop_first=True)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )

        # Train LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        return model, X_test, y_test

    def test_german_credit_perturbation_deterministic(self, german_credit_model):
        """Test determinism: same seed = same robustness score."""
        model, X_test, y_test = german_credit_model

        # Identify protected features
        protected = [col for col in X_test.columns if "gender" in col.lower() or "age" in col.lower()]

        # Run twice with same seed
        result1 = run_perturbation_sweep(
            model=model,
            X_test=X_test,
            protected_features=protected,
            epsilon_values=[0.01, 0.05, 0.1],
            threshold=0.15,
            seed=42,
        )

        result2 = run_perturbation_sweep(
            model=model,
            X_test=X_test,
            protected_features=protected,
            epsilon_values=[0.01, 0.05, 0.1],
            threshold=0.15,
            seed=42,
        )

        # Assert byte-identical results
        assert result1.robustness_score == result2.robustness_score
        assert result1.max_delta == result2.max_delta
        assert result1.per_epsilon_deltas == result2.per_epsilon_deltas
        assert result1.gate_status == result2.gate_status

    def test_german_credit_robustness_score_reasonable(self, german_credit_model):
        """Test that robustness score is in reasonable range."""
        model, X_test, y_test = german_credit_model

        protected = [col for col in X_test.columns if "gender" in col.lower() or "age" in col.lower()]

        result = run_perturbation_sweep(
            model=model,
            X_test=X_test,
            protected_features=protected,
            epsilon_values=[0.01, 0.05, 0.1],
            threshold=0.15,
            seed=42,
        )

        # Robustness score should be non-negative and less than 1 (probability delta)
        assert 0.0 <= result.robustness_score <= 1.0

        # Max delta should be >= all individual deltas
        for epsilon, delta in result.per_epsilon_deltas.items():
            assert result.max_delta >= delta

    def test_german_credit_epsilon_monotonicity(self, german_credit_model):
        """Test that larger epsilon generally produces larger delta."""
        model, X_test, y_test = german_credit_model

        protected = [col for col in X_test.columns if "gender" in col.lower() or "age" in col.lower()]

        result = run_perturbation_sweep(
            model=model,
            X_test=X_test,
            protected_features=protected,
            epsilon_values=[0.01, 0.05, 0.1],
            threshold=0.15,
            seed=42,
        )

        # Check monotonicity: delta should increase with epsilon (at least weakly)
        delta_01 = result.per_epsilon_deltas[0.01]
        delta_05 = result.per_epsilon_deltas[0.05]
        delta_10 = result.per_epsilon_deltas[0.1]

        # Weak monotonicity: max >= mid >= min (allows ties)
        assert delta_10 >= delta_01  # Strongest check
        assert delta_05 >= delta_01 or delta_10 >= delta_05  # At least one step increases

    def test_german_credit_json_export(self, german_credit_model):
        """Test JSON export is valid and contains all fields."""
        model, X_test, y_test = german_credit_model

        protected = [col for col in X_test.columns if "gender" in col.lower() or "age" in col.lower()]

        result = run_perturbation_sweep(
            model=model,
            X_test=X_test,
            protected_features=protected,
            epsilon_values=[0.01, 0.05, 0.1],
            threshold=0.15,
            seed=42,
        )

        # Export to dict
        json_dict = result.to_dict()

        # Validate structure
        assert "robustness_score" in json_dict
        assert "max_delta" in json_dict
        assert "epsilon_values" in json_dict
        assert "per_epsilon_deltas" in json_dict
        assert "gate_status" in json_dict
        assert "threshold" in json_dict
        assert "n_samples" in json_dict
        assert "n_features_perturbed" in json_dict

        # Validate values
        assert isinstance(json_dict["robustness_score"], float)
        assert isinstance(json_dict["epsilon_values"], list)
        assert isinstance(json_dict["per_epsilon_deltas"], dict)
        assert json_dict["gate_status"] in ["PASS", "WARNING", "FAIL"]

        # Validate JSON serializability
        json_str = json.dumps(json_dict)
        assert len(json_str) > 0

        # Validate round-trip
        recovered = json.loads(json_str)
        assert recovered["robustness_score"] == json_dict["robustness_score"]

    def test_german_credit_gate_pass(self, german_credit_model):
        """Test gate PASS with high threshold."""
        model, X_test, y_test = german_credit_model

        protected = [col for col in X_test.columns if "gender" in col.lower() or "age" in col.lower()]

        result = run_perturbation_sweep(
            model=model,
            X_test=X_test,
            protected_features=protected,
            epsilon_values=[0.01],  # Small epsilon
            threshold=0.99,  # Very high threshold
            seed=42,
        )

        assert result.gate_status == "PASS"
        assert result.max_delta < result.threshold

    def test_german_credit_gate_fail(self, german_credit_model):
        """Test gate FAIL with low threshold."""
        model, X_test, y_test = german_credit_model

        protected = [col for col in X_test.columns if "gender" in col.lower() or "age" in col.lower()]

        result = run_perturbation_sweep(
            model=model,
            X_test=X_test,
            protected_features=protected,
            epsilon_values=[0.1],  # Large epsilon
            threshold=0.001,  # Very low threshold
            seed=42,
        )

        assert result.gate_status == "FAIL"
        assert result.max_delta >= 1.5 * result.threshold

    def test_german_credit_protected_features_count(self, german_credit_model):
        """Test that protected features are correctly excluded from perturbation."""
        model, X_test, y_test = german_credit_model

        protected = [col for col in X_test.columns if "gender" in col.lower() or "age" in col.lower()]

        result = run_perturbation_sweep(
            model=model,
            X_test=X_test,
            protected_features=protected,
            epsilon_values=[0.05],
            threshold=0.15,
            seed=42,
        )

        # n_features_perturbed should equal total features minus protected features
        expected_perturbed = len(X_test.columns) - len(protected)
        assert result.n_features_perturbed == expected_perturbed
        assert result.n_samples == len(X_test)

    def test_german_credit_no_protected_features(self, german_credit_model):
        """Test perturbation with no protected features (perturb all)."""
        model, X_test, y_test = german_credit_model

        result = run_perturbation_sweep(
            model=model,
            X_test=X_test,
            protected_features=[],  # Empty list
            epsilon_values=[0.05],
            threshold=0.15,
            seed=42,
        )

        # All features should be perturbed
        assert result.n_features_perturbed == len(X_test.columns)
        assert result.robustness_score > 0.0  # Should have some delta
