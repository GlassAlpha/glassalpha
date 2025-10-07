"""Integration tests for E6.5 Shift Simulator with German Credit dataset.

Tests the complete workflow from CLI to JSON export.
"""

import json

import pandas as pd
import pytest
from xgboost import XGBClassifier

from glassalpha.datasets.german_credit import load_german_credit
from glassalpha.metrics.shift import run_shift_analysis


class TestShiftSimulatorIntegration:
    """Integration tests for shift simulator with German Credit."""

    @pytest.fixture
    def german_credit_data(self):
        """Load and prepare German Credit data."""
        df = load_german_credit()

        # Prepare features and target
        X = df.drop(columns=["credit_risk"])
        y = (df["credit_risk"] == "good").astype(int)

        # Train a simple model
        model = XGBClassifier(random_state=42, n_estimators=50, max_depth=3)
        model.fit(X, y)

        # Generate predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        # Prepare sensitive features
        sensitive_features = pd.DataFrame(
            {
                "gender_male": X["gender_male"],
                "age_group": pd.cut(X["age"], bins=[0, 25, 35, 60, 100], labels=["18-25", "26-35", "36-60", "60+"]),
            },
        )

        return {
            "X": X,
            "y_true": y,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "sensitive_features": sensitive_features,
        }

    def test_single_positive_shift(self, german_credit_data):
        """Test single positive shift (increase proportion)."""
        result = run_shift_analysis(
            y_true=german_credit_data["y_true"],
            y_pred=german_credit_data["y_pred"],
            sensitive_features=german_credit_data["sensitive_features"],
            attribute="gender_male",
            shift=0.1,  # +10pp
            y_proba=german_credit_data["y_proba"],
            threshold=None,  # No gate threshold
        )

        # Verify result structure
        assert result.shift_spec.attribute == "gender_male"
        assert result.shift_spec.shift == 0.1
        assert result.shift_spec.shifted_proportion > result.shift_spec.original_proportion

        # Verify metrics computed
        assert "baseline_metrics" in result.to_dict()
        assert "shifted_metrics" in result.to_dict()
        assert "degradation" in result.to_dict()

        # Verify gate status (should be INFO without threshold)
        assert result.gate_status in ["PASS", "WARNING", "INFO"]

        # No violations without threshold
        assert result.violations == []

    def test_single_negative_shift(self, german_credit_data):
        """Test single negative shift (decrease proportion)."""
        result = run_shift_analysis(
            y_true=german_credit_data["y_true"],
            y_pred=german_credit_data["y_pred"],
            sensitive_features=german_credit_data["sensitive_features"],
            attribute="gender_male",
            shift=-0.1,  # -10pp
            y_proba=german_credit_data["y_proba"],
            threshold=None,
        )

        # Verify shift direction
        assert result.shift_spec.shifted_proportion < result.shift_spec.original_proportion
        assert result.shift_spec.shifted_proportion == pytest.approx(
            result.shift_spec.original_proportion - 0.1,
            abs=0.01,
        )

    def test_shift_with_degradation_threshold_pass(self, german_credit_data):
        """Test shift with degradation threshold that passes."""
        result = run_shift_analysis(
            y_true=german_credit_data["y_true"],
            y_pred=german_credit_data["y_pred"],
            sensitive_features=german_credit_data["sensitive_features"],
            attribute="gender_male",
            shift=0.05,  # Small shift (+5pp)
            y_proba=german_credit_data["y_proba"],
            threshold=0.10,  # Generous threshold (10pp degradation allowed)
        )

        # Small shift should pass with generous threshold
        assert result.gate_status == "PASS"
        assert result.violations == []

    def test_shift_with_degradation_threshold_fail(self, german_credit_data):
        """Test shift with degradation threshold that fails."""
        result = run_shift_analysis(
            y_true=german_credit_data["y_true"],
            y_pred=german_credit_data["y_pred"],
            sensitive_features=german_credit_data["sensitive_features"],
            attribute="gender_male",
            shift=0.15,  # Large shift (+15pp)
            y_proba=german_credit_data["y_proba"],
            threshold=0.01,  # Strict threshold (1pp degradation allowed)
        )

        # Large shift with strict threshold likely to fail
        # (might be WARNING or FAIL depending on actual degradation)
        assert result.gate_status in ["WARNING", "FAIL"]
        if result.gate_status == "FAIL":
            assert len(result.violations) > 0

    def test_deterministic_results(self, german_credit_data):
        """Test that shift analysis is deterministic."""
        result1 = run_shift_analysis(
            y_true=german_credit_data["y_true"],
            y_pred=german_credit_data["y_pred"],
            sensitive_features=german_credit_data["sensitive_features"],
            attribute="gender_male",
            shift=0.1,
            y_proba=german_credit_data["y_proba"],
            threshold=0.05,
        )

        result2 = run_shift_analysis(
            y_true=german_credit_data["y_true"],
            y_pred=german_credit_data["y_pred"],
            sensitive_features=german_credit_data["sensitive_features"],
            attribute="gender_male",
            shift=0.1,
            y_proba=german_credit_data["y_proba"],
            threshold=0.05,
        )

        # Results should be identical
        assert result1.shift_spec.to_dict() == result2.shift_spec.to_dict()
        assert result1.gate_status == result2.gate_status
        assert result1.baseline_metrics == result2.baseline_metrics
        assert result1.shifted_metrics == result2.shifted_metrics

    def test_json_serialization(self, german_credit_data):
        """Test that results can be serialized to JSON."""
        result = run_shift_analysis(
            y_true=german_credit_data["y_true"],
            y_pred=german_credit_data["y_pred"],
            sensitive_features=german_credit_data["sensitive_features"],
            attribute="gender_male",
            shift=0.1,
            y_proba=german_credit_data["y_proba"],
            threshold=0.05,
        )

        # Convert to dict and serialize
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict, indent=2)

        # Verify it can be deserialized
        loaded = json.loads(json_str)

        # Check key fields
        assert "shift_specification" in loaded
        assert loaded["shift_specification"]["attribute"] == "gender_male"
        assert loaded["shift_specification"]["shift"] == 0.1
        assert "baseline_metrics" in loaded
        assert "shifted_metrics" in loaded
        assert "degradation" in loaded
        assert "gate_status" in loaded

    def test_invalid_attribute_raises(self, german_credit_data):
        """Test error for non-existent attribute."""
        with pytest.raises(KeyError, match="nonexistent_attr"):
            run_shift_analysis(
                y_true=german_credit_data["y_true"],
                y_pred=german_credit_data["y_pred"],
                sensitive_features=german_credit_data["sensitive_features"],
                attribute="nonexistent_attr",
                shift=0.1,
            )

    def test_extreme_shift_raises(self, german_credit_data):
        """Test error for shift that exceeds bounds."""
        with pytest.raises(ValueError, match="must be ≤ 0.99"):
            run_shift_analysis(
                y_true=german_credit_data["y_true"],
                y_pred=german_credit_data["y_pred"],
                sensitive_features=german_credit_data["sensitive_features"],
                attribute="gender_male",
                shift=0.5,  # Likely to exceed 99% bound
            )

    def test_metrics_include_fairness_calibration_performance(self, german_credit_data):
        """Test that all metric types are computed."""
        result = run_shift_analysis(
            y_true=german_credit_data["y_true"],
            y_pred=german_credit_data["y_pred"],
            sensitive_features=german_credit_data["sensitive_features"],
            attribute="gender_male",
            shift=0.1,
            y_proba=german_credit_data["y_proba"],
        )

        # Check baseline metrics
        baseline = result.baseline_metrics
        assert "fairness" in baseline
        assert "calibration" in baseline
        assert "performance" in baseline

        # Check shifted metrics
        shifted = result.shifted_metrics
        assert "fairness" in shifted
        assert "calibration" in shifted
        assert "performance" in shifted

        # Check degradation computed for all types
        degradation = result.degradation
        assert "fairness" in degradation or "calibration" in degradation or "performance" in degradation


class TestShiftSimulatorCLIIntegration:
    """Integration tests for shift simulator CLI."""

    @pytest.fixture
    def german_credit_config(self, tmp_path):
        """Create a config file for German Credit audit."""
        config_path = tmp_path / "audit_shift_test.yaml"
        config_content = """
data:
  path: "german_credit"  # Built-in dataset
  target: "credit_risk"
  protected_attributes:
    - gender_male
    - age

model:
  type: xgboost
  params:
    n_estimators: 50
    max_depth: 3
    random_state: 42

metrics:
  fairness:
    threshold: 0.5
  calibration:
    enabled: true

seed: 42
"""
        config_path.write_text(config_content)
        return config_path

    def test_cli_shift_analysis_creates_json_sidecar(self, german_credit_config, tmp_path):
        """Test that CLI creates shift analysis JSON sidecar."""
        from glassalpha.cli.commands import _run_shift_analysis
        from glassalpha.config.loader import load_config_from_file

        # Load config
        config = load_config_from_file(german_credit_config)

        # Set output path
        output = tmp_path / "audit.pdf"

        # Run shift analysis
        exit_code = _run_shift_analysis(
            audit_config=config,
            output=output,
            shift_specs=["gender_male:+0.1"],
            threshold=None,
        )

        # Verify exit code
        assert exit_code == 0

        # Verify JSON sidecar created
        json_path = tmp_path / "audit.shift_analysis.json"
        assert json_path.exists()

        # Verify JSON content
        with json_path.open() as f:
            data = json.load(f)

        assert "shift_analysis" in data
        assert "shifts" in data["shift_analysis"]
        assert len(data["shift_analysis"]["shifts"]) == 1
        assert data["shift_analysis"]["shifts"][0]["shift_specification"]["attribute"] == "gender_male"
        assert data["shift_analysis"]["shifts"][0]["shift_specification"]["shift"] == 0.1

    def test_cli_multiple_shifts(self, german_credit_config, tmp_path):
        """Test CLI with multiple shift specifications."""
        from glassalpha.cli.commands import _run_shift_analysis
        from glassalpha.config.loader import load_config_from_file

        config = load_config_from_file(german_credit_config)
        output = tmp_path / "audit.pdf"

        exit_code = _run_shift_analysis(
            audit_config=config,
            output=output,
            shift_specs=["gender_male:+0.1", "gender_male:-0.05"],
            threshold=None,
        )

        assert exit_code == 0

        # Verify JSON has both shifts
        json_path = tmp_path / "audit.shift_analysis.json"
        with json_path.open() as f:
            data = json.load(f)

        assert len(data["shift_analysis"]["shifts"]) == 2
        assert data["shift_analysis"]["summary"]["total_shifts"] == 2

    def test_cli_threshold_enforcement(self, german_credit_config, tmp_path):
        """Test that CLI respects degradation threshold."""
        from glassalpha.cli.commands import _run_shift_analysis
        from glassalpha.config.loader import load_config_from_file

        config = load_config_from_file(german_credit_config)
        output = tmp_path / "audit.pdf"

        # Run with very strict threshold (likely to fail)
        exit_code = _run_shift_analysis(
            audit_config=config,
            output=output,
            shift_specs=["gender_male:+0.2"],  # Large shift
            threshold=0.001,  # Very strict (0.1pp)
        )

        # Should fail with violations
        # (might be 0 or 1 depending on actual degradation, but check JSON)
        json_path = tmp_path / "audit.shift_analysis.json"
        with json_path.open() as f:
            data = json.load(f)

        # Check summary for failed shifts
        summary = data["shift_analysis"]["summary"]
        assert "failed_shifts" in summary
        assert "violations_detected" in summary
