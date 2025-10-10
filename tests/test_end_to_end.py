"""Comprehensive end-to-end testing for GlassAlpha audit pipeline.

This module provides extensive testing of the complete audit workflow:
- Configuration loading and validation
- Data processing and model training
- Explanation generation and metric computation
- Report generation and PDF output
- Reproducibility and determinism
- Performance characteristics
- Error handling and edge cases
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from glassalpha.config.schema import AuditConfig
from glassalpha.datasets import load_german_credit
from glassalpha.pipeline.audit import run_audit_pipeline
from glassalpha.report import render_audit_pdf
from glassalpha.utils import hash_config, hash_dataframe


class TestEndToEndWorkflow:
    """Test complete audit workflow from configuration to PDF generation."""

    @pytest.fixture(scope="class")
    def setup_components(self):
        """Ensure all required components are imported and registered."""
        # Import all component modules to trigger registration
        return "components_loaded"

    @pytest.fixture(scope="class")
    def test_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = Path(tempfile.mkdtemp(prefix="glassalpha_e2e_"))
        yield temp_dir
        # Cleanup after all tests
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture(scope="class")
    def german_credit_data(self, test_data_dir):
        """Prepare German Credit dataset for testing."""
        data = load_german_credit()
        data_path = test_data_dir / "german_credit.csv"
        data.to_csv(data_path, index=False)
        return data_path, data

    @pytest.fixture
    def base_config(self, german_credit_data):
        """Base audit configuration for testing."""
        data_path, _ = german_credit_data
        return {
            "audit_profile": "tabular_compliance",
            "reproducibility": {"random_seed": 42},
            "data": {
                "dataset": "custom",
                "path": str(data_path),
                "target_column": "credit_risk",
                "feature_columns": [
                    "checking_account_status",
                    "duration_months",
                    "credit_amount",
                    "age_years",
                    "gender",
                ],
                "protected_attributes": ["gender"],
            },
            "model": {
                "type": "xgboost",
                "params": {
                    "objective": "binary:logistic",
                    "n_estimators": 20,  # Fast for testing
                    "max_depth": 3,
                },
            },
            "explainers": {
                "strategy": "first_compatible",
                "priority": ["treeshap"],
                "config": {"treeshap": {"max_samples": 50}},
            },
            "metrics": {
                "performance": {"metrics": ["accuracy", "precision", "recall"]},
                "fairness": {"metrics": ["demographic_parity"]},
            },
        }

    def test_basic_end_to_end_audit(self, setup_components, base_config, test_data_dir):
        """Test complete audit pipeline from config to results."""
        # Create configuration
        config = AuditConfig(**base_config)

        # Run audit pipeline
        start_time = time.time()
        results = run_audit_pipeline(config)
        execution_time = time.time() - start_time

        # Validate results
        assert results.success, f"Audit failed: {results.error_message}"
        assert execution_time < 30.0, f"Audit took too long: {execution_time:.2f}s"

        # Check core components
        assert results.model_performance, "No performance metrics generated"
        assert results.explanations, "No explanations generated"
        assert results.data_summary, "No data summary generated"
        assert results.selected_components, "No components tracked"

        # Check performance metrics
        perf_metrics = results.model_performance
        assert "accuracy" in perf_metrics, "Accuracy metric missing"

        # Handle different metric formats (dict with value or direct numeric)
        accuracy_value = perf_metrics["accuracy"]
        if isinstance(accuracy_value, dict):
            accuracy_value = accuracy_value.get("value", accuracy_value.get("accuracy", 0))

        assert isinstance(accuracy_value, (int, float)), "Accuracy should be numeric"
        assert 0.0 <= accuracy_value <= 1.0, f"Invalid accuracy range: {accuracy_value}"

        # Check explanations
        explanations = results.explanations
        assert "global_importance" in explanations, "Global importance missing"

        # SHAP values might be under different keys
        has_shap = (
            "shap_values" in explanations
            or "local_explanations_sample" in explanations
            or len(explanations.get("global_importance", {})) > 0
        )
        assert has_shap, f"No SHAP data found. Available keys: {list(explanations.keys())}"

        # Check data summary
        data_summary = results.data_summary
        assert data_summary, "No data summary generated"

        # Check for shape info (different possible formats)
        if "n_samples" in data_summary:
            assert data_summary["n_samples"] == 1000, "Incorrect sample count"
        elif "shape" in data_summary:
            rows, cols = data_summary["shape"]
            assert rows == 1000, f"Incorrect sample count: {rows}"
        elif "columns" in data_summary:
            # At least verify we have column information
            assert len(data_summary["columns"]) > 0, "No column information"
        else:
            # Just verify the summary contains some useful info
            assert len(data_summary) > 0, "Empty data summary"

    def test_reproducibility_determinism(self, setup_components, base_config, test_data_dir):
        """Test that identical configurations produce identical results."""
        config = AuditConfig(**base_config)

        # Run audit twice with same configuration
        results1 = run_audit_pipeline(config)
        results2 = run_audit_pipeline(config)

        # Both should succeed
        assert results1.success and results2.success, "One or both audits failed"

        # Compare key results for determinism
        # Handle different metric formats
        def get_accuracy_value(result):
            acc = result.model_performance["accuracy"]
            return acc.get("value", acc.get("accuracy", acc)) if isinstance(acc, dict) else acc

        acc1 = get_accuracy_value(results1)
        acc2 = get_accuracy_value(results2)
        assert acc1 == acc2, f"Non-deterministic performance metrics: {acc1} vs {acc2}"

        # Compare SHAP values (should be identical with same seed)
        shap1 = results1.explanations.get("global_importance", {})
        shap2 = results2.explanations.get("global_importance", {})

        if shap1 and shap2:
            for feature in shap1:
                if feature in shap2:
                    assert abs(shap1[feature] - shap2[feature]) < 1e-10, (
                        f"Non-deterministic SHAP values for {feature}: {shap1[feature]} vs {shap2[feature]}"
                    )

        # Data hashes should be identical
        assert results1.data_summary.get("data_hash") == results2.data_summary.get("data_hash"), (
            "Non-deterministic data processing"
        )

    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason="WeasyPrint PDF generation crashes on macOS due to known rendering issues",
    )
    def test_pdf_generation_and_content(self, setup_components, base_config, test_data_dir):
        """Test PDF generation and content validation."""
        config = AuditConfig(**base_config)
        results = run_audit_pipeline(config)

        assert results.success, f"Audit failed: {results.error_message}"

        # Generate PDF
        pdf_path = test_data_dir / "test_audit.pdf"
        start_time = time.time()

        generated_path = render_audit_pdf(
            audit_results=results,
            output_path=pdf_path,
            report_title="End-to-End Test Audit",
        )

        pdf_time = time.time() - start_time

        # Validate PDF file
        assert generated_path.exists(), "PDF file not created"
        assert generated_path == pdf_path, "Incorrect PDF path returned"

        file_size = pdf_path.stat().st_size
        assert file_size > 100000, f"PDF too small: {file_size} bytes"  # At least 100KB
        assert file_size < 5000000, f"PDF too large: {file_size} bytes"  # Less than 5MB
        assert pdf_time < 10.0, f"PDF generation too slow: {pdf_time:.2f}s"

        # Validate PDF header (basic format check)
        with open(pdf_path, "rb") as f:
            header = f.read(4)
            assert header == b"%PDF", "Invalid PDF format"

    def test_multiple_model_types(self, setup_components, german_credit_data, test_data_dir):
        """Test audit pipeline with different model types."""
        data_path, _ = german_credit_data

        model_configs = [
            {"type": "xgboost", "params": {"objective": "binary:logistic", "n_estimators": 10}},
            {"type": "logistic_regression", "params": {"max_iter": 2000, "solver": "lbfgs"}},
        ]

        # Add LightGBM if available
        try:
            import lightgbm

            model_configs.append(
                {"type": "lightgbm", "params": {"objective": "binary", "num_leaves": 10, "n_estimators": 10}},
            )
        except ImportError:
            pass

        for _i, model_config in enumerate(model_configs):
            config_dict = {
                "audit_profile": "tabular_compliance",
                "reproducibility": {"random_seed": 42},
                "data": {
                    "dataset": "custom",
                    "path": str(data_path),
                    "target_column": "credit_risk",
                    "feature_columns": ["checking_account_status", "duration_months", "credit_amount"],
                    "protected_attributes": (
                        ["gender"]
                        if "gender" in ["checking_account_status", "duration_months", "credit_amount"]
                        else []
                    ),
                },
                "model": model_config,
                "explainers": {"strategy": "first_compatible", "priority": ["treeshap", "kernelshap"]},
                "metrics": {"performance": {"metrics": ["accuracy"]}},
            }

            config = AuditConfig(**config_dict)
            results = run_audit_pipeline(config)

            assert results.success, f"Audit failed for {model_config['type']}: {results.error_message}"
            assert results.model_performance, f"No metrics for {model_config['type']}"
            assert "accuracy" in results.model_performance, f"No accuracy for {model_config['type']}"

            # Generate PDF to ensure end-to-end works (skip on macOS due to WeasyPrint issues)
            if sys.platform != "darwin":
                pdf_path = test_data_dir / f"audit_{model_config['type']}.pdf"
                render_audit_pdf(results, pdf_path)
                assert pdf_path.exists(), f"PDF not generated for {model_config['type']}"

    def test_performance_requirements(self, setup_components, base_config):
        """Test that audit meets performance requirements."""
        config = AuditConfig(**base_config)

        # Measure multiple runs for consistency
        times = []
        for i in range(3):
            start_time = time.time()
            results = run_audit_pipeline(config)
            execution_time = time.time() - start_time

            assert results.success, f"Run {i + 1} failed: {results.error_message}"
            times.append(execution_time)

        # Performance requirements
        avg_time = sum(times) / len(times)
        max_time = max(times)

        assert avg_time < 15.0, f"Average execution too slow: {avg_time:.2f}s"
        assert max_time < 30.0, f"Worst case too slow: {max_time:.2f}s"

        # Consistency check (times shouldn't vary too much)
        time_std = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        assert time_std < avg_time * 0.5, f"Inconsistent execution times: std={time_std:.2f}s"

    def test_configuration_variations(self, setup_components, german_credit_data, test_data_dir):
        """Test various configuration combinations."""
        data_path, _ = german_credit_data

        # Test with minimal configuration
        minimal_config = {
            "audit_profile": "tabular_compliance",
            "reproducibility": {"random_seed": 42},
            "data": {
                "dataset": "custom",
                "path": str(data_path),
                "target_column": "credit_risk",
                "feature_columns": ["checking_account_status", "duration_months"],
            },
            "model": {"type": "logistic_regression"},
        }

        config = AuditConfig(**minimal_config)
        results = run_audit_pipeline(config)
        assert results.success, "Minimal configuration failed"

        # Test with comprehensive configuration
        comprehensive_config = {
            "audit_profile": "tabular_compliance",
            "reproducibility": {"random_seed": 123},
            "data": {
                "dataset": "custom",
                "path": str(data_path),
                "target_column": "credit_risk",
                "feature_columns": [
                    "checking_account_status",
                    "duration_months",
                    "credit_amount",
                    "savings_account",
                    "employment_duration",
                    "age_years",
                    "gender",
                ],
                "protected_attributes": (
                    ["gender", "age_group"] if "age_group" in load_german_credit().columns else ["gender"]
                ),
            },
            "model": {
                "type": "xgboost",
                "params": {"objective": "binary:logistic", "n_estimators": 50, "max_depth": 6},
            },
            "explainers": {
                "strategy": "first_compatible",
                "priority": ["treeshap", "kernelshap"],
                "config": {"treeshap": {"max_samples": 200}},
            },
            "metrics": {
                "performance": {"metrics": ["accuracy", "precision", "recall", "f1", "auc_roc"]},
                "fairness": {"metrics": ["demographic_parity", "equal_opportunity"]},
            },
        }

        config = AuditConfig(**comprehensive_config)
        results = run_audit_pipeline(config)
        assert results.success, "Comprehensive configuration failed"

        # Verify more metrics computed with comprehensive config
        assert len(results.model_performance) >= 5, "Expected more performance metrics"

    def test_error_handling_scenarios(self, setup_components, test_data_dir):
        """Test proper error handling for invalid scenarios."""
        # Test 1: Invalid data path
        invalid_data_config = {
            "audit_profile": "tabular_compliance",
            "reproducibility": {"random_seed": 42},
            "data": {
                "dataset": "custom",
                "path": "/nonexistent/path.csv",
                "target_column": "target",
                "feature_columns": ["feature1"],
            },
            "model": {"type": "logistic_regression"},
        }

        config = AuditConfig(**invalid_data_config)
        results = run_audit_pipeline(config)
        assert not results.success, "Should fail with invalid data path"
        assert "file not found" in results.error_message.lower(), "Should contain file error description"

        # Test 2: Missing target column
        data = load_german_credit()
        temp_data_path = test_data_dir / "invalid_data.csv"
        data.drop(columns=["credit_risk"]).to_csv(temp_data_path, index=False)

        missing_target_config = {
            "audit_profile": "tabular_compliance",
            "reproducibility": {"random_seed": 42},
            "data": {
                "dataset": "custom",
                "path": str(temp_data_path),
                "target_column": "credit_risk",  # Column doesn't exist
                "feature_columns": ["duration_months"],
            },
            "model": {"type": "logistic_regression"},
        }

        config = AuditConfig(**missing_target_config)
        results = run_audit_pipeline(config)
        assert not results.success, "Should fail with missing target column"

        # Test 3: Invalid model type
        invalid_model_config = {
            "audit_profile": "tabular_compliance",
            "reproducibility": {"random_seed": 42},
            "data": {
                "dataset": "custom",
                "path": str(temp_data_path),
                "target_column": "credit_risk",
                "feature_columns": ["duration_months"],
            },
            "model": {"type": "nonexistent_model"},
        }

        config = AuditConfig(**invalid_model_config)
        results = run_audit_pipeline(config)
        assert not results.success, "Should fail with invalid model type"

    def test_manifest_generation_and_content(self, setup_components, base_config):
        """Test audit manifest generation and content validation."""
        config = AuditConfig(**base_config)
        results = run_audit_pipeline(config)

        assert results.success, f"Audit failed: {results.error_message}"
        assert results.manifest, "No manifest generated"

        manifest = results.manifest

        # Check required manifest fields
        required_fields = ["audit_id", "seeds", "config_hash", "data_hash", "selected_components"]
        for field in required_fields:
            assert field in manifest, f"Missing manifest field: {field}"

        # Check seeds
        assert "master_seed" in manifest["seeds"], "Missing master seed"
        assert manifest["seeds"]["master_seed"] == 42, "Incorrect master seed"

        # Check selected components
        components = manifest["selected_components"]
        assert len(components) > 0, "No components tracked"
        assert any(comp.get("type") == "model" for comp in components.values()), "No model tracked"

        # Check execution info
        assert "execution_info" in manifest, "Missing execution info"
        assert "start_time" in manifest["execution_info"], "Missing start time"
        assert "end_time" in manifest["execution_info"], "Missing end time"

    def test_data_integrity_and_hashing(self, german_credit_data):
        """Test data integrity verification and hashing consistency."""
        data_path, original_data = german_credit_data

        # Test data hash consistency
        hash1 = hash_dataframe(original_data)
        hash2 = hash_dataframe(original_data)
        assert hash1 == hash2, "Data hash not deterministic"

        # Test configuration hash consistency
        config_dict = {
            "audit_profile": "tabular_compliance",
            "model": {"type": "xgboost"},
            "reproducibility": {"random_seed": 42},
        }

        config_hash1 = hash_config(config_dict)
        config_hash2 = hash_config(config_dict)
        assert config_hash1 == config_hash2, "Config hash not deterministic"

        # Test that different data produces different hashes
        modified_data = original_data.copy()
        modified_data.iloc[0, 0] = "modified_value"

        modified_hash = hash_dataframe(modified_data)
        assert hash1 != modified_hash, "Data modification not detected"


class TestCLIEndToEnd:
    """Test end-to-end functionality through CLI interface."""

    @pytest.fixture(scope="class")
    def setup_components(self):
        """Ensure all required components are imported and registered."""
        # Import all component modules to trigger registration
        return "components_loaded"

    @pytest.fixture(scope="class")
    def cli_test_dir(self):
        """Create temporary directory for CLI testing."""
        temp_dir = Path(tempfile.mkdtemp(prefix="glassalpha_cli_e2e_"))

        # Prepare test data and configuration
        data = load_german_credit()
        data_path = temp_dir / "test_data.csv"
        data.to_csv(data_path, index=False)

        config_dict = {
            "audit_profile": "tabular_compliance",
            "reproducibility": {"random_seed": 42},
            "data": {
                "dataset": "custom",
                "path": str(data_path),
                "target_column": "credit_risk",
                "feature_columns": ["checking_account_status", "duration_months", "credit_amount"],
                "protected_attributes": [],  # Simplified for CLI testing
            },
            "model": {"type": "xgboost", "params": {"objective": "binary:logistic", "n_estimators": 10}},
            "explainers": {"strategy": "first_compatible", "priority": ["treeshap"]},
            "metrics": {"performance": {"metrics": ["accuracy"]}},
        }

        import yaml

        config_path = temp_dir / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        yield temp_dir, config_path, data_path

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cli_audit_command_success(self, setup_components, cli_test_dir):
        """Test successful CLI audit execution."""
        temp_dir, config_path, data_path = cli_test_dir
        output_pdf = temp_dir / "cli_audit_test.pdf"

        # Set up environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.parent / "src")

        # Run CLI command
        cmd = [
            sys.executable,
            "-m",
            "glassalpha.cli.main",
            "audit",
            "--config",
            str(config_path),
            "--output",
            str(output_pdf),
        ]

        start_time = time.time()
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", env=env, timeout=60)
        execution_time = time.time() - start_time

        # Validate command success
        assert result.returncode == 0, f"CLI command failed: {result.stderr}"
        assert "Audit Report Generated Successfully" in result.stdout, "Success message not found"
        assert output_pdf.exists(), "PDF output not created"

        # Validate timing
        assert execution_time < 45.0, f"CLI execution too slow: {execution_time:.2f}s"

        # Validate PDF
        file_size = output_pdf.stat().st_size
        assert file_size > 50000, f"PDF too small: {file_size} bytes"

    def test_cli_validation_command(self, setup_components, cli_test_dir):
        """Test CLI configuration validation."""
        temp_dir, config_path, data_path = cli_test_dir

        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.parent / "src")

        cmd = [sys.executable, "-m", "glassalpha.cli.main", "validate", "--config", str(config_path)]

        result = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", env=env, timeout=30)

        assert result.returncode == 0, f"Validation failed: {result.stderr}"
        assert "Configuration is valid" in result.stdout, "Validation success message not found"

    def test_cli_component_listing(self, setup_components):
        """Test CLI component listing functionality."""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.parent / "src")

        cmd = [sys.executable, "-m", "glassalpha.cli.main", "list"]

        result = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", env=env, timeout=30)

        assert result.returncode == 0, f"Component listing failed: {result.stderr}"
        assert "Available Components" in result.stdout, "Component listing header not found"
        assert "MODELS:" in result.stdout, "Models section not found"

    def test_cli_error_handling(self, setup_components, cli_test_dir):
        """Test CLI error handling with invalid configurations."""
        temp_dir, config_path, data_path = cli_test_dir

        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.parent / "src")

        # Test with nonexistent config file
        cmd = [
            sys.executable,
            "-m",
            "glassalpha.cli.main",
            "audit",
            "--config",
            "/nonexistent/config.yaml",
            "--output",
            "test.pdf",
        ]

        result = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", env=env, timeout=30)

        assert result.returncode != 0, "Should fail with nonexistent config"
        assert "does not exist" in result.stderr, "Should show file error message"


class TestScalabilityAndLimits:
    """Test system behavior under various scales and edge conditions."""

    @pytest.fixture(scope="class")
    def setup_components(self):
        """Ensure all required components are imported and registered."""
        # Import all component modules to trigger registration
        return "components_loaded"

    def test_small_dataset_handling(self, setup_components, test_data_dir):
        """Test audit with very small datasets."""
        # Create tiny dataset (10 samples)
        data = load_german_credit().sample(n=10, random_state=42)
        small_data_path = test_data_dir / "small_data.csv"
        data.to_csv(small_data_path, index=False)

        config_dict = {
            "audit_profile": "tabular_compliance",
            "reproducibility": {"random_seed": 42},
            "data": {
                "dataset": "custom",
                "path": str(small_data_path),
                "target_column": "credit_risk",
                "feature_columns": ["checking_account_status", "duration_months"],
            },
            "model": {"type": "logistic_regression"},
            "explainers": {"strategy": "first_compatible", "priority": ["kernelshap"]},
            "metrics": {"performance": {"metrics": ["accuracy"]}},
        }

        config = AuditConfig(**config_dict)
        results = run_audit_pipeline(config)

        # Should handle small data gracefully
        assert results.success, f"Small dataset audit failed: {results.error_message}"
        assert results.model_performance, "No performance metrics for small dataset"

    @pytest.fixture(scope="class")
    def test_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = Path(tempfile.mkdtemp(prefix="glassalpha_scale_"))
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_feature_scaling_behavior(self, setup_components, test_data_dir):
        """Test audit with different numbers of features."""
        base_data = load_german_credit()

        # Test with minimal features (2)
        minimal_features = ["duration_months", "credit_amount"]
        minimal_config = self._create_config_with_features(base_data, minimal_features, test_data_dir, "minimal")

        config = AuditConfig(**minimal_config)
        results = run_audit_pipeline(config)
        assert results.success, "Minimal features audit failed"

        # Test with many features (10+)
        many_features = [
            "checking_account_status",
            "duration_months",
            "credit_history",
            "purpose",
            "credit_amount",
            "savings_account",
            "employment_duration",
            "installment_rate",
            "personal_status_sex",
            "other_debtors",
        ]
        many_config = self._create_config_with_features(base_data, many_features, test_data_dir, "many")

        config = AuditConfig(**many_config)
        results = run_audit_pipeline(config)
        assert results.success, "Many features audit failed"

    def _create_config_with_features(self, data, features, test_dir, suffix):
        """Helper to create config with specific feature set."""
        data_subset = data[features + ["credit_risk"]]
        data_path = test_dir / f"data_{suffix}.csv"
        data_subset.to_csv(data_path, index=False)

        return {
            "audit_profile": "tabular_compliance",
            "reproducibility": {"random_seed": 42},
            "data": {
                "dataset": "custom",
                "path": str(data_path),
                "target_column": "credit_risk",
                "feature_columns": features,
            },
            "model": {"type": "xgboost", "params": {"n_estimators": 10}},
            "explainers": {"priority": ["treeshap"]},
            "metrics": {"performance": {"metrics": ["accuracy"]}},
        }


# Helper functions for test utilities
def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def compare_pdf_metadata(pdf1_path: Path, pdf2_path: Path) -> dict:
    """Compare metadata and basic properties of two PDF files."""
    comparison = {
        "size_match": pdf1_path.stat().st_size == pdf2_path.stat().st_size,
        "hash_match": calculate_file_hash(pdf1_path) == calculate_file_hash(pdf2_path),
        "size_diff": abs(pdf1_path.stat().st_size - pdf2_path.stat().st_size),
        "pdf1_size": pdf1_path.stat().st_size,
        "pdf2_size": pdf2_path.stat().st_size,
    }
    return comparison


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])
