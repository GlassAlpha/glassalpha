"""End-to-end integration tests for preprocessing artifact verification.

This module tests the complete workflow from artifact creation through audit
generation with preprocessing verification.
"""

import json
from pathlib import Path

import pandas as pd
import pytest


class TestPreprocessingIntegration:
    """Integration tests for preprocessing artifact verification in audit pipeline."""

    def test_german_credit_audit_with_artifact(self, tmp_path):
        """Test complete audit pipeline with German Credit preprocessing artifact.

        This test validates:
        1. Artifact loading and validation
        2. Hash verification
        3. Data transformation
        4. Report generation with preprocessing section
        5. Manifest completeness

        """
        from glassalpha.config import load_config_from_file
        from glassalpha.pipeline.audit import AuditPipeline

        # Use the test config with preprocessing
        config_path = Path(__file__).parent / "test_config_preprocessing.yaml"
        assert config_path.exists(), f"Test config not found: {config_path}"

        # Load config
        config = load_config_from_file(config_path)

        # Override output paths to tmp_path
        output_html = tmp_path / "audit.html"
        output_manifest = tmp_path / "audit.manifest.json"

        # Create pipeline and run audit
        pipeline = AuditPipeline(config)
        results = pipeline.run()

        # Assert audit succeeded
        assert results.success, f"Audit failed: {results.error_message}"

        # Verify preprocessing info is in execution_info
        assert "preprocessing" in results.execution_info, "Missing preprocessing info in execution_info"
        preprocessing_info = results.execution_info["preprocessing"]

        # Verify artifact mode was used
        assert preprocessing_info["mode"] == "artifact", "Expected artifact mode"

        # Verify hashes are present
        assert "file_hash" in preprocessing_info, "Missing file_hash"
        assert "params_hash" in preprocessing_info, "Missing params_hash"
        assert preprocessing_info["file_hash"].startswith("sha256:"), "Invalid file_hash format"
        assert preprocessing_info["params_hash"].startswith("sha256:"), "Invalid params_hash format"

        # Verify manifest is present and has components
        assert "manifest" in preprocessing_info, "Missing manifest in preprocessing_info"
        manifest = preprocessing_info["manifest"]
        assert "components" in manifest, "Missing components in manifest"
        assert len(manifest["components"]) > 0, "No components in manifest"

        # Verify expected components are present
        component_names = [c["name"] for c in manifest["components"]]
        assert any("imputer" in name for name in component_names), "Missing imputer component"
        assert any("scaler" in name for name in component_names), "Missing scaler component"
        assert any("onehot" in name for name in component_names), "Missing onehot component"

        # Generate HTML report
        from glassalpha.report.renderer import AuditReportRenderer

        renderer = AuditReportRenderer()
        html_content = renderer.render_audit_report(
            results,
            template_name="standard_audit.html",
            output_path=output_html,
        )

        # Verify HTML was generated
        assert output_html.exists(), "HTML report not generated"
        assert len(html_content) > 0, "Empty HTML report"

        # Verify preprocessing section is in HTML
        assert "Preprocessing Verification" in html_content, "Missing preprocessing section in HTML"
        assert "Production Artifact Verified" in html_content, "Missing artifact verification banner"
        # Template strips 'sha256:' prefix for cleaner display
        assert preprocessing_info["file_hash"].replace("sha256:", "") in html_content, "File hash not in HTML"
        assert preprocessing_info["params_hash"].replace("sha256:", "") in html_content, "Params hash not in HTML"

        # Verify component details are in HTML
        assert "Component Details" in html_content, "Missing components section"
        assert "Preprocessing Pipeline" in html_content, "Missing pipeline diagram section"
        for component in manifest["components"]:
            assert component["name"] in html_content, f"Component {component['name']} not in HTML"

        # Save and verify manifest JSON
        manifest_data = results.manifest
        with output_manifest.open("w") as f:
            json.dump(manifest_data, f, indent=2)

        assert output_manifest.exists(), "Manifest JSON not generated"

        # Reload and verify manifest structure
        with output_manifest.open() as f:
            loaded_manifest = json.load(f)

        # Verify manifest has preprocessing component
        assert "components" in loaded_manifest, "Missing components in manifest"
        components = loaded_manifest.get("components", {})
        assert "preprocessing" in components, "Preprocessing not tracked in manifest"
        assert components["preprocessing"]["type"] == "artifact", "Expected artifact type in manifest"

    def test_auto_mode_warning_in_report(self, tmp_path):
        """Test that auto mode produces clear warnings in report.

        This test validates that when preprocessing is in auto mode,
        the report contains prominent non-compliance warnings.

        """
        from glassalpha.config import load_config_from_file
        from glassalpha.pipeline.audit import AuditPipeline

        # Use quickstart config which doesn't have preprocessing configured
        config_path = Path(__file__).parent.parent.parent / "src" / "glassalpha" / "data" / "configs" / "quickstart.yaml"
        assert config_path.exists(), f"Quickstart config not found: {config_path}"

        config = load_config_from_file(config_path)

        # Run audit (will use auto mode)
        pipeline = AuditPipeline(config)
        results = pipeline.run()

        assert results.success, f"Audit failed: {results.error_message}"

        # Verify auto mode was used
        if "preprocessing" in results.execution_info:
            preprocessing_info = results.execution_info["preprocessing"]
            assert preprocessing_info["mode"] == "auto", "Expected auto mode"

            # Generate HTML report
            from glassalpha.report.renderer import AuditReportRenderer

            renderer = AuditReportRenderer()
            html_content = renderer.render_audit_report(
                results,
                template_name="standard_audit.html",
            )

            # Verify warning is prominent in HTML
            assert "WARNING: Non-Compliant Preprocessing Mode" in html_content, "Missing auto mode warning"
            assert "AUTO preprocessing mode" in html_content, "Missing auto mode text"
            assert "NOT suitable for regulatory compliance" in html_content, "Missing compliance warning"

    def test_unknown_category_detection(self, tmp_path):
        """Test that unknown categories are detected and reported.

        This test validates that when audit data contains categories not seen
        during training, they are detected and reported with appropriate warnings.

        """
        # This test requires creating modified test data with unknown categories
        # For now, we'll skip this as it requires more complex setup
        pytest.skip("Requires test data with unknown categories - deferred to edge case tests")

    def test_hash_mismatch_fails_in_strict_mode(self, tmp_path):
        """Test that hash mismatches cause failure in strict mode.

        This test validates that when expected hashes don't match actual hashes,
        the audit fails appropriately in strict mode.

        """
        from glassalpha.config import load_config_from_file

        config_path = Path(__file__).parent.parent.parent / "src" / "glassalpha" / "data" / "configs" / "german_credit.yaml"
        config = load_config_from_file(config_path)

        # Enable preprocessing and set up artifact path for hash validation
        config.preprocessing.mode = "artifact"
        config.preprocessing.artifact_path = str(tmp_path / "test_artifact.pkl")
        config.preprocessing.expected_file_hash = "sha256:wrong_hash_intentionally_invalid"
        config.preprocessing.fail_on_mismatch = True

        # Create a proper sklearn preprocessing artifact file that can handle mixed data types
        import joblib
        import numpy as np
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        # Create preprocessing pipeline that handles both numeric and categorical features
        # This mimics what the actual preprocessing system does
        numeric_features = [
            "duration_months",
            "credit_amount",
            "installment_rate",
            "present_residence_since",
            "age_years",
            "existing_credits_count",
            "dependents_count",
        ]
        categorical_features = [
            "checking_account_status",
            "credit_history",
            "purpose",
            "savings_account",
            "employment_duration",
            "personal_status_sex",
            "other_debtors",
            "property",
            "other_installment_plans",
            "housing",
            "job",
            "telephone",
            "foreign_worker",
            "gender",
            "age_group",
        ]

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ]
        )

        # Fit on dummy data that matches the expected structure
        dummy_data = {}
        for col in numeric_features:
            dummy_data[col] = np.random.randn(10)
        for col in categorical_features:
            dummy_data[col] = np.random.choice(["A", "B", "C"], 10)

        dummy_df = pd.DataFrame(dummy_data)
        preprocessor.fit(dummy_df)

        joblib.dump(preprocessor, config.preprocessing.artifact_path)

        from glassalpha.pipeline.audit import AuditPipeline

        pipeline = AuditPipeline(config)

        # Should fail due to hash mismatch
        results = pipeline.run()
        assert not results.success, "Pipeline should fail with hash mismatch"
        assert "file hash mismatch" in results.error_message, (
            f"Expected hash mismatch error, got: {results.error_message}"
        )

    def test_preprocessing_manifest_in_audit_manifest(self, tmp_path):
        """Test that preprocessing details are captured in audit manifest.

        This test validates that the audit manifest includes complete preprocessing
        information for reproducibility.

        """
        from glassalpha.config import load_config_from_file
        from glassalpha.pipeline.audit import AuditPipeline

        # Create a minimal config for this test instead of using german_credit.yaml
        # which has complex column requirements that conflict with our test artifact
        from glassalpha.config import AuditConfig

        config = AuditConfig(
            audit_profile="tabular_compliance",
            model={"type": "logistic_regression", "params": {"random_state": 42}},
            data={
                "dataset": "custom",
                "path": str(tmp_path / "dummy_data.csv"),
                "target_column": "target",
                "protected_attributes": ["group"]
            },
            explainers={"strategy": "first_compatible", "priority": ["noop"]},
            metrics={"performance": ["accuracy"], "fairness": ["demographic_parity"]},
            reproducibility={"random_seed": 42, "deterministic": True},
            manifest={"enabled": True, "include_git_sha": False, "include_config_hash": False, "include_data_hash": False}
        )

        # Enable preprocessing for this test
        config.preprocessing.mode = "artifact"
        config.preprocessing.artifact_path = str(tmp_path / "test_artifact.pkl")
        # Remove hash check since we're creating our own artifact
        config.preprocessing.expected_file_hash = None

        # Create a simple preprocessing artifact for testing
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        # Create simple preprocessing pipeline for test data
        numeric_features = ["age", "income"]
        categorical_features = ["group"]

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ]
        )

        # Create dummy data that matches our test config
        dummy_data = {
            "age": np.random.randn(10),
            "income": np.random.randn(10),
            "group": np.random.choice(["A", "B"], 10),
            "target": np.random.choice([0, 1], 10)
        }

        dummy_df = pd.DataFrame(dummy_data)
        preprocessor.fit(dummy_df)

        # Save artifact
        joblib.dump(preprocessor, config.preprocessing.artifact_path)

        # Create dummy data file for the config
        dummy_df.to_csv(tmp_path / "dummy_data.csv", index=False)

        pipeline = AuditPipeline(config)
        results = pipeline.run()

        assert results.success, f"Audit failed: {results.error_message}"

        # Get manifest
        manifest = results.manifest

        # Verify preprocessing is tracked as a component
        assert "components" in manifest, "Missing components in manifest"
        components = manifest["components"]
        assert "preprocessing" in components, "Preprocessing not in manifest components"

        # Verify component has details
        prep_component = components["preprocessing"]
        assert "details" in prep_component, "Missing details in preprocessing component"
        details = prep_component["details"]
        assert "artifact_path" in details, "Missing artifact_path"
        assert "file_hash" in details, "Missing file_hash"
        assert "params_hash" in details, "Missing params_hash"


class TestPreprocessingCLIIntegration:
    """Integration tests for preprocessing CLI commands."""

    def test_cli_commands_work_with_real_artifact(self, tmp_path):
        """Test that all CLI commands work with the German Credit artifact."""
        import subprocess
        import sys

        artifact_path = Path(__file__).parent.parent.parent / "artifacts" / "german_credit_preprocessor.joblib"
        if not artifact_path.exists():
            pytest.skip(f"German Credit artifact not found: {artifact_path}")

        # Test hash command
        result = subprocess.run(
            [sys.executable, "-m", "glassalpha", "prep", "hash", str(artifact_path)],
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
        )
        assert result.returncode == 0, f"hash command failed: {result.stderr}"
        assert "File hash:" in result.stdout, "Missing file hash in output"

        # Test hash command with params
        result = subprocess.run(
            [sys.executable, "-m", "glassalpha", "prep", "hash", str(artifact_path), "--params"],
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
        )
        assert result.returncode == 0, f"hash --params command failed: {result.stderr}"
        assert "File hash:" in result.stdout, "Missing file hash in output"
        assert "Params hash:" in result.stdout, "Missing params hash in output"

        # Test inspect command
        result = subprocess.run(
            [sys.executable, "-m", "glassalpha", "prep", "inspect", str(artifact_path)],
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
        )
        assert result.returncode == 0, f"inspect command failed: {result.stderr}"
        assert "PREPROCESSING ARTIFACT MANIFEST" in result.stdout, "Missing manifest header"
        assert "Preprocessing Components" in result.stdout, "Missing components section"

        # Test validate command
        result = subprocess.run(
            [sys.executable, "-m", "glassalpha", "prep", "validate", str(artifact_path)],
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
        )
        assert result.returncode == 0, f"validate command failed: {result.stderr}"
        assert "VALIDATION PASSED" in result.stdout, "Validation did not pass"


@pytest.mark.slow
class TestPreprocessingPerformance:
    """Performance tests for preprocessing artifact verification."""

    def test_preprocessing_overhead_is_minimal(self, tmp_path):
        """Test that preprocessing artifact verification adds minimal overhead.

        Target: <1 second overhead for artifact loading and validation.

        """
        import time

        from glassalpha.preprocessing import (
            compute_file_hash,
            compute_params_hash,
            extract_sklearn_manifest,
            load_artifact,
            validate_classes,
        )

        artifact_path = Path(__file__).parent.parent.parent / "artifacts" / "german_credit_preprocessor.joblib"
        if not artifact_path.exists():
            pytest.skip(f"German Credit artifact not found: {artifact_path}")

        # Measure file hash computation (should be very fast)
        start = time.time()
        file_hash = compute_file_hash(artifact_path)
        file_hash_time = time.time() - start
        assert file_hash_time < 0.1, f"File hash too slow: {file_hash_time:.3f}s"

        # Measure artifact loading
        start = time.time()
        artifact = load_artifact(artifact_path)
        load_time = time.time() - start
        assert load_time < 1.0, f"Artifact loading too slow: {load_time:.3f}s"  # Loosened for CI variability

        # Measure class validation
        start = time.time()
        validate_classes(artifact)
        validate_time = time.time() - start
        assert validate_time < 0.1, f"Class validation too slow: {validate_time:.3f}s"

        # Measure manifest extraction
        start = time.time()
        manifest = extract_sklearn_manifest(artifact)
        extract_time = time.time() - start
        assert extract_time < 0.2, f"Manifest extraction too slow: {extract_time:.3f}s"

        # Measure params hash computation
        start = time.time()
        params_hash = compute_params_hash(manifest)
        params_hash_time = time.time() - start
        assert params_hash_time < 0.1, f"Params hash too slow: {params_hash_time:.3f}s"

        # Total overhead should be < 1 second
        total_time = file_hash_time + load_time + validate_time + extract_time + params_hash_time
        assert total_time < 1.0, f"Total preprocessing overhead too high: {total_time:.3f}s"

        print(f"\n✓ Preprocessing overhead: {total_time:.3f}s")
        print(f"  - File hash: {file_hash_time:.3f}s")
        print(f"  - Load artifact: {load_time:.3f}s")
        print(f"  - Validate classes: {validate_time:.3f}s")
        print(f"  - Extract manifest: {extract_time:.3f}s")
        print(f"  - Params hash: {params_hash_time:.3f}s")
