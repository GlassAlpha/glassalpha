"""Integration tests for evidence pack workflow.

Tests the complete audit → export → verify workflow end-to-end.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from glassalpha.api import from_model
from glassalpha.evidence import create_evidence_pack, verify_evidence_pack


class TestEvidencePackWorkflow:
    """Test complete audit → export → verify workflow."""

    def test_full_workflow_audit_to_pack_to_verify(self, tmp_path):
        """Complete workflow: audit → export → verify succeeds."""
        # Create test data
        import numpy as np
        import pandas as pd

        # Simple binary classification data
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, n_samples),
                "feature2": np.random.normal(0, 1, n_samples),
            }
        )
        y = (X["feature1"] + X["feature2"] > 0).astype(int)

        # Create simple model
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Generate audit
        result = from_model(
            model=model,
            X=X,
            y=y,
            random_seed=42,
            explain=True,
            calibration=True,
        )

        # Save audit report
        report_path = tmp_path / "audit_report.html"
        result.to_html(report_path)

        # Export evidence pack
        pack_path = create_evidence_pack(report_path)

        # Verify evidence pack
        verification_result = verify_evidence_pack(pack_path)

        # All should succeed
        assert verification_result.is_valid
        assert verification_result.artifacts_verified > 0
        assert len(verification_result.errors) == 0

    def test_workflow_with_protected_attributes(self, tmp_path):
        """Workflow with fairness analysis (protected attributes)."""
        import numpy as np
        import pandas as pd

        # Create data with protected attribute
        np.random.seed(42)
        n_samples = 200
        X = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, n_samples),
                "feature2": np.random.normal(0, 1, n_samples),
            }
        )
        y = (X["feature1"] + X["feature2"] > 0).astype(int)
        gender = np.random.choice([0, 1], n_samples)  # Protected attribute

        # Create model
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Generate audit with protected attributes
        result = from_model(
            model=model,
            X=X,
            y=y,
            protected_attributes={"gender": gender},
            random_seed=42,
        )

        # Save report
        report_path = tmp_path / "audit_report.html"
        result.to_html(report_path)

        # Export and verify
        pack_path = create_evidence_pack(report_path)
        verification_result = verify_evidence_pack(pack_path)

        assert verification_result.is_valid

    def test_workflow_with_shift_analysis(self, tmp_path):
        """Workflow including shift analysis artifacts."""
        import numpy as np
        import pandas as pd

        # Create test data
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, n_samples),
                "feature2": np.random.normal(0, 1, n_samples),
            }
        )
        y = (X["feature1"] + X["feature2"] > 0).astype(int)

        # Create model
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Generate audit
        result = from_model(
            model=model,
            X=X,
            y=y,
            random_seed=42,
        )

        # Save report
        report_path = tmp_path / "audit_report.html"
        result.to_html(report_path)

        # Create shift analysis file (simulating shift analysis)
        shift_path = report_path.with_suffix(".shift_analysis.json")
        shift_data = {
            "shift_scenarios": [
                {"shift": "gender:+0.1", "fairness_degradation": 0.02},
                {"shift": "age:+0.05", "fairness_degradation": 0.01},
            ]
        }
        shift_path.write_text(json.dumps(shift_data))

        # Export evidence pack (should include shift analysis)
        pack_path = create_evidence_pack(report_path)

        # Verify pack includes shift analysis
        import zipfile

        with zipfile.ZipFile(pack_path, "r") as zipf:
            files = zipf.namelist()
            assert "audit_report.shift_analysis.json" in files

        # Verify the pack
        verification_result = verify_evidence_pack(pack_path)
        assert verification_result.is_valid

    def test_workflow_with_custom_config(self, tmp_path):
        """Workflow with custom config file."""
        import numpy as np
        import pandas as pd

        # Create test data
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, n_samples),
                "feature2": np.random.normal(0, 1, n_samples),
            }
        )
        y = (X["feature1"] + X["feature2"] > 0).astype(int)

        # Create model
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Generate audit
        result = from_model(
            model=model,
            X=X,
            y=y,
            random_seed=42,
        )

        # Save report
        report_path = tmp_path / "audit_report.html"
        result.to_html(report_path)

        # Create config file
        config_path = tmp_path / "config.yaml"
        config_content = """
model:
  type: sklearn.LogisticRegression

data:
  X: test_data.csv
  y: test_labels.csv

audit:
  random_seed: 42
  explain: true
"""
        config_path.write_text(config_content)

        # Export with config
        pack_path = create_evidence_pack(report_path)

        # Verify config is included
        import zipfile

        with zipfile.ZipFile(pack_path, "r") as zipf:
            files = zipf.namelist()
            assert "config.yaml" in files

        # Verify the pack
        verification_result = verify_evidence_pack(pack_path)
        assert verification_result.is_valid

    def test_workflow_determinism_across_runs(self, tmp_path, monkeypatch):
        """Evidence pack is deterministic across multiple runs."""
        # Set SOURCE_DATE_EPOCH for deterministic timestamps
        monkeypatch.setenv("SOURCE_DATE_EPOCH", "1609459200")  # 2021-01-01 00:00:00 UTC

        import numpy as np
        import pandas as pd

        # Create test data
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, n_samples),
                "feature2": np.random.normal(0, 1, n_samples),
            }
        )
        y = (X["feature1"] + X["feature2"] > 0).astype(int)

        # Create model
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Generate audit
        result = from_model(
            model=model,
            X=X,
            y=y,
            random_seed=42,
        )

        # Save report
        report_path = tmp_path / "audit_report.html"
        result.to_html(report_path)

        # Export multiple times
        pack_paths = []
        for i in range(3):
            pack_path = create_evidence_pack(report_path)
            pack_paths.append(pack_path)

        # All packs should be identical
        pack_contents = []
        for pack_path in pack_paths:
            pack_contents.append(pack_path.read_bytes())

        # Compare all pairs
        for i in range(len(pack_contents)):
            for j in range(i + 1, len(pack_contents)):
                assert pack_contents[i] == pack_contents[j]

    def test_workflow_with_auditresult_to_evidence_pack(self, tmp_path):
        """AuditResult.to_evidence_pack() method works correctly."""
        import numpy as np
        import pandas as pd

        # Create test data
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, n_samples),
                "feature2": np.random.normal(0, 1, n_samples),
            }
        )
        y = (X["feature1"] + X["feature2"] > 0).astype(int)

        # Create model
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Generate audit
        result = from_model(
            model=model,
            X=X,
            y=y,
            random_seed=42,
        )

        # Use AuditResult.to_evidence_pack() method
        pack_path = result.to_evidence_pack(tmp_path / "result_pack.zip")

        # Verify pack was created
        assert pack_path.exists()
        assert pack_path.suffix == ".zip"

        # Verify the pack
        verification_result = verify_evidence_pack(pack_path)
        assert verification_result.is_valid

    def test_workflow_error_handling_missing_manifest(self, tmp_path):
        """Workflow handles missing manifest gracefully."""
        # Create audit report without manifest
        report_path = tmp_path / "audit_report.html"
        report_path.write_text("<html>Audit report</html>")

        # Create manifest file manually (but then delete it to simulate missing)
        manifest_path = report_path.with_suffix(".manifest.json")
        manifest_path.write_text('{"test": "data"}')

        # Remove manifest to simulate missing case
        manifest_path.unlink()

        # Export should still work (creates policy stub)
        pack_path = create_evidence_pack(report_path)

        # Verify pack was created (should be valid even without manifest)
        verification_result = verify_evidence_pack(pack_path)
        assert verification_result.is_valid
        assert len(verification_result.warnings) > 0  # Should have warning about missing manifest

        # Should contain policy stub
        import zipfile

        with zipfile.ZipFile(pack_path, "r") as zipf:
            policy_content = zipf.read("policy_decision.json").decode("utf-8")
            policy_data = json.loads(policy_content)
            assert policy_data["status"] == "not_configured"

    def test_workflow_with_pdf_report(self, tmp_path):
        """Workflow works with PDF reports."""
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LogisticRegression

        # Create test data
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, n_samples),
                "feature2": np.random.normal(0, 1, n_samples),
            }
        )
        y = (X["feature1"] + X["feature2"] > 0).astype(int)

        # Create model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Generate audit
        result = from_model(model=model, X=X, y=y, random_seed=42)

        # Save as PDF (this will create manifest sidecar)
        report_path = tmp_path / "audit_report.pdf"
        result.to_pdf(report_path)

        # Export should work with PDF
        pack_path = create_evidence_pack(report_path)

        # Verify pack includes PDF
        import zipfile

        with zipfile.ZipFile(pack_path, "r") as zipf:
            files = zipf.namelist()
            assert "audit_report.pdf" in files

        # Verify the pack
        verification_result = verify_evidence_pack(pack_path)
        assert verification_result.is_valid

    def test_workflow_verification_exit_codes(self, tmp_path):
        """Verification returns correct exit codes for CLI."""
        # Valid pack
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, n_samples),
                "feature2": np.random.normal(0, 1, n_samples),
            }
        )
        y = (X["feature1"] + X["feature2"] > 0).astype(int)

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        result = from_model(model=model, X=X, y=y, random_seed=42)

        report_path = tmp_path / "audit_report.html"
        result.to_html(report_path)

        pack_path = create_evidence_pack(report_path)
        verification_result = verify_evidence_pack(pack_path)

        # Valid pack should pass verification
        assert verification_result.is_valid

        # Invalid pack should fail
        invalid_pack = tmp_path / "invalid_pack.zip"
        invalid_pack.write_bytes(b"not a zip file")

        invalid_result = verify_evidence_pack(invalid_pack)
        assert not invalid_result.is_valid
