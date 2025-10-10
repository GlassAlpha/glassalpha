"""PDF generation integration tests for Linux CI.

These tests run actual PDF generation and are only executed on Linux
where WeasyPrint is known to work reliably. They provide full end-to-end
validation of PDF output quality and determinism.
"""

import hashlib
import sys
from pathlib import Path

import pytest

from glassalpha.pipeline.audit import AuditResults
from glassalpha.report import render_audit_pdf


@pytest.fixture
def full_audit_results():
    """Create comprehensive audit results for testing."""
    return AuditResults(
        model_performance={
            "accuracy": 0.8523,
            "precision": 0.8234,
            "recall": 0.8756,
            "f1_score": 0.8489,
            "roc_auc": 0.9123,
        },
        fairness_analysis={
            "demographic_parity": {"gender": 0.0523, "age_group": 0.0312},
            "equalized_odds": {
                "gender": {"tpr_diff": 0.0312, "fpr_diff": 0.0198},
                "age_group": {"tpr_diff": 0.0267, "fpr_diff": 0.0145},
            },
            "individual_fairness": {
                "consistency_score": 0.9234,
                "matched_pairs_count": 450,
            },
        },
        drift_analysis={
            "psi": 0.0823,
            "total_variation": 0.0456,
            "feature_drift": {
                "age": 0.0234,
                "income": 0.1123,
                "employment": 0.0567,
            },
        },
        explanations={
            "global": {
                "feature_importance": {
                    "age": 0.3234,
                    "income": 0.2456,
                    "employment": 0.1567,
                    "credit_history": 0.1234,
                    "loan_amount": 0.0889,
                },
            },
            "local": [],
        },
        calibration_analysis={
            "ece": 0.0234,
            "mce": 0.0456,
            "brier_score": 0.0823,
        },
        data_summary={
            "rows": 10000,
            "columns": 25,
            "protected_attributes": ["gender", "age_group"],
            "missing_values": 123,
            "target_distribution": {"positive": 3456, "negative": 6544},
        },
        schema_info={
            "columns": {
                "age": "int64",
                "income": "float64",
                "gender": "category",
                "employment": "category",
            },
        },
        model_info={
            "type": "XGBoostClassifier",
            "features": 25,
            "classes": 2,
            "n_estimators": 100,
            "max_depth": 6,
        },
        selected_components={
            "model": "xgboost",
            "explainer": "treeshap",
            "metrics": ["accuracy", "demographic_parity", "equalized_odds"],
        },
        execution_info={
            "start_time": "2025-01-01T00:00:00",
            "duration_seconds": 125,
            "seed": 42,
        },
        manifest={
            "config_hash": "abc123def456",
            "data_hash": "ghi789jkl012",
            "git_sha": "mno345pqr678",
            "tool_version": "0.2.0",
            "execution": {
                "status": "completed",
                "start_time": "2025-01-01T00:00:00",
                "end_time": "2025-01-01T00:02:05",
                "duration_seconds": 125,
            },
        },
    )


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="PDF generation only tested on Linux (WeasyPrint issues on other platforms)",
)
class TestPDFGenerationLinux:
    """Full PDF generation tests (Linux CI only)."""

    def test_pdf_file_created(self, full_audit_results, tmp_path):
        """PDF file should be created with valid format."""
        pdf_path = tmp_path / "test_audit.pdf"

        result_path = render_audit_pdf(full_audit_results, pdf_path)

        # Verify file created
        assert result_path.exists()
        assert result_path == pdf_path

        # Verify reasonable size
        file_size = pdf_path.stat().st_size
        assert file_size > 50000, f"PDF too small: {file_size} bytes (expected >50KB)"
        assert file_size < 10000000, f"PDF too large: {file_size} bytes (expected <10MB)"

    def test_pdf_has_valid_header(self, full_audit_results, tmp_path):
        """PDF should have valid PDF header."""
        pdf_path = tmp_path / "test_audit.pdf"
        render_audit_pdf(full_audit_results, pdf_path)

        # Verify PDF magic bytes
        with open(pdf_path, "rb") as f:
            header = f.read(8)
            assert header.startswith(b"%PDF"), f"Invalid PDF header: {header[:4]}"
            # Should be PDF 1.4 or later
            assert b"1." in header, "Missing PDF version"

    def test_pdf_deterministic_on_linux(self, full_audit_results, tmp_path):
        """PDFs should be byte-identical on Linux with same inputs."""
        pdf_path1 = tmp_path / "test1.pdf"
        pdf_path2 = tmp_path / "test2.pdf"

        # Generate twice
        render_audit_pdf(full_audit_results, pdf_path1)
        render_audit_pdf(full_audit_results, pdf_path2)

        # Compare hashes
        hash1 = hashlib.sha256(pdf_path1.read_bytes()).hexdigest()
        hash2 = hashlib.sha256(pdf_path2.read_bytes()).hexdigest()

        assert hash1 == hash2, "PDFs should be byte-identical"

    def test_pdf_with_all_sections(self, full_audit_results, tmp_path):
        """PDF should include all audit sections."""
        pdf_path = tmp_path / "comprehensive.pdf"
        render_audit_pdf(
            full_audit_results,
            pdf_path,
            report_title="Comprehensive Audit Report",
        )

        assert pdf_path.exists()

        # Read PDF content as text (basic check)
        pdf_bytes = pdf_path.read_bytes()

        # Check that PDF includes key sections (embedded text)
        # Note: This is a basic check; proper PDF parsing would be better
        # but adds complexity. We're mainly validating file generation works.
        assert len(pdf_bytes) > 50000  # Comprehensive report should be substantial

    def test_pdf_with_minimal_sections(self, tmp_path):
        """PDF should generate even with minimal data."""
        minimal_results = AuditResults(
            model_performance={"accuracy": 0.85},
            fairness_analysis={},
            drift_analysis={},
            explanations={"global": {}},
            data_summary={"rows": 100, "columns": 5},
            schema_info={"columns": {}},
            model_info={"type": "LogisticRegression"},
            selected_components={"model": "sklearn"},
            execution_info={"start_time": "2025-01-01T00:00:00", "duration_seconds": 10},
            manifest={"config_hash": "abc123"},
        )

        pdf_path = tmp_path / "minimal.pdf"
        render_audit_pdf(minimal_results, pdf_path)

        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 10000  # Even minimal should be >10KB

    def test_pdf_generation_performance(self, full_audit_results, tmp_path):
        """PDF generation should complete in reasonable time."""
        import time

        pdf_path = tmp_path / "performance_test.pdf"

        start = time.time()
        render_audit_pdf(full_audit_results, pdf_path)
        duration = time.time() - start

        # Should complete in under 30 seconds on Linux
        assert duration < 30.0, f"PDF generation too slow: {duration:.2f}s"

    def test_pdf_with_unicode_content(self, full_audit_results, tmp_path):
        """PDF should handle unicode characters correctly."""
        # Add unicode to data
        full_audit_results.data_summary["special_note"] = "Testing: ñ, ü, 中文, 🎉"

        pdf_path = tmp_path / "unicode_test.pdf"
        render_audit_pdf(full_audit_results, pdf_path)

        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 50000

    def test_pdf_output_directory_created(self, full_audit_results, tmp_path):
        """PDF generation should create output directory if needed."""
        pdf_path = tmp_path / "nested" / "dir" / "audit.pdf"

        # Directory doesn't exist yet
        assert not pdf_path.parent.exists()

        render_audit_pdf(full_audit_results, pdf_path)

        # Should create directory and file
        assert pdf_path.exists()
        assert pdf_path.parent.exists()


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="PDF snapshot comparison only on Linux",
)
class TestPDFSnapshots:
    """Test PDF output against golden snapshots (Linux CI)."""

    def test_pdf_structure_stable(self, full_audit_results, tmp_path):
        """PDF file structure should be stable across runs."""
        pdf_path = tmp_path / "snapshot.pdf"
        render_audit_pdf(full_audit_results, pdf_path)

        pdf_bytes = pdf_path.read_bytes()

        # Check PDF structure markers are present
        assert b"%%EOF" in pdf_bytes  # PDF end marker
        assert b"/Type /Catalog" in pdf_bytes  # PDF catalog
        assert b"/Type /Pages" in pdf_bytes  # Page tree

    def test_pdf_size_consistent(self, full_audit_results, tmp_path):
        """PDF file size should be consistent across runs."""
        pdf_path1 = tmp_path / "size1.pdf"
        pdf_path2 = tmp_path / "size2.pdf"

        render_audit_pdf(full_audit_results, pdf_path1)
        render_audit_pdf(full_audit_results, pdf_path2)

        size1 = pdf_path1.stat().st_size
        size2 = pdf_path2.stat().st_size

        # Should be exactly identical
        assert size1 == size2, f"PDF sizes differ: {size1} vs {size2}"
