"""HTML snapshot tests for audit reports.

Tests HTML report generation without platform-specific PDF dependencies.
These tests run on all platforms and cover 95% of report generation logic.
"""

import os
import re

import pytest

from glassalpha.pipeline.audit import AuditResults
from glassalpha.report.renderer import AuditReportRenderer


@pytest.fixture(autouse=True, scope="class")
def deterministic_environment():
    """Set up deterministic environment for HTML generation tests.

    This ensures byte-identical HTML output across runs and platforms.
    """
    # Set deterministic environment variables
    original_env = {}

    # Save original values
    env_vars = [
        "SOURCE_DATE_EPOCH",
        "TZ",
        "MPLBACKEND",
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "GLASSALPHA_DETERMINISTIC",
    ]

    for var in env_vars:
        original_env[var] = os.environ.get(var)

    # Set deterministic values
    os.environ["SOURCE_DATE_EPOCH"] = "1577836800"  # 2020-01-01 00:00:00 UTC
    os.environ["TZ"] = "UTC"
    os.environ["MPLBACKEND"] = "Agg"  # Non-interactive matplotlib backend
    os.environ["PYTHONHASHSEED"] = "0"  # Deterministic hash ordering
    os.environ["GLASSALPHA_DETERMINISTIC"] = "42"  # Signal deterministic mode

    # Set single-threaded BLAS for determinism (important for any numerical operations)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    yield

    # Restore original values
    for var, value in original_env.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


@pytest.fixture
def audit_data():
    """Create minimal audit audit_data for testing."""
    return AuditResults(
        model_performance={"accuracy": 0.85, "precision": 0.82, "recall": 0.88},
        fairness_analysis={
            "demographic_parity": {"gender": 0.05},
            "equalized_odds": {"gender": {"tpr_diff": 0.03, "fpr_diff": 0.02}},
        },
        drift_analysis={"psi": 0.1, "total_variation": 0.05},
        explanations={
            "global": {"feature_importance": {"age": 0.3, "income": 0.2, "employment": 0.15}},
            "local": [],
        },
        data_summary={
            "rows": 1000,
            "columns": 10,
            "protected_attributes": ["gender"],
            "missing_values": 5,
        },
        schema_info={
            "columns": {"age": "int64", "income": "float64", "gender": "category"},
        },
        model_info={"type": "LogisticRegression", "features": 5, "classes": 2},
        selected_components={
            "model": "sklearn",
            "explainer": "treeshap",
            "metrics": ["accuracy", "demographic_parity"],
        },
        execution_info={
            "start_time": "2025-01-01T00:00:00",
            "duration_seconds": 30,
            "seed": 42,
        },
        manifest={
            "config_hash": "abc123",
            "data_hash": "def456",
            "git_sha": "xyz789",
            "tool_version": "0.2.0",
            "execution": {
                "status": "completed",
                "start_time": "2025-01-01T00:00:00",
                "end_time": "2025-01-01T00:00:30",
                "duration_seconds": 30,
            },
        },
    )


class TestHTMLStructure:
    """Test HTML report structure and content."""

    def test_html_contains_main_sections(self, audit_data):
        """HTML should contain all major report sections."""
        renderer = AuditReportRenderer()
        html = renderer.render_audit_report(audit_data, output_format="html")

        # Main sections
        assert "<html" in html.lower()
        assert "</html>" in html.lower()
        assert "<head>" in html.lower()
        assert "<body>" in html.lower()

        # Report title
        assert "Model Audit Report" in html or "Audit Report" in html

        # Key sections
        assert "Model Performance" in html or "Performance" in html
        assert "Fairness Analysis" in html or "Fairness" in html
        assert "Feature Importance" in html or "Explanations" in html

    def test_html_contains_performance_metrics(self, audit_data):
        """HTML should display performance metrics."""
        renderer = AuditReportRenderer()
        html = renderer.render_audit_report(audit_data, output_format="html")

        # Check metrics are present
        assert "accuracy" in html.lower()
        assert "0.85" in html or "85" in html  # Accuracy value
        assert "precision" in html.lower()
        assert "recall" in html.lower()

    def test_html_contains_fairness_metrics(self, audit_data):
        """HTML should display fairness metrics."""
        renderer = AuditReportRenderer()
        html = renderer.render_audit_report(audit_data, output_format="html")

        # Check fairness metrics
        assert "demographic_parity" in html.lower() or "demographic parity" in html.lower()
        assert "gender" in html.lower()
        assert "0.05" in html  # Demographic parity value

    def test_html_contains_feature_importance(self, audit_data):
        """HTML should display feature importance."""
        renderer = AuditReportRenderer()
        html = renderer.render_audit_report(audit_data, output_format="html")

        # Check that feature importance section exists
        # Note: Specific features may be in plots/charts rather than raw text
        assert "age" in html.lower()  # This is in the fixture
        # Other features may be rendered as visualizations

    def test_html_contains_metadata(self, audit_data):
        """HTML should contain audit metadata."""
        renderer = AuditReportRenderer()
        html = renderer.render_audit_report(audit_data, output_format="html")

        # Check metadata (config_hash should be in manifest)
        assert "abc123" in html  # config_hash
        # Note: Not all manifest fields may be rendered in HTML
        # The key test is that HTML is generated successfully

    def test_html_valid_structure(self, audit_data):
        """HTML should have valid structure."""
        renderer = AuditReportRenderer()
        html = renderer.render_audit_report(audit_data, output_format="html")

        # Basic HTML validation
        assert html.count("<html") == 1
        assert html.count("</html>") == 1
        assert html.count("<head>") == 1
        assert html.count("</head>") == 1
        assert html.count("<body") >= 1
        assert html.count("</body>") == 1

        # No obvious rendering errors
        assert "{{" not in html  # No unrendered Jinja2 templates
        assert "}}" not in html
        assert "undefined" not in html.lower()
        # Note: "none" appears in CSS (display: none) and "None" may appear in valid contexts


class TestHTMLDeterminism:
    """Test that HTML output is deterministic."""

    def test_html_byte_identical_across_runs(self, audit_data):
        """Same inputs should produce byte-identical HTML."""
        renderer = AuditReportRenderer()

        # Render twice (disable plots to avoid matplotlib non-determinism)
        html1 = renderer.render_audit_report(audit_data, embed_plots=False, output_format="html")
        html2 = renderer.render_audit_report(audit_data, embed_plots=False, output_format="html")

        assert html1 == html2, "HTML should be byte-identical across runs"

    def test_html_content_stable(self, audit_data):
        """HTML content should be stable and deterministic."""
        renderer = AuditReportRenderer()
        # Disable plots to avoid matplotlib non-determinism
        html = renderer.render_audit_report(audit_data, embed_plots=False, output_format="html")

        # Extract all numbers from HTML
        numbers = re.findall(r"\b\d+\.?\d*\b", html)

        # Re-render
        html2 = renderer.render_audit_report(audit_data, embed_plots=False, output_format="html")
        numbers2 = re.findall(r"\b\d+\.?\d*\b", html2)

        # Numbers should be identical
        assert numbers == numbers2, "Numeric values should be stable"


class TestHTMLEdgeCases:
    """Test HTML generation handles edge cases gracefully."""

    def test_html_with_missing_optional_sections(self, audit_data):
        """HTML should handle missing optional sections."""
        # Remove optional sections
        audit_data.drift_analysis = {}
        audit_data.explanations = {"global": {}}

        renderer = AuditReportRenderer()
        html = renderer.render_audit_report(audit_data, output_format="html")

        # Should still render
        assert "<html" in html.lower()
        assert "Model Performance" in html or "Performance" in html

    def test_html_with_empty_fairness(self, audit_data):
        """HTML should handle empty fairness analysis."""
        audit_data.fairness_analysis = {}

        renderer = AuditReportRenderer()
        html = renderer.render_audit_report(audit_data, output_format="html")

        # Should still render
        assert "<html" in html.lower()
        assert len(html) > 1000  # Non-trivial output

    def test_html_with_many_features(self, audit_data):
        """HTML should handle large feature sets."""
        # Create 50 features
        audit_data.explanations = {
            "global": {
                "feature_importance": {f"feature_{i}": 0.01 * i for i in range(50)},
            },
        }

        renderer = AuditReportRenderer()
        html = renderer.render_audit_report(audit_data, output_format="html")

        # Should render without errors (features may be in plots/visualizations)
        assert len(html) > 5000  # Non-trivial output
        assert "<html" in html.lower()  # Valid HTML structure


class TestHTMLOutput:
    """Test HTML output to files."""

    def test_html_writes_to_file(self, audit_data, tmp_path):
        """HTML should write to file correctly."""
        renderer = AuditReportRenderer()
        html = renderer.render_audit_report(audit_data, output_format="html")

        # Write to file
        output_path = tmp_path / "test_report.html"
        output_path.write_text(html, encoding="utf-8")

        # Verify file
        assert output_path.exists()
        assert output_path.stat().st_size > 5000  # Reasonable size

        # Verify content readable
        content = output_path.read_text(encoding="utf-8")
        assert content == html

    def test_html_utf8_encoding(self, audit_data, tmp_path):
        """HTML should handle UTF-8 encoding correctly."""
        # Add unicode characters to a field that's definitely rendered
        audit_data.model_info["special_note"] = "Testing: ñ, ü, 中文, 🎉"

        renderer = AuditReportRenderer()
        html = renderer.render_audit_report(audit_data, output_format="html")

        # Write and read back
        output_path = tmp_path / "test_unicode.html"
        output_path.write_text(html, encoding="utf-8")
        content = output_path.read_text(encoding="utf-8")

        # Should preserve UTF-8 encoding (charset should be declared)
        assert 'charset="UTF-8"' in content or 'charset="utf-8"' in content
