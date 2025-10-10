"""Golden report snapshot tests - template regression prevention.

These tests capture key sections of generated reports to catch breaking changes
without requiring 100% coverage of the renderer. Focus on high-risk template areas
that customers see directly.
"""

import pytest


class TestGoldenReportSnapshots:
    """Test report generation critical paths that prevent customer-visible regressions."""

    def test_standard_audit_template_has_required_sections(self):
        """Standard audit template must contain required sections for regulatory compliance."""
        try:
            from importlib.resources import files
        except ImportError:
            pytest.skip("importlib.resources not available")

        template_files = files("glassalpha.report.templates")
        template = template_files.joinpath("standard_audit.html")

        if not template.is_file():
            pytest.fail("standard_audit.html template not found in package")

        content = template.read_text(encoding="utf-8")

        # Critical sections that customers/regulators expect
        required_sections = [
            "<!DOCTYPE html>",  # Valid HTML document
            "<html",  # HTML structure
            "<head>",  # Proper HTML head
            "<title>",  # Document title
            "Model Audit Report",  # Expected title content
            "<body>",  # HTML body
            "Explainability",  # Explainability section
            "Performance",  # Performance metrics section
            "Feature",  # Feature information
        ]

        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)

        assert not missing_sections, f"Template missing required sections: {missing_sections}"

    def test_report_renderer_basic_smoke_test(self):
        """Report renderer must be importable and create basic HTML."""
        try:
            from glassalpha.config.schema import AuditConfig
            from glassalpha.pipeline.audit import AuditResults
            from glassalpha.report.renderer import AuditReportRenderer
        except ImportError as e:
            pytest.skip(f"Report components not available: {e}")

        # Create minimal test data with explicit config contract
        config = AuditConfig(
            data={
                "dataset": "custom",  # Explicit: custom dataset
                "path": "dummy.csv",  # Dummy path (no actual file read in this smoke test)
            },
            model={"type": "logistic_regression"},
            audit_profile="tabular_compliance",
        )

        # Create minimal audit results with proper manifest structure
        from glassalpha.utils.manifest import AuditManifest

        manifest = AuditManifest(
            audit_id="test-audit-123",
            config=config.to_dict(),
            audit_profile=config.audit_profile,
        )
        manifest.execution.status = "completed"  # Set proper status for template

        results = AuditResults(
            model_info={"type": "logistic_regression", "version": "test"},
            explanations={"global_importance": {"feature_1": 0.8, "feature_2": 0.2}},
            manifest=manifest.model_dump(),  # Convert to dict for compatibility
            success=True,
        )

        # Test renderer can be created
        renderer = AuditReportRenderer()
        assert renderer is not None, "Renderer should be creatable"

        # Test basic rendering doesn't crash
        try:
            html_content = renderer.render_audit_report(results)
            assert isinstance(html_content, str), "Renderer should return string"
            assert len(html_content) > 0, "Rendered content should not be empty"
            assert "html" in html_content.lower(), "Content should contain HTML"
        except Exception as e:
            # If rendering fails due to missing dependencies, that's OK for this smoke test
            # We just want to ensure the basic structure is sound
            if "shap" in str(e).lower() or "matplotlib" in str(e).lower():
                pytest.skip(f"Rendering failed due to missing optional dependency: {e}")
            else:
                raise  # Re-raise unexpected errors
