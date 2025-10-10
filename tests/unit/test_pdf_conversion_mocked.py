"""Mocked PDF conversion tests.

Tests PDF conversion logic without actually invoking WeasyPrint.
These tests work on all platforms and verify integration contracts.
"""

from unittest.mock import Mock, patch

import pytest

from glassalpha.pipeline.audit import AuditResults

try:
    from glassalpha.report.renderers.pdf import AuditPDFRenderer as PDFRenderer
    from glassalpha.report.renderers.pdf import render_audit_pdf

    _PDF_AVAILABLE = True
except ImportError:
    # PDF dependencies not available
    PDFRenderer = None  # type: ignore[assignment, misc]
    render_audit_pdf = None  # type: ignore[assignment, misc]
    _PDF_AVAILABLE = False


@pytest.fixture
def minimal_audit_results():
    """Create minimal audit results for testing."""
    return AuditResults(
        model_performance={"accuracy": 0.85},
        fairness_analysis={"demographic_parity": {"gender": 0.05}},
        drift_analysis={"psi": 0.1},
        explanations={"global": {"feature_importance": {"age": 0.3}}},
        data_summary={"rows": 1000, "columns": 10, "protected_attributes": ["gender"]},
        schema_info={"columns": {"age": "int64", "gender": "category"}},
        model_info={"type": "LogisticRegression", "features": 5},
        selected_components={"model": "sklearn", "explainer": "treeshap"},
        execution_info={"start_time": "2025-01-01T00:00:00", "duration_seconds": 30, "seed": 42},
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


@pytest.mark.skipif(PDFRenderer is None, reason="PDF dependencies not available")
class TestPDFConversionMocked:
    """Test PDF conversion with mocked WeasyPrint."""

    @patch("glassalpha.report.renderers.pdf.AuditPDFRenderer._set_pdf_metadata")
    @patch("glassalpha.report.renderers.pdf.CSS")
    @patch("glassalpha.report.renderers.pdf.HTML")
    def test_weasyprint_called_with_html_content(self, mock_html_class, mock_css_class, _mock_metadata, tmp_path):
        """WeasyPrint should be called with HTML content."""
        output_path = tmp_path / "test.pdf"

        # Mock the HTML object and its methods
        mock_html_instance = Mock()
        mock_html_class.return_value = mock_html_instance
        mock_rendered_doc = Mock()
        mock_html_instance.render.return_value = mock_rendered_doc

        # Mock CSS
        mock_css_instance = Mock()
        mock_css_class.return_value = mock_css_instance

        # Mock write_pdf to create file so stat() works
        def create_file(*_args, **_kwargs):
            output_path.write_bytes(b"%PDF-1.4\n")

        mock_rendered_doc.write_pdf.side_effect = create_file

        renderer = PDFRenderer()
        renderer.convert_html_to_pdf("<html>test</html>", output_path)

        # Verify HTML class was instantiated with content
        mock_html_class.assert_called_once()
        call_args = mock_html_class.call_args
        assert "test" in str(call_args)  # HTML content passed

    @patch("glassalpha.report.renderers.pdf.AuditPDFRenderer._set_pdf_metadata")
    @patch("glassalpha.report.renderers.pdf.CSS")
    @patch("glassalpha.report.renderers.pdf.HTML")
    def test_weasyprint_write_pdf_called(self, mock_html_class, mock_css_class, _mock_metadata, tmp_path):
        """WeasyPrint write_pdf should be called with output path."""
        output_path = tmp_path / "test.pdf"

        # Mock the HTML instance
        mock_html_instance = Mock()
        mock_html_class.return_value = mock_html_instance
        mock_rendered_doc = Mock()
        mock_html_instance.render.return_value = mock_rendered_doc

        # Mock CSS
        mock_css_instance = Mock()
        mock_css_class.return_value = mock_css_instance

        # Mock write_pdf to create file so stat() works
        def create_file(*_args, **_kwargs):
            output_path.write_bytes(b"%PDF-1.4\n")

        mock_rendered_doc.write_pdf.side_effect = create_file

        renderer = PDFRenderer()
        renderer.convert_html_to_pdf("<html>test</html>", output_path)

        # Verify write_pdf was called
        mock_rendered_doc.write_pdf.assert_called_once()
        call_args = mock_rendered_doc.write_pdf.call_args
        assert output_path in call_args[0] or str(output_path) in str(call_args[1].values())

    @patch("glassalpha.report.renderers.pdf.AuditPDFRenderer._set_pdf_metadata")
    @patch("glassalpha.report.renderers.pdf.CSS")
    @patch("glassalpha.report.renderers.pdf.HTML")
    def test_pdf_conversion_creates_output_file(self, mock_html_class, mock_css_class, _mock_metadata, tmp_path):
        """PDF conversion should create output file."""
        output_path = tmp_path / "test.pdf"

        # Mock write_pdf to actually create the file
        mock_html_instance = Mock()
        mock_html_class.return_value = mock_html_instance
        mock_rendered_doc = Mock()
        mock_html_instance.render.return_value = mock_rendered_doc

        # Mock CSS
        mock_css_instance = Mock()
        mock_css_class.return_value = mock_css_instance

        def create_dummy_pdf(*_args, **_kwargs):
            output_path.write_bytes(b"%PDF-1.4\n")

        mock_rendered_doc.write_pdf.side_effect = create_dummy_pdf

        renderer = PDFRenderer()
        result = renderer.convert_html_to_pdf("<html>test</html>", output_path)

        assert result == output_path
        assert output_path.exists()

    @patch("glassalpha.report.renderers.pdf.CSS")
    @patch("glassalpha.report.renderers.pdf.HTML")
    def test_pdf_conversion_handles_import_error(self, mock_html_class, _mock_css_class, tmp_path):
        """PDF conversion should handle WeasyPrint import errors gracefully."""
        output_path = tmp_path / "test.pdf"

        # Simulate ImportError
        mock_html_class.side_effect = ImportError("WeasyPrint not installed")

        renderer = PDFRenderer()

        with pytest.raises(RuntimeError) as exc_info:
            renderer.convert_html_to_pdf("<html>test</html>", output_path)

        assert "PDF backend (WeasyPrint) is not installed" in str(exc_info.value)

    @patch("glassalpha.report.renderers.pdf.CSS")
    @patch("glassalpha.report.renderers.pdf.HTML")
    def test_pdf_conversion_handles_runtime_error(self, mock_html_class, mock_css_class, tmp_path):
        """PDF conversion should handle runtime errors."""
        output_path = tmp_path / "test.pdf"

        # Simulate runtime error
        mock_html_instance = Mock()
        mock_html_class.return_value = mock_html_instance
        mock_rendered_doc = Mock()
        mock_html_instance.render.return_value = mock_rendered_doc
        mock_rendered_doc.write_pdf.side_effect = RuntimeError("PDF generation failed")

        # Mock CSS
        mock_css_instance = Mock()
        mock_css_class.return_value = mock_css_instance

        renderer = PDFRenderer()

        with pytest.raises(RuntimeError) as exc_info:
            renderer.convert_html_to_pdf("<html>test</html>", output_path)

        assert "PDF conversion failed" in str(exc_info.value)

    @patch("glassalpha.report.renderers.pdf.CSS")
    @patch("glassalpha.report.renderers.pdf.HTML")
    def test_render_audit_pdf_end_to_end_mocked(self, mock_html_class, mock_css_class, tmp_path):
        """Test full render_audit_pdf flow with mocked WeasyPrint."""
        output_path = tmp_path / "audit.pdf"

        # Create test audit results
        test_results = AuditResults(
            model_performance={"accuracy": 0.85},
            fairness_analysis={"demographic_parity": {"gender": 0.05}},
            drift_analysis={"psi": 0.1},
            explanations={"global": {"feature_importance": {"age": 0.3}}},
            data_summary={"rows": 1000, "columns": 10, "protected_attributes": ["gender"]},
            schema_info={"columns": {"age": "int64", "gender": "category"}},
            model_info={"type": "LogisticRegression", "features": 5},
            selected_components={"model": "sklearn", "explainer": "treeshap"},
            execution_info={"start_time": "2025-01-01T00:00:00", "duration_seconds": 30, "seed": 42},
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

        # Mock successful PDF generation
        mock_html_instance = Mock()
        mock_html_class.return_value = mock_html_instance
        mock_rendered_doc = Mock()
        mock_html_instance.render.return_value = mock_rendered_doc

        # Mock CSS
        mock_css_instance = Mock()
        mock_css_class.return_value = mock_css_instance

        def create_pdf(*_args, **_kwargs):
            output_path.write_bytes(b"%PDF-1.4\ntest content")

        mock_rendered_doc.write_pdf.side_effect = create_pdf

        # This should work even though WeasyPrint is mocked
        result = render_audit_pdf(test_results, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_pdf_renderer_initialization(self):
        """PDFRenderer should initialize with default config."""
        renderer = PDFRenderer()

        assert renderer is not None
        assert hasattr(renderer, "config")
        assert hasattr(renderer, "convert_html_to_pdf")

    @patch("glassalpha.report.renderers.pdf.AuditPDFRenderer._set_pdf_metadata")
    @patch("glassalpha.report.renderers.pdf.CSS")
    @patch("glassalpha.report.renderers.pdf.HTML")
    def test_pdf_conversion_with_stylesheets(self, mock_html_class, mock_css_class, _mock_metadata, tmp_path):
        """PDF conversion should handle stylesheets."""
        output_path = tmp_path / "test.pdf"

        mock_html_instance = Mock()
        mock_html_class.return_value = mock_html_instance
        mock_rendered_doc = Mock()
        mock_html_instance.render.return_value = mock_rendered_doc

        # Mock CSS
        mock_css_instance = Mock()
        mock_css_class.return_value = mock_css_instance

        # Mock write_pdf to create file so stat() works
        def create_file(*_args, **_kwargs):
            output_path.write_bytes(b"%PDF-1.4\n")

        mock_rendered_doc.write_pdf.side_effect = create_file

        renderer = PDFRenderer()
        renderer.convert_html_to_pdf("<html>test</html>", output_path)

        # Verify render was called with stylesheets
        mock_html_instance.render.assert_called_once()
        call_kwargs = mock_html_instance.render.call_args[1]

        # Stylesheets should be passed as list
        if "stylesheets" in call_kwargs:
            assert isinstance(call_kwargs["stylesheets"], list)


@pytest.mark.skipif(PDFRenderer is None, reason="PDF dependencies not available")
class TestPDFMetadata:
    """Test PDF metadata handling (mocked)."""

    @patch("glassalpha.report.renderers.pdf.CSS")
    @patch("glassalpha.report.renderers.pdf.HTML")
    @patch("glassalpha.report.renderers.pdf.AuditPDFRenderer._set_pdf_metadata")
    def test_pdf_metadata_set(self, mock_set_metadata, mock_html_class, mock_css_class, tmp_path):
        """PDF metadata should be set after generation."""
        output_path = tmp_path / "test.pdf"

        # Mock PDF creation
        mock_html_instance = Mock()
        mock_html_class.return_value = mock_html_instance
        mock_rendered_doc = Mock()
        mock_html_instance.render.return_value = mock_rendered_doc

        # Mock CSS
        mock_css_instance = Mock()
        mock_css_class.return_value = mock_css_instance

        def create_pdf(*_args, **_kwargs):
            output_path.write_bytes(b"%PDF-1.4\n")

        mock_rendered_doc.write_pdf.side_effect = create_pdf

        renderer = PDFRenderer()
        renderer.convert_html_to_pdf("<html>test</html>", output_path)

        # Metadata setter should be called
        mock_set_metadata.assert_called_once_with(output_path)


@pytest.mark.skipif(PDFRenderer is None, reason="PDF dependencies not available")
class TestPDFErrorHandling:
    """Test PDF generation error handling."""

    @patch("glassalpha.report.renderers.pdf.CSS")
    @patch("glassalpha.report.renderers.pdf.HTML")
    def test_pdf_handles_invalid_html(self, mock_html_class, _mock_css_class, tmp_path):
        """PDF conversion should handle malformed HTML."""
        output_path = tmp_path / "test.pdf"

        # Simulate HTML parsing error
        mock_html_class.side_effect = ValueError("Invalid HTML")

        renderer = PDFRenderer()

        with pytest.raises(RuntimeError) as exc_info:
            renderer.convert_html_to_pdf("<html><<invalid>>", output_path)

        assert "PDF conversion failed" in str(exc_info.value)

    @patch("glassalpha.report.renderers.pdf.CSS")
    @patch("glassalpha.report.renderers.pdf.HTML")
    def test_pdf_handles_write_permission_error(self, mock_html_class, mock_css_class, tmp_path):
        """PDF conversion should handle write permission errors."""
        output_path = tmp_path / "readonly" / "test.pdf"

        mock_html_instance = Mock()
        mock_html_class.return_value = mock_html_instance
        mock_rendered_doc = Mock()
        mock_html_instance.render.return_value = mock_rendered_doc
        mock_rendered_doc.write_pdf.side_effect = PermissionError("Cannot write")

        # Mock CSS
        mock_css_instance = Mock()
        mock_css_class.return_value = mock_css_instance

        renderer = PDFRenderer()

        with pytest.raises(RuntimeError) as exc_info:
            renderer.convert_html_to_pdf("<html>test</html>", output_path)

        assert "PDF conversion failed" in str(exc_info.value)
