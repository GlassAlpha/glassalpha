"""Professional PDF renderer for audit reports using WeasyPrint.

This module provides high-quality PDF generation from HTML audit templates,
with professional styling, page layout, and regulatory compliance features.
"""

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weasyprint import CSS, HTML

from ...pipeline.audit import AuditResults
from ..renderer import AuditReportRenderer

# Suppress WeasyPrint warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="weasyprint")

# Suppress specific known unsupported CSS properties via logging filter


class WeasyPrintCSSFilter(logging.Filter):
    """Filter out known unsupported CSS property warnings from WeasyPrint."""

    def filter(self, record):
        if (
            record.levelno == logging.WARNING
            and "weasyprint" in record.name
            and "Ignored" in record.getMessage()
            and "unknown property" in record.getMessage()
        ):
            return False
        return True


# Apply filter to weasyprint logger
weasyprint_logger = logging.getLogger("weasyprint")
weasyprint_logger.addFilter(WeasyPrintCSSFilter())

logger = logging.getLogger(__name__)


@dataclass
class PDFConfig:
    """Configuration for PDF generation."""

    # Page layout
    page_size: str = "A4"  # A4, Letter, Legal, A3
    page_orientation: str = "portrait"  # portrait, landscape

    # Margins (in inches)
    margin_top: float = 0.75
    margin_bottom: float = 0.75
    margin_left: float = 0.75
    margin_right: float = 0.75

    # PDF metadata
    title: str = "ML Model Audit Report"
    author: str = "GlassAlpha"
    subject: str = "Machine Learning Model Audit"
    creator: str = "GlassAlpha Audit Toolkit"

    # Quality settings
    optimize_size: bool = True
    image_quality: int = 95  # JPEG quality for embedded images

    # Advanced options
    enable_forms: bool = False
    enable_javascript: bool = False
    enable_annotations: bool = False


class AuditPDFRenderer:
    """Professional PDF renderer for audit reports.

    Note: PDF generation is "best effort" for determinism. HTML reports
    are byte-identical across runs (verified), but PDF rendering may have
    minor variations due to WeasyPrint's layout engine. PDFs are suitable
    for human review and regulatory submission but should not be used for
    cryptographic verification of audit reproducibility.
    """

    def __init__(self, config: PDFConfig | None = None):
        """Initialize PDF renderer with configuration.

        Args:
            config: PDF generation configuration

        """
        self.config = config or PDFConfig()
        self.html_renderer = AuditReportRenderer()

        logger.info(f"Initialized PDF renderer with page size: {self.config.page_size}")

    def render_audit_pdf(
        self,
        audit_results: AuditResults,
        output_path: Path,
        template_name: str = "standard_audit.html",
        show_progress: bool = True,
        progress_callback: Callable[[str, int], None] | None = None,
        **template_vars: Any,
    ) -> Path:
        """Render complete audit report as PDF.

        Args:
            audit_results: Complete audit results from pipeline
            output_path: Path where PDF should be saved
            template_name: HTML template to use
            show_progress: Show progress indicator during rendering
            progress_callback: Optional callback(message, percent) for progress updates
            **template_vars: Additional template variables

        Returns:
            Path to generated PDF file

        """
        logger.info(f"Rendering audit PDF to: {output_path}")

        # Validate output path before proceeding
        from ..renderer import validate_output_path

        output_path = Path(output_path)
        validate_output_path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html_content = None

        try:
            # Generate HTML content with embedded plots
            if show_progress:
                progress_callback("Converting HTML template...", 10) if progress_callback else None

            html_content = self.html_renderer.render_audit_report(
                audit_results=audit_results,
                template_name=template_name,
                embed_plots=True,  # Essential for self-contained PDF
                **template_vars,
            )

            logger.debug(f"Generated HTML content: {len(html_content):,} characters")

            if show_progress:
                progress_callback("Rendering pages (this may take 1-2 minutes)...", 40) if progress_callback else None

            # Create WeasyPrint HTML object
            html_doc = HTML(string=html_content, base_url=str(self.html_renderer.template_dir))

            # Generate additional CSS for PDF optimization
            pdf_css = self._generate_pdf_css()
            css_doc = CSS(string=pdf_css)

            # Render PDF with custom configuration
            # NOTE: This call can hang with certain HTML/CSS combinations
            # The timeout is enforced at the CLI level via ThreadPoolExecutor
            logger.debug("Starting WeasyPrint render (this may take 30-60 seconds)...")
            pdf_document = html_doc.render(stylesheets=[css_doc])
            logger.debug("WeasyPrint render complete")

            if show_progress:
                progress_callback("Writing file and normalizing metadata...", 80) if progress_callback else None

            # Write PDF to file with metadata
            pdf_document.write_pdf(
                target=str(output_path),
                pdf_version="1.4",  # Good compatibility
                pdf_identifier=False,  # Deterministic output
                custom_metadata=True,  # Use HTML meta tags for deterministic timestamps
                pdf_variant=None,  # Disable PDF/A for stability (was causing hangs)
                presentational_hints=True,
                optimize_images=self.config.optimize_size,
                jpeg_quality=self.config.image_quality,
            )

            # Set PDF metadata
            self._set_pdf_metadata(output_path)

            # Normalize PDF metadata for byte-identical determinism
            from glassalpha.utils.determinism import normalize_pdf_metadata

            normalize_pdf_metadata(output_path)

            file_size = output_path.stat().st_size
            logger.info(f"Successfully generated PDF: {output_path} ({file_size:,} bytes)")

            if show_progress:
                progress_callback(
                    f"PDF generated successfully: {output_path.name} ({file_size:,} bytes)",
                    100,
                ) if progress_callback else None

            return output_path

        except ImportError as e:
            # Graceful fallback: write HTML next to requested PDF and explain
            html_path = str(Path(output_path).with_suffix(".html"))
            if html_content:
                Path(html_path).write_text(html_content, encoding="utf-8")
                logger.warning(f"PDF backend not available. Wrote HTML to {html_path}")

            raise RuntimeError(
                f"PDF backend (WeasyPrint) is not installed. "
                f"Wrote HTML report to {html_path} instead.\n\n"
                "To enable PDF generation:\n"
                "  pip install 'glassalpha[pdf]'\n"
                "  # or: pip install weasyprint\n\n"
                "Note: HTML reports are fully functional and can be printed as PDF.",
            ) from e

        except Exception as e:
            # Save HTML on any error for debugging
            if html_content:
                debug_html = Path(output_path).with_suffix(".debug.html")
                try:
                    debug_html.write_text(html_content, encoding="utf-8")
                    logger.error(f"PDF generation failed. Saved HTML to {debug_html} for debugging")
                except Exception as write_error:
                    logger.error(f"Failed to save debug HTML: {write_error}")

            logger.error(f"Failed to generate PDF: {e}")
            raise RuntimeError(f"PDF generation failed: {e}") from e

        finally:
            # P2 fix: issue #4 - Explicit resource cleanup for multiprocessing
            # Note: WeasyPrint uses loky for multiprocessing which can leave semaphore warnings
            # This is a known cosmetic issue and doesn't affect functionality
            # See: https://github.com/Kozea/WeasyPrint/issues/
            import gc

            gc.collect()

    def _generate_pdf_css(self) -> str:
        """Generate additional CSS optimized for PDF output.

        Returns:
            CSS string for PDF-specific styling

        """
        css = f"""
        /* PDF-specific styling */
        @page {{
            size: {self.config.page_size} {self.config.page_orientation};
            margin-top: {self.config.margin_top}in;
            margin-bottom: {self.config.margin_bottom}in;
            margin-left: {self.config.margin_left}in;
            margin-right: {self.config.margin_right}in;

            @top-left {{
                content: "{self.config.title}";
                font-size: 9pt;
                color: #666;
                font-family: 'Segoe UI', Arial, sans-serif;
            }}

            @top-right {{
                content: "Page " counter(page) " of " counter(pages);
                font-size: 9pt;
                color: #666;
                font-family: 'Segoe UI', Arial, sans-serif;
            }}

            @bottom-center {{
                content: "Generated by GlassAlpha - Confidential";
                font-size: 8pt;
                color: #999;
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
        }}

        /* Enhanced print styles */
        body {{
            font-size: 10pt;
            line-height: 1.4;
            color: #000;
            background: white;
        }}

        /* Ensure proper page breaks */
        .page-break {{
            page-break-before: always;
            break-before: page;
        }}

        .no-break {{
            page-break-inside: avoid;
            break-inside: avoid;
        }}

        /* Optimize tables for PDF */
        table {{
            page-break-inside: avoid;
            border-collapse: collapse;
            width: 100%;
        }}

        thead {{
            display: table-header-group;
        }}

        tfoot {{
            display: table-footer-group;
        }}

        tr {{
            page-break-inside: avoid;
            break-inside: avoid;
        }}

        /* Enhanced section styling */
        .section {{
            page-break-inside: avoid;
            break-inside: avoid;
            margin-bottom: 1.5rem;
        }}

        h1, h2, h3, h4 {{
            page-break-after: avoid;
            break-after: avoid;
            color: #2E3440;
        }}

        /* Optimize images for PDF */
        img {{
            max-width: 100%;
            height: auto;
            page-break-inside: avoid;
            break-inside: avoid;
        }}

        .plot-container {{
            page-break-inside: avoid;
            break-inside: avoid;
            margin: 1rem 0;
            text-align: center;
        }}

        /* Status indicators - ensure good contrast */
        .status-indicator {{
            font-weight: bold;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-size: 8pt;
        }}

        .status-pass {{
            background-color: #A3BE8C !important;
            color: white !important;
        }}

        .status-fail {{
            background-color: #BF616A !important;
            color: white !important;
        }}

        .status-warning {{
            background-color: #EBCB8B !important;
            color: #2E3440 !important;
        }}

        /* Callout boxes */
        .callout {{
            page-break-inside: avoid;
            break-inside: avoid;
            border: 1pt solid #ddd;
            border-radius: 4pt;
            padding: 0.75rem;
            margin: 0.5rem 0;
        }}

        .callout-warning {{
            border-left: 4pt solid #EBCB8B !important;
            background-color: #fff8e6 !important;
        }}

        .callout-danger {{
            border-left: 4pt solid #BF616A !important;
            background-color: #ffebee !important;
        }}

        .callout-success {{
            border-left: 4pt solid #A3BE8C !important;
            background-color: #e8f5e8 !important;
        }}

        .callout-info {{
            border-left: 4pt solid #5E81AC !important;
            background-color: #e3f2fd !important;
        }}

        /* Code and monospace */
        code {{
            font-family: 'Courier New', 'Monaco', monospace;
            font-size: 8pt;
            background-color: #f5f5f5;
            padding: 0.1rem 0.2rem;
            border-radius: 2pt;
            border: 1pt solid #ddd;
        }}

        /* Two-column layout for PDF */
        .two-column {{
            display: block; /* Simplified for PDF */
        }}

        .two-column > div {{
            margin-bottom: 1rem;
        }}

        /* Footer styling */
        .footer {{
            border-top: 1pt solid #ddd;
            margin-top: 2rem;
            padding-top: 1rem;
            font-size: 8pt;
            color: #666;
        }}

        /* Hyperlink styling for PDF */
        a {{
            color: #5E81AC;
            text-decoration: none;
        }}

        /* Badge styling */
        .badge {{
            font-weight: bold;
            padding: 0.1rem 0.3rem;
            border-radius: 2pt;
            font-size: 7pt;
            text-transform: uppercase;
        }}

        .badge-primary {{
            background-color: #2E3440 !important;
            color: white !important;
        }}

        .badge-success {{
            background-color: #A3BE8C !important;
            color: white !important;
        }}

        .badge-warning {{
            background-color: #EBCB8B !important;
            color: #2E3440 !important;
        }}

        .badge-danger {{
            background-color: #BF616A !important;
            color: white !important;
        }}
        """

        return css

    def _set_pdf_metadata(self, pdf_path: Path) -> None:
        """Set PDF metadata for professional appearance.

        Args:
            pdf_path: Path to the generated PDF file

        """
        try:
            # Import PyPDF2 for metadata manipulation
            try:
                # Try modern pypdf first, then fall back to older PyPDF2
                try:
                    from pypdf import PdfReader, PdfWriter  # pypdf 3.x
                except ImportError:
                    from PyPDF2 import PdfReader, PdfWriter  # pypdf 2.x fallback

                # Read the PDF
                with open(pdf_path, "rb") as file:
                    reader = PdfReader(file)
                    writer = PdfWriter()

                    # Copy all pages
                    for page in reader.pages:
                        writer.add_page(page)

                    # Add metadata
                    metadata = {
                        "/Title": self.config.title,
                        "/Author": self.config.author,
                        "/Subject": self.config.subject,
                        "/Creator": self.config.creator,
                        "/Producer": "GlassAlpha PDF Renderer",
                        "/Keywords": "ML,Audit,Compliance,SHAP,Fairness,Bias,AI",
                    }
                    writer.add_metadata(metadata)

                    # Write the updated PDF
                    with open(pdf_path, "wb") as output_file:
                        writer.write(output_file)

                logger.debug("PDF metadata set successfully")

            except ImportError:
                logger.debug("pypdf not available - PDF metadata not set (optional)")

        except Exception as e:
            logger.warning(f"Failed to set PDF metadata: {e}")

    def convert_html_to_pdf(self, html_content: str, output_path: Path, base_url: str | None = None) -> Path:
        """Convert HTML content directly to PDF.

        Args:
            html_content: HTML content string
            output_path: Path where PDF should be saved
            base_url: Base URL for resolving relative paths

        Returns:
            Path to generated PDF file

        """
        logger.info(f"Converting HTML to PDF: {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Create WeasyPrint HTML object
            html_doc = HTML(string=html_content, base_url=base_url)

            # Generate PDF CSS
            pdf_css = self._generate_pdf_css()
            css_doc = CSS(string=pdf_css)

            # Render to PDF
            pdf_document = html_doc.render(stylesheets=[css_doc])

            pdf_document.write_pdf(
                target=str(output_path),
                pdf_version="1.4",
                pdf_identifier=False,
                optimize_images=self.config.optimize_size,
                jpeg_quality=self.config.image_quality,
            )

            # Set metadata
            self._set_pdf_metadata(output_path)

            # Normalize PDF metadata for byte-identical determinism
            from glassalpha.utils.determinism import normalize_pdf_metadata

            normalize_pdf_metadata(output_path)

            file_size = output_path.stat().st_size
            logger.info(f"HTML to PDF conversion complete: {file_size:,} bytes")

            return output_path

        except ImportError as e:
            # Graceful fallback: write HTML next to requested PDF and explain
            html_path = str(Path(output_path).with_suffix(".html"))
            Path(html_path).write_text(html_content, encoding="utf-8")
            logger.warning(f"PDF backend not available. Wrote HTML to {html_path}")

            raise RuntimeError(
                f"PDF backend (WeasyPrint) is not installed. "
                f"Wrote HTML report to {html_path} instead.\n\n"
                "To enable PDF generation:\n"
                "  pip install 'glassalpha[pdf]'\n"
                "  # or: pip install weasyprint\n\n"
                "Note: HTML reports are fully functional and can be printed as PDF.",
            ) from e

        except Exception as e:
            logger.error(f"HTML to PDF conversion failed: {e}")
            raise RuntimeError(f"PDF conversion failed: {e}") from e


def render_audit_pdf(
    audit_results: AuditResults,
    output_path: Path,
    config: PDFConfig | None = None,
    template_name: str = "standard_audit.html",
    **template_vars: Any,
) -> Path:
    """Convenience function to render audit report as PDF.

    Args:
        audit_results: Complete audit results from pipeline
        output_path: Path where PDF should be saved
        config: PDF generation configuration
        template_name: HTML template to use
        **template_vars: Additional template variables

    Returns:
        Path to generated PDF file

    """
    renderer = AuditPDFRenderer(config=config)
    return renderer.render_audit_pdf(
        audit_results=audit_results,
        output_path=output_path,
        template_name=template_name,
        **template_vars,
    )
