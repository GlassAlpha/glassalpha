"""WeasyPrint PDF exporter - fallback for air-gapped environments."""

import logging
from pathlib import Path

from .base import ExportRequest, PdfExporter

logger = logging.getLogger(__name__)


class WeasyPdfExporter(PdfExporter):
    """Simplified WeasyPrint fallback with reduced complexity."""

    @property
    def name(self) -> str:
        return "WeasyPrint (fallback)"

    def export(self, req: ExportRequest, out_path: Path) -> None:
        """Export HTML to PDF using WeasyPrint (simplified mode)."""
        try:
            from weasyprint import CSS, HTML
        except ImportError as e:
            raise RuntimeError(
                "WeasyPrint not installed. Install with: pip install 'glassalpha[pdf_weasyprint]'"
            ) from e

        logger.warning(f"Using {self.name} - output may be simplified for reliability")

        try:
            # Simplified CSS to avoid WeasyPrint hang issues
            simplified_css = CSS(
                string="""
                @page {
                    size: A4;
                    margin: 0.5in;
                }
                body {
                    font-family: Arial, sans-serif;
                    font-size: 10pt;
                }
                /* Disable complex features that cause hangs */
                * {
                    box-shadow: none !important;
                    text-shadow: none !important;
                }
                img {
                    max-width: 100%;
                    page-break-inside: avoid;
                }
                table {
                    page-break-inside: avoid;
                }
            """
            )

            HTML(filename=str(req.html_path)).write_pdf(
                str(out_path),
                stylesheets=[simplified_css],
            )

            logger.info(f"PDF exported with fallback: {out_path.stat().st_size} bytes")

        except Exception as e:
            raise RuntimeError(f"WeasyPrint fallback failed: {e}") from e
