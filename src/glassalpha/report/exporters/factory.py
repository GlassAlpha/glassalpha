"""Factory for selecting PDF export backend."""

import logging
import os

from .base import ExportProfile, PdfExporter

logger = logging.getLogger(__name__)


def get_exporter(profile: ExportProfile | None = None) -> PdfExporter:
    """Get appropriate PDF exporter based on profile and availability.

    Args:
        profile: Export profile (defaults to FAST)

    Returns:
        PdfExporter instance

    Priority:
        1. STRICT profile → WeasyPrint (pure Python)
        2. Environment variable GA_PDF_ENGINE
        3. Playwright if available (default)
        4. WeasyPrint fallback
    """
    profile = profile or ExportProfile.FAST

    # STRICT profile always uses WeasyPrint
    if profile == ExportProfile.STRICT:
        from .pdf_weasy import WeasyPdfExporter

        logger.info("Using WeasyPrint (strict profile)")
        return WeasyPdfExporter()

    # Check environment override
    engine = os.environ.get("GA_PDF_ENGINE", "").lower()
    if engine == "weasyprint":
        from .pdf_weasy import WeasyPdfExporter

        logger.info("Using WeasyPrint (GA_PDF_ENGINE override)")
        return WeasyPdfExporter()

    # Try Playwright first (recommended)
    try:
        from .pdf_chrome import ChromePdfExporter

        logger.info("Using Playwright/Chromium (primary backend)")
        return ChromePdfExporter()
    except ImportError:
        logger.warning("Playwright not available, falling back to WeasyPrint")

    # Fallback to WeasyPrint
    try:
        from .pdf_weasy import WeasyPdfExporter

        logger.info("Using WeasyPrint (fallback)")
        return WeasyPdfExporter()
    except ImportError:
        raise RuntimeError(
            "No PDF backend available. Install with:\n"
            "  pip install 'glassalpha[pdf]'  # Playwright (recommended)\n"
            "  pip install 'glassalpha[pdf_weasyprint]'  # Pure Python fallback"
        )
