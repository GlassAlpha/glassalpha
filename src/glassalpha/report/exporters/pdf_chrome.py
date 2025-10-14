"""Playwright/Chromium PDF exporter - primary backend."""

import logging
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

from .base import ExportRequest, PdfExporter

logger = logging.getLogger(__name__)


class ChromePdfExporter(PdfExporter):
    """Fast, reliable PDF export using Headless Chrome."""

    @property
    def name(self) -> str:
        return "Playwright (Chromium)"

    def export(self, req: ExportRequest, out_path: Path) -> None:
        """Export HTML to PDF using Chrome's print engine."""
        html_url = req.html_path.resolve().as_uri()

        logger.info(f"Exporting PDF with {self.name}: {out_path.name}")

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
                try:
                    page = browser.new_page()
                    page.goto(html_url, wait_until="networkidle", timeout=60000)

                    margins = req.margins or {
                        "top": "0.5in",
                        "bottom": "0.5in",
                        "left": "0.5in",
                        "right": "0.5in",
                    }

                    page.pdf(
                        path=str(out_path),
                        format=req.page_format,
                        print_background=True,
                        scale=1.0,
                        margin=margins,
                    )

                    logger.info(f"PDF exported successfully: {out_path.stat().st_size} bytes")
                finally:
                    browser.close()

        except PlaywrightTimeout:
            raise RuntimeError(
                "Chrome PDF export timed out after 60 seconds. "
                "Try reducing report complexity or using 'strict' profile."
            )
        except Exception as e:
            raise RuntimeError(f"Chrome PDF export failed: {e}") from e
