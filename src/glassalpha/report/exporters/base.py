"""Base interface for PDF exporters."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ExportProfile(str, Enum):
    """PDF export quality profiles."""

    FAST = "fast"  # 150 DPI, optimized for speed
    PRINT = "print"  # 200 DPI, high quality
    STRICT = "strict"  # WeasyPrint fallback, pure Python


@dataclass(frozen=True)
class ExportRequest:
    """Request for PDF export."""

    html_path: Path
    assets_dir: Path
    profile: ExportProfile = ExportProfile.FAST
    page_format: str = "A4"
    margins: dict[str, str] | None = None


class PdfExporter:
    """Base interface for PDF export backends."""

    def export(self, req: ExportRequest, out_path: Path) -> None:
        """Export HTML to PDF.

        Args:
            req: Export configuration
            out_path: Destination PDF path

        Raises:
            RuntimeError: If export fails
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Backend name for logging."""
        raise NotImplementedError
