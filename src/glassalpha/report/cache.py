"""Content-hash caching for PDF exports."""

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_export_key(html_path: Path, assets_manifest: dict) -> str:
    """Compute stable hash of HTML + assets for caching.

    Args:
        html_path: Path to HTML file
        assets_manifest: Dict of asset paths and metadata

    Returns:
        16-character hex hash (SHA256 prefix)
    """
    h = hashlib.sha256()

    # Hash HTML content
    h.update(html_path.read_bytes())

    # Hash assets manifest (sorted for stability)
    h.update(json.dumps(assets_manifest, sort_keys=True).encode())

    return h.hexdigest()[:16]


def get_cache_dir() -> Path:
    """Get PDF cache directory."""
    from platformdirs import user_cache_dir

    cache_dir = Path(user_cache_dir("glassalpha")) / "pdf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cached_pdf(key: str) -> Path | None:
    """Return cached PDF if exists, else None."""
    pdf_path = get_cache_dir() / f"{key}.pdf"
    if pdf_path.exists():
        logger.info(f"Cache hit: {key}")
        return pdf_path
    return None


def cache_pdf(key: str, pdf_path: Path) -> None:
    """Store PDF in cache."""
    cached_path = get_cache_dir() / f"{key}.pdf"

    import shutil

    shutil.copy(pdf_path, cached_path)

    logger.info(f"Cached PDF: {key}")


def clear_cache() -> int:
    """Clear PDF cache and return number of files removed."""
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return 0

    count = 0
    for pdf in cache_dir.glob("*.pdf"):
        pdf.unlink()
        count += 1

    logger.info(f"Cleared {count} cached PDFs")
    return count
