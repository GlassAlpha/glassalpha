"""High-level PDF export API with caching and optimization."""

import logging
import time
from pathlib import Path

from .cache import cache_pdf, compute_export_key, get_cached_pdf
from .exporters import ExportProfile, ExportRequest, get_exporter
from .images import optimize_assets

logger = logging.getLogger(__name__)


def export_pdf(
    html_path: Path,
    output_path: Path,
    profile: str = "fast",
    use_cache: bool = True,
) -> tuple[Path, bool]:
    """Export HTML to PDF with caching and optimization.

    Args:
        html_path: Path to HTML report
        output_path: Destination PDF path
        profile: Export profile (fast/print/strict)
        use_cache: Enable content-hash caching

    Returns:
        Tuple of (output_path, cache_hit)
    """
    start_time = time.time()

    # Parse profile
    try:
        export_profile = ExportProfile(profile)
    except ValueError:
        logger.warning(f"Unknown profile '{profile}', using 'fast'")
        export_profile = ExportProfile.FAST

    # Check cache first
    cache_hit = False
    if use_cache:
        # Build assets manifest
        assets_dir = html_path.parent
        assets = {
            "plots": [str(p.relative_to(assets_dir)) for p in assets_dir.glob("*.png")],
        }

        cache_key = compute_export_key(html_path, assets)
        cached_pdf = get_cached_pdf(cache_key)

        if cached_pdf:
            import shutil

            shutil.copy(cached_pdf, output_path)
            logger.info(f"PDF export from cache: {time.time() - start_time:.1f}s")
            return output_path, True

    # Optimize images
    assets_dir = html_path.parent
    optimize_assets(assets_dir, profile=profile)

    # Select exporter
    exporter = get_exporter(export_profile)
    logger.info(f"PDF backend: {exporter.name}")

    # Export PDF
    req = ExportRequest(
        html_path=html_path,
        assets_dir=assets_dir,
        profile=export_profile,
    )

    exporter.export(req, output_path)

    # Cache result
    if use_cache:
        cache_pdf(cache_key, output_path)

    elapsed = time.time() - start_time
    logger.info(f"PDF export completed: {elapsed:.1f}s")

    return output_path, False
