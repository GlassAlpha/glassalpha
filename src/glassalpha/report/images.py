"""Image optimization for PDF export."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def optimize_plot_for_pdf(
    img_path: Path,
    max_width: int = 1500,
    quality: int = 90,
) -> None:
    """Right-size plot for PDF without visible quality loss.

    Args:
        img_path: Path to PNG image
        max_width: Maximum width in pixels (1500 = 150 DPI @ 10")
        quality: JPEG quality for optimization (85-95 recommended)
    """
    try:
        from PIL import Image
    except ImportError:
        logger.debug("PIL not available, skipping image optimization")
        return

    try:
        img = Image.open(img_path)

        if img.width <= max_width:
            return  # Already optimal size

        # Calculate new dimensions maintaining aspect ratio
        scale = max_width / img.width
        new_height = int(img.height * scale)

        # Resize with high-quality algorithm
        img_resized = img.resize((max_width, new_height), Image.LANCZOS)

        # Save with optimization
        img_resized.save(img_path, optimize=True, quality=quality)

        logger.debug(f"Optimized {img_path.name}: {img.width}x{img.height} → {max_width}x{new_height}")

    except Exception as e:
        logger.warning(f"Failed to optimize {img_path.name}: {e}")


def optimize_assets(
    assets_dir: Path,
    profile: str = "fast",
) -> None:
    """Optimize all images in assets directory.

    Args:
        assets_dir: Directory containing plot images
        profile: Export profile (fast=1500px, print=2000px)
    """
    max_width = {
        "fast": 1500,  # 150 DPI @ 10" wide
        "print": 2000,  # 200 DPI @ 10" wide
        "strict": 1500,  # Same as fast for fallback
    }.get(profile, 1500)

    png_files = list(assets_dir.glob("*.png"))
    if not png_files:
        return

    logger.info(f"Optimizing {len(png_files)} images for {profile} profile")

    for png_path in png_files:
        optimize_plot_for_pdf(png_path, max_width=max_width)
