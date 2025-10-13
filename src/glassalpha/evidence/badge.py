"""SVG badge generation for evidence packs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_badge_svg(manifest_data: dict[str, Any]) -> str:
    """Generate badge SVG from manifest data.

    Args:
        manifest_data: Manifest dictionary with performance and fairness metrics

    Returns:
        SVG string for badge display
    """
    # Extract metrics for badge
    accuracy = manifest_data.get("performance", {}).get("accuracy", 0)
    fairness = manifest_data.get("fairness", {}).get("demographic_parity_max_diff", 0)

    # Determine status based on thresholds
    status = "PASS" if accuracy > 0.8 and fairness < 0.05 else "FAIL"
    status_symbol = "✓" if status == "PASS" else "✗"

    # Format metrics for display
    acc_str = f"{accuracy:.3f}" if isinstance(accuracy, (int, float)) else "N/A"
    fair_str = f"{fairness:.3f}" if isinstance(fairness, (int, float)) else "N/A"

    return f"""<svg width="200" height="40" xmlns="http://www.w3.org/2000/svg">
  <rect width="200" height="40" fill="#f0f0f0" stroke="#ccc" stroke-width="1"/>
  <text x="10" y="15" font-family="Arial" font-size="12" fill="#333">
    {status} {status_symbol} | Acc: {acc_str} | ΔDP: {fair_str}
  </text>
</svg>"""


def save_badge_svg(manifest_path: Path, output_path: Path) -> None:
    """Generate and save badge SVG from manifest file.

    Args:
        manifest_path: Path to manifest JSON file
        output_path: Path where to save badge SVG
    """
    # Read manifest data
    import json

    with open(manifest_path, encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Generate badge content
    badge_content = generate_badge_svg(manifest_data)

    # Save to file
    output_path.write_text(badge_content, encoding="utf-8")
