"""Report generation package for GlassAlpha.

Provides functionality for generating audit reports including:
- HTML report rendering
- PDF report generation (optional dependency)
- Plotting and visualization (optional dependency)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from glassalpha.pipeline.audit import AuditResults

from .renderer import AuditReportRenderer, render_audit_report

try:
    from .plots import (
        AuditPlotter,
        create_fairness_plots,
        create_performance_plots,
        create_shap_plots,
        plot_drift_analysis,
    )

    _PLOTS_AVAILABLE = True
except ImportError:
    _PLOTS_AVAILABLE = False

try:
    from .renderers import (
        AuditPDFRenderer,
        render_audit_pdf,
    )
    from .renderers.pdf import PDFConfig

    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False


def render_audit_html(
    audit_results: AuditResults,
    output_path: Path | str,
    report_title: str = "Model Audit Report",
) -> Path:
    """Render audit report as HTML (platform-independent).

    This is the recommended format for development and cross-platform
    compatibility. HTML reports include all audit content and can be
    viewed in any browser.

    Args:
        audit_results: Results from audit pipeline
        output_path: Path where HTML file should be written
        report_title: Title for the audit report

    Returns:
        Path to generated HTML file

    Example:
        >>> from glassalpha.pipeline.audit import run_audit_pipeline
        >>> from glassalpha.report import render_audit_html
        >>> results = run_audit_pipeline(config)
        >>> html_path = render_audit_html(results, "audit.html")

    Note:
        HTML generation works on all platforms without additional dependencies.
        For PDF output, use render_audit_pdf() (requires WeasyPrint).

    """
    output_path = Path(output_path)
    renderer = AuditReportRenderer()
    html_content = renderer.render_audit_report(
        audit_results,
        output_format="html",
        report_title=report_title,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


# Public API
__all__ = [
    "AuditReportRenderer",
    "render_audit_html",
    "render_audit_report",
]

if _PLOTS_AVAILABLE:
    __all__ += [
        "AuditPlotter",
        "create_fairness_plots",
        "create_performance_plots",
        "create_shap_plots",
        "plot_drift_analysis",
    ]

if _PDF_AVAILABLE:
    __all__ += [
        "AuditPDFRenderer",
        "PDFConfig",
        "render_audit_pdf",
    ]
