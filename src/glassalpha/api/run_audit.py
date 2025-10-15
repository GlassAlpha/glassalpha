"""run_audit() entry point - run complete audit pipeline from configuration file.

This is the programmatic equivalent of `glassalpha audit` CLI command.
Use this function when you want to run audits from Python scripts without
spawning subprocess calls.

LAZY IMPORTS: numpy and pandas are imported inside functions to enable
basic module imports without scientific dependencies installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def run_audit(
    config_path: Path | str,
    output_path: Path | str,
    *,
    strict: bool | None = None,
    profile: str | None = None,
    override_config: Path | str | None = None,
    progress_callback: Any | None = None,
    show_progress: bool = True,
) -> Path:
    """Run complete audit pipeline from configuration file and generate report.

    This is the programmatic equivalent of `glassalpha audit` CLI command.
    Use this function when you want to run audits from Python scripts without
    spawning subprocess calls.

    Args:
        config_path: Path to audit configuration YAML file
        output_path: Path where report should be saved (PDF or HTML)
        strict: Enable strict mode for regulatory compliance (optional)
        profile: Override audit profile from config (optional)
        override_config: Additional config file to override settings (optional)
        progress_callback: Optional callback(message, percent) for progress updates
        show_progress: Show progress messages to console (default: True)

    Returns:
        Path: Path to generated report file

    Raises:
        FileNotFoundError: If config_path does not exist
        ValueError: If configuration is invalid
        RuntimeError: If audit pipeline fails

    Examples:
        Basic usage:
        >>> from glassalpha.api import run_audit
        >>> report_path = run_audit("audit.yaml", "report.pdf")
        >>> print(f"Report generated: {report_path}")

        With custom progress callback:
        >>> def show_progress(msg, pct):
        ...     print(f"[{pct:3d}%] {msg}")
        >>> report_path = run_audit("audit.yaml", "report.pdf", progress_callback=show_progress)

        With strict mode:
        >>> report_path = run_audit("prod.yaml", "report.pdf", strict=True)

        With profile override:
        >>> report_path = run_audit(
        ...     "base.yaml",
        ...     "report.pdf",
        ...     profile="tabular_compliance"
        ... )

    """
    import os
    from datetime import datetime

    from glassalpha.config import load_config_from_file
    from glassalpha.pipeline.audit import run_audit_pipeline

    # Convert to Path objects
    config_path = Path(config_path)
    output_path = Path(output_path)
    if override_config:
        override_config = Path(override_config)

    # Validate config file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    # Validate override config if provided
    if override_config and not override_config.exists():
        raise FileNotFoundError(f"Override configuration file does not exist: {override_config}")

    # Ensure output directory exists
    output_dir = output_path.parent if output_path.parent != Path() else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if output directory is writable
    if not os.access(output_dir, os.W_OK):
        raise RuntimeError(f"Output directory is not writable: {output_dir}")

    # Load configuration
    audit_config = load_config_from_file(
        config_path,
        profile_name=profile,
        strict=strict,
    )

    # Resolve save_path relative to config file directory
    if hasattr(audit_config.model, "save_path") and audit_config.model.save_path:
        config_dir = config_path.parent
        # If save_path is relative, make it absolute relative to config directory
        if not audit_config.model.save_path.is_absolute():
            audit_config.model.save_path = config_dir / audit_config.model.save_path

    # Determine explainer selection (explicit dispatch)
    from glassalpha.explain import select_explainer

    selected_explainer = select_explainer(
        model_type=audit_config.model.type,
        config=audit_config.model_dump(),
    )

    # Create default progress callback if none provided but show_progress=True
    if progress_callback is None and show_progress:

        def default_progress(message: str, percent: int) -> None:
            """Default console progress reporter."""
            print(f"  [{percent:3d}%] {message}")

        progress_callback = default_progress

    # Run audit pipeline with progress feedback
    audit_results = run_audit_pipeline(
        audit_config,
        selected_explainer=selected_explainer,
        progress_callback=progress_callback,
    )

    if not audit_results.success:
        raise RuntimeError(f"Audit pipeline failed: {audit_results.error_message}")

    # Determine output format from extension or config
    output_format = output_path.suffix.lower().lstrip(".")
    if output_format not in ("pdf", "html"):
        # Fall back to config
        output_format = (
            getattr(audit_config.report, "output_format", "html") if hasattr(audit_config, "report") else "html"
        )

    # Generate report
    if output_format == "pdf":
        # Check PDF dependencies
        try:
            from glassalpha.report import _PDF_AVAILABLE
        except ImportError:
            _PDF_AVAILABLE = False

        if not _PDF_AVAILABLE:
            # Fall back to HTML
            output_format = "html"
            if output_path.suffix.lower() == ".pdf":
                output_path = output_path.with_suffix(".html")

    if output_format == "pdf":
        try:
            from glassalpha.report import PDFConfig, render_audit_pdf
        except ImportError:
            raise RuntimeError(
                "PDF generation requires additional dependencies.\n"
                "Install with: pip install 'glassalpha[all]'\n"
                "Or use HTML output: --output audit.html",
            ) from None

        pdf_config = PDFConfig(
            page_size="A4",
            title="ML Model Audit Report",
            author="GlassAlpha",
            subject="Machine Learning Model Compliance Assessment",
            optimize_size=True,
        )

        # Use deterministic timestamp if SOURCE_DATE_EPOCH is set
        import os

        source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
        if source_date_epoch:
            from datetime import UTC, datetime

            report_date = datetime.fromtimestamp(int(source_date_epoch), tz=UTC).strftime("%Y-%m-%d")
            generation_date = datetime.fromtimestamp(int(source_date_epoch), tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            report_date = datetime.now().strftime("%Y-%m-%d")
            generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        pdf_path = render_audit_pdf(
            audit_results=audit_results,
            output_path=output_path,
            config=pdf_config,
            report_title=f"ML Model Audit Report - {report_date}",
            generation_date=generation_date,
            show_progress=show_progress,
            progress_callback=progress_callback,
        )

        # Generate manifest sidecar if available
        if hasattr(audit_results, "execution_info") and "provenance_manifest" in audit_results.execution_info:
            from glassalpha.provenance import write_manifest_sidecar

            try:
                write_manifest_sidecar(
                    audit_results.execution_info["provenance_manifest"],
                    output_path,
                )
            except Exception:
                pass  # Non-critical

        return pdf_path
    # HTML output
    from glassalpha.report import render_audit_report

    # Get compact setting from config (defaults to True to avoid massive files)
    compact = audit_config.runtime.compact_report if hasattr(audit_config, "runtime") else True

    render_audit_report(
        audit_results=audit_results,
        output_path=output_path,
        compact=compact,
        report_title=f"ML Model Audit Report - {datetime.now().strftime('%Y-%m-%d')}",
        generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    # Generate manifest sidecar if available
    if hasattr(audit_results, "execution_info") and "provenance_manifest" in audit_results.execution_info:
        from glassalpha.provenance import write_manifest_sidecar

        try:
            write_manifest_sidecar(
                audit_results.execution_info["provenance_manifest"],
                output_path,
            )
        except Exception:
            pass  # Non-critical

    return output_path


__all__ = ["run_audit"]
