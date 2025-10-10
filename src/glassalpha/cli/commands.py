"""CLI commands for GlassAlpha.

This module implements the main commands available in the CLI,
including the core audit command with strict mode support.

ARCHITECTURE NOTE: Exception handling in CLI commands intentionally
suppresses Python tracebacks (using 'from None') to provide clean
user-facing error messages. This is the correct pattern for CLI tools.
Do not "fix" this by adding 'from e' - users don't need stack traces.
"""

import logging
import os
import sys
from pathlib import Path

import typer

from .defaults import get_smart_defaults
from .exit_codes import ExitCode

logger = logging.getLogger(__name__)


def _output_error(message: str) -> None:
    """Output error message to stderr.

    Args:
        message: Error message to display

    """
    typer.echo(message, err=True)


def print_banner(title: str = "GlassAlpha Audit Generation") -> None:
    """Print a standardized banner for CLI commands."""
    typer.echo(title)
    typer.echo("=" * 40)


def _ensure_docs_if_pdf(output_path: str) -> None:
    """Check if PDF output is requested and ensure docs dependencies are available.

    Args:
        output_path: Path to the output file

    Raises:
        SystemExit: If PDF is requested but jinja2 is not available

    """
    from pathlib import Path

    if Path(output_path).suffix.lower() == ".pdf":
        try:
            import weasyprint  # noqa: F401
        except ImportError:
            try:
                import reportlab  # noqa: F401
            except ImportError:
                raise SystemExit(
                    "PDF backend (WeasyPrint) is not installed.\n\n"
                    "To enable PDF generation:\n"
                    "  pip install 'glassalpha[pdf]'\n"
                    "  # or: pip install weasyprint\n\n"
                    "Note: Use --output audit.html to generate HTML reports instead.",
                )


def _bootstrap_components() -> None:
    """Bootstrap basic built-in components for CLI operation.

    This imports the core built-ins that should always be available,
    ensuring the registry has basic models and explainers before
    preflight checks run.
    """
    logger.debug("Bootstrapping basic built-in components")

    # Import core to ensure PassThroughModel is registered (via noop_components)
    try:
        from ..core import PassThroughModel  # noqa: F401 - auto-registered by noop_components

        logger.debug("PassThroughModel available")
    except ImportError as e:
        logger.error(f"Failed to import PassThroughModel: {e}")
        raise typer.Exit(ExitCode.SYSTEM_ERROR) from e

    # Import sklearn models if available (they're optional)
    try:
        from ..models.tabular import sklearn  # noqa: F401 - registers LogisticRegression, etc.

        logger.debug("sklearn models imported")
    except ImportError as e:
        logger.warning(f"sklearn models not available: {e}. Will use passthrough model only.")

    # Import basic explainers
    try:
        from ..explain import (
            coefficients,  # noqa: F401 - registers CoefficientsExplainer
            noop,  # noqa: F401 - registers NoOpExplainer
        )

        logger.debug("Basic explainers imported")
    except ImportError as e:
        logger.warning(f"Failed to import basic explainers: {e}")

    # Import basic metrics
    try:
        from ..metrics.performance import classification  # noqa: F401 - registers accuracy, etc.

        logger.debug("Basic metrics imported")
    except ImportError as e:
        logger.warning(f"Failed to import basic metrics: {e}")

    # Call discover() to ensure entry points are also registered
    from ..core.registry import ModelRegistry
    from ..explain.registry import ExplainerRegistry
    from ..metrics.registry import MetricRegistry
    from ..profiles.registry import ProfileRegistry

    ModelRegistry.discover()
    ExplainerRegistry.discover()
    MetricRegistry.discover()
    ProfileRegistry.discover()

    logger.debug("Component bootstrap completed")


def _run_audit_pipeline(
    config,
    output_path: Path,
    selected_explainer: str | None = None,
    compact: bool = False,
    requested_model: str | None = None,
):
    """Execute the complete audit pipeline and generate report in specified format.

    Args:
        config: Validated audit configuration
        output_path: Path where report should be saved
        selected_explainer: Optional explainer override
        compact: If True, generate compact report (excludes matched pairs from HTML)
        requested_model: Originally requested model type (for fallback tracking)

    Returns:
        tuple: (audit_results, trained_model) where trained_model is the model used in the audit

    """
    import time

    # Import here to avoid circular imports and startup overhead
    from ..pipeline.audit import run_audit_pipeline
    from ..report import render_audit_report
    from ..utils.progress import get_progress_bar

    start_time = time.time()

    # Determine output format from config and check PDF availability
    # Priority: config > file extension > default (HTML)
    if (
        hasattr(config, "report")
        and hasattr(config.report, "output_format")
        and config.report.output_format is not None
    ):
        # Explicit config setting takes precedence
        output_format = config.report.output_format
    # Auto-detect from file extension
    elif output_path.suffix.lower() == ".pdf":
        output_format = "pdf"
    elif output_path.suffix.lower() in [".html", ".htm"]:
        output_format = "html"
    else:
        # Default to HTML for speed and fewer dependencies
        output_format = "html"

    # Check for format/extension mismatch and resolve it
    # CLI flag (--output extension) overrides config for better UX
    if output_format == "pdf" and output_path.suffix.lower() in [".html", ".htm"]:
        typer.secho(
            f"⚠️  Format mismatch: config specifies PDF but output file is {output_path.suffix}",
            fg=typer.colors.YELLOW,
        )
        typer.echo("   Using HTML format to match file extension (CLI flag overrides config)\n")
        output_format = "html"
    elif output_format == "html" and output_path.suffix.lower() == ".pdf":
        typer.secho(
            f"⚠️  Format mismatch: config specifies HTML but output file is {output_path.suffix}",
            fg=typer.colors.YELLOW,
        )
        typer.echo("   Using PDF format to match file extension (CLI flag overrides config)\n")
        output_format = "pdf"

    # Inform user about format inference if not explicitly configured
    if hasattr(config, "report") and hasattr(config.report, "output_format") and config.report.output_format is None:
        if output_path.suffix.lower() == ".pdf":
            typer.echo("ℹ️  Output format inferred from extension: PDF")
        elif output_path.suffix.lower() in [".html", ".htm"]:
            typer.echo("ℹ️  Output format inferred from extension: HTML")

    # Check if PDF backend is available
    try:
        from ..report import _PDF_AVAILABLE
    except ImportError:
        _PDF_AVAILABLE = False

    # Fallback to HTML if PDF requested but not available
    if output_format == "pdf" and not _PDF_AVAILABLE:
        typer.echo("⚠️  PDF backend not available. Generating HTML report instead.")
        typer.echo("💡 To enable PDF reports: pip install 'glassalpha[pdf]' or pip install weasyprint\n")
        output_format = "html"
        # Update output path extension if needed
        if output_path.suffix.lower() == ".pdf":
            output_path = output_path.with_suffix(".html")
    elif output_format == "pdf" and _PDF_AVAILABLE:
        # Warn about potential slow PDF generation
        typer.secho(
            "⚠️  PDF generation may take 1-2 minutes for complex audits. Consider using HTML format for faster results.",
            fg=typer.colors.YELLOW,
        )
        typer.echo("   Tip: Set 'output_format: html' in your config for faster generation.\n")

    try:
        # Step 1: Run audit pipeline with progress bar
        # Check if progress should be shown (respects strict mode, env vars)
        show_progress = not config.strict_mode

        # Create progress bar (auto-detects notebook vs terminal)
        with get_progress_bar(total=100, desc="Running audit", disable=not show_progress, leave=False) as pbar:

            def update_progress(message: str, percent: int):
                """Update progress bar with current step."""
                pbar.set_description(f"Audit: {message}")
                pbar.n = percent
                pbar.refresh()

            # Run the actual audit pipeline with progress callback
            audit_results = run_audit_pipeline(
                config,
                progress_callback=update_progress,
                selected_explainer=selected_explainer,
                requested_model=requested_model,
            )

        if not audit_results.success:
            typer.secho(
                f"Audit pipeline failed: {audit_results.error_message}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        pipeline_time = time.time() - start_time
        typer.secho(f"Audit pipeline completed in {pipeline_time:.2f}s", fg=typer.colors.GREEN)

        # Show audit summary
        _display_audit_summary(audit_results)

        # Step 2: Generate report in specified format
        if output_format == "pdf":
            # Import PDF dependencies only when needed
            try:
                from ..report import PDFConfig, render_audit_pdf
            except ImportError:
                _output_error(
                    "PDF generation requires additional dependencies.\n"
                    "Install with: pip install 'glassalpha[docs]'\n"
                    "Falling back to HTML output...",
                )
                output_format = "html"
                output_path = output_path.with_suffix(".html")

            # Print format-specific message after fallback handling
            if output_format == "pdf":
                typer.echo(f"\nGenerating PDF report: {output_path}")
                typer.echo("⏳ PDF generation in progress... (this may take 1-3 minutes)")

                # Create PDF configuration
                pdf_config = PDFConfig(
                    page_size="A4",
                    title="ML Model Audit Report",
                    author="GlassAlpha",
                    subject="Machine Learning Model Compliance Assessment",
                    optimize_size=True,
                )

                # Generate PDF with timeout protection - MUST be wrapped in deterministic context
                pdf_start = time.time()

                # Use deterministic timestamp for PDF generation
                from glassalpha.utils.determinism import get_deterministic_timestamp

                seed = audit_results.execution_info.get("random_seed") if audit_results.execution_info else None
                deterministic_ts = get_deterministic_timestamp(seed=seed)
                report_date = deterministic_ts.strftime("%Y-%m-%d")
                generation_date = deterministic_ts.strftime("%Y-%m-%d %H:%M:%S UTC")

                # Add timeout protection for PDF generation using concurrent.futures
                import concurrent.futures

                pdf_path = None

                def pdf_generation_worker():
                    """Worker function for PDF generation that can run in a thread."""
                    try:
                        # PDF generation must happen within deterministic context
                        from ..utils.determinism import deterministic

                        with deterministic(seed=seed or 42, strict=True):
                            pdf_path = render_audit_pdf(
                                audit_results=audit_results,
                                output_path=output_path,
                                config=pdf_config,
                                report_title=f"ML Model Audit Report - {report_date}",
                                generation_date=generation_date,
                            )
                        return pdf_path
                    except Exception as e:
                        raise e

            try:
                # Run PDF generation with timeout using ThreadPoolExecutor
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(pdf_generation_worker)
                    pdf_path = future.result(timeout=300)  # 5 minutes timeout

                if pdf_path is None:
                    raise RuntimeError("PDF generation failed - no output path returned")

            except concurrent.futures.TimeoutError:
                typer.secho("\n❌ PDF generation timed out after 5 minutes", fg=typer.colors.RED)
                typer.echo("💡 Try using HTML format for faster results:")
                typer.echo("   glassalpha audit --config your_config.yaml --output report.html")
                typer.echo("   # or set 'output_format: html' in your config")
                raise typer.Exit(code=1)
            except Exception as e:
                typer.secho(f"\n❌ PDF generation failed: {e}", fg=typer.colors.RED)
                typer.echo("💡 Try using HTML format as a fallback:")
                typer.echo("   glassalpha audit --config your_config.yaml --output report.html")
                typer.echo("   # or set 'output_format: html' in your config")
                raise typer.Exit(code=1)

            pdf_time = time.time() - pdf_start

            # Validate PDF was actually created (P0 fix: issue #1, #3)
            if not pdf_path.exists():
                typer.secho("\n❌ PDF generation failed - file was not created", fg=typer.colors.RED, err=True)
                typer.echo("\nPossible causes:", err=True)
                typer.echo("  • WeasyPrint dependencies missing or misconfigured", err=True)
                typer.echo("  • System font issues", err=True)
                typer.echo("  • Insufficient memory or disk space", err=True)
                typer.echo("\nSolutions:", err=True)
                typer.echo("  1. Use HTML output instead: --output report.html", err=True)
                typer.echo("  2. Check PDF dependencies: glassalpha doctor", err=True)
                typer.echo("  3. Install PDF dependencies: pip install 'glassalpha[pdf]'", err=True)
                raise typer.Exit(ExitCode.SYSTEM_ERROR)

            file_size = pdf_path.stat().st_size

            # Generate manifest sidecar if provenance manifest is available
            manifest_path = None
            if hasattr(audit_results, "execution_info") and "provenance_manifest" in audit_results.execution_info:
                from ..provenance import write_manifest_sidecar  # noqa: PLC0415

                try:
                    manifest_path = write_manifest_sidecar(
                        audit_results.execution_info["provenance_manifest"],
                        output_path,
                    )
                    typer.echo(f"Manifest: {manifest_path}")
                except Exception as e:
                    logger.warning(f"Failed to write manifest sidecar: {e}")

            # Success message
            total_time = time.time() - start_time
            typer.echo("\nAudit Report Generated Successfully!")

        elif output_format == "html":
            typer.echo(f"\nGenerating HTML report: {output_path}")

            # Generate HTML
            html_start = time.time()
            # Use deterministic timestamp if SOURCE_DATE_EPOCH is set

            # Use deterministic timestamp for reproducibility
            from glassalpha.utils.determinism import get_deterministic_timestamp  # noqa: PLC0415

            seed = audit_results.execution_info.get("random_seed") if audit_results.execution_info else None
            deterministic_ts = get_deterministic_timestamp(seed=seed)
            report_date = deterministic_ts.strftime("%Y-%m-%d")
            generation_date = deterministic_ts.strftime("%Y-%m-%d %H:%M:%S UTC")

            html_content = render_audit_report(
                audit_results=audit_results,
                output_path=output_path,
                compact=compact,
                report_title=f"ML Model Audit Report - {report_date}",
                generation_date=generation_date,
            )

            html_time = time.time() - html_start
            file_size = len(html_content.encode("utf-8"))

            # Generate manifest sidecar if provenance manifest is available
            manifest_path = None
            if hasattr(audit_results, "execution_info") and "provenance_manifest" in audit_results.execution_info:
                from ..provenance import write_manifest_sidecar  # noqa: PLC0415

                try:
                    manifest_path = write_manifest_sidecar(
                        audit_results.execution_info["provenance_manifest"],
                        output_path,
                    )
                    typer.echo(f"Manifest: {manifest_path}")
                except Exception as e:
                    logger.warning(f"Failed to write manifest sidecar: {e}")

            # Success message
            total_time = time.time() - start_time
            typer.echo("\nAudit Report Generated Successfully!")

        else:
            typer.secho(
                f"Error: Unsupported output format '{output_format}'\n\n"
                f"Supported formats:\n"
                f"  - html (recommended): --output audit.html\n"
                f"  - pdf (requires weasyprint): --output audit.pdf\n\n"
                f"Fix: Change output file extension to .html or .pdf",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        typer.echo("=" * 50)

        # Show final output information
        if output_format == "pdf":
            typer.secho(f"Output: {output_path}", fg=typer.colors.GREEN)
            typer.echo(f"Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
            typer.echo(f"Total time: {total_time:.2f}s")
            typer.echo(f"   Pipeline: {pipeline_time:.2f}s")
            typer.echo(f"   PDF generation: {pdf_time:.2f}s")
        elif output_format == "html":
            typer.secho(f"Output: {output_path}", fg=typer.colors.GREEN)
            file_size_mb = file_size / (1024 * 1024)
            typer.echo(f"Size: {file_size:,} bytes ({file_size_mb:.1f} MB)")

            # Enhanced file size warnings with actionable tips
            if file_size_mb > 50:
                # Very large file (>50MB) - critical warning
                typer.secho(
                    f"⚠️  Large report size ({file_size_mb:.1f} MB) detected",
                    fg=typer.colors.YELLOW,
                )
                typer.echo("💡 Note: You used --full-report which includes all individual fairness pairs")
                typer.echo("   • Use default (compact mode) for <1MB reports")
                typer.echo("   • Use --fast for development (reduces bootstrap samples)")
                typer.echo("   • Full data always available in manifest.json sidecar")
            elif file_size_mb > 1:
                # Moderate file (1-50MB) - gentle warning
                typer.secho(
                    f"⚠️  Report size ({file_size_mb:.1f} MB) is larger than expected",
                    fg=typer.colors.YELLOW,
                )
                typer.echo("💡 Tip: Default compact mode should produce <1MB reports")

            typer.echo(f"Total time: {total_time:.2f}s")
            typer.echo(f"   Pipeline: {pipeline_time:.2f}s")
            typer.echo(f"   HTML generation: {html_time:.2f}s")

        typer.echo("\nThe audit report is ready for review and regulatory submission.")

        # Regulatory compliance message
        if config.strict_mode:
            typer.secho(
                "\nStrict mode: Report meets regulatory compliance requirements",
                fg=typer.colors.YELLOW,
            )

        return audit_results

    except Exception as e:
        typer.secho(f"Audit failed: {e!s}", fg=typer.colors.RED, err=True)

        # Show more details in verbose mode
        if "--verbose" in sys.argv or "-v" in sys.argv:
            logger.exception("Detailed audit failure information:")

        raise typer.Exit(ExitCode.USER_ERROR)


def _display_audit_summary(audit_results) -> None:
    """Display a summary of audit results."""
    typer.echo("\nAudit Summary:")

    # Model performance
    if audit_results.model_performance:
        perf_count = len(
            [m for m in audit_results.model_performance.values() if isinstance(m, dict) and "error" not in m],
        )
        typer.echo(f"  Performance metrics: {perf_count} computed")

        # Show key metrics
        for name, result in audit_results.model_performance.items():
            if isinstance(result, dict) and "accuracy" in result:
                accuracy = result["accuracy"]

                # Warn about suspiciously high accuracy (>98%) which may indicate data leakage or overfitting
                if accuracy > 0.98:
                    status = "SUSPICIOUS"
                    typer.echo(f"     {status} {name}: {accuracy:.1%}")
                    typer.secho(
                        "     ⚠️  Accuracy >98% may indicate data leakage or overfitting",
                        fg=typer.colors.YELLOW,
                    )
                else:
                    status = "GOOD" if accuracy > 0.8 else "OK" if accuracy > 0.6 else "POOR"
                    typer.echo(f"     {status} {name}: {accuracy:.1%}")
                break

    # Fairness analysis
    if audit_results.fairness_analysis:
        # Check if fairness analysis was skipped
        if audit_results.fairness_analysis.get("skipped"):
            typer.echo("  Fairness: SKIPPED (no protected attributes configured)")
        else:
            bias_detected = []
            total_metrics = 0
            failed_metrics = 0

            for attr, metrics in audit_results.fairness_analysis.items():
                for metric, result in metrics.items():
                    total_metrics += 1
                    if isinstance(result, dict):
                        if "error" in result:
                            failed_metrics += 1
                        elif result.get("is_fair") is False:
                            bias_detected.append(f"{attr}.{metric}")

            computed_metrics = total_metrics - failed_metrics
            typer.echo(f"  Fairness metrics: {computed_metrics}/{total_metrics} computed")

            if bias_detected:
                typer.secho(f"     WARNING: Bias detected in: {', '.join(bias_detected[:2])}", fg=typer.colors.YELLOW)
            elif computed_metrics > 0:
                typer.secho("     No bias detected", fg=typer.colors.GREEN)

    # SHAP explanations
    if audit_results.explanations:
        has_importance = "global_importance" in audit_results.explanations

        if has_importance:
            typer.echo("  Explanations: Global feature importance available")

            # Show top feature
            importance = audit_results.explanations.get("global_importance", {})
            if importance:
                top_feature = max(importance.items(), key=lambda x: abs(x[1]))
                typer.echo(f"     Most important: {top_feature[0]} ({top_feature[1]:+.3f})")
        else:
            typer.echo("  Explanations: Not available")

    # Data summary
    if audit_results.data_summary and "shape" in audit_results.data_summary:
        rows, cols = audit_results.data_summary["shape"]
        typer.echo(f"  Dataset: {rows:,} samples, {cols} features")

    # Selected components with fallback indication
    if audit_results.selected_components:
        typer.echo(f"  Components: {len(audit_results.selected_components)} selected")

        # Show model (with fallback indication if applicable)
        model_info = audit_results.selected_components.get("model")
        if model_info:
            model_name = model_info.get("name", "unknown")
            requested_model = model_info.get("requested")

            if requested_model and requested_model != model_name:
                typer.secho(
                    f"     Model: {model_name} (fallback from {requested_model})",
                    fg=typer.colors.YELLOW,
                )
            else:
                typer.echo(f"     Model: {model_name}")

        # Show explainer (with reasoning if available)
        explainer_info = audit_results.selected_components.get("explainer")
        if explainer_info:
            explainer_name = explainer_info.get("name", "unknown")
            reason = explainer_info.get("reason", "")

            if reason:
                typer.echo(f"     Explainer: {explainer_name} ({reason})")
            else:
                typer.echo(f"     Explainer: {explainer_name}")

        # Show preprocessing mode if available
        prep_info = audit_results.selected_components.get("preprocessing")
        if prep_info:
            prep_mode = prep_info.get("mode", "unknown")
            if prep_mode == "auto":
                typer.secho(
                    f"     Preprocessing: {prep_mode} (not production-ready)",
                    fg=typer.colors.YELLOW,
                )
            else:
                typer.echo(f"     Preprocessing: {prep_mode}")


def _binarize_sensitive_features_for_shift(
    sensitive_features,
    shift_specs: list[str],
):
    """Binarize sensitive features for shift analysis.

    Shift analysis requires binary attributes (0/1), but datasets may have
    categorical attributes that need to be binarized.

    Args:
        sensitive_features: DataFrame with sensitive attributes
        shift_specs: List of shift specifications to determine which attributes to binarize

    Returns:
        DataFrame with binarized attributes for shift analysis

    """
    # Create a copy to avoid modifying the original
    binarized = sensitive_features.copy()

    # Get unique attribute names from shift specs
    shift_attributes = set()
    for spec in shift_specs:
        try:
            from ..metrics.shift import parse_shift_spec

            attribute, _ = parse_shift_spec(spec)
            shift_attributes.add(attribute)
        except ValueError:
            # Skip invalid specs
            continue

    # Binarize each attribute needed for shift analysis
    for attribute in shift_attributes:
        if attribute not in binarized.columns:
            # Provide helpful error message listing available attributes
            available = ", ".join(binarized.columns.tolist())
            raise ValueError(
                f"Attribute '{attribute}' not found in sensitive_features. "
                f"Available: [{available}]. "
                f"Add '{attribute}' to 'protected_attributes' in your config to enable shift analysis.",
            )

        col = binarized[attribute]

        # Skip if already binary (0/1 or boolean)
        if col.dtype in [bool, "int8", "int16", "int32", "int64"]:
            unique_vals = col.unique()
            if set(unique_vals).issubset({0, 1}):
                continue  # Already binary

        # Handle categorical attributes
        if col.dtype == "object" or col.dtype.name == "category":
            unique_vals = col.dropna().unique()

            # For binary categorical (e.g., "male"/"female")
            if len(unique_vals) == 2:
                # Map first value to 0, second to 1
                val1, val2 = unique_vals
                binarized[attribute] = col.map({val1: 0, val2: 1}).astype(int)

            # For multi-class categorical, create binary indicators for each class
            elif len(unique_vals) > 2:
                # For shift analysis, we need to choose a reference category
                # Use the most frequent as reference (0) and others as 1
                value_counts = col.value_counts()
                reference_val = value_counts.index[0]

                # Create binary indicator (1 if not reference, 0 if reference)
                binarized[attribute] = (col != reference_val).astype(int)

                typer.echo(f"  Binarized '{attribute}': reference='{reference_val}', others=1")

            else:
                typer.secho(f"Warning: Cannot binarize '{attribute}' - insufficient categories", fg=typer.colors.YELLOW)

        # Handle numeric attributes (e.g., age)
        elif col.dtype in ["int64", "float64"]:
            # For numeric attributes, create binary based on median split
            median_val = col.median()
            binarized[attribute] = (col > median_val).astype(int)

            typer.echo(f"  Binarized '{attribute}': median={median_val:.1f}, above=1")

        else:
            typer.secho(
                f"Warning: Cannot binarize '{attribute}' - unsupported dtype {col.dtype}",
                fg=typer.colors.YELLOW,
            )

    return binarized


def _validate_shift_specs_against_config(
    shift_specs: list[str],
    protected_attributes: list[str],
) -> None:
    """Validate shift specs reference valid protected attributes.

    Raises:
        ValueError: If any shift attribute not in protected_attributes

    """
    from ..metrics.shift import parse_shift_spec

    invalid_attrs = []
    for spec in shift_specs:
        try:
            attribute, _ = parse_shift_spec(spec)
            if attribute not in protected_attributes:
                invalid_attrs.append(attribute)
        except ValueError as e:
            raise ValueError(f"Invalid shift spec '{spec}': {e}") from e

    if invalid_attrs:
        available = ", ".join(protected_attributes)
        invalid_list = ", ".join(invalid_attrs)
        raise ValueError(
            f"Shift analysis failed: Attributes [{invalid_list}] not found in "
            f"protected_attributes.\n"
            f"Available: [{available}]\n\n"
            f"Fix: Add missing attributes to 'protected_attributes' in your config:\n"
            + "".join(f"  protected_attributes:\n    - {attr}\n" for attr in invalid_attrs),
        )


def _run_shift_analysis(
    audit_config,
    output: Path,
    shift_specs: list[str],
    threshold: float | None,
) -> int:
    """Run demographic shift analysis and export results.

    Args:
        audit_config: Validated audit configuration
        output: Base output path (for JSON sidecar)
        shift_specs: List of shift specifications (e.g., ["gender:+0.1", "age:-0.05"])
        threshold: Degradation threshold for failure (optional)

    Returns:
        Exit code (0 = pass, 1 = violations detected)

    """
    import json

    from ..data import TabularDataLoader
    from ..metrics.shift import parse_shift_spec, run_shift_analysis
    from ..models import load_model_from_config

    try:
        # Validate shift specs BEFORE any work
        _validate_shift_specs_against_config(
            shift_specs,
            audit_config.data.protected_attributes or [],
        )

        # Load dataset using same loader as audit pipeline
        typer.echo("\nLoading test data...")
        data_loader = TabularDataLoader()

        # Handle built-in datasets vs file paths
        if audit_config.data.dataset == "german_credit":
            from ..datasets import get_german_credit_schema, load_german_credit

            data = load_german_credit()
            schema = get_german_credit_schema()
        else:
            # Load from file path
            from ..data.tabular import TabularDataSchema

            data = data_loader.load(audit_config.data.path, audit_config.data)
            # Convert DataConfig to TabularDataSchema
            schema = TabularDataSchema(
                target=audit_config.data.target_column,
                features=audit_config.data.feature_columns
                or [col for col in data.columns if col != audit_config.data.target_column],
                sensitive_features=audit_config.data.protected_attributes,
                categorical_features=None,  # Optional: will be inferred if not provided
                numeric_features=None,  # Optional: will be inferred if not provided
            )

        # Extract features, target, and sensitive features
        X_test, y_test, sensitive_features = data_loader.extract_features_target(data, schema)

        # Preprocess features for model training (handle categorical encoding)
        from ..utils.preprocessing import preprocess_auto

        X_test = preprocess_auto(X_test)

        # Binarize sensitive features for shift analysis (shift analysis requires binary attributes)
        sensitive_features_binarized = _binarize_sensitive_features_for_shift(sensitive_features, shift_specs)

        # Load model using same approach as audit pipeline
        typer.echo("Loading model...")
        model = load_model_from_config(audit_config.model)

        # Train model if not loaded from file
        if not hasattr(model, "_is_fitted") or not model._is_fitted:
            typer.echo("Training model...")
            from ..pipeline.train import train_from_config

            # Create a minimal config for training
            train_config = type(
                "TrainConfig",
                (),
                {
                    "model": audit_config.model,
                    "reproducibility": audit_config.reproducibility,
                },
            )()
            model = train_from_config(train_config, X_test, y_test)

        # Generate predictions
        typer.echo("Generating predictions...")

        # Binary classification predictions
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            # Handle binary classification (get positive class probability)
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
        else:
            y_proba = None

        y_pred = model.predict(X_test)

        # Convert to binary if needed
        if hasattr(y_pred, "squeeze"):
            y_pred = y_pred.squeeze()
        y_pred = (y_pred > 0.5).astype(int) if y_pred.dtype == float else y_pred.astype(int)

        # Run shift analysis for each shift specification
        results = []
        has_violations = False

        for shift_spec in shift_specs:
            try:
                # Parse shift specification
                attribute, shift_value = parse_shift_spec(shift_spec)

                typer.echo(f"\nAnalyzing shift: {attribute} {shift_value:+.2f} ({shift_value * 100:+.0f}pp)")

                # Run shift analysis
                result = run_shift_analysis(
                    y_true=y_test,
                    y_pred=y_pred,
                    sensitive_features=sensitive_features_binarized,
                    attribute=attribute,
                    shift=shift_value,
                    y_proba=y_proba,
                    threshold=threshold,
                )

                # Display result
                typer.echo(f"  Original proportion: {result.shift_spec.original_proportion:.3f}")
                typer.echo(f"  Shifted proportion:  {result.shift_spec.shifted_proportion:.3f}")
                typer.echo(f"  Gate status: {result.gate_status}")

                if result.violations:
                    has_violations = True
                    typer.secho("  Violations:", fg=typer.colors.RED)
                    for violation in result.violations:
                        typer.secho(f"    • {violation}", fg=typer.colors.RED)
                else:
                    typer.secho("  ✓ No violations detected", fg=typer.colors.GREEN)

                # Add to results
                results.append(result.to_dict())

            except ValueError as e:
                typer.secho(f"\n✗ Failed to process shift '{shift_spec}': {e}", fg=typer.colors.RED)
                return ExitCode.USER_ERROR

        # Export results to JSON sidecar
        shift_json_path = output.with_suffix(".shift_analysis.json")

        export_data = {
            "shift_analysis": {
                "threshold": threshold,
                "shifts": results,
                "summary": {
                    "total_shifts": len(results),
                    "violations_detected": has_violations,
                    "failed_shifts": sum(1 for r in results if r["gate_status"] == "FAIL"),
                    "warning_shifts": sum(1 for r in results if r["gate_status"] == "WARNING"),
                },
            },
        }

        with shift_json_path.open("w") as f:
            json.dump(export_data, f, indent=2)

        typer.echo(f"\n📄 Shift analysis results: {shift_json_path}")

        # Return exit code based on violations
        if has_violations and threshold is not None:
            return ExitCode.VALIDATION_ERROR
        return 0

    except ValueError:
        # Re-raise validation errors with our detailed error message
        raise
    except Exception as e:
        # Only catch unexpected errors
        typer.secho(f"\n✗ Unexpected error in shift analysis: {e}", fg=typer.colors.RED)
        if "--verbose" in sys.argv or "-v" in sys.argv:
            logger.exception("Detailed shift analysis failure:")
        return ExitCode.SYSTEM_ERROR  # Not user error


def audit(  # pragma: no cover
    # Typer requires function calls in defaults - this is the documented pattern
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to audit configuration YAML file (auto-detects glassalpha.yaml, audit.yaml, config.yaml)",
        # Remove exists=True to handle file checking manually for better error messages
        file_okay=True,
        dir_okay=False,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Path for output report (defaults to {config_name}.html)",
    ),
    strict: bool | None = typer.Option(
        None,
        "--strict",
        "-s",
        help="Enable strict mode for regulatory compliance (auto-enabled for prod*/production* configs). Allows built-in datasets.",
    ),
    strict_full: bool = typer.Option(
        False,
        "--strict-full",
        help="Enable full strict mode for maximum regulatory compliance. Requires explicit data schemas and model paths. Disallows built-in datasets.",
    ),
    repro: bool | None = typer.Option(
        None,
        "--repro",
        help="Enable deterministic reproduction mode (auto-enabled in CI and for test* configs)",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        "-p",
        help="Override audit profile",
    ),
    override_config: Path | None = typer.Option(
        None,
        "--override",
        help="Additional config file to override settings",
        file_okay=True,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate configuration without generating report",
    ),
    no_fallback: bool = typer.Option(
        False,
        "--no-fallback",
        help="Fail if requested components are unavailable (no automatic fallbacks)",
    ),
    show_defaults: bool = typer.Option(
        False,
        "--show-defaults",
        help="Show inferred defaults and exit (useful for debugging)",
    ),
    check_output: bool = typer.Option(
        False,
        "--check-output",
        help="Check output paths are writable and exit (pre-flight validation)",
    ),
    check_shift: list[str] = typer.Option(
        [],
        "--check-shift",
        help="Test model robustness under demographic shifts (e.g., 'gender:+0.1'). Can specify multiple.",
    ),
    fail_on_degradation: float | None = typer.Option(
        None,
        "--fail-on-degradation",
        help="Exit with error if any metric degrades by more than this threshold (e.g., 0.05 for 5pp).",
    ),
    save_model: Path | None = typer.Option(
        None,
        "--save-model",
        help="Save the trained model to the specified path (e.g., model.pkl). Required for reasons/recourse commands.",
    ),
    fast: bool = typer.Option(
        False,
        "--fast",
        help="Fast demo mode: reduce bootstrap samples to 100 for lightning-quick audits (~2-3s vs ~5-7s)",
    ),
    compact_report: bool = typer.Option(
        True,
        "--compact-report/--full-report",
        help="Generate compact report (<1MB, default) by excluding individual fairness matched pairs from HTML. Use --full-report for complete data (may be 50-100MB). Full data always saved in manifest.json.",
    ),
):
    """Generate a compliance audit report (HTML/PDF) with optional shift testing.

    This is the main command for GlassAlpha. It loads a configuration file,
    runs the audit pipeline, and generates a deterministic audit report.

    Smart Defaults:
        If no --config is provided, searches for: glassalpha.yaml, audit.yaml, config.yaml
        If no --output is provided, uses {config_name}.html
        Strict mode auto-enables for prod*/production* configs
        Repro mode auto-enables in CI environments and for test* configs

    Examples:
        # Minimal usage (uses smart defaults)
        glassalpha audit

        # Explicit paths
        glassalpha audit --config audit.yaml --output report.html

        # See what defaults would be used
        glassalpha audit --show-defaults

        # Check output paths before running audit
        glassalpha audit --check-output

        # Strict mode for regulatory compliance
        glassalpha audit --config production.yaml  # Auto-enables strict!

        # Override specific settings
        glassalpha audit -c base.yaml --override custom.yaml

        # Fail if components unavailable (no fallbacks)
        glassalpha audit --no-fallback

        # Stress test for demographic shifts (E6.5)
        glassalpha audit --check-shift gender:+0.1

        # Multiple shifts with degradation threshold
        glassalpha audit --check-shift gender:+0.1 --check-shift age:-0.05 --fail-on-degradation 0.05

    """
    try:
        # Apply smart defaults
        try:
            defaults = get_smart_defaults(
                config=config,
                output=output,
                strict=strict if strict is not None else None,
                repro=repro if repro is not None else None,
            )
        except ValueError as e:
            _output_error(f"Configuration error: {e}")
            raise typer.Exit(ExitCode.USER_ERROR) from None

        # Extract resolved values
        config = defaults["config"]
        output = defaults["output"]
        strict = defaults["strict"]
        repro = defaults["repro"]

        # Handle strict_full override - this takes precedence over regular strict mode
        if strict_full:
            strict = True  # Enable strict mode

        # Show defaults if requested
        if show_defaults:
            typer.echo("Inferred defaults:")
            typer.echo(f"  config: {config}")
            typer.echo(f"  output: {output}")
            typer.echo(f"  strict: {strict}")
            typer.echo(f"  repro:  {repro}")
            return

        # Check file existence early with specific error message
        if not config.exists():
            _output_error(
                f"Configuration file does not exist: {config}\n\n"
                f"Quick fixes:\n"
                f"  1. Create a config: glassalpha init\n"
                f"  2. List datasets: glassalpha datasets list\n"
                f"  3. Use example template: glassalpha init --template quickstart\n\n"
                f"Examples:\n"
                f"  glassalpha init --template quickstart --output my-audit.yaml\n"
                f"  glassalpha audit --config my-audit.yaml --output report.html",
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        # Check override config if provided
        if override_config and not override_config.exists():
            _output_error(
                f"Override configuration file does not exist: {override_config}\n\n"
                f"Fix: Check file path and spelling, or remove --override flag",
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        # Validate output directory exists before doing any work
        output_dir = output.parent if output.parent != Path() else Path.cwd()
        if not output_dir.exists():
            _output_error(f"Output directory does not exist: {output_dir}. Create it with: mkdir -p {output_dir}")
            raise typer.Exit(ExitCode.USER_ERROR)

        # Check if output directory is writable
        if not os.access(output_dir, os.W_OK):
            _output_error(f"Output directory is not writable: {output_dir}")
            raise typer.Exit(ExitCode.SYSTEM_ERROR)

        # Validate manifest sidecar path will be writable
        manifest_path = output.with_suffix(".manifest.json")
        if manifest_path.exists() and not os.access(manifest_path, os.W_OK):
            _output_error(
                f"Cannot overwrite existing manifest (read-only): {manifest_path}. "
                "Make the file writable or remove it before running audit",
            )
            raise typer.Exit(ExitCode.SYSTEM_ERROR)

        # Check output paths and exit if requested
        if check_output:
            typer.secho("✓ Output path validation:", fg=typer.colors.GREEN, bold=True)
            typer.echo(f"  Output file:      {output}")
            typer.echo(f"  Output directory: {output_dir}")
            typer.echo(f"  Manifest sidecar: {manifest_path}")
            typer.echo()
            typer.secho("✓ All output paths are writable", fg=typer.colors.GREEN)

            # Show what would be created
            if output.exists():
                typer.echo(f"  Note: {output.name} will be overwritten")
            if manifest_path.exists():
                typer.echo(f"  Note: {manifest_path.name} will be overwritten")

            return

        # Import here to avoid circular imports
        from ..config import load_config_from_file
        from ..core import list_components
        from .preflight import preflight_check_dependencies, preflight_check_model

        # Bootstrap basic components before any preflight checks
        _bootstrap_components()

        print_banner()

        # Preflight checks - ensure dependencies are available
        if not preflight_check_dependencies():
            raise typer.Exit(ExitCode.VALIDATION_ERROR)

        # Load configuration - this doesn't need heavy ML libraries
        typer.echo(f"Loading configuration from: {config}")
        if override_config:
            typer.echo(f"Applying overrides from: {override_config}")

        audit_config = load_config_from_file(
            config,
            override_path=override_config,
            profile_name=profile,
            strict=strict,
            strict_full=strict_full,
        )

        # Apply fast mode if requested (reduces bootstrap samples for quick demos)
        if fast:
            if not hasattr(audit_config, "metrics") or audit_config.metrics is None:
                from ..config.schema import MetricsConfig

                # Create MetricsConfig but preserve any existing settings from YAML
                metrics_config = MetricsConfig()
                if hasattr(audit_config, "metrics") and audit_config.metrics is not None:
                    # Copy existing settings
                    for attr in [
                        "performance",
                        "fairness",
                        "drift",
                        "stability",
                        "custom",
                        "compute_confidence_intervals",
                    ]:
                        if hasattr(audit_config.metrics, attr):
                            setattr(metrics_config, attr, getattr(audit_config.metrics, attr))

                audit_config.metrics = metrics_config

            # Reduce bootstrap samples for fast mode
            audit_config.metrics.n_bootstrap = 100
            typer.secho("⚡ Fast mode enabled - using 100 bootstrap samples for quick demo", fg=typer.colors.CYAN)

        # Show progress indication before starting pipeline
        typer.echo()
        typer.secho("⏱️  Running audit pipeline...", fg=typer.colors.CYAN)
        if not fast:
            typer.echo("   Estimated time: 5-7 seconds (use --fast for 2-3 seconds)")
        else:
            typer.echo("   Estimated time: 2-3 seconds")
        typer.echo()

        # Validate model availability and apply fallbacks (or fail if no_fallback is set)
        audit_config, requested_model = preflight_check_model(audit_config, allow_fallback=not no_fallback)

        # Determine explainer selection early for consistent display
        from ..explain.registry import ExplainerRegistry

        selected_explainer = ExplainerRegistry.find_compatible(audit_config.model.type, audit_config.model_dump())
        typer.echo(f"Explainer: {selected_explainer}")

        # Apply repro mode if requested
        if repro:
            from ..runtime import set_repro  # noqa: PLC0415

            # Use config seed if available, otherwise default
            seed = (
                getattr(audit_config.reproducibility, "random_seed", 42)
                if hasattr(audit_config, "reproducibility")
                else 42
            )

            typer.echo("🔒 Enabling deterministic reproduction mode...")
            repro_status = set_repro(
                seed=seed,
                strict=True,  # Always use strict mode with --repro flag
                thread_control=True,  # Control threads for determinism
                warn_on_failure=True,
            )

            successful = sum(1 for control in repro_status["controls"].values() if control.get("success", False))
            total = len(repro_status["controls"])
            typer.echo(f"   Configured {successful}/{total} determinism controls")

            if successful < total:
                typer.secho(
                    "⚠️  Some determinism controls failed - results may not be fully reproducible",
                    fg=typer.colors.YELLOW,
                )

        # Report configuration
        typer.echo(f"Audit profile: {audit_config.audit_profile}")
        typer.echo(f"Strict mode: {'ENABLED' if audit_config.strict_mode else 'disabled'}")
        typer.echo(f"Repro mode: {'ENABLED' if repro else 'disabled'}")

        if audit_config.strict_mode:
            typer.secho("⚠️  Strict mode enabled - enforcing regulatory compliance", fg=typer.colors.YELLOW)

        if repro:
            typer.secho("🔒 Repro mode enabled - results will be deterministic", fg=typer.colors.BLUE)

        # Validate components exist
        available = list_components()
        model_type = audit_config.model.type

        if model_type not in available.get("models", []) and model_type != "passthrough":
            typer.secho(f"Warning: Model type '{model_type}' not found in registry", fg=typer.colors.YELLOW)

        if dry_run:
            typer.echo()
            typer.secho("✓ Configuration validation passed", fg=typer.colors.GREEN)
            typer.echo()
            typer.echo("Configuration summary:")
            typer.echo(f"  Profile: {audit_config.audit_profile}")
            typer.echo(f"  Model: {audit_config.model.type}")
            typer.echo(f"  Explainer: {selected_explainer or 'auto-detect'}")
            typer.echo(f"  Data: {getattr(audit_config.data, 'dataset', 'custom')}")
            typer.echo()
            typer.echo("Dry run complete - no report generated")
            typer.echo()
            typer.echo("Next step:")
            typer.echo(f"  glassalpha audit --config {config} --output {output}")
            return

        # Determine output format with priority: config > file extension > default (HTML)
        if (
            hasattr(audit_config, "report")
            and hasattr(audit_config.report, "output_format")
            and audit_config.report.output_format is not None
        ):
            # Explicit config setting takes precedence
            output_format = audit_config.report.output_format
        # Auto-detect from file extension
        elif output.suffix.lower() == ".pdf":
            output_format = "pdf"
        elif output.suffix.lower() in [".html", ".htm"]:
            output_format = "html"
        else:
            # Default to HTML for speed and fewer dependencies
            output_format = "html"

        # Inform user about format inference if not explicitly configured
        if (
            hasattr(audit_config, "report")
            and hasattr(audit_config.report, "output_format")
            and audit_config.report.output_format is None
        ):
            if output.suffix.lower() == ".pdf":
                typer.echo("ℹ️  Output format inferred from extension: PDF")
            elif output.suffix.lower() in [".html", ".htm"]:
                typer.echo("ℹ️  Output format inferred from extension: HTML")
        if output_format == "pdf":
            _ensure_docs_if_pdf(str(output))

        # Warn if output file already exists
        if output.exists():
            file_size = output.stat().st_size / 1024  # KB
            typer.echo()
            typer.secho(
                "⚠️  Output file exists and will be overwritten:",
                fg=typer.colors.YELLOW,
            )
            typer.echo(f"   {output} ({file_size:.1f} KB)")
            typer.echo()

        # Run audit pipeline in deterministic context
        typer.echo("Running audit pipeline...")

        # Use deterministic context for entire pipeline execution
        from ..utils.determinism import deterministic

        seed = (
            audit_config.reproducibility.random_seed
            if hasattr(audit_config, "reproducibility")
            and audit_config.reproducibility
            and hasattr(audit_config.reproducibility, "random_seed")
            else 42
        )

        # Get strict mode from config, defaulting appropriately
        strict_mode = getattr(config, "strict_mode", False)

        with deterministic(seed=seed, strict=strict_mode):
            audit_results = _run_audit_pipeline(
                audit_config,
                output,
                selected_explainer,
                compact=compact_report,
                requested_model=requested_model,
            )

        # Save model if requested
        if save_model and audit_results:
            try:
                import joblib

                model_to_save = audit_results.trained_model
                if model_to_save is not None:
                    save_model.parent.mkdir(parents=True, exist_ok=True)

                    # Extract the underlying sklearn model from the wrapper for compatibility
                    underlying_model = model_to_save
                    if hasattr(model_to_save, "model"):
                        # It's a wrapper - extract the underlying model
                        underlying_model = model_to_save.model

                    # Enhanced preprocessing info for reasons/recourse compatibility
                    preprocessing_info = None
                    if hasattr(audit_results, "execution_info") and "preprocessing" in audit_results.execution_info:
                        preprocessing_info = audit_results.execution_info["preprocessing"]

                    # Extract complete preprocessing metadata for reasons/recourse
                    complete_preprocessing_info = None
                    if preprocessing_info:
                        complete_preprocessing_info = {
                            "mode": preprocessing_info.get("mode", "auto"),
                            "artifact_path": preprocessing_info.get("artifact_path")
                            if preprocessing_info.get("mode") == "artifact"
                            else None,
                            "categorical_cols": preprocessing_info.get("categorical_cols", []),
                            "numeric_cols": preprocessing_info.get("numeric_cols", []),
                            "feature_dtypes": preprocessing_info.get("feature_dtypes", {}),
                            "feature_names_before": preprocessing_info.get("feature_names_before", []),
                            "feature_names_after": preprocessing_info.get("feature_names_after", []),
                        }

                    model_artifact = {
                        "model": underlying_model,
                        "feature_names": (
                            list(audit_results.data_info.get("feature_columns", []))
                            if hasattr(audit_results, "data_info") and audit_results.data_info
                            else None
                        ),
                        "preprocessing": complete_preprocessing_info,
                        "target_column": audit_config.data.target_column,
                        "protected_attributes": audit_config.data.protected_attributes,
                        "glassalpha_version": "0.2.0",
                    }

                    joblib.dump(model_artifact, save_model)
                    typer.secho(f"✓ Model saved to: {save_model}", fg=typer.colors.GREEN)
                else:
                    typer.secho("⚠️  No model available to save", fg=typer.colors.YELLOW)
            except Exception as e:
                typer.secho(f"⚠️  Failed to save model: {e}", fg=typer.colors.YELLOW)

        # Run shift analysis if requested (E6.5)
        if check_shift:
            typer.echo("\n" + "=" * 60)
            typer.echo("DEMOGRAPHIC SHIFT ANALYSIS (E6.5)")
            typer.echo("=" * 60)

            try:
                shift_exit_code = _run_shift_analysis(
                    audit_config=audit_config,
                    output=output,
                    shift_specs=check_shift,
                    threshold=fail_on_degradation,
                )
            except ValueError as e:
                # Clear validation error - show and exit immediately
                typer.secho(f"\n{e}", fg=typer.colors.RED)
                raise typer.Exit(ExitCode.USER_ERROR)

            # If shift analysis detected violations and we should fail, exit with error
            if shift_exit_code != 0:
                typer.secho(
                    "\n✗ Shift analysis detected metric degradation exceeding threshold",
                    fg=typer.colors.RED,
                    bold=True,
                )
                raise typer.Exit(shift_exit_code)
            typer.secho(
                "\n✓ Shift analysis complete - no violations detected",
                fg=typer.colors.GREEN,
            )

    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        # Use 'from None' to suppress Python traceback for clean CLI UX
        # Users should see "File not found", not internal stack traces
        raise typer.Exit(ExitCode.USER_ERROR) from None
    except ValueError as e:
        # Check if this is a StrictModeError (which inherits from ValueError)
        # StrictModeError should return VALIDATION_ERROR for CI integration
        from ..config.strict import StrictModeError

        if isinstance(e, StrictModeError):
            typer.secho(f"Strict mode validation failed: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(ExitCode.VALIDATION_ERROR) from None

        # Other ValueErrors are user configuration errors
        typer.secho(f"Configuration error: {e}", fg=typer.colors.RED, err=True)
        # Intentional: Clean error message for end users
        raise typer.Exit(ExitCode.USER_ERROR) from None
    except Exception as e:
        if "--verbose" in sys.argv or "-v" in sys.argv:
            logger.exception("Audit failed")
        typer.secho(f"Audit failed: {e}", fg=typer.colors.RED, err=True)
        # CLI design: Hide Python internals from users (verbose mode shows full details)
        raise typer.Exit(ExitCode.USER_ERROR) from None


def doctor():  # pragma: no cover
    """Check environment and optional features.

    This command diagnoses the current environment and shows what optional
    features are available and how to enable them.

    Examples:
        # Basic environment check
        glassalpha doctor

        # Verbose output
        glassalpha doctor --verbose

    """
    import importlib.util
    import platform

    typer.echo("GlassAlpha Environment Check")
    typer.echo("=" * 40)

    # Basic environment info
    typer.echo("Environment")
    typer.echo(f"  Python: {sys.version}")
    typer.echo(f"  OS: {platform.system()} {platform.machine()}")
    typer.echo()

    # Core features - always available
    typer.echo("Core Features (always available)")
    typer.echo("-" * 20)
    typer.echo("  ✅ LogisticRegression (scikit-learn)")
    typer.echo("  ✅ NoOp explainers (baseline)")
    typer.echo("  ✅ HTML reports (jinja2)")
    typer.echo("  ✅ Basic metrics (performance, fairness)")
    typer.echo()

    # Optional features check
    typer.echo("Optional Features")
    typer.echo("-" * 20)

    # Check all components
    has_shap = importlib.util.find_spec("shap") is not None
    has_xgboost = importlib.util.find_spec("xgboost") is not None
    has_lightgbm = importlib.util.find_spec("lightgbm") is not None
    has_matplotlib = importlib.util.find_spec("matplotlib") is not None

    # PDF backend check
    has_pdf_backend = False
    pdf_backend_name = None
    try:
        import weasyprint  # noqa: F401

        has_pdf_backend = True
        pdf_backend_name = "weasyprint"
    except ImportError:
        try:
            import reportlab  # noqa: F401

            has_pdf_backend = True
            pdf_backend_name = "reportlab"
        except ImportError:
            pass

    # Group: SHAP + Tree models (they come together in [explain] extra)
    has_all_explain = has_shap and has_xgboost and has_lightgbm
    if has_all_explain:
        typer.echo("  SHAP + tree models: ✅ installed")
        typer.echo("    (includes SHAP, XGBoost, LightGBM)")
    else:
        typer.echo("  SHAP + tree models: ❌ not installed")
        # Show what's partially there if any
        installed_parts = []
        if has_shap:
            installed_parts.append("SHAP")
        if has_xgboost:
            installed_parts.append("XGBoost")
        if has_lightgbm:
            installed_parts.append("LightGBM")
        if installed_parts:
            typer.echo(f"    (partially installed: {', '.join(installed_parts)})")

    # Templating (always available)
    typer.echo("  Templating: ✅ installed (jinja2)")

    # PDF backend
    if has_pdf_backend:
        typer.echo(f"  PDF generation: ✅ installed ({pdf_backend_name})")
    else:
        typer.echo("  PDF generation: ❌ not installed")

    # Visualization
    if has_matplotlib:
        typer.echo("  Visualization: ✅ installed (matplotlib)")
    else:
        typer.echo("  Visualization: ❌ not installed")

    typer.echo()

    # Status summary and next steps
    typer.echo("Status & Next Steps")
    typer.echo("-" * 20)

    missing_features = []

    # Check what's missing
    if not has_all_explain:
        missing_features.append("SHAP + tree models")
    if not has_pdf_backend:
        missing_features.append("PDF generation")
    if not has_matplotlib:
        missing_features.append("visualization")

    # Show appropriate message
    if not missing_features:
        typer.echo("  ✅ All optional features installed!")
        typer.echo()
    else:
        typer.echo("  Missing features:")
        typer.echo()

        # Show specific install commands for what's missing
        if not has_all_explain:
            typer.echo("  📦 For SHAP + tree models (XGBoost, LightGBM):")
            typer.echo("     pip install 'glassalpha[explain]'")
            typer.echo()

        if not has_pdf_backend:
            typer.echo("  📄 For PDF reports:")
            typer.echo("     pip install 'glassalpha[docs]'")
            typer.echo()

        if not has_matplotlib:
            typer.echo("  📊 For enhanced plots:")
            typer.echo("     pip install 'glassalpha[viz]'")
            typer.echo()

        # Show quick install if multiple things missing
        if len(missing_features) > 1:
            typer.echo("  💡 Or install everything at once:")
            typer.echo("     pip install 'glassalpha[all]'")
            typer.echo()

    # Smart recommendation based on what's installed
    if has_pdf_backend:
        suggested_command = "glassalpha audit --config quickstart.yaml --output quickstart.pdf"
    else:
        suggested_command = "glassalpha audit --config quickstart.yaml --output quickstart.html"

    typer.echo(f"Ready to run: {suggested_command}")
    typer.echo()


def validate(  # pragma: no cover
    config_path: Path | None = typer.Argument(
        None,
        help="Path to configuration file to validate",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file to validate (alternative to positional arg)",
        exists=True,
        file_okay=True,
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        "-p",
        help="Validate against specific profile",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate for strict mode compliance (allows built-in datasets)",
    ),
    strict_full: bool = typer.Option(
        False,
        "--strict-full",
        help="Validate for full strict mode compliance (requires explicit schemas, disallows built-in datasets)",
    ),
    strict_validation: bool = typer.Option(
        False,
        "--strict-validation",
        help="Enforce runtime availability checks (recommended for production)",
    ),
):
    """Validate a configuration file.

    This command checks if a configuration file is valid without
    running the audit pipeline.

    Examples:
        # Basic validation (positional argument)
        glassalpha validate config.yaml

        # Basic validation (option syntax)
        glassalpha validate --config audit.yaml

        # Validate for specific profile
        glassalpha validate -c audit.yaml --profile tabular_compliance

        # Check strict mode compliance
        glassalpha validate -c audit.yaml --strict

        # Enforce runtime checks (production-ready)
        glassalpha validate -c audit.yaml --strict-validation

    """
    try:
        from ..config import load_config_from_file
        from ..core.registry import ModelRegistry  # Already correct location
        from ..explain.registry import ExplainerRegistry

        # Use positional arg if provided, otherwise fall back to --config
        config_to_validate = config_path or config

        # Auto-detect config if not provided
        if config_to_validate is None:
            from .defaults import get_smart_defaults

            defaults = get_smart_defaults()
            config_to_validate = defaults["config"]
            if config_to_validate is None:
                typer.echo("Error: No configuration file specified", fg=typer.colors.RED, err=True)
                typer.echo("\nUsage:")
                typer.echo("  glassalpha validate config.yaml")
                typer.echo("  glassalpha validate --config config.yaml")
                raise typer.Exit(ExitCode.USER_ERROR.value)
            typer.echo(f"Auto-detected config: {config_to_validate}")

        typer.echo(f"Validating configuration: {config_to_validate}")

        # Load and validate
        audit_config = load_config_from_file(
            config_to_validate,
            profile_name=profile,
            strict=strict,
            strict_full=strict_full,
        )

        typer.echo(f"Profile: {audit_config.audit_profile}")
        typer.echo(f"Model type: {audit_config.model.type}")
        strict_mode_desc = "not checked"
        if strict_full:
            strict_mode_desc = "full strict mode valid"
        elif strict:
            strict_mode_desc = "strict mode valid (quick mode allowed)"
        typer.echo(f"Strict mode: {strict_mode_desc}")

        # Semantic validation
        validation_errors = []
        validation_warnings = []

        # 1. Check if data source exists (if path specified)
        if hasattr(audit_config.data, "path") and audit_config.data.path:
            from pathlib import Path as PathLib

            data_path = PathLib(audit_config.data.path)
            if not data_path.exists():
                validation_errors.append(
                    f"Data file not found: {data_path}\n"
                    f"Fix: Create the file or use a built-in dataset:\n"
                    f"  data:\n"
                    f"    dataset: german_credit  # or adult_income",
                )

        # 2. Check if protected attributes are specified for fairness metrics
        if hasattr(audit_config.metrics, "fairness") and audit_config.metrics.fairness:
            if not hasattr(audit_config.data, "protected_attributes") or not audit_config.data.protected_attributes:
                validation_warnings.append(
                    "Fairness metrics requested but no protected_attributes specified.\n"
                    "Add protected_attributes to data section:\n"
                    "  data:\n"
                    "    protected_attributes:\n"
                    "      - gender\n"
                    "      - age_group",
                )

        # 3. Check model/explainer compatibility
        model_type = audit_config.model.type
        if hasattr(audit_config.explainers, "priority") and audit_config.explainers.priority:
            explainer_priorities = audit_config.explainers.priority
            # Check for common incompatibilities
            if model_type == "logistic_regression" and "treeshap" in explainer_priorities:
                validation_warnings.append(
                    f"Explainer 'treeshap' is not compatible with '{model_type}'.\n"
                    f"Recommend: Use 'coefficients' explainer for linear models:\n"
                    f"  explainers:\n"
                    f"    priority: [coefficients]",
                )
            elif model_type in ["xgboost", "lightgbm"] and "coefficients" in explainer_priorities:
                validation_warnings.append(
                    f"Explainer 'coefficients' is not ideal for '{model_type}'.\n"
                    f"Recommend: Use 'treeshap' for tree models:\n"
                    f"  explainers:\n"
                    f"    priority: [treeshap]",
                )

        # 4. Check model availability
        available_models = ModelRegistry.available_plugins()
        if not available_models.get(audit_config.model.type, False):
            msg = (
                f"Model '{audit_config.model.type}' is not available. "
                f"Install with: pip install 'glassalpha[explain]' (for xgboost/lightgbm)"
            )
            if strict_validation:
                validation_errors.append(msg)
            else:
                validation_warnings.append(msg + " (Will fallback to logistic_regression)")

        # 5. Check explainer availability and compatibility
        if audit_config.explainers.priority:
            available_explainers = ExplainerRegistry.available_plugins()
            requested_explainers = audit_config.explainers.priority

            available_requested = [e for e in requested_explainers if available_explainers.get(e, False)]

            if not available_requested:
                msg = (
                    f"None of the requested explainers {requested_explainers} are available. "
                    f"Install with: pip install 'glassalpha[explain]'"
                )
                if strict_validation:
                    validation_errors.append(msg)
                else:
                    validation_warnings.append(msg + " (Will fallback to permutation explainer)")
            else:
                # Check model/explainer compatibility for available explainers
                model_type = audit_config.model.type
                if "treeshap" in requested_explainers and model_type not in ["xgboost", "lightgbm", "random_forest"]:
                    msg = (
                        f"TreeSHAP requested but model type '{model_type}' is not a tree model. "
                        "Consider using 'coefficients' (for linear) or 'permutation' (universal)."
                    )
                    if strict_validation:
                        validation_errors.append(msg)
                    else:
                        validation_warnings.append(msg)

                # Check other explainer compatibility issues
                if "coefficients" in requested_explainers and model_type not in [
                    "logistic_regression",
                    "linear_regression",
                ]:
                    msg = f"Coefficients explainer requested but model type '{model_type}' doesn't have coefficients."
                    if strict_validation:
                        validation_errors.append(msg)
                    else:
                        validation_warnings.append(msg)

        # Check dataset and validate schema if specified
        if audit_config.data.path and audit_config.data.dataset == "custom":
            data_path = Path(audit_config.data.path).expanduser()
            if not data_path.exists():
                msg = f"Data file not found: {data_path}"
                if strict_validation:
                    validation_errors.append(msg)
                else:
                    validation_warnings.append(msg)
            else:
                # Validate dataset schema if file exists
                try:
                    from ..data.tabular import TabularDataLoader, TabularDataSchema

                    # Load data to validate schema
                    loader = TabularDataLoader()
                    df = loader.load(data_path)

                    # Build schema from config
                    schema = TabularDataSchema(
                        target=audit_config.data.target_column,
                        features=audit_config.data.feature_columns or [],
                        sensitive_features=audit_config.data.protected_attributes or [],
                    )

                    # Validate schema
                    loader.validate_schema(df, schema)
                    typer.echo(f"  ✓ Dataset schema validated ({len(df)} rows, {len(df.columns)} columns)")

                except ValueError as e:
                    msg = f"Dataset schema validation failed: {e}"
                    if strict_validation:
                        validation_errors.append(msg)
                    else:
                        validation_warnings.append(msg)
                except Exception as e:
                    msg = f"Error loading dataset for validation: {e}"
                    if strict_validation:
                        validation_errors.append(msg)
                    else:
                        validation_warnings.append(msg)
        elif audit_config.data.dataset and audit_config.data.dataset != "custom":
            # Built-in dataset - validate feature columns if specified
            if audit_config.data.feature_columns:
                try:
                    from ..data.registry import DatasetRegistry

                    # Try to load dataset info
                    dataset_info = DatasetRegistry.get(audit_config.data.dataset)
                    if dataset_info:
                        # Could validate against known schema here if we store it
                        typer.echo(f"  ✓ Using built-in dataset: {audit_config.data.dataset}")
                except Exception:
                    # Built-in dataset validation is optional - don't fail if it doesn't work
                    pass

        # Report validation errors
        if validation_errors:
            typer.echo()
            typer.secho("Validation failed with errors:", fg=typer.colors.RED, err=True)
            for error in validation_errors:
                typer.secho(f"  • {error}", fg=typer.colors.RED, err=True)
            typer.echo()
            typer.secho(
                "Tip: Run without --strict-validation to see warnings instead of errors",
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(ExitCode.VALIDATION_ERROR)

        # Report validation results
        typer.secho("Configuration is valid", fg=typer.colors.GREEN)

        # Show runtime warnings
        if validation_warnings:
            typer.echo()
            typer.secho("Runtime warnings:", fg=typer.colors.YELLOW)
            for warning in validation_warnings:
                typer.secho(f"  • {warning}", fg=typer.colors.YELLOW)
            typer.echo()
            if not strict_validation:
                typer.secho(
                    "Tip: Add --strict-validation to treat warnings as errors (recommended for production)",
                    fg=typer.colors.CYAN,
                )

        # Show other warnings
        if not audit_config.reproducibility.random_seed:
            typer.secho("Warning: No random seed specified - results may vary", fg=typer.colors.YELLOW)

        if not audit_config.data.protected_attributes:
            typer.secho(
                "Warning: No protected attributes - fairness analysis limited",
                fg=typer.colors.YELLOW,
            )

    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        # CLI UX: Clean error messages, no Python tracebacks for users
        raise typer.Exit(ExitCode.USER_ERROR) from None
    except ValueError as e:
        # All ValueErrors in validation context are validation failures
        # (including StrictModeError which inherits from ValueError)
        typer.secho(f"Validation failed: {e}", fg=typer.colors.RED, err=True)
        # Intentional: User-friendly validation errors
        raise typer.Exit(ExitCode.VALIDATION_ERROR) from None
    except Exception as e:
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED, err=True)
        # Design choice: Hide implementation details from end users
        raise typer.Exit(ExitCode.USER_ERROR) from None


def docs(  # pragma: no cover
    topic: str | None = typer.Argument(
        None,
        help="Documentation topic (e.g., 'model-parameters', 'quickstart', 'cli')",
    ),
    open_browser: bool = typer.Option(
        True,
        "--open/--no-open",
        help="Open in browser",
    ),
):
    """Open documentation in browser.

    Opens the GlassAlpha documentation website. You can optionally specify
    a topic to jump directly to that section.

    Examples:
        # Open docs home
        glassalpha docs

        # Open specific topic
        glassalpha docs model-parameters

        # Just print URL without opening
        glassalpha docs quickstart --no-open

    """
    import webbrowser

    base_url = "https://glassalpha.com"

    # Build URL based on topic
    if topic:
        # Normalize topic (replace underscores with hyphens)
        topic_normalized = topic.replace("_", "-")

        # Special cases for common topics
        if topic_normalized in ["quickstart", "installation", "configuration", "overview", "datasets", "custom-data"]:
            url = f"{base_url}/getting-started/{topic_normalized}/"
        elif topic_normalized in ["cli", "troubleshooting", "faq", "contributing", "api"]:
            url = f"{base_url}/reference/{topic_normalized}/"
        else:
            # Default to guides section for most topics
            url = f"{base_url}/guides/{topic_normalized}/"
    else:
        url = base_url

    # Open in browser or just print URL
    if open_browser:
        try:
            webbrowser.open(url)
            typer.echo(f"📖 Opening documentation: {url}")
        except Exception as e:
            typer.secho(f"Could not open browser: {e}", fg=typer.colors.YELLOW)
            typer.echo(f"Documentation URL: {url}")
    else:
        typer.echo(f"Documentation URL: {url}")


def list_components_cmd(  # pragma: no cover
    component_type: str | None = typer.Argument(
        None,
        help="Component type to list (models, explainers, metrics, profiles)",
    ),
    include_enterprise: bool = typer.Option(
        False,
        "--include-enterprise",
        "-e",
        help="Include enterprise components",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show component details",
    ),
):
    """List available components with runtime availability status.

    Shows registered models, explainers, metrics, and audit profiles.
    Indicates which components are available vs require additional dependencies.

    Examples:
        # List all components
        glassalpha list

        # List specific type
        glassalpha list models

        # Include enterprise components
        glassalpha list --include-enterprise

    """
    import importlib.util

    from ..core import list_components

    components = list_components(component_type=component_type, include_enterprise=include_enterprise)

    if not components:
        typer.echo(f"No components found for type: {component_type}")
        return

    typer.echo("Available Components")
    typer.echo("=" * 40)

    # Check dependencies
    has_shap = importlib.util.find_spec("shap") is not None
    has_xgboost = importlib.util.find_spec("xgboost") is not None
    has_lightgbm = importlib.util.find_spec("lightgbm") is not None

    for comp_type, items in components.items():
        typer.echo(f"\n{comp_type.upper()}:")

        if not items:
            typer.echo("  (none registered)")
        else:
            for item in sorted(items):
                # Determine availability status
                status = "✅"
                note = ""

                if comp_type == "models":
                    if (item == "xgboost" and not has_xgboost) or (item == "lightgbm" and not has_lightgbm):
                        status = "⚠️"
                        note = " (requires: pip install 'glassalpha[explain]')"
                elif comp_type == "explainers":
                    if item in ("treeshap", "kernelshap") and not has_shap:
                        status = "⚠️"
                        note = " (requires: pip install 'glassalpha[explain]')"

                if verbose:
                    typer.echo(f"  {status} {item}{note}")
                else:
                    typer.echo(f"  {status} {item}{note}")


def reasons(  # pragma: no cover
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model file (.pkl, .joblib). Generate with: glassalpha audit --save-model model.pkl",
        exists=True,
        file_okay=True,
    ),
    data: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to test data file (CSV)",
        exists=True,
        file_okay=True,
    ),
    instance: int = typer.Option(
        ...,
        "--instance",
        "-i",
        help="Row index of instance to explain (0-based)",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to reason codes configuration YAML",
        file_okay=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Path for output notice file (defaults to stdout)",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        "-t",
        help="Decision threshold for approved/denied",
    ),
    top_n: int = typer.Option(
        4,
        "--top-n",
        "-n",
        help="Number of reason codes to generate (ECOA typical: 4)",
    ),
    format: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: 'text' or 'json'",
    ),
):
    """Generate ECOA-compliant reason codes for adverse action notice.

    This command extracts top-N negative feature contributions from a trained model
    to explain why a specific instance was denied (or approved). Output is formatted
    as an ECOA-compliant adverse action notice.

    Requirements:
        - Trained model with SHAP-compatible architecture
        - Test dataset with same features as training
        - Instance index to explain

    Examples:
        # Generate reason codes for instance 42
        glassalpha reasons \\
            --model models/german_credit.pkl \\
            --data data/test.csv \\
            --instance 42 \\
            --output notices/instance_42.txt

        # With custom config
        glassalpha reasons -m model.pkl -d test.csv -i 10 -c config.yaml

        # JSON output
        glassalpha reasons -m model.pkl -d test.csv -i 5 --format json

        # Custom threshold and top-N
        glassalpha reasons -m model.pkl -d test.csv -i 0 --threshold 0.6 --top-n 3

    """
    import json

    import joblib

    try:
        import pandas as pd

        # Load configuration if provided
        protected_attributes = None
        organization = "[Organization Name]"
        contact_info = "[Contact Information]"
        seed = 42

        if config and config.exists():
            from ..config import load_config_from_file

            cfg = load_config_from_file(config)
            protected_attributes = getattr(cfg.data, "protected_attributes", None) if hasattr(cfg, "data") else None
            seed = getattr(cfg.reproducibility, "random_seed", 42) if hasattr(cfg, "reproducibility") else 42
            # Load organization info from config if available
            if hasattr(cfg, "reason_codes"):
                organization = getattr(cfg.reason_codes, "organization", organization)
                contact_info = getattr(cfg.reason_codes, "contact_info", contact_info)

        typer.echo(f"Loading model from: {model}")
        # Use joblib for loading (matches saving with joblib.dump in audit command)
        loaded = joblib.load(model)

        # Handle both old format (model only) and new format (dict with metadata)
        if isinstance(loaded, dict) and "model" in loaded:
            model_obj = loaded["model"]
            expected_features = loaded.get("feature_names")
            preprocessing_info = loaded.get("preprocessing")
        else:
            model_obj = loaded
            expected_features = None
            preprocessing_info = None

        typer.echo(f"Loading data from: {data}")
        df = pd.read_csv(data)

        # Apply preprocessing if available
        if preprocessing_info:
            typer.echo("Applying preprocessing to match model training...")

            def _apply_preprocessing_from_model_artifact(
                X: "pd.DataFrame",
                preprocessing_info: dict | None,
            ) -> "pd.DataFrame":
                """Apply the same preprocessing that was used during model training."""
                if preprocessing_info is None:
                    return X
                mode = preprocessing_info.get("mode", "auto")

                if mode == "artifact":
                    # Load preprocessing artifact and apply it
                    artifact_path = preprocessing_info.get("artifact_path")
                    if artifact_path:
                        try:
                            import joblib

                            preprocessor = joblib.load(artifact_path)
                            logger.info(f"Applying preprocessing artifact from: {artifact_path}")

                            # Apply preprocessing (assumes target column is not in X)
                            # Use transform, not fit_transform to match training exactly
                            X_transformed = preprocessor.transform(X)

                            # Get expected feature names from preprocessing info or artifact
                            expected_features = preprocessing_info.get("feature_names_after")
                            if expected_features:
                                # Use stored feature names for consistency
                                sanitized_feature_names = [
                                    str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                                    for name in expected_features
                                ]
                            else:
                                # Fallback: get from artifact
                                feature_names = preprocessor.get_feature_names_out()
                                sanitized_feature_names = [
                                    str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                                    for name in feature_names
                                ]

                            # Validate feature count matches expectations
                            if len(sanitized_feature_names) != X_transformed.shape[1]:
                                raise ValueError(
                                    f"Feature count mismatch: expected {len(sanitized_feature_names)} "
                                    f"features but got {X_transformed.shape[1]} from preprocessing",
                                )

                            # Return as DataFrame with proper column names
                            return pd.DataFrame(X_transformed, columns=sanitized_feature_names, index=X.index)
                        except Exception as e:
                            logger.exception(f"Failed to apply preprocessing artifact: {e}")
                            raise ValueError(
                                f"Could not apply preprocessing artifact: {e}. "
                                "Ensure the preprocessing artifact is accessible and matches the training data.",
                            )
                    else:
                        raise ValueError("Artifact mode specified but no artifact_path provided in model metadata")
                elif mode == "auto":
                    return _apply_auto_preprocessing_from_metadata(X, preprocessing_info)
                else:
                    raise ValueError(f"Unknown preprocessing mode: {mode}")

            def _apply_auto_preprocessing_from_metadata(X: "pd.DataFrame", preprocessing_info: dict) -> "pd.DataFrame":
                """Apply auto preprocessing using stored metadata from training."""
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder

                categorical_cols = preprocessing_info.get("categorical_cols", [])
                numeric_cols = preprocessing_info.get("numeric_cols", [])
                feature_dtypes = preprocessing_info.get("feature_dtypes", {})

                logger.debug(
                    f"Applying auto preprocessing from metadata: {len(categorical_cols)} categorical, {len(numeric_cols)} numeric columns",
                )

                # Validate that expected columns exist in input data
                missing_cols = set(categorical_cols + numeric_cols) - set(X.columns)
                if missing_cols:
                    raise ValueError(
                        f"Missing columns in test data: {missing_cols}. "
                        f"Expected columns: {sorted(categorical_cols + numeric_cols)}. "
                        f"Actual columns: {sorted(X.columns)}. "
                        "Ensure test data matches training data structure.",
                    )

                if not categorical_cols and not numeric_cols:
                    return X

                transformers = []
                if categorical_cols:
                    transformers.append(
                        (
                            "categorical",
                            OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None),
                            categorical_cols,
                        ),
                    )
                if numeric_cols:
                    transformers.append(("numeric", "passthrough", numeric_cols))

                preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

                # Use training-time preprocessing info to recreate the same transformation
                # We need to fit on training-like data or use stored parameters
                # For now, fit on the test data but this may not be identical to training
                # TODO: Store and use training preprocessor parameters for perfect reproducibility

                X_transformed = preprocessor.fit_transform(X)
                feature_names = []

                # Reconstruct feature names based on training metadata
                if categorical_cols:
                    cat_transformer = preprocessor.named_transformers_["categorical"]
                    if hasattr(cat_transformer, "get_feature_names_out"):
                        # Use stored feature names if available
                        stored_cat_features = preprocessing_info.get("feature_names_after", [])
                        if stored_cat_features:
                            # Filter for categorical features only
                            feature_names.extend(
                                [
                                    f
                                    for f in stored_cat_features
                                    if any(f.startswith(c + "_") for c in categorical_cols)
                                ],
                            )
                        else:
                            cat_features = cat_transformer.get_feature_names_out(categorical_cols)
                            feature_names.extend(cat_features)
                    else:
                        # Fallback: reconstruct from categories
                        for i, col in enumerate(categorical_cols):
                            unique_vals = cat_transformer.categories_[i]
                            feature_names.extend([f"{col}_{val}" for val in unique_vals])

                if numeric_cols:
                    feature_names.extend(numeric_cols)

                # Validate feature count matches
                if len(feature_names) != X_transformed.shape[1]:
                    logger.warning(
                        f"Feature count mismatch: expected {len(feature_names)} "
                        f"but got {X_transformed.shape[1]} from preprocessing. "
                        "Using generic feature names.",
                    )
                    # Fallback to generic names
                    feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

                sanitized_feature_names = [
                    str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                    for name in feature_names
                ]

                return pd.DataFrame(X_transformed, columns=sanitized_feature_names, index=X.index)

            def _apply_auto_preprocessing(X: "pd.DataFrame", preprocessing_info: dict) -> "pd.DataFrame":
                """Apply auto preprocessing using stored metadata from training."""
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder

                categorical_cols = preprocessing_info.get("categorical_cols", [])
                numeric_cols = preprocessing_info.get("numeric_cols", [])
                feature_dtypes = preprocessing_info.get("feature_dtypes", {})

                logger.debug(
                    f"Applying auto preprocessing: {len(categorical_cols)} categorical, {len(numeric_cols)} numeric columns",
                )

                # Validate that expected columns exist in input data
                missing_cols = set(categorical_cols + numeric_cols) - set(X.columns)
                if missing_cols:
                    raise ValueError(
                        f"Missing columns in test data: {missing_cols}. "
                        f"Expected columns: {sorted(categorical_cols + numeric_cols)}. "
                        f"Actual columns: {sorted(X.columns)}. "
                        "Ensure test data matches training data structure.",
                    )

                if not categorical_cols and not numeric_cols:
                    return X

                transformers = []
                if categorical_cols:
                    transformers.append(
                        (
                            "categorical",
                            OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None),
                            categorical_cols,
                        ),
                    )
                if numeric_cols:
                    transformers.append(("numeric", "passthrough", numeric_cols))

                preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

                try:
                    X_transformed = preprocessor.fit_transform(X)
                    feature_names = []

                    # Reconstruct feature names based on training metadata
                    if categorical_cols:
                        cat_transformer = preprocessor.named_transformers_["categorical"]
                        if hasattr(cat_transformer, "get_feature_names_out"):
                            # Use stored feature names if available
                            stored_cat_features = preprocessing_info.get("feature_names_after", [])
                            if stored_cat_features:
                                # Filter for categorical features only
                                feature_names.extend(
                                    [
                                        f
                                        for f in stored_cat_features
                                        if any(f.startswith(c + "_") for c in categorical_cols)
                                    ],
                                )
                            else:
                                cat_features = cat_transformer.get_feature_names_out(categorical_cols)
                                feature_names.extend(cat_features)
                        else:
                            # Fallback: reconstruct from categories
                            for i, col in enumerate(categorical_cols):
                                unique_vals = cat_transformer.categories_[i]
                                feature_names.extend([f"{col}_{val}" for val in unique_vals])

                    if numeric_cols:
                        feature_names.extend(numeric_cols)

                    # Validate feature count matches
                    if len(feature_names) != X_transformed.shape[1]:
                        logger.warning(
                            f"Feature count mismatch: expected {len(feature_names)} "
                            f"but got {X_transformed.shape[1]} from preprocessing. "
                            "Using generic feature names.",
                        )
                        # Fallback to generic names
                        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

                    sanitized_feature_names = [
                        str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                        for name in feature_names
                    ]

                    X_processed = pd.DataFrame(X_transformed, columns=sanitized_feature_names, index=X.index)
                    logger.info(f"Preprocessed {len(categorical_cols)} categorical columns with OneHotEncoder")
                    logger.info(f"Final feature count: {len(sanitized_feature_names)} (from {len(X.columns)} original)")
                    return X_processed

                except Exception as e:
                    logger.exception(f"Preprocessing failed: {e}")
                    raise ValueError(
                        f"Could not apply auto preprocessing: {e}. Ensure test data matches training data structure.",
                    )

            df = _apply_preprocessing_from_model_artifact(df, preprocessing_info)

        # Validate feature alignment if metadata available
        if expected_features is not None:
            available_features = list(df.columns)
            if set(expected_features) - set(available_features):
                missing = set(expected_features) - set(available_features)
                typer.secho(
                    f"Error: Model expects {len(expected_features)} features but data only has {len(available_features)} columns.\n"
                    f"Missing features: {sorted(list(missing))[:10]}{'...' if len(missing) > 10 else ''}\n\n"
                    f"Model was trained on features: {expected_features[:5]}...\n"
                    f"Data has columns: {available_features[:5]}...\n\n"
                    f"Fix: Ensure data file has the same columns as the training data.\n"
                    f"• Check column names match exactly (case-sensitive)\n"
                    f"• Verify no columns were renamed or removed\n"
                    f"• Use the same data preprocessing pipeline",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(ExitCode.USER_ERROR)

            # Reorder columns to match training order
            df = df[expected_features]

        if instance < 0 or instance >= len(df):
            typer.secho(
                f"Error: Instance index {instance} is out of range in data file.\n"
                f"Data has {len(df)} rows (valid indices: 0-{len(df) - 1}).\n\n"
                f"Fix: Choose an instance index between 0 and {len(df) - 1}, or check that your data file contains the expected number of rows.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        # Get instance
        X_instance = df.iloc[[instance]].drop(columns=["target"], errors="ignore")
        feature_names = X_instance.columns.tolist()
        feature_values = X_instance.iloc[0]

        typer.echo(f"Generating SHAP explanations for instance {instance}...")

        # Auto-encode categorical columns for SHAP compatibility
        X_instance_encoded = X_instance.copy()
        categorical_cols = X_instance.select_dtypes(include=["object", "category", "string"]).columns

        if len(categorical_cols) > 0:
            typer.echo(f"Auto-encoding {len(categorical_cols)} categorical columns for SHAP compatibility...")
            from sklearn.preprocessing import LabelEncoder

            for col in categorical_cols:
                if X_instance_encoded[col].dtype in ["object", "category", "string"]:
                    # Convert to string first to handle any data types
                    X_instance_encoded[col] = X_instance_encoded[col].astype(str)
                    le = LabelEncoder()
                    X_instance_encoded[col] = le.fit_transform(X_instance_encoded[col])

        # Get prediction (with better error handling for feature mismatch)
        try:
            # Convert DataFrame to numpy for XGBoost compatibility
            X_numpy = X_instance_encoded.values if hasattr(X_instance_encoded, "values") else X_instance_encoded

            if hasattr(model_obj, "predict_proba"):
                prediction = float(model_obj.predict_proba(X_numpy)[0, 1])
            else:
                prediction = float(model_obj.predict(X_numpy)[0])
        except ValueError as e:
            if "Too many missing features" in str(e):
                typer.secho(
                    f"Error: Model expects {len(expected_features) if expected_features else 'unknown'} features but data has {len(X_instance_encoded.columns)} columns",
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.echo("\nThis usually means:")
                typer.echo("  • The model was trained on encoded/preprocessed data")
                typer.echo("  • But you're providing raw data with categorical columns")
                typer.echo("\nSolutions:")
                typer.echo("  1. Use the same dataset that was used for training")
                typer.echo("  2. Apply the same preprocessing (encoding) that was used during training")
                typer.echo("  3. Retrain the model on raw data if preprocessing isn't available")
                typer.echo("\nFor help with preprocessing, see: https://glassalpha.com/guides/preprocessing/")
                raise typer.Exit(ExitCode.USER_ERROR) from None
            raise

        # Extract native model from wrapper if needed
        native_model = model_obj
        if hasattr(model_obj, "model"):
            # GlassAlpha wrapper - extract underlying model
            native_model = model_obj.model
            typer.echo(f"Extracted native model from wrapper: {type(native_model).__name__}")

        # Generate feature contributions (SHAP or coefficients fallback)
        shap_values = None
        try:
            # Try SHAP first (best for tree models and non-linear models)
            import shap

            try:
                explainer = shap.TreeExplainer(native_model)
                # Convert DataFrame to numpy for SHAP compatibility
                X_shap = X_instance_encoded.values if hasattr(X_instance_encoded, "values") else X_instance_encoded
                shap_values = explainer.shap_values(X_shap)

                # Handle multi-output case (binary classification)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Positive class

                # Flatten to 1D
                if len(shap_values.shape) > 1:
                    shap_values = shap_values[0]
            except Exception as tree_error:
                # TreeSHAP failed, try KernelSHAP
                error_msg = str(tree_error).lower()
                if any(
                    phrase in error_msg
                    for phrase in [
                        "model type not yet supported",
                        "treeexplainer",
                        "does not support treeexplainer",
                        "linear model",
                        "logisticregression",
                    ]
                ):
                    typer.secho(
                        f"Note: TreeSHAP not compatible with {type(native_model).__name__}. Using KernelSHAP...",
                        fg=typer.colors.CYAN,
                    )
                    # Convert to numpy for compatibility
                    X_sample = (
                        X_instance_encoded.values if hasattr(X_instance_encoded, "values") else X_instance_encoded
                    )
                    explainer = shap.KernelExplainer(model_obj.predict_proba, shap.sample(X_sample, 100))
                    shap_values = explainer.shap_values(X_sample)[0, :, 1]
                else:
                    raise

        except ImportError:
            # SHAP not available - try coefficient-based fallback for linear models
            if hasattr(native_model, "coef_"):
                typer.secho(
                    "Note: SHAP not installed. Using coefficient-based explanations for linear model.",
                    fg=typer.colors.CYAN,
                )
                typer.echo("  For better explanations: pip install 'glassalpha[explain]'")
                typer.echo()

                # Use coefficients as feature importance
                coefficients = native_model.coef_[0] if len(native_model.coef_.shape) > 1 else native_model.coef_

                # Convert to contributions: coef * (feature_value - mean)
                # This approximates local explanation for linear models
                X_numpy = X_instance_encoded.values[0] if hasattr(X_instance_encoded, "values") else X_instance_encoded
                shap_values = coefficients * X_numpy
            else:
                typer.secho(
                    "Error: SHAP not installed and model doesn't support coefficient extraction.",
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.echo()
                typer.echo("Solutions:")
                typer.echo("  1. Install SHAP: pip install 'glassalpha[explain]'")
                typer.echo("  2. Use a model with coefficients (LogisticRegression)")
                raise typer.Exit(ExitCode.USER_ERROR)
        except Exception as e:
            if "Too many missing features" in str(e):
                typer.secho(
                    f"Error: Model expects {len(expected_features) if expected_features else 'unknown'} features but data has {len(X_instance_encoded.columns)} columns",
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.echo("\nThis usually means:")
                typer.echo("  • The model was trained on encoded/preprocessed data")
                typer.echo("  • But you're providing raw data with categorical columns")
                typer.echo("\nSolutions:")
                typer.echo("  1. Use the same dataset that was used for training")
                typer.echo("  2. Apply the same preprocessing (encoding) that was used during training")
                typer.echo("  3. Retrain the model on raw data if preprocessing isn't available")
                typer.echo("\nFor help with preprocessing, see: https://glassalpha.com/guides/preprocessing/")
            else:
                typer.secho(
                    f"Error generating explanations: {e}",
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.echo("\nTip: Ensure model is compatible (XGBoost, LightGBM, RandomForest, LogisticRegression)")
                typer.echo("For linear models without SHAP: pip install 'glassalpha[explain]'")
            raise typer.Exit(ExitCode.USER_ERROR) from None

        # Extract reason codes
        from ..explain.reason_codes import extract_reason_codes, format_adverse_action_notice

        typer.echo("Extracting top-N negative contributions...")
        result = extract_reason_codes(
            shap_values=shap_values,
            feature_names=feature_names,
            feature_values=feature_values,
            instance_id=instance,
            prediction=prediction,
            threshold=threshold,
            top_n=top_n,
            protected_attributes=protected_attributes,
            seed=seed,
        )

        # Format output
        if format == "json":
            output_dict = {
                "instance_id": result.instance_id,
                "prediction": result.prediction,
                "decision": result.decision,
                "reason_codes": [
                    {
                        "rank": code.rank,
                        "feature": code.feature,
                        "contribution": code.contribution,
                        "feature_value": code.feature_value,
                    }
                    for code in result.reason_codes
                ],
                "excluded_features": result.excluded_features,
                "timestamp": result.timestamp,
                "model_hash": result.model_hash,
                "seed": result.seed,
            }
            output_text = json.dumps(output_dict, indent=2, sort_keys=True)
        else:
            # Text format (ECOA notice)
            output_text = format_adverse_action_notice(
                result=result,
                organization=organization,
                contact_info=contact_info,
            )

        # Write or print output
        if output:
            output.write_text(output_text)
            typer.secho(
                "\n✅ Reason codes generated successfully!",
                fg=typer.colors.GREEN,
            )
            typer.echo(f"Output: {output}")
        else:
            typer.echo("\n" + "=" * 60)
            typer.echo(output_text)
            typer.echo("=" * 60)

        # Show summary
        typer.echo(f"\nInstance: {result.instance_id}")
        typer.echo(f"Prediction: {result.prediction:.1%}")
        typer.echo(f"Decision: {result.decision.upper()}")
        typer.echo(f"Reason codes: {len(result.reason_codes)}")
        if result.excluded_features:
            typer.echo(f"Protected attributes excluded: {len(result.excluded_features)}")

    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR) from None
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR) from None
    except Exception as e:
        if "--verbose" in sys.argv or "-v" in sys.argv:
            logger.exception("Reason code generation failed")
        typer.secho(f"Failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR) from None


def recourse(  # pragma: no cover
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model file (.pkl, .joblib). Generate with: glassalpha audit --save-model model.pkl",
        exists=True,
        file_okay=True,
    ),
    data: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to test data file (CSV)",
        exists=True,
        file_okay=True,
    ),
    instance: int = typer.Option(
        ...,
        "--instance",
        "-i",
        help="Row index of instance to explain (0-based)",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to recourse configuration YAML",
        file_okay=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Path for output recommendations file (JSON, defaults to stdout)",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        "-t",
        help="Decision threshold for approved/denied",
    ),
    top_n: int = typer.Option(
        5,
        "--top-n",
        "-n",
        help="Number of counterfactual recommendations to generate",
    ),
    force_recourse: bool = typer.Option(
        False,
        "--force-recourse",
        help="Generate recourse recommendations even for approved instances (for testing)",
    ),
):
    """Generate ECOA-compliant counterfactual recourse recommendations.

    This command generates feasible counterfactual recommendations with policy constraints
    for individuals receiving adverse decisions. Supports immutable features, monotonic
    constraints, and cost-weighted optimization.

    Requirements:
        - Trained model with SHAP-compatible architecture
        - Test dataset with same features as training
        - Instance index to explain (must be denied: prediction < threshold)
        - Configuration file with policy constraints (recommended)

    Examples:
        # Generate recourse for denied instance
        glassalpha recourse \\
            --model models/german_credit.pkl \\
            --data data/test.csv \\
            --instance 42 \\
            --config configs/recourse_german_credit.yaml \\
            --output recourse/instance_42.json

        # With custom threshold and top-N
        glassalpha recourse -m model.pkl -d test.csv -i 10 -c config.yaml --top-n 3

        # Output to stdout
        glassalpha recourse -m model.pkl -d test.csv -i 5 -c config.yaml

    Configuration File:
        The config file should include:
        - recourse.immutable_features: list of features that cannot be changed
        - recourse.monotonic_constraints: directional constraints (increase_only, decrease_only)
        - recourse.cost_function: cost function for optimization (weighted_l1)
        - data.protected_attributes: list of protected attributes to exclude
        - reproducibility.random_seed: seed for deterministic results

    """
    import json

    import joblib

    try:
        import pandas as pd

        # Load configuration
        immutable_features: list[str] = []
        monotonic_constraints: dict[str, str] = {}
        feature_costs: dict[str, float] = {}
        feature_bounds: dict[str, tuple[float, float]] = {}
        seed = 42

        if config and config.exists():
            from ..config import load_config_from_file

            cfg = load_config_from_file(config)

            # Load recourse config
            if hasattr(cfg, "recourse"):
                immutable_features = list(getattr(cfg.recourse, "immutable_features", []))
                raw_constraints = getattr(cfg.recourse, "monotonic_constraints", {})
                # Convert monotonic constraints to dict[str, str] for API
                monotonic_constraints = {str(k): str(v) for k, v in raw_constraints.items()}

            # Load seed
            seed = getattr(cfg.reproducibility, "random_seed", 42) if hasattr(cfg, "reproducibility") else 42
        else:
            typer.secho(
                "Warning: No config provided. Using default policy (no constraints).",
                fg=typer.colors.YELLOW,
            )

        typer.echo(f"Loading model from: {model}")
        # Use joblib for loading (matches saving with joblib.dump in audit command)
        import joblib

        loaded = joblib.load(model)

        # Handle both old format (model only) and new format (dict with metadata)
        if isinstance(loaded, dict) and "model" in loaded:
            model_obj = loaded["model"]
            expected_features = loaded.get("feature_names")
            preprocessing_info = loaded.get("preprocessing")
        else:
            model_obj = loaded
            expected_features = None
            preprocessing_info = None

        typer.echo(f"Loading data from: {data}")
        df = pd.read_csv(data)

        # Apply preprocessing if available
        if preprocessing_info:
            typer.echo("Applying preprocessing to match model training...")

            def _apply_preprocessing_from_model_artifact(
                X: "pd.DataFrame",
                preprocessing_info: dict | None,
            ) -> "pd.DataFrame":
                """Apply the same preprocessing that was used during model training."""
                if preprocessing_info is None:
                    return X
                mode = preprocessing_info.get("mode", "auto")

                if mode == "artifact":
                    # Load preprocessing artifact and apply it
                    artifact_path = preprocessing_info.get("artifact_path")
                    if artifact_path:
                        try:
                            import joblib

                            preprocessor = joblib.load(artifact_path)
                            logger.info(f"Applying preprocessing artifact from: {artifact_path}")

                            # Apply preprocessing (assumes target column is not in X)
                            # Use transform, not fit_transform to match training exactly
                            X_transformed = preprocessor.transform(X)

                            # Get expected feature names from preprocessing info or artifact
                            expected_features = preprocessing_info.get("feature_names_after")
                            if expected_features:
                                # Use stored feature names for consistency
                                sanitized_feature_names = [
                                    str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                                    for name in expected_features
                                ]
                            else:
                                # Fallback: get from artifact
                                feature_names = preprocessor.get_feature_names_out()
                                sanitized_feature_names = [
                                    str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                                    for name in feature_names
                                ]

                            # Validate feature count matches expectations
                            if len(sanitized_feature_names) != X_transformed.shape[1]:
                                raise ValueError(
                                    f"Feature count mismatch: expected {len(sanitized_feature_names)} "
                                    f"features but got {X_transformed.shape[1]} from preprocessing",
                                )

                            # Return as DataFrame with proper column names
                            return pd.DataFrame(X_transformed, columns=sanitized_feature_names, index=X.index)
                        except Exception as e:
                            logger.exception(f"Failed to apply preprocessing artifact: {e}")
                            raise ValueError(
                                f"Could not apply preprocessing artifact: {e}. "
                                "Ensure the preprocessing artifact is accessible and matches the training data.",
                            )
                    else:
                        raise ValueError("Artifact mode specified but no artifact_path provided in model metadata")
                elif mode == "auto":
                    return _apply_auto_preprocessing_from_metadata(X, preprocessing_info)
                else:
                    raise ValueError(f"Unknown preprocessing mode: {mode}")

            def _apply_auto_preprocessing_from_metadata(X: "pd.DataFrame", preprocessing_info: dict) -> "pd.DataFrame":
                """Apply auto preprocessing using stored metadata from training."""
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder

                categorical_cols = preprocessing_info.get("categorical_cols", [])
                numeric_cols = preprocessing_info.get("numeric_cols", [])
                feature_dtypes = preprocessing_info.get("feature_dtypes", {})

                logger.debug(
                    f"Applying auto preprocessing from metadata: {len(categorical_cols)} categorical, {len(numeric_cols)} numeric columns",
                )

                # Validate that expected columns exist in input data
                missing_cols = set(categorical_cols + numeric_cols) - set(X.columns)
                if missing_cols:
                    raise ValueError(
                        f"Missing columns in test data: {missing_cols}. "
                        f"Expected columns: {sorted(categorical_cols + numeric_cols)}. "
                        f"Actual columns: {sorted(X.columns)}. "
                        "Ensure test data matches training data structure.",
                    )

                if not categorical_cols and not numeric_cols:
                    return X

                transformers = []
                if categorical_cols:
                    transformers.append(
                        (
                            "categorical",
                            OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None),
                            categorical_cols,
                        ),
                    )
                if numeric_cols:
                    transformers.append(("numeric", "passthrough", numeric_cols))

                preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

                # Use training-time preprocessing info to recreate the same transformation
                # We need to fit on training-like data or use stored parameters
                # For now, fit on the test data but this may not be identical to training
                # TODO: Store and use training preprocessor parameters for perfect reproducibility

                X_transformed = preprocessor.fit_transform(X)
                feature_names = []

                # Reconstruct feature names based on training metadata
                if categorical_cols:
                    cat_transformer = preprocessor.named_transformers_["categorical"]
                    if hasattr(cat_transformer, "get_feature_names_out"):
                        # Use stored feature names if available
                        stored_cat_features = preprocessing_info.get("feature_names_after", [])
                        if stored_cat_features:
                            # Filter for categorical features only
                            feature_names.extend(
                                [
                                    f
                                    for f in stored_cat_features
                                    if any(f.startswith(c + "_") for c in categorical_cols)
                                ],
                            )
                        else:
                            cat_features = cat_transformer.get_feature_names_out(categorical_cols)
                            feature_names.extend(cat_features)
                    else:
                        # Fallback: reconstruct from categories
                        for i, col in enumerate(categorical_cols):
                            unique_vals = cat_transformer.categories_[i]
                            feature_names.extend([f"{col}_{val}" for val in unique_vals])

                if numeric_cols:
                    feature_names.extend(numeric_cols)

                # Validate feature count matches
                if len(feature_names) != X_transformed.shape[1]:
                    logger.warning(
                        f"Feature count mismatch: expected {len(feature_names)} "
                        f"but got {X_transformed.shape[1]} from preprocessing. "
                        "Using generic feature names.",
                    )
                    # Fallback to generic names
                    feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

                sanitized_feature_names = [
                    str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                    for name in feature_names
                ]

                return pd.DataFrame(X_transformed, columns=sanitized_feature_names, index=X.index)

            def _apply_auto_preprocessing(X: "pd.DataFrame", preprocessing_info: dict) -> "pd.DataFrame":
                """Apply auto preprocessing using stored metadata from training."""
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder

                categorical_cols = preprocessing_info.get("categorical_cols", [])
                numeric_cols = preprocessing_info.get("numeric_cols", [])
                feature_dtypes = preprocessing_info.get("feature_dtypes", {})

                logger.debug(
                    f"Applying auto preprocessing: {len(categorical_cols)} categorical, {len(numeric_cols)} numeric columns",
                )

                # Validate that expected columns exist in input data
                missing_cols = set(categorical_cols + numeric_cols) - set(X.columns)
                if missing_cols:
                    raise ValueError(
                        f"Missing columns in test data: {missing_cols}. "
                        f"Expected columns: {sorted(categorical_cols + numeric_cols)}. "
                        f"Actual columns: {sorted(X.columns)}. "
                        "Ensure test data matches training data structure.",
                    )

                if not categorical_cols and not numeric_cols:
                    return X

                transformers = []
                if categorical_cols:
                    transformers.append(
                        (
                            "categorical",
                            OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None),
                            categorical_cols,
                        ),
                    )
                if numeric_cols:
                    transformers.append(("numeric", "passthrough", numeric_cols))

                preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

                try:
                    X_transformed = preprocessor.fit_transform(X)
                    feature_names = []

                    # Reconstruct feature names based on training metadata
                    if categorical_cols:
                        cat_transformer = preprocessor.named_transformers_["categorical"]
                        if hasattr(cat_transformer, "get_feature_names_out"):
                            # Use stored feature names if available
                            stored_cat_features = preprocessing_info.get("feature_names_after", [])
                            if stored_cat_features:
                                # Filter for categorical features only
                                feature_names.extend(
                                    [
                                        f
                                        for f in stored_cat_features
                                        if any(f.startswith(c + "_") for c in categorical_cols)
                                    ],
                                )
                            else:
                                cat_features = cat_transformer.get_feature_names_out(categorical_cols)
                                feature_names.extend(cat_features)
                        else:
                            # Fallback: reconstruct from categories
                            for i, col in enumerate(categorical_cols):
                                unique_vals = cat_transformer.categories_[i]
                                feature_names.extend([f"{col}_{val}" for val in unique_vals])

                    if numeric_cols:
                        feature_names.extend(numeric_cols)

                    # Validate feature count matches
                    if len(feature_names) != X_transformed.shape[1]:
                        logger.warning(
                            f"Feature count mismatch: expected {len(feature_names)} "
                            f"but got {X_transformed.shape[1]} from preprocessing. "
                            "Using generic feature names.",
                        )
                        # Fallback to generic names
                        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

                    sanitized_feature_names = [
                        str(name).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                        for name in feature_names
                    ]

                    X_processed = pd.DataFrame(X_transformed, columns=sanitized_feature_names, index=X.index)
                    logger.info(f"Preprocessed {len(categorical_cols)} categorical columns with OneHotEncoder")
                    logger.info(f"Final feature count: {len(sanitized_feature_names)} (from {len(X.columns)} original)")
                    return X_processed

                except Exception as e:
                    logger.exception(f"Preprocessing failed: {e}")
                    raise ValueError(
                        f"Could not apply auto preprocessing: {e}. Ensure test data matches training data structure.",
                    )

            df = _apply_preprocessing_from_model_artifact(df, preprocessing_info)

        # Validate feature alignment if metadata available
        if expected_features is not None:
            available_features = list(df.columns)
            if set(expected_features) - set(available_features):
                missing = set(expected_features) - set(available_features)
                typer.secho(
                    f"Error: Model expects {len(expected_features)} features but data only has {len(available_features)} columns.\n"
                    f"Missing features: {sorted(list(missing))[:10]}{'...' if len(missing) > 10 else ''}\n\n"
                    f"Model was trained on features: {expected_features[:5]}...\n"
                    f"Data has columns: {available_features[:5]}...\n\n"
                    f"Fix: Ensure data file has the same columns as the training data.\n"
                    f"• Check column names match exactly (case-sensitive)\n"
                    f"• Verify no columns were renamed or removed\n"
                    f"• Use the same data preprocessing pipeline",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(ExitCode.USER_ERROR)

            # Reorder columns to match training order
            df = df[expected_features]

        if instance < 0 or instance >= len(df):
            typer.secho(
                f"Error: Instance index {instance} is out of range in data file.\n"
                f"Data has {len(df)} rows (valid indices: 0-{len(df) - 1}).\n\n"
                f"Fix: Choose an instance index between 0 and {len(df) - 1}, or check that your data file contains the expected number of rows.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        # Get instance
        X_instance = df.iloc[[instance]].drop(columns=["target"], errors="ignore")
        feature_names = X_instance.columns.tolist()
        feature_values_series = X_instance.iloc[0]

        typer.echo(f"Generating SHAP explanations for instance {instance}...")

        # Auto-encode categorical columns for SHAP compatibility
        X_instance_encoded = X_instance.copy()
        categorical_cols = X_instance.select_dtypes(include=["object", "category", "string"]).columns

        if len(categorical_cols) > 0:
            typer.echo(f"Auto-encoding {len(categorical_cols)} categorical columns for SHAP compatibility...")
            from sklearn.preprocessing import LabelEncoder

            for col in categorical_cols:
                if X_instance_encoded[col].dtype in ["object", "category", "string"]:
                    # Convert to string first to handle any data types
                    X_instance_encoded[col] = X_instance_encoded[col].astype(str)
                    le = LabelEncoder()
                    X_instance_encoded[col] = le.fit_transform(X_instance_encoded[col])

        # Get prediction (with better error handling for feature mismatch)
        try:
            if hasattr(model_obj, "predict_proba"):
                prediction = float(model_obj.predict_proba(X_instance_encoded)[0, 1])
            else:
                prediction = float(model_obj.predict(X_instance_encoded)[0])
        except ValueError as e:
            if "Too many missing features" in str(e):
                typer.secho(
                    f"Error: Model expects {len(expected_features) if expected_features else 'unknown'} features but data has {len(X_instance_encoded.columns)} columns",
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.echo("\nThis usually means:")
                typer.echo("  • The model was trained on encoded/preprocessed data")
                typer.echo("  • But you're providing raw data with categorical columns")
                typer.echo("\nSolutions:")
                typer.echo("  1. Use the same dataset that was used for training")
                typer.echo("  2. Apply the same preprocessing (encoding) that was used during training")
                typer.echo("  3. Retrain the model on raw data if preprocessing isn't available")
                typer.echo("\nFor help with preprocessing, see: https://glassalpha.com/guides/preprocessing/")
                raise typer.Exit(ExitCode.USER_ERROR) from None
            raise

        # Check if instance is already approved
        if prediction >= threshold and not force_recourse:
            typer.secho(
                f"\nInstance {instance} is already approved (prediction={prediction:.1%} >= threshold={threshold:.1%})",
                fg=typer.colors.YELLOW,
            )
            typer.echo("No recourse needed.")
            typer.echo("\nTo generate recourse anyway (for testing), use --force-recourse flag.")
            raise typer.Exit(ExitCode.SUCCESS)

        # Extract native model from wrapper if needed
        native_model = model_obj
        if hasattr(model_obj, "model"):
            # GlassAlpha wrapper - extract underlying model
            native_model = model_obj.model
            typer.echo(f"Extracted native model from wrapper: {type(native_model).__name__}")

        # Generate SHAP values (use TreeSHAP for tree models)
        try:
            import shap

            explainer = shap.TreeExplainer(native_model)
            # Convert DataFrame to numpy for SHAP compatibility
            X_shap = X_instance_encoded.values if hasattr(X_instance_encoded, "values") else X_instance_encoded
            shap_values = explainer.shap_values(X_shap)

            # Handle multi-output case (binary classification)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class

            # Flatten to 1D
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]

        except Exception as e:
            if "Too many missing features" in str(e):
                typer.secho(
                    f"Error: Model expects {len(expected_features) if expected_features else 'unknown'} features but data has {len(X_instance_encoded.columns)} columns",
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.echo("\nThis usually means:")
                typer.echo("  • The model was trained on encoded/preprocessed data")
                typer.echo("  • But you're providing raw data with categorical columns")
                typer.echo("\nSolutions:")
                typer.echo("  1. Use the same dataset that was used for training")
                typer.echo("  2. Apply the same preprocessing (encoding) that was used during training")
                typer.echo("  3. Retrain the model on raw data if preprocessing isn't available")
                typer.echo("\nFor help with preprocessing, see: https://glassalpha.com/guides/preprocessing/")
                raise typer.Exit(ExitCode.USER_ERROR) from None
            # Check if this is a TreeSHAP compatibility error for non-tree models
            error_msg = str(e).lower()
            if any(
                phrase in error_msg
                for phrase in [
                    "model type not yet supported",
                    "treeexplainer",
                    "does not support treeexplainer",
                    "linear model",
                    "logisticregression",
                ]
            ):
                typer.secho(
                    f"Warning: TreeSHAP not compatible with {type(native_model).__name__}. Using KernelSHAP instead...",
                    fg=typer.colors.YELLOW,
                )
                # Fallback to KernelSHAP (use original model_obj for predict interface)
                import shap

                explainer = shap.KernelExplainer(model_obj.predict_proba, shap.sample(X_instance_encoded, 100))
                shap_values = explainer.shap_values(X_instance_encoded)[0, :, 1]
            else:
                typer.secho(
                    f"Error generating SHAP values: {e}",
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.echo("\nTip: Ensure model is TreeSHAP-compatible (XGBoost, LightGBM, RandomForest)")
                typer.echo("If using a wrapped model, save the native model directly instead.")
                raise typer.Exit(ExitCode.USER_ERROR) from None

        typer.echo("Generating counterfactual recommendations...")

        # Build policy constraints
        from ..explain.policy import PolicyConstraints

        policy = PolicyConstraints(
            immutable_features=immutable_features,
            monotonic_constraints=monotonic_constraints,
            feature_costs=feature_costs if feature_costs else dict.fromkeys(feature_names, 1.0),
            feature_bounds=feature_bounds,
        )

        # Generate recourse
        from ..explain.recourse import generate_recourse

        result = generate_recourse(
            model=model_obj,
            feature_values=feature_values_series,
            shap_values=shap_values,
            feature_names=feature_names,
            instance_id=instance,
            original_prediction=prediction,
            threshold=threshold,
            policy_constraints=policy,
            top_n=top_n,
            seed=seed,
        )

        # Format output as JSON
        output_dict = {
            "instance_id": result.instance_id,
            "original_prediction": result.original_prediction,
            "threshold": result.threshold,
            "recommendations": [
                {
                    "rank": rec.rank,
                    "feature_changes": {
                        feature: {"old": old_val, "new": new_val}
                        for feature, (old_val, new_val) in rec.feature_changes.items()
                    },
                    "total_cost": rec.total_cost,
                    "predicted_probability": rec.predicted_probability,
                    "feasible": rec.feasible,
                }
                for rec in result.recommendations
            ],
            "policy_constraints": {
                "immutable_features": result.policy_constraints.immutable_features,
                "monotonic_constraints": result.policy_constraints.monotonic_constraints,
            },
            "seed": result.seed,
            "total_candidates": result.total_candidates,
            "feasible_candidates": result.feasible_candidates,
        }
        output_text = json.dumps(output_dict, indent=2, sort_keys=True)

        # Write or print output
        if output:
            output.write_text(output_text)
            typer.secho(
                "\n✅ Recourse recommendations generated successfully!",
                fg=typer.colors.GREEN,
            )
            typer.echo(f"Output: {output}")
        else:
            typer.echo("\n" + "=" * 60)
            typer.echo(output_text)
            typer.echo("=" * 60)

        # Show summary
        typer.echo(f"\nInstance: {result.instance_id}")
        typer.echo(f"Original prediction: {result.original_prediction:.1%}")
        typer.echo(f"Threshold: {result.threshold:.1%}")
        typer.echo(f"Recommendations: {len(result.recommendations)}")
        typer.echo(f"Total candidates evaluated: {result.total_candidates}")
        typer.echo(f"Feasible candidates: {result.feasible_candidates}")

        if len(result.recommendations) == 0:
            typer.secho(
                "\n⚠️  No feasible recourse found. Try:",
                fg=typer.colors.YELLOW,
            )
            typer.echo("  - Relaxing monotonic constraints")
            typer.echo("  - Reducing immutable features")
            typer.echo("  - Increasing feature bounds")

    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR) from None
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR) from None
    except Exception as e:
        if "--verbose" in sys.argv or "-v" in sys.argv:
            logger.exception("Recourse generation failed")
        typer.secho(f"Failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR) from None
