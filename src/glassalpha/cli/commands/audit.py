"""Audit command and related helpers.

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

from glassalpha.cli._output import safe_echo
from glassalpha.cli.exit_codes import ExitCode

from ._shared import (
    bootstrap_components,
    ensure_docs_if_pdf,
    is_ci_environment,
    output_error,
    print_banner,
)

logger = logging.getLogger(__name__)


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
        Audit results object

    """
    import time

    # Import here to avoid circular imports and startup overhead
    from glassalpha.pipeline.audit import run_audit_pipeline
    from glassalpha.report import render_audit_report
    from glassalpha.utils.progress import get_progress_bar

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
        from glassalpha.report import _PDF_AVAILABLE
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
        strict_mode_progress = getattr(config.runtime, "strict_mode", False) if hasattr(config, "runtime") else False
        show_progress = not strict_mode_progress

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
                from glassalpha.report import PDFConfig, render_audit_pdf
            except ImportError:
                output_error(
                    "PDF generation requires additional dependencies.\n"
                    "Install with: pip install 'glassalpha[docs]'\n"
                    "Falling back to HTML output...",
                )
                output_format = "html"
                output_path = output_path.with_suffix(".html")

            # Print format-specific message after fallback handling
            # Note: PDF determinism is "best effort" - layout engine may have minor variations
            if output_format == "pdf":
                typer.echo(f"\nGenerating PDF report: {output_path}")

                # Warn user about PDF generation issues and offer alternative
                typer.echo()
                typer.secho("⚠️  WARNING: PDF generation can be slow and may hang", fg=typer.colors.YELLOW)
                typer.echo()
                typer.echo("Known issues:")
                typer.echo("  • Can take 2-5 minutes for complex reports")
                typer.echo("  • May hang indefinitely on some systems")
                typer.echo("  • WeasyPrint rendering engine limitations")
                typer.echo()
                typer.secho("💡 Recommended alternative:", fg=typer.colors.CYAN)
                typer.echo("  1. Generate HTML first: --output report.html  (fast, reliable)")
                typer.echo("  2. Convert to PDF if needed: glassalpha html-to-pdf report.html")
                typer.echo()
                typer.echo("Press Ctrl+C within 10 seconds to use HTML format instead...")
                typer.echo()

                # Give user 10 seconds to cancel
                import time

                pdf_cancelled = False
                try:
                    for i in range(10, 0, -1):
                        typer.echo(f"  Starting PDF generation in {i}...", err=True)
                        time.sleep(1)
                except KeyboardInterrupt:
                    pdf_cancelled = True
                    typer.echo()
                    typer.secho("⏩ Switched to HTML format", fg=typer.colors.CYAN)
                    output_format = "html"
                    output_path = output_path.with_suffix(".html")

                if not pdf_cancelled:
                    typer.echo()
                    typer.echo("⏳ PDF generation in progress... (this may take 1-3 minutes)")

                # Create PDF configuration
                pdf_config = PDFConfig(
                    page_size="A4",
                    title="ML Model Audit Report",
                    author="GlassAlpha",
                    subject="Machine Learning Model Compliance Assessment",
                    optimize_size=True,
                )

                # Only proceed with PDF generation if user didn't cancel
                if not pdf_cancelled:
                    # Use deterministic timestamp for PDF generation
                    from glassalpha.utils.determinism import get_deterministic_timestamp

                    seed = audit_results.execution_info.get("random_seed") if audit_results.execution_info else None
                    deterministic_ts = get_deterministic_timestamp(seed=seed)
                    report_date = deterministic_ts.strftime("%Y-%m-%d")
                    generation_date = deterministic_ts.strftime("%Y-%m-%d %H:%M:%S UTC")

                    # Add timeout protection for PDF generation using concurrent.futures
                    import concurrent.futures

                    # Create progress queue for thread communication BEFORE worker definition
                    import queue

                    progress_queue = queue.Queue()

                    def pdf_progress_relay(message: str, percent: int) -> None:
                        """Progress callback that writes to shared queue."""
                        progress_queue.put((message, percent))

                    pdf_path = None

                    def pdf_generation_worker():
                        """Worker function for PDF generation that can run in a thread."""
                        try:
                            # PDF generation inherits deterministic context from parent thread
                            # (thread limits already set before ThreadPoolExecutor creation)
                            pdf_path = render_audit_pdf(
                                audit_results=audit_results,
                                output_path=output_path,
                                config=pdf_config,
                                report_title=f"ML Model Audit Report - {report_date}",
                                generation_date=generation_date,
                                show_progress=True,  # Show progress in CLI
                                progress_callback=pdf_progress_relay,  # Pass progress to main thread
                            )
                            return pdf_path
                        except Exception as e:
                            raise e

            if output_format == "pdf" and not pdf_cancelled:
                try:
                    # Run PDF generation with timeout using ThreadPoolExecutor
                    # Wrap in deterministic context BEFORE creating threads to avoid deadlocks
                    pdf_start_time = time.time()
                    timeout_seconds = 300  # 5 minutes max

                    from glassalpha.utils.determinism import deterministic

                    with deterministic(seed=seed or 42, strict=True):
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(pdf_generation_worker)

                            # Monitor progress from queue while PDF generates
                            typer.echo("⏳ PDF generation in progress...", err=True)
                            completed = False
                            last_progress_time = time.time()

                            while not completed:
                                # Check overall timeout
                                elapsed = time.time() - pdf_start_time
                                if elapsed > timeout_seconds:
                                    typer.echo("  [TIMEOUT] PDF generation exceeded 5 minutes", err=True)
                                    future.cancel()
                                    break

                                try:
                                    message, percent = progress_queue.get(timeout=0.5)
                                    typer.echo(f"  [{percent:3d}%] {message}", err=True)
                                    last_progress_time = time.time()
                                    if "successfully" in message or "complete" in message:
                                        completed = True
                                except queue.Empty:
                                    # Check if stuck (no progress for 90 seconds)
                                    if time.time() - last_progress_time > 90:
                                        typer.echo("  [WARNING] No progress for 90 seconds, still running...", err=True)
                                        last_progress_time = time.time()  # Reset to avoid spam
                                    continue
                                except Exception:
                                    # Queue might be closed or other error
                                    break

                            # Get final result with remaining timeout
                            remaining_timeout = max(1, timeout_seconds - (time.time() - pdf_start_time))
                            pdf_path = future.result(timeout=remaining_timeout)

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

                pdf_time = time.time() - pdf_start_time

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
                from glassalpha.provenance import write_manifest_sidecar

                try:
                    manifest_path = write_manifest_sidecar(
                        audit_results.execution_info["provenance_manifest"],
                        output_path,
                    )
                    typer.echo(f"Manifest: {manifest_path}")
                except Exception as e:
                    logger.warning(f"Failed to write manifest sidecar: {e}")

        elif output_format == "html":
            typer.echo(f"\nGenerating HTML report: {output_path}")

            # Generate HTML
            html_start = time.time()
            # Use deterministic timestamp if SOURCE_DATE_EPOCH is set

            # Use deterministic timestamp for reproducibility
            from glassalpha.utils.determinism import get_deterministic_timestamp

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
                from glassalpha.provenance import write_manifest_sidecar

                try:
                    manifest_path = write_manifest_sidecar(
                        audit_results.execution_info["provenance_manifest"],
                        output_path,
                    )
                except Exception as e:
                    logger.warning(f"Failed to write manifest sidecar: {e}")

            # Success message with detailed paths
            if manifest_path and manifest_path.exists():
                safe_echo(f"Manifest: {manifest_path}")

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

        # Calculate final timing
        total_time = time.time() - start_time

        # Show final output information
        typer.echo("\nAudit Report Generated Successfully!")
        typer.echo("=" * 50)

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
                typer.echo("   • Enable runtime.fast_mode in config for faster iteration (reduces bootstrap samples)")
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

        if manifest_path and manifest_path.exists():
            typer.echo(f"\nManifest: {manifest_path}")

        typer.echo("\nThe audit report is ready for review and regulatory submission.")

        # Show "What now?" guidance for new users
        typer.echo("\n" + "=" * 50)
        typer.secho("Next Steps:", fg=typer.colors.CYAN, bold=True)
        typer.echo()
        if output_format == "html":
            typer.echo("  📄 View report:")
            typer.echo(f"     open {output_path}  # macOS")
            typer.echo(f"     xdg-open {output_path}  # Linux")
            typer.echo()
        typer.echo("  🔍 Try advanced features:")
        typer.echo("     glassalpha audit --check-shift gender:+0.1  # Test demographic shifts")
        # Show reason codes example if model save path was configured
        if hasattr(config.model, "save_path") and config.model.save_path:
            model_save_path = config.model.save_path
            typer.echo(f"     glassalpha reasons -m {model_save_path} -d models/test_data.csv -i 0")
        typer.echo(f"     glassalpha export-evidence-pack {output_path}  # Create verification package")
        typer.echo()
        typer.echo("  📖 Learn more:")
        typer.echo("     glassalpha docs quickstart")
        typer.echo("=" * 50)

        # Regulatory compliance message
        if getattr(getattr(config, "runtime", None), "strict_mode", False):
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
            from glassalpha.metrics.shift import parse_shift_spec

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
    from glassalpha.metrics.shift import parse_shift_spec

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

    from glassalpha.data import TabularDataLoader
    from glassalpha.metrics.shift import parse_shift_spec, run_shift_analysis
    from glassalpha.models import load_model_from_config

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
            from glassalpha.datasets import get_german_credit_schema, load_german_credit

            data = load_german_credit()
            schema = get_german_credit_schema()
        else:
            # Load from file path
            from glassalpha.data.tabular import TabularDataSchema

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
        from glassalpha.utils.preprocessing import preprocess_auto

        X_test = preprocess_auto(X_test)

        # Binarize sensitive features for shift analysis (shift analysis requires binary attributes)
        sensitive_features_binarized = _binarize_sensitive_features_for_shift(sensitive_features, shift_specs)

        # Load model using same approach as audit pipeline
        typer.echo("Loading model...")
        model = load_model_from_config(audit_config.model)

        # Train model if not loaded from file
        if not hasattr(model, "_is_fitted") or not model._is_fitted:
            typer.echo("Training model...")
            from glassalpha.pipeline.train import train_from_config

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

                # Display result with clear before/after proportions
                orig_pct = result.shift_spec.original_proportion * 100
                shifted_pct = result.shift_spec.shifted_proportion * 100
                typer.echo(
                    f"\nAnalyzing shift: {attribute} {shift_value:+.2f} ({shift_value * 100:+.0f} percentage points)"
                )
                typer.secho(
                    f"  {orig_pct:.1f}% → {shifted_pct:.1f}% (change: {shifted_pct - orig_pct:+.1f}pp)",
                    fg=typer.colors.CYAN,
                )
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


def _check_available_components() -> dict[str, list[str]]:
    """Check available components based on runtime dependencies."""
    import importlib.util

    available = {
        "models": [],
        "explainers": [],
        "metrics": [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc_roc",
            "demographic_parity",
            "equal_opportunity",
            "equalized_odds",
        ],
        "profiles": ["tabular_compliance"],
    }

    # Check model dependencies
    if importlib.util.find_spec("sklearn"):
        available["models"].append("logistic_regression")
    if importlib.util.find_spec("xgboost"):
        available["models"].append("xgboost")
    if importlib.util.find_spec("lightgbm"):
        available["models"].append("lightgbm")

    # Check explainer dependencies
    available["explainers"].append("coefficients")  # Always available
    if importlib.util.find_spec("shap"):
        available["explainers"].extend(["treeshap", "kernelshap"])

    return available


def audit(  # pragma: no cover
    # Typer requires function calls in defaults - this is the documented pattern
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to audit configuration YAML file (auto-detects glassalpha.yaml, audit.yaml, audit_config.yaml, config.yaml)",
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
        help="Enable strict mode for regulatory compliance (auto-enabled for prod*/production* configs)",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        "-p",
        help="Override audit profile",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate configuration without generating report",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
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
):
    """Generate a compliance audit report (HTML/PDF) with optional shift testing.

    This is the main command for GlassAlpha. It loads a configuration file,
    runs the audit pipeline, and generates a deterministic audit report.

    Smart Defaults:
        If no --config is provided, searches for: glassalpha.yaml, audit.yaml, audit_config.yaml, config.yaml
        If no --output is provided, uses {config_name}.html
        Strict mode auto-enables for prod*/production* configs
        Repro mode auto-enables in CI environments

    Configuration:
        Runtime options (fast mode, compact report, fallback behavior) are configured
        in the config file under 'runtime:' section. See documentation for details.

    Examples:
        # Minimal usage (uses smart defaults)
        glassalpha audit

        # Explicit paths
        glassalpha audit --config audit.yaml --output report.html

        # Strict mode for regulatory compliance
        glassalpha audit --config production.yaml  # Auto-enables strict!

        # Validate configuration without running audit
        glassalpha audit --dry-run

        # Stress test for demographic shifts (E6.5)
        glassalpha audit --check-shift gender:+0.1

        # Multiple shifts with degradation threshold
        glassalpha audit --check-shift gender:+0.1 --check-shift age:-0.05 --fail-on-degradation 0.05

    """
    try:
        # Track if config was auto-detected for user messaging
        config_was_auto_detected = config is None

        # Auto-detect config if not provided
        if config is None:
            # Search for config files in current and parent directories (up to 3 levels)
            # Note: audit_config.yaml is generated by quickstart command
            config_candidates = ["glassalpha.yaml", "audit.yaml", "audit_config.yaml", "config.yaml"]
            config = None
            search_start = Path.cwd()
            current = search_start
            searched_dirs = []

            # Search current and up to 3 parent directories
            for _ in range(4):
                searched_dirs.append(str(current))
                for candidate in config_candidates:
                    candidate_path = current / candidate
                    if candidate_path.exists():
                        config = candidate_path
                        break

                if config is not None:
                    break

                # Stop at filesystem root
                if current.parent == current:
                    break
                current = current.parent

            if config is None:
                searched_files = ["glassalpha.yaml", "audit.yaml", "audit_config.yaml", "config.yaml"]
                output_error(
                    f"Configuration file not found\n\n"
                    f"Searched directories:\n  {chr(10).join('  ' + d for d in searched_dirs[:3])}\n\n"
                    f"Searched for: {', '.join(searched_files)}\n\n"
                    "Quick fixes:\n"
                    "  1. Generate project: glassalpha quickstart\n"
                    "  2. Specify config path: glassalpha audit --config /path/to/config.yaml\n\n"
                    "Examples:\n"
                    "  glassalpha quickstart  # Creates project with audit_config.yaml\n"
                    "  glassalpha audit --config audit_config.yaml --output report.html",
                )
                raise typer.Exit(ExitCode.USER_ERROR)

        # Auto-detect output path if not provided
        if output is None:
            # Default to {config_name}.html
            output = config.with_suffix(".html")

        # Auto-detect CI environment for repro mode
        repro = is_ci_environment()

        # Check file existence early with specific error message
        if not config.exists():
            current_dir = Path.cwd()

            # Build helpful error message
            error_msg = f"""Configuration file not found: {config}

📂 Current directory: {current_dir}
📍 Searched path: {config.absolute()}
"""

            # List any YAML files in current directory
            yaml_files = list(current_dir.glob("*.yaml")) + list(current_dir.glob("*.yml"))
            if yaml_files:
                error_msg += "\n📋 Available config files:\n"
                for f in yaml_files[:5]:  # Show max 5
                    error_msg += f"  - {f.name}\n"
                if len(yaml_files) > 5:
                    error_msg += f"  ... and {len(yaml_files) - 5} more\n"

                # Smart suggestion: find likely config files
                likely_configs = [
                    f
                    for f in yaml_files
                    if any(keyword in f.name.lower() for keyword in ["audit", "config", "glassalpha"])
                ]
                if likely_configs:
                    error_msg += f"\n💡 Did you mean: glassalpha audit --config {likely_configs[0].name}\n"
            else:
                error_msg += "\n⚠️  No YAML config files found in current directory\n"

            error_msg += """
💡 Quick fixes:
  1. Change to project directory:
     cd /path/to/project && glassalpha audit

  2. Use full config path:
     glassalpha audit --config /full/path/to/audit.yaml

  3. Create config in current directory:
     glassalpha quickstart

📖 Examples:
  glassalpha quickstart
  glassalpha audit --config audit_config.yaml
  glassalpha audit --config ../other-project/audit.yaml"""

            output_error(error_msg)
            raise typer.Exit(code=ExitCode.USER_ERROR)

        # Validate output directory exists before doing any work
        output_dir = output.parent if output.parent != Path() else Path.cwd()
        if not output_dir.exists():
            output_error(f"Output directory does not exist: {output_dir}. Create it with: mkdir -p {output_dir}")
            raise typer.Exit(code=ExitCode.USER_ERROR)

        # Check if output directory is writable
        if not os.access(output_dir, os.W_OK):
            output_error(f"Output directory is not writable: {output_dir}")
            raise typer.Exit(code=ExitCode.SYSTEM_ERROR)

        # Validate manifest sidecar path will be writable
        manifest_path = output.with_suffix(".manifest.json")
        if manifest_path.exists() and not os.access(manifest_path, os.W_OK):
            output_error(
                f"Cannot overwrite existing manifest (read-only): {manifest_path}. "
                "Make the file writable or remove it before running audit",
            )
            raise typer.Exit(code=ExitCode.SYSTEM_ERROR)

        # Import here to avoid circular imports
        from glassalpha.cli.preflight import preflight_check_dependencies, preflight_check_model
        from glassalpha.config import load_config

        # Bootstrap basic components before any preflight checks
        bootstrap_components()

        print_banner()

        # Preflight checks - ensure dependencies are available
        if not preflight_check_dependencies():
            raise typer.Exit(ExitCode.VALIDATION_ERROR)

        # Load configuration - this doesn't need heavy ML libraries
        if config_was_auto_detected:
            typer.echo(f"Loading configuration from: {config} (auto-detected)")
        else:
            typer.echo(f"Loading configuration from: {config}")

        audit_config = load_config(
            config,
            profile_name=profile,
            strict=strict,
        )

        # Validate dataset name early if using built-in dataset
        if audit_config.data.dataset and audit_config.data.dataset not in ["custom", None]:
            from glassalpha.datasets import validate_dataset_name

            is_valid, error_msg = validate_dataset_name(audit_config.data.dataset)
            if not is_valid:
                output_error(f"Configuration error: {error_msg}")
                raise typer.Exit(code=ExitCode.USER_ERROR)

        # Apply fast mode if requested (reduces bootstrap samples for quick demos)
        if audit_config.runtime.fast_mode:
            if not hasattr(audit_config, "metrics") or audit_config.metrics is None:
                from glassalpha.config.schema import MetricsConfig

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

        # Validate shift specifications early before running pipeline
        if check_shift:
            for shift_spec in check_shift:
                if ":" not in shift_spec:
                    typer.secho(
                        f"❌ Invalid shift specification: '{shift_spec}'\n\n"
                        f"Expected format: 'attribute:shift' (e.g., 'gender:+0.1', 'age:-0.05')\n\n"
                        f"Examples:\n"
                        f"  glassalpha audit --check-shift gender:+0.1\n"
                        f"  glassalpha audit --check-shift race:-0.05 --check-shift age:+0.1\n",
                        fg=typer.colors.RED,
                        err=True,
                    )
                    raise typer.Exit(ExitCode.USER_ERROR)

                attribute, shift_str = shift_spec.split(":", 1)
                try:
                    shift_value = float(shift_str)
                    if abs(shift_value) > 0.5:
                        typer.secho(
                            f"⚠️  Warning: Large shift value ({shift_value}) for '{attribute}'\n"
                            f"   Typical range: -0.2 to +0.2 (±20 percentage points)\n"
                            f"   Values >0.5 may produce unrealistic distributions",
                            fg=typer.colors.YELLOW,
                        )
                except ValueError:
                    typer.secho(
                        f"❌ Invalid shift value: '{shift_str}' for attribute '{attribute}'\n\n"
                        f"Shift value must be a number (e.g., +0.1, -0.05)\n\n"
                        f"Examples:\n"
                        f"  gender:+0.1   # Increase gender=1 proportion by 10pp\n"
                        f"  age:-0.05     # Decrease age=senior proportion by 5pp\n",
                        fg=typer.colors.RED,
                        err=True,
                    )
                    raise typer.Exit(ExitCode.USER_ERROR)

        # Show progress indication before starting pipeline
        typer.echo()
        typer.secho("⏱️  Running audit pipeline...", fg=typer.colors.CYAN)
        if not audit_config.runtime.fast_mode:
            typer.echo("   Estimated time: 5-7 seconds (set runtime.fast_mode=true for 2-3 seconds)")
        else:
            typer.echo("   Estimated time: 2-3 seconds")
        typer.echo()

        # Validate model availability and apply fallbacks (or fail if no_fallback is set)
        audit_config, requested_model = preflight_check_model(
            audit_config,
            allow_fallback=not audit_config.runtime.no_fallback,
        )

        # Determine explainer selection early for consistent display (explicit dispatch)
        from glassalpha.explain import select_explainer

        selected_explainer = select_explainer(model_type=audit_config.model.type, config=audit_config.model_dump())
        typer.echo(f"Explainer: {type(selected_explainer).__name__}")

        # Apply repro mode if requested
        if repro:
            from glassalpha.runtime import set_repro

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
        strict_mode_display = (
            getattr(audit_config.runtime, "strict_mode", False) if hasattr(audit_config, "runtime") else False
        )
        typer.echo(f"Strict mode: {'ENABLED' if strict_mode_display else 'disabled'}")
        typer.echo(f"Repro mode: {'ENABLED' if repro else 'disabled'}")

        strict_mode_enabled = (
            getattr(audit_config.runtime, "strict_mode", False) if hasattr(audit_config, "runtime") else False
        )
        if strict_mode_enabled:
            typer.secho("⚠️  Strict mode enabled - enforcing regulatory compliance", fg=typer.colors.YELLOW)

        if repro:
            typer.secho("🔒 Repro mode enabled - results will be deterministic", fg=typer.colors.BLUE)

        # Validate components exist
        available = _check_available_components()
        model_type = audit_config.model.type

        if model_type not in available.get("models", []) and model_type != "passthrough":
            typer.secho(f"Warning: Model type '{model_type}' not found in registry", fg=typer.colors.YELLOW)

        if dry_run:
            typer.echo()
            typer.secho("✓ Configuration validation passed", fg=typer.colors.GREEN)
            typer.echo()
            typer.echo("Will execute:")
            typer.echo(f"  Profile: {audit_config.audit_profile}")
            typer.echo(f"  Model: {audit_config.model.type}")
            typer.echo(f"  Explainer: {selected_explainer or 'auto-detect'}")
            typer.echo(f"  Data: {getattr(audit_config.data, 'dataset', 'custom')}")

            # Show shift testing if configured
            if check_shift:
                typer.echo()
                typer.echo("Shift tests:")
                for shift_spec in check_shift:
                    attribute, shift_value = shift_spec.split(":", 1)
                    typer.echo(f"  - {attribute}: {shift_value} ({float(shift_value) * 100:+.0f} percentage points)")
                if fail_on_degradation:
                    typer.echo(
                        f"  Degradation gate: {fail_on_degradation} ({fail_on_degradation * 100:.0f}pp threshold)"
                    )

            # Estimate timing
            typer.echo()
            if not audit_config.runtime.fast_mode:
                typer.echo("Estimated time: 12-15 seconds")
            else:
                typer.echo("Estimated time: 2-3 seconds (fast mode enabled)")

            typer.echo()
            typer.echo("Dry run complete - no report generated")
            typer.echo()
            typer.echo("Next step:")
            command = f"glassalpha audit --config {config} --output {output}"
            if check_shift:
                for shift_spec in check_shift:
                    command += f" --check-shift {shift_spec}"
            if fail_on_degradation:
                command += f" --fail-on-degradation {fail_on_degradation}"
            typer.echo(f"  {command}")
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
            ensure_docs_if_pdf(str(output))

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
        from glassalpha.utils.determinism import deterministic

        seed = (
            audit_config.reproducibility.random_seed
            if hasattr(audit_config, "reproducibility")
            and audit_config.reproducibility
            and hasattr(audit_config.reproducibility, "random_seed")
            else 42
        )

        # Get strict mode from config, defaulting appropriately
        strict_mode = getattr(config.runtime, "strict_mode", False) if hasattr(config, "runtime") else False

        with deterministic(seed=seed, strict=strict_mode):
            audit_results = _run_audit_pipeline(
                audit_config,
                output,
                selected_explainer,
                compact=audit_config.runtime.compact_report,
                requested_model=requested_model,
            )

        # Save model if requested (from config)
        save_model_path = audit_config.model.save_path if hasattr(audit_config.model, "save_path") else None
        if save_model_path and audit_results:
            try:
                import joblib

                model_to_save = audit_results.trained_model
                if model_to_save is not None:
                    save_model_path.parent.mkdir(parents=True, exist_ok=True)

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

                    # Extract feature names from data_summary
                    feature_names = None
                    if hasattr(audit_results, "data_summary") and audit_results.data_summary:
                        feature_names = audit_results.data_summary.get("feature_names")

                    model_artifact = {
                        "model": underlying_model,
                        "feature_names": feature_names,
                        "preprocessing": complete_preprocessing_info,
                        "target_column": audit_config.data.target_column,
                        "protected_attributes": audit_config.data.protected_attributes,
                        "glassalpha_version": "0.2.0",
                    }

                    try:
                        joblib.dump(model_artifact, save_model_path)

                        # Show detailed model save info with prominent visibility
                        model_size = save_model_path.stat().st_size
                        typer.echo()
                        typer.echo("=" * 50)
                        typer.secho(f"✓ Model saved to: {save_model_path}", fg=typer.colors.GREEN, bold=True)
                        typer.echo(f"  Size: {model_size:,} bytes ({model_size / 1024:.1f} KB)")
                        typer.echo(f"  Type: {audit_config.model.type}")
                        if feature_names:
                            typer.echo(f"  Features: {len(feature_names)}")
                        typer.echo("=" * 50)

                        # Show next steps for using the model
                        test_data_hint = "models/test_data.csv"  # Standard location from audit
                        typer.echo()
                        typer.secho("Next steps:", fg=typer.colors.CYAN, bold=True)
                        typer.echo(
                            f"  Generate reason codes: glassalpha reasons -m {save_model_path} -d {test_data_hint} -i 0"
                        )
                        typer.echo(
                            f"  Generate recourse: glassalpha recourse -m {save_model_path} -d {test_data_hint} -i 0"
                        )
                    except Exception as e:
                        # Only show warning if actual save failed
                        typer.secho(f"⚠️  Failed to save model: {e}", fg=typer.colors.YELLOW)
                else:
                    typer.secho("⚠️  No model available to save", fg=typer.colors.YELLOW)
            except Exception as e:
                # Catch errors in model extraction/preparation
                typer.secho(f"⚠️  Failed to prepare model for saving: {e}", fg=typer.colors.YELLOW)

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

    except typer.Exit:
        # Re-raise typer.Exit without modification - these are intentional exits with custom messages
        raise
    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        # Use 'from None' to suppress Python traceback for clean CLI UX
        # Users should see "File not found", not internal stack traces
        raise typer.Exit(ExitCode.USER_ERROR) from None
    except ValueError as e:
        # Check if this is a strict mode validation error (from Pydantic validator)
        # Strict mode validation errors should return VALIDATION_ERROR for CI integration
        error_msg = str(e)
        if "Strict mode validation failed:" in error_msg:
            typer.secho(f"Strict mode validation failed: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(ExitCode.VALIDATION_ERROR) from None

        # Other ValueErrors are user configuration errors
        typer.secho(f"Configuration error: {e}", fg=typer.colors.RED, err=True)
        # Intentional: Clean error message for end users
        raise typer.Exit(ExitCode.USER_ERROR) from None
    except Exception as e:
        # Show clean error message (traceback only in verbose mode)
        typer.secho(f"Audit failed: {e}", fg=typer.colors.RED, err=True)
        if verbose or "--verbose" in sys.argv or "-v" in sys.argv:
            logger.exception("Full traceback (verbose mode enabled)")
        # CLI design: Hide Python internals from users by default
        raise typer.Exit(ExitCode.USER_ERROR) from None
