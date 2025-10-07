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
from typing import Any

import typer

from .defaults import get_smart_defaults
from .exit_codes import ExitCode
from .json_error import JSONErrorOutput, should_use_json_output

logger = logging.getLogger(__name__)


def _output_error(
    exit_code: int,
    error_type: str,
    message: str,
    details: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    use_typer: bool = True,
) -> None:
    """Output error message (JSON or human-readable based on mode).

    Args:
        exit_code: Exit code to use
        error_type: Error type (CONFIG, DATA, MODEL, etc.)
        message: Error message
        details: Additional error details
        context: Contextual information
        use_typer: Use typer.echo for human output (vs plain print)

    """
    if should_use_json_output():
        JSONErrorOutput.output_error(
            exit_code=exit_code,
            error_type=error_type,
            message=message,
            details=details,
            context=context,
        )
    # Human-readable output
    elif use_typer:
        typer.echo(message, err=True)
    else:
        print(message, file=sys.stderr)


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
                    "PDF requested but no PDF backend found.\n"
                    'Install: pip install "glassalpha[docs]"\n'
                    "Or use: --output audit.html",
                )


def _ascii(s: str) -> str:
    """Convert Unicode characters to ASCII equivalents for CLI compatibility."""
    return (
        s.replace("✓", "OK")
        .replace("•", "*")
        .replace("—", "-")
        .replace("–", "-")
        .replace("…", "...")
        .replace(""", '"').replace(""", '"')
        .replace("'", "'")
        .replace("❌", "X")
        .replace("⚠️", "!")
        .replace("🎉", "")
        .replace("📊", "")
        .replace("📁", "")
        .replace("⏱️", "")
        .replace("🛡️", "")
        .replace("⚖️", "")
        .replace("🔍", "")
        .replace("📋", "")
        .replace("🔧", "")
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


def _run_audit_pipeline(config, output_path: Path, selected_explainer: str | None = None) -> None:
    """Execute the complete audit pipeline and generate report in specified format.

    Args:
        config: Validated audit configuration
        output_path: Path where report should be saved

    """
    import time
    from datetime import datetime

    # Import here to avoid circular imports and startup overhead
    from ..pipeline.audit import run_audit_pipeline
    from ..report import render_audit_report

    start_time = time.time()

    # Determine output format from config and check PDF availability
    output_format = getattr(config.report, "output_format", "pdf") if hasattr(config, "report") else "pdf"

    # Check if PDF backend is available
    try:
        from ..report import _PDF_AVAILABLE
    except ImportError:
        _PDF_AVAILABLE = False

    # Fallback to HTML if PDF requested but not available
    if output_format == "pdf" and not _PDF_AVAILABLE:
        typer.echo("⚠️  PDF backend not available. Generating HTML report instead.")
        typer.echo("💡 To enable PDF reports: pip install 'glassalpha[docs]'\n")
        output_format = "html"
        # Update output path extension if needed
        if output_path.suffix.lower() == ".pdf":
            output_path = output_path.with_suffix(".html")

    try:
        # Step 1: Run audit pipeline
        typer.echo("  Loading data and initializing components...")

        # Run the actual audit pipeline
        audit_results = run_audit_pipeline(config, selected_explainer=selected_explainer)

        if not audit_results.success:
            typer.secho(
                _ascii(f"❌ Audit pipeline failed: {audit_results.error_message}"),
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        pipeline_time = time.time() - start_time
        typer.secho(_ascii(f"✓ Audit pipeline completed in {pipeline_time:.2f}s"), fg=typer.colors.GREEN)

        # Show audit summary
        _display_audit_summary(audit_results)

        # Step 2: Generate report in specified format
        if output_format == "pdf":
            # Import PDF dependencies only when needed
            from ..report import PDFConfig, render_audit_pdf

            typer.echo(f"\nGenerating PDF report: {output_path}")

            # Create PDF configuration
            pdf_config = PDFConfig(
                page_size="A4",
                title="ML Model Audit Report",
                author="GlassAlpha",
                subject="Machine Learning Model Compliance Assessment",
                optimize_size=True,
            )

            # Generate PDF
            pdf_start = time.time()
            pdf_path = render_audit_pdf(
                audit_results=audit_results,
                output_path=output_path,
                config=pdf_config,
                report_title=f"ML Model Audit Report - {datetime.now().strftime('%Y-%m-%d')}",
                generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            )

            pdf_time = time.time() - pdf_start
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
                    typer.echo(_ascii(f"📋 Manifest: {manifest_path}"))
                except Exception as e:
                    logger.warning(f"Failed to write manifest sidecar: {e}")

            # Success message
            total_time = time.time() - start_time
            typer.echo(_ascii("\n🎉 Audit Report Generated Successfully!"))

        elif output_format == "html":
            typer.echo(f"\nGenerating HTML report: {output_path}")

            # Generate HTML
            html_start = time.time()
            html_content = render_audit_report(
                audit_results=audit_results,
                output_path=output_path,
                report_title=f"ML Model Audit Report - {datetime.now().strftime('%Y-%m-%d')}",
                generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
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
                    typer.echo(_ascii(f"📋 Manifest: {manifest_path}"))
                except Exception as e:
                    logger.warning(f"Failed to write manifest sidecar: {e}")

            # Success message
            total_time = time.time() - start_time
            typer.echo(_ascii("\n🎉 Audit Report Generated Successfully!"))

        else:
            typer.secho(f"Error: Unsupported output format '{output_format}'", fg=typer.colors.RED, err=True)
            raise typer.Exit(ExitCode.USER_ERROR)

        typer.echo("=" * 50)

        # Show final output information
        if output_format == "pdf":
            typer.secho(_ascii(f"📁 Output: {output_path}"), fg=typer.colors.GREEN)
            typer.echo(_ascii(f"📊 Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)"))
            typer.echo(_ascii(f"⏱️  Total time: {total_time:.2f}s"))
            typer.echo(_ascii(f"   • Pipeline: {pipeline_time:.2f}s"))
            typer.echo(_ascii(f"   • PDF generation: {pdf_time:.2f}s"))
        elif output_format == "html":
            typer.secho(_ascii(f"📁 Output: {output_path}"), fg=typer.colors.GREEN)
            typer.echo(_ascii(f"📊 Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)"))
            typer.echo(_ascii(f"⏱️  Total time: {total_time:.2f}s"))
            typer.echo(_ascii(f"   • Pipeline: {pipeline_time:.2f}s"))
            typer.echo(_ascii(f"   • HTML generation: {html_time:.2f}s"))

        typer.echo(_ascii("\nThe audit report is ready for review and regulatory submission."))

        # Regulatory compliance message
        if config.strict_mode:
            typer.secho(
                _ascii("\n🛡️  Strict mode: Report meets regulatory compliance requirements"),
                fg=typer.colors.YELLOW,
            )

    except Exception as e:
        typer.secho(_ascii(f"\n❌ Audit failed: {e!s}"), fg=typer.colors.RED, err=True)

        # Show more details in verbose mode
        if "--verbose" in sys.argv or "-v" in sys.argv:
            logger.exception("Detailed audit failure information:")

        raise typer.Exit(ExitCode.USER_ERROR)


def _display_audit_summary(audit_results) -> None:
    """Display a summary of audit results."""
    typer.echo(_ascii("\n📊 Audit Summary:"))

    # Model performance
    if audit_results.model_performance:
        perf_count = len(
            [m for m in audit_results.model_performance.values() if isinstance(m, dict) and "error" not in m],
        )
        typer.echo(_ascii(f"  ✅ Performance metrics: {perf_count} computed"))

        # Show key metrics
        for name, result in audit_results.model_performance.items():
            if isinstance(result, dict) and "accuracy" in result:
                accuracy = result["accuracy"]
                status = "✅" if accuracy > 0.8 else "⚠️" if accuracy > 0.6 else "❌"
                typer.echo(_ascii(f"     {status} {name}: {accuracy:.1%}"))
                break

    # Fairness analysis
    if audit_results.fairness_analysis:
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
        typer.echo(_ascii(f"  ⚖️  Fairness metrics: {computed_metrics}/{total_metrics} computed"))

        if bias_detected:
            typer.secho(_ascii(f"     ⚠️  Bias detected in: {', '.join(bias_detected[:2])}"), fg=typer.colors.YELLOW)
        elif computed_metrics > 0:
            typer.secho(_ascii("     ✅ No bias detected"), fg=typer.colors.GREEN)

    # SHAP explanations
    if audit_results.explanations:
        has_importance = "global_importance" in audit_results.explanations

        if has_importance:
            typer.echo(_ascii("  🔍 Explanations: ✅ Global feature importance"))

            # Show top feature
            importance = audit_results.explanations.get("global_importance", {})
            if importance:
                top_feature = max(importance.items(), key=lambda x: abs(x[1]))
                typer.echo(f"     Most important: {top_feature[0]} ({top_feature[1]:+.3f})")
        else:
            typer.echo(_ascii("  🔍 Explanations: ❌ Not available"))

    # Data summary
    if audit_results.data_summary and "shape" in audit_results.data_summary:
        rows, cols = audit_results.data_summary["shape"]
        typer.echo(_ascii(f"  📋 Dataset: {rows:,} samples, {cols} features"))

    # Selected components with fallback indication
    if audit_results.selected_components:
        typer.echo(_ascii(f"  🔧 Components: {len(audit_results.selected_components)} selected"))

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
        # Load dataset using same loader as audit pipeline
        typer.echo("\nLoading test data...")
        data_loader = TabularDataLoader()

        # Load data
        data = data_loader.load(audit_config.data.path, audit_config.data)

        # Extract features, target, and sensitive features
        X_test, y_test, sensitive_features = data_loader.extract_features_target(data, audit_config.data)

        # Load model using same approach as audit pipeline
        typer.echo("Loading model...")
        model = load_model_from_config(audit_config.model)

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
                    sensitive_features=sensitive_features,
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

    except Exception as e:
        typer.secho(f"\n✗ Shift analysis failed: {e}", fg=typer.colors.RED)
        if "--verbose" in sys.argv or "-v" in sys.argv:
            logger.exception("Detailed shift analysis failure:")
        return ExitCode.USER_ERROR


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
        help="Enable strict mode for regulatory compliance (auto-enabled for prod*/production* configs)",
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
):
    """Generate a compliance audit PDF report with optional shift testing.

    This is the main command for GlassAlpha. It loads a configuration file,
    runs the audit pipeline, and generates a deterministic PDF report.

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
            _output_error(
                exit_code=ExitCode.USER_ERROR,
                error_type="CONFIG",
                message=str(e),
                details={"tip": "Create a config with 'glassalpha init'"},
            )
            raise typer.Exit(ExitCode.USER_ERROR) from None

        # Extract resolved values
        config = defaults["config"]
        output = defaults["output"]
        strict = defaults["strict"]
        repro = defaults["repro"]

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
                exit_code=ExitCode.USER_ERROR,
                error_type="CONFIG",
                message=f"File '{config}' does not exist.",
                context={"config_path": str(config)},
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        # Check override config if provided
        if override_config and not override_config.exists():
            _output_error(
                exit_code=ExitCode.USER_ERROR,
                error_type="CONFIG",
                message=f"Override file '{override_config}' does not exist.",
                context={"override_path": str(override_config)},
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        # Validate output directory exists before doing any work
        output_dir = output.parent if output.parent != Path() else Path.cwd()
        if not output_dir.exists():
            _output_error(
                exit_code=ExitCode.USER_ERROR,
                error_type="SYSTEM",
                message=f"Output directory does not exist: {output_dir}",
                details={"hint": f"Create it with: mkdir -p {output_dir}"},
                context={"output_dir": str(output_dir)},
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        # Check if output directory is writable
        if not os.access(output_dir, os.W_OK):
            _output_error(
                exit_code=ExitCode.SYSTEM_ERROR,
                error_type="SYSTEM",
                message=f"Output directory is not writable: {output_dir}",
                context={"output_dir": str(output_dir)},
            )
            raise typer.Exit(ExitCode.SYSTEM_ERROR)

        # Validate manifest sidecar path will be writable
        manifest_path = output.with_suffix(".manifest.json")
        if manifest_path.exists() and not os.access(manifest_path, os.W_OK):
            _output_error(
                exit_code=ExitCode.SYSTEM_ERROR,
                error_type="SYSTEM",
                message=f"Cannot overwrite existing manifest (read-only): {manifest_path}",
                details={"tip": "Make the file writable or remove it before running audit"},
                context={"manifest_path": str(manifest_path)},
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

        audit_config = load_config_from_file(config, override_path=override_config, profile_name=profile, strict=strict)

        # Validate model availability and apply fallbacks (or fail if no_fallback is set)
        audit_config = preflight_check_model(audit_config, allow_fallback=not no_fallback)

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
            typer.secho(_ascii("✓ Configuration valid (dry run - no report generated)"), fg=typer.colors.GREEN)
            return

        # Check PDF dependencies if PDF output requested
        output_format = (
            getattr(audit_config.report, "output_format", "pdf") if hasattr(audit_config, "report") else "pdf"
        )
        if output_format == "pdf":
            _ensure_docs_if_pdf(str(output))

        # Run audit pipeline
        typer.echo("\nRunning audit pipeline...")
        _run_audit_pipeline(audit_config, output, selected_explainer)

        # Run shift analysis if requested (E6.5)
        if check_shift:
            typer.echo("\n" + "=" * 60)
            typer.echo("DEMOGRAPHIC SHIFT ANALYSIS (E6.5)")
            typer.echo("=" * 60)

            shift_exit_code = _run_shift_analysis(
                audit_config=audit_config,
                output=output,
                shift_specs=check_shift,
                threshold=fail_on_degradation,
            )

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
        suggested_command = "glassalpha audit --config configs/quickstart.yaml --output quickstart.pdf"
    else:
        suggested_command = "glassalpha audit --config configs/quickstart.yaml --output quickstart.html"

    typer.echo(f"Ready to run: {suggested_command}")
    typer.echo()


def _validate_model_params(config: Any) -> list[str]:  # noqa: ANN401
    """Check model parameters for common issues using capability-based validation.

    Args:
        config: Audit configuration

    Returns:
        List of warning messages

    """
    warnings = []
    model_type = config.model.type
    params = config.model.params if hasattr(config.model, "params") else {}

    # Get model class to access its capabilities
    from glassalpha.core.registry import ModelRegistry  # noqa: PLC0415

    try:
        model_class = ModelRegistry.get(model_type)
        if not model_class:
            return warnings

        # Get parameter rules from model capabilities
        capabilities = getattr(model_class, "capabilities", {})
        param_rules = capabilities.get("parameter_rules", {})

        # Validate each parameter against its rules
        for param_name, param_value in params.items():
            if param_name not in param_rules:
                continue

            rule = param_rules[param_name]
            param_type = rule.get("type")
            description = rule.get("description", param_name)

            # Check special values (e.g., LightGBM max_depth=-1)
            special_values = rule.get("special_values", {})
            if param_value in special_values:
                # This is a valid special value, skip other checks
                continue

            # Type validation
            if param_type == "int" and not isinstance(param_value, int):
                warnings.append(f"{param_name}: {param_value} - {description} must be an integer")
                continue
            if param_type == "float" and not isinstance(param_value, (int, float)):
                warnings.append(f"{param_name}: {param_value} - {description} must be a number")
                continue

            # Range validation
            if "min" in rule:
                min_val = rule["min"]
                exclusive_min = rule.get("exclusive_min", False)
                if exclusive_min:
                    if param_value <= min_val:
                        warnings.append(f"{param_name}: {param_value} - {description} must be > {min_val}")
                elif param_value < min_val:
                    warnings.append(f"{param_name}: {param_value} - {description} must be >= {min_val}")

            if "max" in rule:
                max_val = rule["max"]
                exclusive_max = rule.get("exclusive_max", False)
                if exclusive_max:
                    if param_value >= max_val:
                        warnings.append(f"{param_name}: {param_value} - {description} must be < {max_val}")
                elif param_value > max_val:
                    warnings.append(f"{param_name}: {param_value} - {description} must be <= {max_val}")

            # Typical range warnings (not errors)
            if "typical_range" in rule:
                typical_min, typical_max = rule["typical_range"]
                if param_value < typical_min or param_value > typical_max:
                    warnings.append(
                        f"{param_name}: {param_value} - Typically between {typical_min} and {typical_max} (unusual value)",
                    )

    except (KeyError, ImportError, AttributeError):
        # If model class can't be loaded, skip validation
        # This handles cases where dependencies aren't installed
        pass

    return warnings


def validate(  # pragma: no cover
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to configuration file to validate",
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
        help="Validate for strict mode compliance",
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
        # Basic validation
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

        typer.echo(f"Validating configuration: {config}")

        # Load and validate
        audit_config = load_config_from_file(config, profile_name=profile, strict=strict)

        typer.echo(f"Profile: {audit_config.audit_profile}")
        typer.echo(f"Model type: {audit_config.model.type}")
        typer.echo(f"Strict mode: {'valid' if strict else 'not checked'}")

        # Runtime availability validation
        validation_errors = []
        validation_warnings = []

        # Check model availability
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

        # Check explainer availability and compatibility
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
                # Check model/explainer compatibility
                model_type = audit_config.model.type
                if "treeshap" in requested_explainers and model_type not in ["xgboost", "lightgbm", "random_forest"]:
                    validation_warnings.append(
                        f"TreeSHAP requested but model type '{model_type}' is not a tree model. "
                        "Consider using 'coefficients' (for linear) or 'permutation' (universal).",
                    )

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
                    pass  # Built-in dataset validation is optional

        # Report validation errors
        if validation_errors:
            typer.echo()
            typer.secho(_ascii("✗ Validation failed with errors:"), fg=typer.colors.RED, err=True)
            for error in validation_errors:
                typer.secho(f"  • {error}", fg=typer.colors.RED, err=True)
            typer.echo()
            typer.secho(
                "Tip: Run without --strict-validation to see warnings instead of errors",
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(ExitCode.VALIDATION_ERROR)

        # Report validation results
        typer.secho(_ascii("\n✓ Configuration is valid"), fg=typer.colors.GREEN)

        # Show runtime warnings
        if validation_warnings:
            typer.echo()
            typer.secho(_ascii("⚠ Runtime warnings:"), fg=typer.colors.YELLOW)
            for warning in validation_warnings:
                typer.secho(f"  • {warning}", fg=typer.colors.YELLOW)
            typer.echo()
            if not strict_validation:
                typer.secho(
                    "Tip: Add --strict-validation to treat warnings as errors (recommended for production)",
                    fg=typer.colors.CYAN,
                )

        # Check model parameters
        param_warnings = _validate_model_params(audit_config)
        if param_warnings:
            typer.echo()
            typer.secho(_ascii("⚠ Parameter warnings:"), fg=typer.colors.YELLOW)
            for warning in param_warnings:
                typer.secho(f"  {warning}", fg=typer.colors.YELLOW)
            typer.echo()

            # Add direct link to relevant section
            model_type = audit_config.model.type.replace("_", "-")
            doc_url = f"https://glassalpha.com/docs/reference/model-parameters/#{model_type}"
            typer.echo(_ascii("💡 See parameter reference:"))
            typer.echo(f"   {doc_url}")
            typer.echo("   Or run: glassalpha docs model-parameters")

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

    base_url = "https://glassalpha.com/docs"

    # Build URL based on topic
    if topic:
        # Normalize topic (replace underscores with hyphens)
        topic_normalized = topic.replace("_", "-")
        url = f"{base_url}/reference/{topic_normalized}/"

        # Special cases for common topics
        if topic_normalized in ["quickstart", "installation"]:
            url = f"{base_url}/getting-started/{topic_normalized}/"
        elif topic_normalized in ["cli", "troubleshooting", "faq"]:
            url = f"{base_url}/reference/{topic_normalized}/"
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
    """List available components.

    Shows registered models, explainers, metrics, and audit profiles.

    Examples:
        # List all components
        glassalpha list

        # List specific type
        glassalpha list models

        # Include enterprise components
        glassalpha list --include-enterprise

    """
    from ..core import list_components

    components = list_components(component_type=component_type, include_enterprise=include_enterprise)

    if not components:
        typer.echo(f"No components found for type: {component_type}")
        return

    typer.echo("Available Components")
    typer.echo("=" * 40)

    for comp_type, items in components.items():
        typer.echo(f"\n{comp_type.upper()}:")

        if not items:
            typer.echo("  (none registered)")
        else:
            for item in sorted(items):
                if verbose:
                    # TODO: Show more details about each component
                    typer.echo(f"  - {item}")
                else:
                    typer.echo(f"  - {item}")


def reasons(  # pragma: no cover
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model file (.pkl, .joblib)",
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
    import pickle

    import pandas as pd

    try:
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
        with open(model, "rb") as f:
            model_obj = pickle.load(f)

        typer.echo(f"Loading data from: {data}")
        df = pd.read_csv(data)

        if instance < 0 or instance >= len(df):
            typer.secho(
                f"Error: Instance {instance} out of range (0-{len(df) - 1})",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        # Get instance
        X_instance = df.iloc[[instance]].drop(columns=["target"], errors="ignore")
        feature_names = X_instance.columns.tolist()
        feature_values = X_instance.iloc[0]

        typer.echo(f"Generating SHAP explanations for instance {instance}...")

        # Get prediction
        if hasattr(model_obj, "predict_proba"):
            prediction = float(model_obj.predict_proba(X_instance)[0, 1])
        else:
            prediction = float(model_obj.predict(X_instance)[0])

        # Generate SHAP values (use TreeSHAP for tree models)
        try:
            import shap

            explainer = shap.TreeExplainer(model_obj)
            shap_values = explainer.shap_values(X_instance)

            # Handle multi-output case (binary classification)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class

            # Flatten to 1D
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]

        except Exception as e:
            typer.secho(
                f"Error generating SHAP values: {e}",
                fg=typer.colors.RED,
                err=True,
            )
            typer.echo("\nTip: Ensure model is TreeSHAP-compatible (XGBoost, LightGBM, RandomForest)")
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
            output_text = json.dumps(output_dict, indent=2)
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
        help="Path to trained model file (.pkl, .joblib)",
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
    import pickle

    import pandas as pd

    try:
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
        with open(model, "rb") as f:
            model_obj = pickle.load(f)

        typer.echo(f"Loading data from: {data}")
        df = pd.read_csv(data)

        if instance < 0 or instance >= len(df):
            typer.secho(
                f"Error: Instance {instance} out of range (0-{len(df) - 1})",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        # Get instance
        X_instance = df.iloc[[instance]].drop(columns=["target"], errors="ignore")
        feature_names = X_instance.columns.tolist()
        feature_values_series = X_instance.iloc[0]

        typer.echo(f"Generating SHAP explanations for instance {instance}...")

        # Get prediction
        if hasattr(model_obj, "predict_proba"):
            prediction = float(model_obj.predict_proba(X_instance)[0, 1])
        else:
            prediction = float(model_obj.predict(X_instance)[0])

        # Check if instance is already approved
        if prediction >= threshold:
            typer.secho(
                f"\nInstance {instance} is already approved (prediction={prediction:.1%} >= threshold={threshold:.1%})",
                fg=typer.colors.YELLOW,
            )
            typer.echo("No recourse needed.")
            raise typer.Exit(ExitCode.SUCCESS)

        # Generate SHAP values (use TreeSHAP for tree models)
        try:
            import shap

            explainer = shap.TreeExplainer(model_obj)
            shap_values = explainer.shap_values(X_instance)

            # Handle multi-output case (binary classification)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class

            # Flatten to 1D
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]

        except Exception as e:
            typer.secho(
                f"Warning: SHAP TreeExplainer failed ({e}). Trying KernelSHAP...",
                fg=typer.colors.YELLOW,
            )
            # Fallback to KernelSHAP
            import shap

            explainer = shap.KernelExplainer(model_obj.predict_proba, shap.sample(X_instance, 100))
            shap_values = explainer.shap_values(X_instance)[0, :, 1]

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
        output_text = json.dumps(output_dict, indent=2)

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
