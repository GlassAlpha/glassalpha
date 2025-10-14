"""System commands: doctor, docs, list.

Commands for environment checking and system information.
"""

import logging
import os
import sys

import typer

logger = logging.getLogger(__name__)


def doctor(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed environment information including package versions and paths",
    ),
):  # pragma: no cover
    """Check environment and optional features.

    This command diagnoses the current environment and shows what optional
    features are available and how to enable them.

    Examples:
        # Basic environment check
        glassalpha doctor

        # Verbose output with package versions
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
    # Note: Either XGBoost OR LightGBM is sufficient with SHAP
    has_tree_explain = has_shap and (has_xgboost or has_lightgbm)
    if has_tree_explain:
        installed_parts = []
        if has_shap:
            installed_parts.append("SHAP")
        if has_xgboost:
            installed_parts.append("XGBoost")
        if has_lightgbm:
            installed_parts.append("LightGBM")
        typer.echo(f"  SHAP + tree models: ✅ installed ({', '.join(installed_parts)})")
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
    if not has_tree_explain:
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
        if not has_tree_explain:
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

    # Check determinism environment variables
    typer.echo("Determinism Check")
    typer.echo("-" * 20)

    env_vars = {
        "TZ": "UTC",
        "MPLBACKEND": "Agg",
        "PYTHONHASHSEED": "0",
        "SOURCE_DATE_EPOCH": "(any value)",
    }

    missing_vars = []
    for var, _expected in env_vars.items():
        value = os.environ.get(var)
        if value:
            typer.echo(f"  ✅ {var}={value}")
        else:
            typer.echo(f"  ❌ {var} not set")
            missing_vars.append(var)

    if missing_vars:
        typer.echo()
        typer.secho("  ⚠️  Determinism Warning:", fg=typer.colors.YELLOW, bold=True)
        typer.echo("  Environment variables not set. Audits may not be byte-identical across runs.")
        typer.echo()
        typer.secho("  🔧 Quick fix:", fg=typer.colors.CYAN)
        typer.echo("    glassalpha setup-env --output .glassalpha-env")
        typer.echo("    source .glassalpha-env")
        typer.echo()
        typer.echo("  Or use in current shell:")
        typer.echo('    eval "$(glassalpha setup-env)"')
        typer.echo()
        typer.echo("  Manual setup:")
        typer.echo("    export TZ=UTC")
        typer.echo("    export MPLBACKEND=Agg")
        typer.echo("    export PYTHONHASHSEED=0")
        typer.echo()
        typer.echo("  Why this matters: Regulators require byte-identical outputs for audit verification.")
        typer.echo("  Learn more: glassalpha docs determinism")
    else:
        typer.echo("  ✅ All determinism variables set!")

    typer.echo()

    # Verbose output - detailed environment information
    if verbose:
        typer.echo("\n" + "=" * 40)
        typer.echo("Detailed Environment Information")
        typer.echo("=" * 40)

        # Python details
        typer.echo("\nPython Environment:")
        typer.echo(f"  Executable: {sys.executable}")
        typer.echo(f"  Version: {sys.version}")
        typer.echo(f"  Path: {sys.path[0]}")

        # Package versions
        typer.echo("\nInstalled Package Versions:")
        packages = [
            "numpy",
            "pandas",
            "scikit-learn",
            "matplotlib",
            "jinja2",
            "xgboost",
            "lightgbm",
            "shap",
            "weasyprint",
            "reportlab",
            "glassalpha",
        ]
        for package in packages:
            try:
                import importlib.metadata

                version = importlib.metadata.version(package)
                typer.echo(f"  {package}: {version}")
            except Exception:
                typer.echo(f"  {package}: not installed")

        # Configuration locations
        typer.echo("\nConfiguration Locations:")
        try:
            from platformdirs import user_config_dir, user_data_dir

            typer.echo(f"  Data dir: {user_data_dir('glassalpha')}")
            typer.echo(f"  Config dir: {user_config_dir('glassalpha')}")
        except ImportError:
            typer.echo("  platformdirs not installed (optional)")

        # Cache locations
        typer.echo("\nCache Directory:")
        try:
            from glassalpha.utils.cache_dirs import resolve_data_root

            typer.echo(f"  Cache: {resolve_data_root()}")
        except Exception as e:
            typer.echo(f"  Cache: unable to resolve ({e})")

        # Environment checks
        typer.echo("\nEnvironment Variables:")
        env_vars = ["PYTHONHASHSEED", "TZ", "MPLBACKEND", "SOURCE_DATE_EPOCH"]
        for var in env_vars:
            value = os.environ.get(var, "(not set)")
            typer.echo(f"  {var}: {value}")

        typer.echo()


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

    components = _check_available_components()

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
