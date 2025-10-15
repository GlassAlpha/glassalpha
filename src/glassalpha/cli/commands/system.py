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
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON for CI integration",
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

        # JSON output for CI integration
        glassalpha doctor --json

    """
    import importlib.util
    import json
    import platform

    # Collect data first
    env_data = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "version": platform.version(),
        },
        "determinism": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "TZ": os.environ.get("TZ"),
            "MPLBACKEND": os.environ.get("MPLBACKEND"),
            "SOURCE_DATE_EPOCH": os.environ.get("SOURCE_DATE_EPOCH"),
        },
        "features": {
            "shap": importlib.util.find_spec("shap") is not None,
            "xgboost": importlib.util.find_spec("xgboost") is not None,
            "lightgbm": importlib.util.find_spec("lightgbm") is not None,
            "matplotlib": importlib.util.find_spec("matplotlib") is not None,
        },
    }

    # Try to get package versions
    try:
        import importlib.metadata

        env_data["versions"] = {}
        for pkg in ["numpy", "pandas", "scikit-learn", "glassalpha"]:
            try:
                env_data["versions"][pkg] = importlib.metadata.version(pkg)
            except Exception:
                pass
    except ImportError:
        pass

    # JSON output mode
    if json_output:
        typer.echo(json.dumps(env_data, indent=2))
        return

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
        typer.echo(f"  PDF export: ✅ installed ({pdf_backend_name})")
    else:
        typer.echo("  PDF export: ❌ not installed")

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
        missing_features.append("PDF export")
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
            typer.echo("  📄 For PDF export:")
            typer.echo("     pip install 'glassalpha[all]'")
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
            missing_vars.append(var)

    if missing_vars:
        typer.echo()
        typer.secho(
            "  ℹ️  Determinism Status: Not configured (optional for development)", fg=typer.colors.CYAN, bold=True
        )
        typer.echo()
        typer.echo("  Why it matters:")
        typer.echo("    Regulators require byte-identical audit outputs for verification.")
        typer.echo("    Without these variables, reports may vary slightly across runs.")
        typer.echo()
        typer.echo("  When to enable:")
        typer.echo("    • Before production audits or regulatory submission")
        typer.echo("    • When sharing audits with compliance officers")
        typer.echo("    • For CI/CD pipelines that verify outputs")
        typer.echo()
        typer.secho("  ✅ Quick setup (your quickstart projects include this automatically):", fg=typer.colors.GREEN)
        typer.echo("    glassalpha quickstart  # Determinism built-in")
        typer.echo()
        typer.secho("  🔧 Manual setup (for existing projects):", fg=typer.colors.YELLOW)
        typer.echo('    eval "$(glassalpha setup-env)"  # Current shell session')
        typer.echo("    glassalpha setup-env >> ~/.bashrc  # Permanent (restart shell after)")
        typer.echo()
        typer.echo("  Learn more: glassalpha docs determinism")
    else:
        typer.echo("  ✅ Determinism configured!")
        typer.echo("     Your audits will be byte-identical across runs.")

    typer.echo()

    # Verbose output - detailed environment information
    if verbose:
        typer.echo("\n" + "=" * 40)
        typer.echo("Detailed Environment Information")
        typer.echo("=" * 40)

        # Python details
        typer.echo("\nPython Environment:")
        typer.echo(f"  Executable: {env_data['python_executable']}")
        typer.echo(f"  Version: {env_data['python_version']}")
        typer.echo(f"  Path: {sys.path[0]}")

        # Package versions
        typer.echo("\nInstalled Package Versions:")
        if "versions" in env_data:
            for package, version in env_data["versions"].items():
                typer.echo(f"  {package}: {version}")
        else:
            typer.echo("  (unable to retrieve package versions)")

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
        for var, value in env_data["determinism"].items():
            typer.echo(f"  {var}: {value or '(not set)'}")

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
                        status = "[WARN]"
                        note = " (requires: pip install 'glassalpha[explain]')"
                elif comp_type == "explainers":
                    if item in ("treeshap", "kernelshap") and not has_shap:
                        status = "[WARN]"
                        note = " (requires: pip install 'glassalpha[explain]')"

                if verbose:
                    typer.echo(f"  {status} {item}{note}")
                else:
                    typer.echo(f"  {status} {item}{note}")
