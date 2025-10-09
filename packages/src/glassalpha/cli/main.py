"""Main CLI application using Typer.

This module sets up the command groups and structure for the GlassAlpha CLI,
enabling future expansion without breaking changes.

ARCHITECTURE NOTE: Uses Typer function-call defaults (B008 lint rule)
which is the documented Typer pattern. Also uses clean CLI exception
handling with 'from None' to hide Python internals from end users.
"""

import logging
import sys
from pathlib import Path

import typer
from platformdirs import user_data_dir

from .. import __version__
from .exit_codes import ExitCode

# Configure logging with WARNING as default (clean output for users)
# User can override with --verbose (INFO) or --quiet (ERROR only)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Main CLI app
app = typer.Typer(
    name="glassalpha",
    help="GlassAlpha - AI Compliance Toolkit",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=True,
    epilog="""For more information, visit: https://glassalpha.com""",
)

# Command groups
# Core functionality (OSS)
datasets_app = typer.Typer(
    help="Dataset management operations",
    no_args_is_help=True,
)

# Preprocessing artifact management (OSS)
from .prep import prep_app


# First-run detection helper
def _show_first_run_tip():
    """Show helpful tip on first run."""
    # Skip for --help, --version, and doctor command
    if "--help" in sys.argv or "-h" in sys.argv or "--version" in sys.argv or "-V" in sys.argv or "doctor" in sys.argv:
        return

    # Check for state file
    state_dir = Path(user_data_dir("glassalpha", "glassalpha"))
    state_file = state_dir / ".first_run_complete"

    if not state_file.exists():
        # Create state directory and file
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file.touch()

        # Show tip
        typer.echo()
        typer.secho("👋 Welcome to GlassAlpha!", fg=typer.colors.BRIGHT_BLUE, bold=True)
        typer.echo("   Run 'glassalpha doctor' to check your environment and see what features are available.")
        typer.echo("   💡 Tip: Use 'glassalpha audit --fast' for lightning-quick demos (2-3s vs 5-7s).")
        typer.echo()


# Version callback
def version_callback(value: bool):
    """Print version and exit."""
    if value:
        typer.echo(f"GlassAlpha version {__version__}")
        raise typer.Exit(ExitCode.SUCCESS)


@app.callback()
def main_callback(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-error output",
    ),
):
    """GlassAlpha - Transparent, auditable, regulator-ready ML audits.

    Use 'glassalpha COMMAND --help' for more information on a command.
    Global flags like --verbose and --quiet apply to all commands.
    """
    # Set logging level based on flags
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)  # INFO shows progress, not DEBUG details

    # First-run detection - show helpful tip once
    _show_first_run_tip()


# Add command groups to main app
# Core functionality (OSS)
app.add_typer(datasets_app, name="datasets")
app.add_typer(prep_app, name="prep")


# Import and register commands
from .commands import audit, docs, doctor, list_components_cmd, reasons, recourse, validate
from .init import init
from .quickstart import quickstart

# Register main commands
app.command()(audit)
app.command("validate")(validate)
app.command("list", help="List available components")(list_components_cmd)
app.command("doctor", help="Check environment and optional features")(doctor)
app.command("docs", help="Open documentation in browser")(docs)
app.command("init", help="Initialize new audit configuration")(init)
app.command("quickstart", help="Generate template audit project")(quickstart)
app.command("reasons", help="Generate ECOA-compliant reason codes")(reasons)
app.command("recourse", help="Generate counterfactual recourse recommendations")(recourse)

# Register datasets commands with lazy loading (Phase 2 performance optimization)
# These import the datasets module only when the command is actually invoked,
# not during --help rendering, saving ~500ms of import time


@datasets_app.command("list")
def list_datasets_lazy():
    """List all available datasets in the registry."""
    from .datasets import list_datasets

    return list_datasets()


@datasets_app.command("info")
def dataset_info_lazy(dataset: str = typer.Argument(..., help="Dataset key to inspect")):
    """Show detailed information about a specific dataset."""
    from .datasets import dataset_info

    return dataset_info(dataset)


@datasets_app.command("cache-dir")
def show_cache_dir_lazy():
    """Show the directory where datasets are cached."""
    from .datasets import show_cache_dir

    return show_cache_dir()


@datasets_app.command("fetch")
def fetch_dataset_lazy(
    dataset: str = typer.Argument(..., help="Dataset key to fetch"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download even if file exists"),
    dest: Path = typer.Option(None, "--dest", help="Custom destination path"),
):
    """Fetch and cache a dataset from the registry."""
    from .datasets import fetch_dataset

    return fetch_dataset(dataset, force, dest)


# Add models command to show available models
@app.command()
def models():
    """Show available models."""
    # Import models to trigger registration

    from ..core import ModelRegistry

    typer.echo("Available Models:")
    typer.echo("=" * 50)
    typer.echo()

    available_models = ModelRegistry.available_plugins()

    if not available_models:
        typer.echo("No models available.")
        return

    # Group models by category
    core_models = []
    tree_models = []
    other_models = []

    for model, available in available_models.items():
        status = "Available" if available else "Not available"

        if model in ["logistic_regression", "sklearn_generic"]:
            core_models.append((model, status, available))
        elif model in ["xgboost", "lightgbm"]:
            tree_models.append((model, status, available))
        elif model != "passthrough":  # Skip internal passthrough model
            other_models.append((model, status, available))

    # Display core models
    if core_models:
        typer.echo("Core Models (always available):")
        for model, status, available in core_models:
            typer.echo(f"  {status}: {model}")
        typer.echo()

    # Display tree models
    if tree_models:
        typer.echo("Tree Models:")
        for model, status, available in tree_models:
            typer.echo(f"  {status}: {model}")
        typer.echo()

    # Display other models if any
    if other_models:
        typer.echo("Other Models:")
        for model, status, available in other_models:
            typer.echo(f"  {status}: {model}")
        typer.echo()


if __name__ == "__main__":  # pragma: no cover
    app()
