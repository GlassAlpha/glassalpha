"""Main CLI application using Typer.

This module sets up the command groups and structure for the GlassAlpha CLI,
enabling future expansion without breaking changes.

ARCHITECTURE NOTES:

1. **Command Separation is Intentional**:
   - `audit` - Main compliance workflow (PDF generation)
   - `reasons` - Separate ECOA adverse action notices (different data format)
   - `recourse` - Separate counterfactual generation (different use case)
   - Each command maps to a distinct regulatory concept and workflow
   - DO NOT merge these - they serve different personas and regulatory requirements

2. **Dual API Design**:
   - CLI + YAML configs for compliance officers (reproducible, auditable)
   - Python `from_model()` API for data scientists (exploratory, no file I/O)
   - Both are necessary for different user workflows

3. **Technical Notes**:
   - Uses Typer function-call defaults (B008 lint rule) - this is the documented Typer pattern
   - Uses clean CLI exception handling with 'from None' to hide Python internals from end users
   - Command names match regulatory terminology (e.g., "reasons" not "explanations")
"""

import logging
import os
import sys
import warnings
from pathlib import Path

import typer
from platformdirs import user_data_dir

from .. import __version__
from .exit_codes import ExitCode

# Configure logging with WARNING as default (clean output for users)
# User can override with --verbose (INFO) or --quiet (ERROR only)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Suppress expected sklearn warnings to keep output clean
# Only show these warnings in verbose mode
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Main CLI app
app = typer.Typer(
    name="glassalpha",
    help="GlassAlpha - AI Compliance Toolkit",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=True,
    epilog="""For more information, visit: https://glassalpha.com""",
)

# Command groups removed - simplified to 3 core commands


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
        typer.secho("Welcome to GlassAlpha", fg=typer.colors.BRIGHT_BLUE, bold=True)
        typer.echo()
        typer.echo("Quick start:")
        typer.echo("  1. Check environment: glassalpha doctor")
        typer.echo("  2. Generate project: glassalpha quickstart")
        typer.echo("  3. Run audit: glassalpha audit")
        typer.echo()
        typer.echo("Tip: Enable fast mode in config (runtime.fast_mode: true) for quicker iteration")
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
        # Quiet mode also disables progress bars
        os.environ["GLASSALPHA_NO_PROGRESS"] = "1"
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)  # INFO shows progress, not DEBUG details
        # Verbose mode disables progress bars (they interfere with log output)
        os.environ["GLASSALPHA_NO_PROGRESS"] = "1"
        # Re-enable warnings in verbose mode for debugging
        warnings.filterwarnings("default")

    # First-run detection - show helpful tip once
    _show_first_run_tip()


# Import and register core commands
from .commands import (
    audit,
    config_cheat_cmd,
    config_list_cmd,
    config_template_cmd,
    docs,
    doctor,
    export_evidence_pack,
    list_components_cmd,
    reasons,
    recourse,
    validate,
    verify_evidence_pack,
)
from .commands.setup_env import setup_env
from .quickstart import quickstart

# Register commands
app.command()(audit)
app.command(name="config-cheat")(config_cheat_cmd)
app.command(name="config-list")(config_list_cmd)
app.command(name="config-template")(config_template_cmd)
app.command()(doctor)
app.command()(docs)
app.command(name="export-evidence-pack")(export_evidence_pack)
app.command()(quickstart)
app.command()(reasons)
app.command()(recourse)
app.command(name="setup-env")(setup_env)
app.command()(validate)
app.command(name="verify-evidence-pack")(verify_evidence_pack)
app.command(name="list")(list_components_cmd)


if __name__ == "__main__":  # pragma: no cover
    app()
