"""Shared utilities for CLI commands.

Common helper functions used across multiple CLI commands.
"""

import logging
import os
from pathlib import Path

import typer

from ..exit_codes import ExitCode

logger = logging.getLogger(__name__)


def _check_and_warn_determinism() -> None:
    """Check if determinism environment variables are set and warn if missing.

    This provides immediate feedback during audit generation, not just in doctor command.
    Only shows warning if variables are actually missing (suppresses noise after setup).
    """
    required_vars = {
        "TZ": "UTC",
        "MPLBACKEND": "Agg",
        "PYTHONHASHSEED": "0",
    }

    missing = [var for var in required_vars if not os.environ.get(var)]

    # Only show warning if variables are missing
    if missing:
        typer.echo()
        typer.secho("⚠️  One-time setup required for reproducible audits", fg=typer.colors.YELLOW, bold=True)
        typer.echo()
        typer.secho("   Run once to enable byte-identical reports:", fg=typer.colors.CYAN)
        typer.echo('     eval "$(glassalpha setup-env)"')
        typer.echo()
        typer.echo("   Or save to file for permanent setup:")
        typer.echo("     glassalpha setup-env >> ~/.bashrc  # or ~/.zshrc")
        typer.echo()
        typer.secho(f"   Missing variables: {', '.join(missing)}", fg=typer.colors.BRIGHT_BLACK)
        typer.echo()


def is_ci_environment() -> bool:
    """Check if running in CI environment.

    Returns:
        True if CI environment variables are detected, False otherwise.

    CI indicators:
        - CI=true (GitHub Actions, Travis CI, etc.)
        - GITHUB_ACTIONS=true (GitHub Actions)
        - GITLAB_CI=true (GitLab CI)
        - JENKINS_URL (Jenkins)
        - BUILDKITE (Buildkite)

    """
    ci_indicators = [
        "CI",  # Generic CI indicator
        "GITHUB_ACTIONS",  # GitHub Actions
        "GITLAB_CI",  # GitLab CI
        "JENKINS_URL",  # Jenkins
        "BUILDKITE",  # Buildkite
        "TRAVIS",  # Travis CI
        "CIRCLECI",  # CircleCI
        "AZURE_HTTP_USER_AGENT",  # Azure DevOps
    ]

    return any(os.environ.get(indicator) for indicator in ci_indicators)


def output_error(message: str) -> None:
    """Output error message to stderr.

    Args:
        message: Error message to display

    """
    typer.echo(message, err=True)


def print_banner(title: str = "GlassAlpha Audit Generation") -> None:
    """Print a standardized banner for CLI commands."""
    typer.echo(title)
    typer.echo("=" * 40)

    # Check determinism environment and warn if not set
    _check_and_warn_determinism()


def ensure_docs_if_pdf(output_path: str) -> None:
    """Check if PDF output is requested and ensure docs dependencies are available.

    Args:
        output_path: Path to the output file

    Raises:
        SystemExit: If PDF is requested but dependencies not available

    """
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
                    "  pip install 'glassalpha[all]'\n"
                    "  # or: pip install weasyprint\n\n"
                    "Note: Use --output audit.html to generate HTML reports instead.",
                )


def bootstrap_components() -> None:
    """Bootstrap basic built-in components for CLI operation.

    This imports the core built-ins that should always be available,
    ensuring basic models and explainers are loaded before commands run.
    """
    logger.debug("Bootstrapping basic built-in components")

    # Import core to ensure PassThroughModel is available
    try:
        from ...models.passthrough import PassThroughModel  # noqa: F401

        logger.debug("PassThroughModel available")
    except ImportError as e:
        logger.error(f"Failed to import PassThroughModel: {e}")
        raise typer.Exit(ExitCode.SYSTEM_ERROR) from e

    # Import sklearn models if available (they're optional) - LAZY IMPORT
    # Note: sklearn is only imported when actually needed, not at CLI startup
    logger.debug("sklearn models will be imported on-demand")

    # Import basic explainers
    try:
        from ...explain import (
            coefficients,  # noqa: F401
            noop,  # noqa: F401
        )

        logger.debug("Basic explainers imported")
    except ImportError as e:
        logger.warning(f"Failed to import basic explainers: {e}")

    # Import basic metrics
    try:
        from ...metrics.performance import classification  # noqa: F401

        logger.debug("Basic metrics imported")
    except ImportError as e:
        logger.warning(f"Failed to import basic metrics: {e}")

    logger.debug("Component bootstrap completed")
