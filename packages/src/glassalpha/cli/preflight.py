"""Preflight checks for CLI commands with dependency validation and fallbacks.

This module handles dependency checking, model availability validation,
and provides graceful fallbacks when optional components are missing.
"""

import logging
from typing import Any

import typer

from ..core.registry import ModelRegistry
from ..explain.registry import ExplainerRegistry
from ..metrics.registry import MetricRegistry
from .exit_codes import ExitCode

logger = logging.getLogger(__name__)


def preflight_check_model(config: Any, allow_fallback: bool = True) -> tuple[Any, str | None]:
    """Validate model availability and provide fallbacks.

    Args:
        config: Audit configuration object
        allow_fallback: If False, fail instead of falling back to alternative models

    Returns:
        Tuple of (modified_config, requested_model_or_none)
        requested_model_or_none is the originally requested model if fallback occurred, None otherwise

    Raises:
        typer.Exit: If no suitable model is available and fallbacks disabled

    """
    if not hasattr(config, "model") or not config.model:
        # Use default model if no model specified
        config.model = _create_default_model_config()
        return config, None

    model_type = config.model.type

    # Check if requested model is available
    available_models = ModelRegistry.available_plugins()

    if not available_models.get(model_type, False):
        # Model not available - check for fallbacks
        if allow_fallback:
            fallback_model = _find_fallback_model(model_type, available_models)
            if fallback_model:
                # Enhanced fallback warning with clear explanation
                typer.echo()
                typer.secho("⚠ Model Fallback Activated", fg=typer.colors.YELLOW, bold=True)
                typer.echo(f"  Requested: {model_type}")
                typer.echo(f"  Using:     {fallback_model}")
                typer.echo()
                typer.echo(f"  Why: '{model_type}' is not installed in your environment")
                typer.echo()

                # Installation instructions
                install_hint = ModelRegistry.get_install_hint(model_type)
                if install_hint:
                    typer.echo(f"  To use {model_type}:")
                    typer.echo(f"    {install_hint}")
                else:
                    typer.echo(f"  To use {model_type}:")
                    typer.echo("    pip install 'glassalpha[explain]'")

                typer.echo()
                typer.secho("  Use --no-fallback to fail instead of falling back", fg=typer.colors.CYAN)
                typer.echo()

                config.model.type = fallback_model
                return config, model_type
            # No fallback available
            _show_installation_error(model_type)
            raise typer.Exit(ExitCode.VALIDATION_ERROR)
        # Fallbacks disabled
        _show_installation_error(model_type)
        raise typer.Exit(ExitCode.VALIDATION_ERROR)

    return config, None


def _create_default_model_config():
    """Create default model configuration for logistic regression."""
    from types import SimpleNamespace

    model_config = SimpleNamespace()
    model_config.type = "logistic_regression"
    model_config.params = {"random_state": 42}
    return model_config


def _find_fallback_model(requested_model: str, available_models: dict[str, bool]) -> str | None:
    """Find a suitable fallback model when requested model is unavailable.

    Args:
        requested_model: The originally requested model type
        available_models: Dictionary of available models

    Returns:
        Fallback model name or None if no suitable fallback

    """
    # Priority order for fallbacks - always prefer more capable models
    fallback_priority = [
        "logistic_regression",  # Most capable baseline
        "passthrough",  # Minimal but always works
    ]

    # Remove requested model from fallback list if it's in there
    fallback_priority = [m for m in fallback_priority if m != requested_model]

    # Return first available fallback
    for model in fallback_priority:
        if available_models.get(model, False):
            return model

    return None


def _show_installation_error(model_type: str):
    """Show helpful installation error message.

    Args:
        model_type: The model type that failed

    """
    hint = ModelRegistry.get_install_hint(model_type)
    if hint:
        typer.secho(
            f"Model '{model_type}' requires an optional dependency. {hint}",
            fg=typer.colors.RED,
            err=True,
        )
    else:
        typer.secho(
            f"Model '{model_type}' is not available. Please check your installation.",
            fg=typer.colors.RED,
            err=True,
        )


def preflight_check_dependencies():
    """Check core dependencies and show warnings for missing optional ones."""
    # Banner is already printed by CLI command - don't duplicate

    # Check model availability
    available_models = ModelRegistry.available_plugins()

    # Check for basic models
    has_logistic = available_models.get("logistic_regression", False)
    has_passthrough = available_models.get("passthrough", False)

    if not (has_logistic or has_passthrough):
        typer.secho(
            "Warning: No models available. Install scikit-learn for basic functionality.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        return False

    # Check explainer compatibility - use a model that's actually available
    test_model = "logistic_regression" if has_logistic else "passthrough"
    try:
        # Try to find a compatible explainer for a basic model type
        # Create a minimal config for the test
        minimal_config = {"explainers": {"strategy": "first_compatible"}}
        ExplainerRegistry.find_compatible(test_model, minimal_config)
        typer.secho("✓ Explainers available and compatible", fg=typer.colors.GREEN)
    except RuntimeError as e:
        # Get install hint for the first available explainer that needs installation
        for name in ExplainerRegistry.names():
            hint = ExplainerRegistry.get_install_hint(name)
            if hint:
                typer.secho(
                    f"Note: {e!s}. {hint}",
                    fg=typer.colors.BLUE,
                    err=True,
                )
                break
        else:
            typer.secho(
                f"Warning: {e!s}",
                fg=typer.colors.YELLOW,
                err=True,
            )

    # Check metric availability
    available_metrics = MetricRegistry.available_plugins()

    # At least one metric should be available for meaningful audits
    metric_available = any(available_metrics.values())
    if not metric_available:
        typer.secho(
            "Warning: No metrics available. Core functionality may be limited.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    return True


def show_available_models():
    """Display available models and their installation status."""
    available_models = ModelRegistry.available_plugins()

    typer.echo("Available models:")
    for model, available in available_models.items():
        status = "✓" if available else "✗"
        hint = ModelRegistry.get_install_hint(model)
        install_info = f" (install with: {hint})" if hint and not available else ""

        typer.echo(f"  {status} {model}{install_info}")

    if not any(available_models.values()):
        typer.secho(
            "No models available. Install models with: pip install 'glassalpha[tabular]'",
            fg=typer.colors.YELLOW,
        )
