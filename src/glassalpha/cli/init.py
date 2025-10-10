"""Interactive configuration wizard for GlassAlpha.

This module provides the `glassalpha init` command which guides users through
creating a configuration file with an interactive questionnaire.
"""

from pathlib import Path

import typer

from .exit_codes import ExitCode

# Create Typer app for init command
app = typer.Typer(help="Initialize new audit configuration")


def _get_template_path(template_name: str) -> str | None:
    """Get the full content of a template file.

    Args:
        template_name: Name of template (quickstart, production, development, testing)

    Returns:
        Template content as string, or None if not found

    """
    try:
        # Try to read from templates directory
        templates_pkg = "glassalpha.templates"
        template_file = f"{template_name}.yaml"

        # Use importlib.resources for Python 3.9+ compatibility
        try:
            # Python 3.9+
            from importlib.resources import files

            template_path = files(templates_pkg).joinpath(template_file)
            return template_path.read_text()
        except (ImportError, AttributeError):
            # Python 3.8 fallback
            import pkg_resources

            return pkg_resources.resource_string(templates_pkg, template_file).decode()

    except Exception:
        return None


@app.command()
def init(
    output: Path = typer.Option(
        Path("audit_config.yaml"),
        "--output",
        "-o",
        help="Output path for generated configuration file",
    ),
    template: str | None = typer.Option(
        None,
        "--template",
        "-t",
        help="Use a specific template (quickstart, production, development, testing)",
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Use interactive mode to customize configuration",
    ),
):
    """Initialize a new audit configuration.

    Creates a new configuration file either by copying a template or through
    an interactive wizard that asks questions about your use case.

    Examples:
        # Interactive wizard (default)
        glassalpha init

        # Use specific template
        glassalpha init --template quickstart

        # Non-interactive with custom output
        glassalpha init --template production --output prod.yaml --no-interactive

    """
    try:
        typer.echo("GlassAlpha Configuration Wizard")
        typer.echo("=" * 40)
        typer.echo()

        # If template specified and non-interactive, just copy template
        if template and not interactive:
            _copy_template(template, output)
            return

        # Interactive mode
        if interactive and not template:
            typer.echo("This wizard will help you create an audit configuration.")
            typer.echo("Press Ctrl+C at any time to cancel.")
            typer.echo()

            # Ask about use case
            template = _ask_use_case()

        # Get template content
        if not template:
            template = "quickstart"  # Default

        template_content = _get_template_path(template)

        if not template_content:
            typer.secho(
                f"Error: Template '{template}' not found",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        # Customize if interactive
        if interactive:
            template_content = _customize_template(template, template_content)

        # Write output file
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(template_content)

        # Success message
        typer.echo()
        typer.secho(f"✓ Created configuration: {output}", fg=typer.colors.GREEN)
        typer.echo()
        typer.echo("Next steps:")
        typer.echo(f"  1. Review and customize: {output}")
        typer.echo(f"  2. Run audit: glassalpha audit --config {output} --output report.html")

        if template == "production":
            typer.echo()
            typer.secho("⚠ Production template requires additional setup:", fg=typer.colors.YELLOW)
            typer.echo("  • Update data paths in the config")
            typer.echo("  • Generate preprocessing hashes: glassalpha prep hash <artifact>")
            typer.echo("  • List all features explicitly")

    except KeyboardInterrupt:
        typer.echo()
        typer.echo("Cancelled.")
        raise typer.Exit(ExitCode.SUCCESS)
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR)


def _copy_template(template_name: str, output: Path) -> None:
    """Copy a template file to output location.

    Args:
        template_name: Name of template to copy
        output: Destination path

    """
    template_content = _get_template_path(template_name)

    if not template_content:
        typer.secho(
            f"Error: Template '{template_name}' not found",
            fg=typer.colors.RED,
            err=True,
        )
        typer.echo()
        typer.echo("Available templates: quickstart, production, development, testing")
        raise typer.Exit(ExitCode.USER_ERROR)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template_content)

    typer.secho(f"✓ Created configuration: {output}", fg=typer.colors.GREEN)
    typer.echo(f"  Based on template: {template_name}")


def _ask_use_case() -> str:
    """Ask user about their use case and return appropriate template.

    Returns:
        Template name (quickstart, production, development, testing)

    """
    typer.echo("What is your use case?")
    typer.echo()
    typer.echo("  1. Quick testing / First time using GlassAlpha")
    typer.echo("  2. Model development / Iteration")
    typer.echo("  3. CI/CD integration / Automated testing")
    typer.echo("  4. Production deployment / Regulatory submission")
    typer.echo()

    try:
        choice = typer.prompt("Select option (1-4)", type=int, default=1, show_default=True)
    except (EOFError, RuntimeError):
        # Handle non-interactive environments or EOF
        typer.echo("Using default: 1")
        choice = 1

    template_map = {
        1: "quickstart",
        2: "development",
        3: "testing",
        4: "production",
    }

    template = template_map.get(choice, "quickstart")

    typer.echo()
    typer.secho(f"Selected: {template}", fg=typer.colors.GREEN)
    typer.echo()

    return template


def _customize_template(template_name: str, template_content: str) -> str:
    """Allow user to customize template through prompts.

    Args:
        template_name: Name of the template being customized
        template_content: Original template content

    Returns:
        Customized template content

    """
    typer.echo("Configuration options (press Enter to keep defaults):")
    typer.echo()

    # Ask about customization
    try:
        customize = typer.confirm("Would you like to customize the configuration?", default=False)
    except (EOFError, RuntimeError):
        # Handle non-interactive environments or EOF
        typer.echo("Using default: No")
        customize = False

    if not customize:
        return template_content

    typer.echo()

    # Common customizations based on template
    if template_name in ("development", "production"):
        # Ask about data path
        try:
            use_custom_data = typer.confirm("Use custom dataset (vs. built-in)?", default=False)
        except (EOFError, RuntimeError):
            use_custom_data = False

        if use_custom_data:
            try:
                data_path = typer.prompt("Data file path", default="data/my_data.csv")
                target_col = typer.prompt("Target column name", default="target")
            except (EOFError, RuntimeError):
                data_path = "data/my_data.csv"
                target_col = "target"

            # Replace dataset with custom path
            template_content = template_content.replace(
                "dataset: german_credit",
                f"# dataset: german_credit  # Switched to custom data\n  path: {data_path}",
            )
            template_content = template_content.replace(
                "target_column: credit_risk",
                f"target_column: {target_col}",
            )

    # Ask about model type
    typer.echo()
    typer.echo("Model type:")
    typer.echo("  1. Logistic Regression (fast)")
    typer.echo("  2. XGBoost (recommended)")
    typer.echo("  3. LightGBM (fast + accurate)")

    try:
        model_choice = typer.prompt("Select model (1-3)", type=int, default=2)
    except (EOFError, RuntimeError):
        typer.echo("Using default: 2")
        model_choice = 2

    model_map = {
        1: "logistic_regression",
        2: "xgboost",
        3: "lightgbm",
    }

    selected_model = model_map.get(model_choice, "xgboost")

    # Update model type in template
    if "type: logistic_regression" in template_content:
        template_content = template_content.replace(
            "type: logistic_regression",
            f"type: {selected_model}",
        )
    elif "type: xgboost" in template_content:
        template_content = template_content.replace(
            "type: xgboost",
            f"type: {selected_model}",
        )

    typer.echo()
    typer.secho("✓ Template customized", fg=typer.colors.GREEN)

    return template_content


# Add command to main app
def register_init_command(main_app: typer.Typer) -> None:
    """Register the init command with the main CLI app.

    Args:
        main_app: Main Typer application instance

    """
    main_app.add_typer(app, name="init")
