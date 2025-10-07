"""QuickStart project generator for GlassAlpha.

This module provides the `glassalpha quickstart` command which scaffolds a complete
audit project with config, directory structure, and example scripts.
"""

from pathlib import Path

import typer

from .exit_codes import ExitCode

# Create Typer app for quickstart command
app = typer.Typer(help="Generate template audit project")


def _get_template_content(template_name: str) -> str | None:
    """Get the full content of a template file.

    Args:
        template_name: Name of template (quickstart, german_credit, adult_income)

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
def quickstart(
    output: Path = typer.Option(
        Path("my-audit-project"),
        "--output",
        "-o",
        help="Output directory for project scaffold",
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Dataset type (german_credit, adult_income)",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model type (xgboost, lightgbm, logistic_regression)",
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Use interactive mode to customize project",
    ),
):
    """Generate a complete audit project with config and directory structure.

    Creates a project directory with:
    - Configuration file (audit_config.yaml)
    - Directory structure (data/, models/, reports/, configs/)
    - Example run script (run_audit.py)
    - README with next steps

    Designed for <60 seconds from install to first audit.

    Examples:
        # Interactive wizard (default)
        glassalpha quickstart

        # German Credit with XGBoost (non-interactive)
        glassalpha quickstart --dataset german_credit --model xgboost --no-interactive

        # Custom project name
        glassalpha quickstart --output credit-audit-2024

    """
    try:
        typer.echo("GlassAlpha QuickStart Generator")
        typer.echo("=" * 40)
        typer.echo()

        # Interactive mode: ask questions
        if interactive:
            if not dataset:
                dataset = _ask_dataset()
            if not model:
                model = _ask_model()

        # Set defaults if not provided
        if not dataset:
            dataset = "german_credit"
        if not model:
            model = "xgboost"

        # Validate inputs
        if dataset not in ("german_credit", "adult_income"):
            typer.secho(
                f"Error: Unknown dataset '{dataset}'. Must be 'german_credit' or 'adult_income'.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        if model not in ("xgboost", "lightgbm", "logistic_regression"):
            typer.secho(
                f"Error: Unknown model '{model}'. Must be 'xgboost', 'lightgbm', or 'logistic_regression'.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(ExitCode.USER_ERROR)

        # Check if output directory already exists
        if output.exists() and any(output.iterdir()):
            overwrite = typer.confirm(
                f"Directory '{output}' already exists and is not empty. Overwrite?",
                default=False,
            )
            if not overwrite:
                typer.echo("Cancelled.")
                raise typer.Exit(ExitCode.SUCCESS)

        # Create project structure
        typer.echo(f"Creating project structure in: {output}")
        typer.echo()

        _create_project_structure(output, dataset, model)

        # Success message
        typer.echo()
        typer.secho("✓ Project created successfully!", fg=typer.colors.GREEN)
        typer.echo()
        typer.echo(f"Project: {output.absolute()}")
        typer.echo()
        typer.echo("Next steps:")
        typer.echo(f"  1. cd {output}")
        typer.echo("  2. python run_audit.py")
        typer.echo()
        typer.echo("Your audit will be generated in: reports/audit_report.pdf")

    except KeyboardInterrupt:
        typer.echo()
        typer.echo("Cancelled.")
        raise typer.Exit(ExitCode.SUCCESS)
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR)


def _ask_dataset() -> str:
    """Ask user about dataset choice.

    Returns:
        Dataset name (german_credit, adult_income)

    """
    typer.echo("Which dataset would you like to start with?")
    typer.echo()
    typer.echo("  1. German Credit (credit risk, 1000 samples)")
    typer.echo("  2. Adult Income (income prediction, 48842 samples)")
    typer.echo()

    choice = typer.prompt("Select dataset (1-2)", type=int, default=1)

    dataset_map = {
        1: "german_credit",
        2: "adult_income",
    }

    dataset = dataset_map.get(choice, "german_credit")

    typer.echo()
    typer.secho(f"Selected: {dataset}", fg=typer.colors.GREEN)
    typer.echo()

    return dataset


def _ask_model() -> str:
    """Ask user about model choice.

    Returns:
        Model name (xgboost, lightgbm, logistic_regression)

    """
    typer.echo("Which model type would you like to use?")
    typer.echo()
    typer.echo("  1. XGBoost (recommended, TreeSHAP explainability)")
    typer.echo("  2. LightGBM (fast, TreeSHAP explainability)")
    typer.echo("  3. Logistic Regression (simple, coefficient explainability)")
    typer.echo()

    choice = typer.prompt("Select model (1-3)", type=int, default=1)

    model_map = {
        1: "xgboost",
        2: "lightgbm",
        3: "logistic_regression",
    }

    model = model_map.get(choice, "xgboost")

    typer.echo()
    typer.secho(f"Selected: {model}", fg=typer.colors.GREEN)
    typer.echo()

    return model


def _create_project_structure(output: Path, dataset: str, model: str) -> None:
    """Create the complete project structure.

    Args:
        output: Project root directory
        dataset: Dataset name
        model: Model type

    """
    # Create directories
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    (output / "reports").mkdir(exist_ok=True)
    (output / "configs").mkdir(exist_ok=True)

    # Create audit config
    _create_audit_config(output / "audit_config.yaml", dataset, model)

    # Create run script
    _create_run_script(output / "run_audit.py", dataset, model)

    # Create README
    _create_readme(output / "README.md", dataset, model)

    # Create .gitignore
    _create_gitignore(output / ".gitignore")

    typer.echo("  ✓ Created directory structure")
    typer.echo("  ✓ Created audit_config.yaml")
    typer.echo("  ✓ Created run_audit.py")
    typer.echo("  ✓ Created README.md")
    typer.echo("  ✓ Created .gitignore")


def _create_audit_config(path: Path, dataset: str, model: str) -> None:
    """Create audit configuration file.

    Args:
        path: Path to config file
        dataset: Dataset name
        model: Model type

    """
    # Map dataset to target column
    target_map = {
        "german_credit": "credit_risk",
        "adult_income": "income",
    }

    # Map dataset to protected attributes
    protected_map = {
        "german_credit": ["gender_male", "age"],
        "adult_income": ["gender", "race", "age"],
    }

    target = target_map.get(dataset, "target")
    protected = protected_map.get(dataset, ["gender", "age"])

    config_content = f"""# GlassAlpha Audit Configuration
# Generated by: glassalpha quickstart

# Data configuration
data:
  dataset: {dataset}  # Built-in dataset
  target_column: {target}
  protected_attributes:
{chr(10).join(f"    - {attr}" for attr in protected)}

# Model configuration
model:
  type: {model}
  params:
    random_state: 42
    {_get_model_params(model)}

# Metrics configuration
metrics:
  fairness:
    enabled: true
    threshold: 0.5  # Decision threshold
  calibration:
    enabled: true
    n_bins: 10
  stability:
    enabled: true
    epsilon_values: [0.01, 0.05, 0.1]

# Reproducibility
seed: 42
strict_mode: false  # Set to true for regulatory audits

# Output configuration
output:
  format: pdf
  include_manifest: true
"""

    path.write_text(config_content)


def _get_model_params(model: str) -> str:
    """Get default model parameters for config.

    Args:
        model: Model type

    Returns:
        YAML-formatted parameter string

    """
    params_map = {
        "xgboost": "n_estimators: 100\n    max_depth: 6",
        "lightgbm": "n_estimators: 100\n    max_depth: 6",
        "logistic_regression": "max_iter: 1000\n    penalty: l2",
    }

    return params_map.get(model, "")


def _create_run_script(path: Path, dataset: str, model: str) -> None:
    """Create example run script.

    Args:
        path: Path to script file
        dataset: Dataset name
        model: Model type

    """
    script_content = f'''#!/usr/bin/env python3
"""Example audit script for {dataset} dataset with {model} model.

This script demonstrates the simplest way to run a GlassAlpha audit.
"""

from pathlib import Path

from glassalpha.cli.commands import audit

if __name__ == "__main__":
    print("=" * 60)
    print("GlassAlpha Audit - {dataset.replace("_", " ").title()}")
    print("=" * 60)
    print()

    # Run audit
    config_path = Path("audit_config.yaml")
    output_path = Path("reports/audit_report.pdf")

    print(f"Configuration: {{config_path}}")
    print(f"Output: {{output_path}}")
    print()
    print("Running audit...")
    print()

    # Call audit command programmatically
    audit(
        config=config_path,
        output=output_path,
        check_shift=[],  # Add --check-shift flags here for shift testing
        fail_on_degradation=None,  # Set threshold for CI/CD gates
    )

    print()
    print("✓ Audit complete!")
    print(f"  Report: {{output_path}}")
    print()
    print("Next steps:")
    print("  1. Open reports/audit_report.pdf to view the audit")
    print("  2. Modify audit_config.yaml to customize metrics")
    print("  3. Try shift testing: --check-shift gender:+0.1")
'''

    path.write_text(script_content)
    path.chmod(0o755)  # Make executable


def _create_readme(path: Path, dataset: str, model: str) -> None:
    """Create project README.

    Args:
        path: Path to README file
        dataset: Dataset name
        model: Model type

    """
    readme_content = f"""# {dataset.replace("_", " ").title()} Audit Project

This project was generated by `glassalpha quickstart` to demonstrate compliance auditing for the {dataset} dataset using a {model} model.

## Quick Start

Run your first audit in 3 commands:

```bash
cd {path.parent.name}
python run_audit.py
open reports/audit_report.pdf
```

## Project Structure

```
{path.parent.name}/
├── audit_config.yaml      # Audit configuration
├── run_audit.py           # Example run script
├── data/                  # (Optional) Custom datasets
├── models/                # (Optional) Pre-trained models
├── reports/               # Generated audit reports
└── configs/               # (Optional) Policy configurations
```

## Configuration

Edit `audit_config.yaml` to customize:

- **Protected attributes**: Which features to analyze for fairness
- **Model parameters**: Hyperparameters for training
- **Metrics**: Which fairness/calibration metrics to compute
- **Seed**: For reproducibility

## Advanced Usage

### Shift Testing (Demographic Robustness)

Test model stability under population shifts:

```bash
glassalpha audit --config audit_config.yaml \\
  --check-shift gender:+0.1 \\
  --check-shift age:-0.05 \\
  --fail-on-degradation 0.05
```

### CI/CD Integration

Add to `.github/workflows/model-validation.yml`:

```yaml
- name: Run compliance audit
  run: |
    glassalpha audit --config audit_config.yaml \\
      --check-shift gender:+0.1 \\
      --fail-on-degradation 0.05
```

### Custom Data

Replace built-in dataset with your own:

1. Add CSV to `data/` directory
2. Update `audit_config.yaml`:
   ```yaml
   data:
     path: data/my_data.csv
     target_column: outcome
     protected_attributes:
       - gender
       - race
   ```

## Documentation

- [User Guides](https://glassalpha.com/guides/)
- [API Reference](https://glassalpha.com/reference/)
- [Shift Testing Guide](https://glassalpha.com/guides/shift-testing/)
- [SR 11-7 Compliance](https://glassalpha.com/compliance/sr-11-7-mapping/)

## Learn More

- **German Credit Tutorial**: [glassalpha.com/examples/german-credit-audit/](https://glassalpha.com/examples/german-credit-audit/)
- **Fairness Metrics**: [glassalpha.com/guides/fairness/](https://glassalpha.com/guides/fairness/)
- **Recourse Generation**: [glassalpha.com/guides/recourse/](https://glassalpha.com/guides/recourse/)

---

Generated by `glassalpha quickstart`
"""

    path.write_text(readme_content)


def _create_gitignore(path: Path) -> None:
    """Create .gitignore file.

    Args:
        path: Path to .gitignore file

    """
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# GlassAlpha
reports/*.pdf
reports/*.html
reports/*.json
*.manifest.json
*.shift_analysis.json

# Data (uncomment if you don't want to commit data)
# data/*.csv
# data/*.parquet

# Models
models/*.pkl
models/*.joblib
models/*.bin

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
"""

    path.write_text(gitignore_content)


# Add command to main app
def register_quickstart_command(main_app: typer.Typer) -> None:
    """Register the quickstart command with the main CLI app.

    Args:
        main_app: Main Typer application instance

    """
    main_app.add_typer(app, name="quickstart")
