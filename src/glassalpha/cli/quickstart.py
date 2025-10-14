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
):
    """Generate a complete audit project with config and directory structure.

    Creates a project directory with:
    - Configuration file (audit.yaml)
    - Directory structure (data/, models/, reports/, configs/)
    - Example run script (run_audit.py)
    - README with next steps

    Designed for <60 seconds from install to first audit.

    Defaults: German Credit + LogisticRegression (works with base install)

    Examples:
        # Zero-config quickstart (recommended for first-time users)
        glassalpha quickstart

        # German Credit with XGBoost (requires: pip install 'glassalpha[explain]')
        glassalpha quickstart --dataset german_credit --model xgboost

        # Adult Income with LightGBM
        glassalpha quickstart --dataset adult_income --model lightgbm

        # Custom project name
        glassalpha quickstart --output credit-audit-2024

    """
    try:
        typer.echo("GlassAlpha QuickStart Generator")
        typer.echo("=" * 40)
        typer.echo()

        # Set smart defaults if not provided
        if not dataset:
            dataset = "german_credit"
        if not model:
            # Check if advanced models are available, default to logistic_regression for base install
            import importlib.util

            has_explain = importlib.util.find_spec("shap") is not None and (
                importlib.util.find_spec("xgboost") is not None or importlib.util.find_spec("lightgbm") is not None
            )
            model = "xgboost" if has_explain else "logistic_regression"

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

        # Handle --output . specially (create in current directory)
        if output == Path("."):
            output = Path.cwd()
            typer.echo(f"Creating project in current directory: {output}")
            typer.echo()

        # Check if output directory already exists
        if output.exists() and any(output.iterdir()):
            try:
                overwrite = typer.confirm(
                    f"Directory '{output}' already exists and is not empty. Overwrite?",
                    default=False,
                )
            except (EOFError, RuntimeError):
                # Handle non-interactive environments - don't overwrite by default
                typer.secho(f"\n❌ Directory '{output}' already exists and is not empty", fg=typer.colors.RED)
                typer.echo()
                typer.secho("💡 Solutions:", fg=typer.colors.CYAN, bold=True)
                typer.echo(f"   • Use a different name: glassalpha quickstart --output {output}-2")
                typer.echo(f"   • Delete existing: rm -rf {output}")
                typer.echo("   • Create in subdirectory: glassalpha quickstart --output my-custom-name")
                typer.echo()
                overwrite = False

            if not overwrite:
                typer.echo("Cancelled.")
                raise typer.Exit(ExitCode.SUCCESS)

        # Create project structure
        typer.echo(f"Creating project structure in: {output}")
        typer.echo()

        _create_project_structure(output, dataset, model)

        # Success message with clear next steps
        typer.echo()
        typer.secho("✓ Project created successfully!", fg=typer.colors.GREEN, bold=True)
        typer.echo()
        typer.echo(f"Project: {output}")
        typer.echo()
        typer.secho("Next steps:", fg=typer.colors.CYAN, bold=True)
        typer.echo(f"  1. cd {output}")
        typer.echo("  2. python run_audit.py")
        typer.echo()
        typer.echo("Your audit will be generated in: reports/audit_report.html")
        typer.echo()

    except KeyboardInterrupt:
        typer.echo()
        typer.echo("Cancelled.")
        raise typer.Exit(ExitCode.SUCCESS)
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USER_ERROR)


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

    # Create audit config (audit.yaml for auto-detection)
    _create_audit_config(output / "audit.yaml", dataset, model)

    # Create run script
    _create_run_script(output / "run_audit.py", dataset, model)

    # Create README
    _create_readme(output / "README.md", dataset, model)

    # Create .gitignore
    _create_gitignore(output / ".gitignore")

    typer.echo("  ✓ Created directory structure")
    typer.echo("  ✓ Created audit.yaml")
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
    # Get schema from dataset modules (ensures correct column names)
    if dataset == "german_credit":
        from ..datasets.german_credit import get_german_credit_schema

        schema = get_german_credit_schema()
        target = schema.target
        protected = list(schema.sensitive_features)[:2] if schema.sensitive_features else ["gender"]
    elif dataset == "adult_income":
        from ..datasets.adult_income import get_adult_income_schema

        schema = get_adult_income_schema()
        target = schema["target_column"]
        protected = schema["protected_attributes"][:2]
    else:
        # Fallback for unknown datasets
        target = "target"
        protected = ["gender"]

    config_content = f"""# GlassAlpha Audit Configuration
# Generated by: glassalpha quickstart

audit_profile: tabular_compliance

# Data configuration
data:
  dataset: {dataset}
  fetch: if_missing
  offline: false
  target_column: {target}
  protected_attributes:
{chr(10).join(f"    - {attr}" for attr in protected)}

# Model configuration
model:
  type: {model}
  save_path: models/{dataset}_{model}_model.pkl  # Auto-save for reasons/recourse
  params:
    random_state: 42
    {_get_model_params(model)}

# Explainers configuration
explainers:
  strategy: first_compatible
  {_get_explainer_config(model)}

# Metrics configuration
metrics:
  performance:
    metrics:
      - accuracy
      - precision
      - recall
      - f1
  fairness:
    metrics:
      - demographic_parity
      - equal_opportunity
      - equalized_odds

# Reproducibility configuration
reproducibility:
  random_seed: 42

# Runtime configuration (performance tuning)
runtime:
  fast_mode: true  # Fast audits (2-3s vs 8-10s). Set false for full bootstrap CIs.

# Report configuration
report:
  template: standard_audit
  output_format: html

# Manifest configuration
manifest:
  enabled: true
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
        "lightgbm": "n_estimators: 100\n    max_depth: 6\n    learning_rate: 0.1",
        "logistic_regression": "max_iter: 5000  # Sufficient for most datasets, prevents convergence warnings",
    }

    return params_map.get(model, "")


def _get_explainer_config(model: str) -> str:
    """Get explainer configuration appropriate for model type.

    Args:
        model: Model type

    Returns:
        YAML-formatted explainer configuration

    """
    if model == "logistic_regression":
        # Use coefficient-based explainers (no extra dependencies needed)
        return """priority:
    - coefficients  # Fast for linear models
    - permutation   # Universal fallback
  config:
    coefficients:
      normalize: true
    permutation:
      n_samples: 100"""
    # Tree models use SHAP explainers
    return """priority:
    - treeshap       # Best for tree models (requires: pip install 'glassalpha[explain]')
    - kernelshap     # SHAP fallback
  config:
    treeshap:
      max_samples: 1000
    kernelshap:
      n_samples: 500"""


def _create_run_script(path: Path, dataset: str, model: str) -> None:
    """Create example run script.

    Args:
        path: Path to script file
        dataset: Dataset name
        model: Model type

    """
    script_content = f'''#!/usr/bin/env python3
"""Example audit script for {dataset} dataset with {model} model.

This script demonstrates the simplest way to run a GlassAlpha audit programmatically.

Requirements:
    - GlassAlpha must be installed: pip install 'glassalpha[all]'
    - For development from source: pip install -e ".[all]" from project root
"""

import sys
from pathlib import Path

# Check if glassalpha is installed
try:
    from glassalpha.api import run_audit
except ImportError:
    print("Error: GlassAlpha not installed.")
    print()
    print("Install with:")
    print("  pip install 'glassalpha[all]'")
    print()
    sys.exit(1)

if __name__ == "__main__":
    print("=" * 60)
    print("GlassAlpha Audit - {dataset.replace("_", " ").title()}")
    print("=" * 60)
    print()

    # Configuration paths
    config_path = Path("audit.yaml")
    output_path = Path("reports/audit_report.html")

    print(f"Configuration: {{config_path}}")
    print(f"Output: {{output_path}}")
    print()
    print("Running audit...")
    print()

    # Run audit using programmatic API
    try:
        report_path = run_audit(
            config_path=config_path,
            output_path=output_path,
        )

        print()
        print("✓ Audit complete!")
        print(f"  Report: {{report_path}}")
        print()
        print("Next steps:")
        print("  1. Open reports/audit_report.html to view the audit")
        print("  2. Modify audit.yaml to customize metrics")
        print("  3. Try CLI for shift testing: glassalpha audit --check-shift gender:+0.1")
        print()

    except Exception as e:
        print(f"✗ Audit failed: {{e}}")
        raise
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
open reports/audit_report.html
```

**Note**: Make sure GlassAlpha is installed first:
```bash
pip install 'glassalpha[all]'
```

## Project Structure

```
{path.parent.name}/
├── audit.yaml             # Audit configuration
├── run_audit.py           # Example run script
├── data/                  # (Optional) Custom datasets
├── models/                # (Optional) Pre-trained models
├── reports/               # Generated audit reports
└── configs/               # (Optional) Policy configurations
```

## Configuration

Edit `audit.yaml` to customize:

- **Protected attributes**: Which features to analyze for fairness
- **Model parameters**: Hyperparameters for training
- **Metrics**: Which fairness/calibration metrics to compute
- **Seed**: For reproducibility

## Advanced Usage

### Shift Testing (Demographic Robustness)

Test model stability under population shifts:

```bash
glassalpha audit \\
  --check-shift gender:+0.1 \\
  --check-shift age:-0.05 \\
  --fail-on-degradation 0.05
```

### CI/CD Integration

Add to `.github/workflows/model-validation.yml`:

```yaml
- name: Run compliance audit
  run: |
    glassalpha audit \\
      --check-shift gender:+0.1 \\
      --fail-on-degradation 0.05
```

### Custom Data

Replace built-in dataset with your own:

1. Add CSV to `data/` directory
2. Update `audit.yaml`:
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
