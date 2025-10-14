"""Configuration commands: validate, list, template, cheat.

Commands for working with configuration files.
"""

import logging
from pathlib import Path

import typer

from glassalpha.cli.exit_codes import ExitCode

logger = logging.getLogger(__name__)


def validate(  # pragma: no cover
    config_path: Path | None = typer.Argument(
        None,
        help="Path to configuration file to validate",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file to validate (alternative to positional arg)",
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
    check_data: bool = typer.Option(
        False,
        "--check-data",
        help="Load and validate actual dataset (checks if target column exists, file is readable)",
    ),
):
    """Validate a configuration file.

    This command checks if a configuration file is valid without
    running the audit pipeline.

    Examples:
        # Basic validation (positional argument)
        glassalpha validate config.yaml

        # Basic validation (option syntax)
        glassalpha validate --config audit.yaml

        # Validate for specific profile
        glassalpha validate -c audit.yaml --profile tabular_compliance

        # Check strict mode compliance
        glassalpha validate -c audit.yaml --strict

        # Enforce runtime checks (production-ready)
        glassalpha validate -c audit.yaml --strict-validation

        # Validate data files exist and are readable
        glassalpha validate -c audit.yaml --check-data

    """
    try:
        from glassalpha.config import load_config

        # Use positional arg if provided, otherwise fall back to --config
        config_to_validate = config_path or config

        # Auto-detect config if not provided
        if config_to_validate is None:
            # Look for config files in current directory
            config_candidates = ["glassalpha.yaml", "audit.yaml", "audit_config.yaml", "config.yaml"]
            config_to_validate = None

            for candidate in config_candidates:
                if Path(candidate).exists():
                    config_to_validate = Path(candidate)
                    break

            if config_to_validate is None:
                typer.echo("Error: No configuration file specified", fg=typer.colors.RED, err=True)
                typer.echo("\nUsage:")
                typer.echo("  glassalpha validate config.yaml")
                typer.echo("  glassalpha validate --config config.yaml")
                typer.echo("\nOr create a project with config file:")
                typer.echo("  glassalpha quickstart")
                raise typer.Exit(ExitCode.USER_ERROR.value)

            typer.echo(f"Auto-detected config: {config_to_validate}")

        typer.echo(f"Validating configuration: {config_to_validate}")

        # Load and validate
        audit_config = load_config(
            config_to_validate,
            profile_name=profile,
            strict=strict,
        )

        typer.echo(f"Profile: {audit_config.audit_profile}")
        typer.echo(f"Model type: {audit_config.model.type}")
        strict_mode_desc = "not checked"
        if strict:
            strict_mode_desc = "strict mode valid"
        typer.echo(f"Strict mode: {strict_mode_desc}")

        # Semantic validation
        validation_errors = []
        validation_warnings = []

        # 1. Check if data source exists (if path specified)
        if hasattr(audit_config.data, "path") and audit_config.data.path:
            from pathlib import Path as PathLib

            data_path = PathLib(audit_config.data.path)
            if not data_path.exists():
                validation_errors.append(
                    f"Data file not found: {data_path}\n"
                    f"Fix: Create the file or use a built-in dataset:\n"
                    f"  data:\n"
                    f"    dataset: german_credit  # or adult_income",
                )

        # 2. Check if protected attributes are specified for fairness metrics
        if hasattr(audit_config.metrics, "fairness") and audit_config.metrics.fairness:
            if not hasattr(audit_config.data, "protected_attributes") or not audit_config.data.protected_attributes:
                validation_warnings.append(
                    "Fairness metrics requested but no protected_attributes specified.\n"
                    "Add protected_attributes to data section:\n"
                    "  data:\n"
                    "    protected_attributes:\n"
                    "      - gender\n"
                    "      - age_group",
                )

        # 3. Check model/explainer compatibility
        model_type = audit_config.model.type
        if hasattr(audit_config.explainers, "priority") and audit_config.explainers.priority:
            explainer_priorities = audit_config.explainers.priority
            # Only warn about the FIRST explainer (which will be used if available)
            first_explainer = explainer_priorities[0]
            # Check for common incompatibilities
            if model_type == "logistic_regression" and first_explainer == "treeshap":
                validation_warnings.append(
                    f"Explainer 'treeshap' is not compatible with '{model_type}'.\n"
                    f"Recommend: Use 'coefficients' explainer for linear models:\n"
                    f"  explainers:\n"
                    f"    priority: [coefficients]",
                )
            elif model_type in ["xgboost", "lightgbm"] and first_explainer == "coefficients":
                validation_warnings.append(
                    f"Explainer 'coefficients' is not ideal for '{model_type}'.\n"
                    f"Recommend: Use 'treeshap' for tree models:\n"
                    f"  explainers:\n"
                    f"    priority: [treeshap]",
                )

        # Direct availability checking
        import importlib.util

        def _check_model_available(model_type: str) -> bool:
            """Check if model type dependencies are installed."""
            model_type = model_type.lower()
            if model_type in ["logistic_regression", "logistic", "sklearn"]:
                return importlib.util.find_spec("sklearn") is not None
            if model_type in ["xgboost", "xgb"]:
                return importlib.util.find_spec("xgboost") is not None
            if model_type in ["lightgbm", "lgb", "lgbm"]:
                return importlib.util.find_spec("lightgbm") is not None
            return False

        def _check_explainer_available(explainer_name: str) -> bool:
            """Check if explainer dependencies are installed."""
            explainer_name = explainer_name.lower()
            if explainer_name in ["treeshap", "kernelshap"]:
                return importlib.util.find_spec("shap") is not None
            if explainer_name in ["coefficients", "coef"]:
                return True  # Always available
            return False

        # 4. Check model availability
        if not _check_model_available(audit_config.model.type):
            msg = (
                f"Model '{audit_config.model.type}' requires additional dependencies. "
                f"Install with: pip install 'glassalpha[{audit_config.model.type}]'"
            )
            if strict_validation:
                validation_errors.append(msg)
            else:
                validation_warnings.append(msg + " (Will fallback to logistic_regression)")

        # 5. Check explainer availability and compatibility
        if audit_config.explainers.priority:
            available_requested = [e for e in audit_config.explainers.priority if _check_explainer_available(e)]

            if not available_requested:
                msg = (
                    f"None of the requested explainers {audit_config.explainers.priority} are available. "
                    f"Install with: pip install 'glassalpha[explain]'"
                )
                if strict_validation:
                    validation_errors.append(msg)
                else:
                    validation_warnings.append(msg + " (Will fallback to permutation explainer)")
            else:
                # Check model/explainer compatibility for FIRST available explainer only
                # (subsequent explainers are fallbacks, so incompatibility is expected)
                model_type = audit_config.model.type
                first_available = available_requested[0] if available_requested else None

                if first_available == "treeshap" and model_type not in ["xgboost", "lightgbm", "random_forest"]:
                    msg = (
                        f"TreeSHAP requested but model type '{model_type}' is not a tree model. "
                        "Consider using 'coefficients' (for linear) or 'permutation' (universal)."
                    )
                    if strict_validation:
                        validation_errors.append(msg)
                    else:
                        validation_warnings.append(msg)

                # Check other explainer compatibility issues for first available only
                if first_available == "coefficients" and model_type not in [
                    "logistic_regression",
                    "linear_regression",
                ]:
                    msg = f"Coefficients explainer requested but model type '{model_type}' doesn't have coefficients."
                    if strict_validation:
                        validation_errors.append(msg)
                    else:
                        validation_warnings.append(msg)

        # Check dataset and validate schema if --check-data is specified
        if check_data:
            typer.echo("\nValidating data files...")

            if audit_config.data.path and audit_config.data.dataset == "custom":
                data_path = Path(audit_config.data.path).expanduser()
                if not data_path.exists():
                    validation_errors.append(f"Data file not found: {data_path}")
                else:
                    # Validate dataset schema if file exists
                    try:
                        from glassalpha.data.tabular import TabularDataLoader

                        # Load data to validate schema
                        loader = TabularDataLoader()
                        df = loader.load(data_path)

                        typer.echo(f"  ✓ Data file readable: {data_path}")
                        typer.echo(f"  ✓ Loaded: {len(df)} rows × {len(df.columns)} columns")

                        # Check if target column exists
                        if audit_config.data.target_column:
                            if audit_config.data.target_column not in df.columns:
                                validation_errors.append(
                                    f"Target column '{audit_config.data.target_column}' not found in data.\n"
                                    f"    Available columns: {list(df.columns)[:10]}...",
                                )
                            else:
                                typer.echo(f"  ✓ Target column exists: {audit_config.data.target_column}")

                        # Check if protected attributes exist
                        if audit_config.data.protected_attributes:
                            missing_protected = [
                                attr for attr in audit_config.data.protected_attributes if attr not in df.columns
                            ]
                            if missing_protected:
                                validation_warnings.append(
                                    f"Protected attributes not found in data: {missing_protected}\n"
                                    f"    Available columns: {list(df.columns)[:10]}...",
                                )
                            else:
                                typer.echo(f"  ✓ Protected attributes exist: {audit_config.data.protected_attributes}")

                    except ValueError as e:
                        validation_errors.append(f"Dataset schema validation failed: {e}")
                    except Exception as e:
                        validation_errors.append(f"Error loading dataset: {e}")
            elif audit_config.data.dataset and audit_config.data.dataset != "custom":
                # Built-in dataset - try to load and validate
                try:
                    if audit_config.data.dataset in ["german_credit", "adult_income"]:
                        typer.echo(f"  ✓ Using built-in dataset: {audit_config.data.dataset}")

                        # Try to load the dataset to verify it exists
                        from glassalpha.utils.cache_dirs import resolve_data_root

                        cache_root = resolve_data_root()

                        # Map dataset name to expected cache path
                        dataset_name = audit_config.data.dataset
                        if dataset_name == "german_credit":
                            expected_path = cache_root / "german_credit_processed.csv"
                        elif dataset_name == "adult_income":
                            expected_path = cache_root / "adult_income.csv"
                        else:
                            # Generic fallback for custom datasets
                            expected_path = cache_root / f"{dataset_name}.csv"

                        if expected_path.exists():
                            typer.echo(f"  ✓ Dataset cached locally: {expected_path}")
                        else:
                            validation_warnings.append(
                                f"Built-in dataset '{audit_config.data.dataset}' not cached yet.\n"
                                f"    Run: glassalpha datasets fetch {audit_config.data.dataset}",
                            )
                    else:
                        validation_errors.append(f"Unknown built-in dataset: {audit_config.data.dataset}")
                except Exception as e:
                    validation_warnings.append(f"Could not verify built-in dataset: {e}")
            else:
                validation_warnings.append("No data source specified in configuration")
        elif audit_config.data.path and audit_config.data.dataset == "custom":
            # Basic file existence check even without --check-data
            data_path = Path(audit_config.data.path).expanduser()
            if not data_path.exists():
                msg = f"Data file not found: {data_path}"
                if strict_validation:
                    validation_errors.append(msg)
                else:
                    validation_warnings.append(msg)

        # Report validation errors
        if validation_errors:
            typer.echo()
            typer.secho("Validation failed with errors:", fg=typer.colors.RED, err=True)
            for error in validation_errors:
                typer.secho(f"  • {error}", fg=typer.colors.RED, err=True)
            typer.echo()
            typer.secho(
                "Tip: Run without --strict-validation to see warnings instead of errors",
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(ExitCode.VALIDATION_ERROR)

        # Report validation results
        typer.secho("Configuration is valid", fg=typer.colors.GREEN)

        # Show runtime warnings
        if validation_warnings:
            typer.echo()
            typer.secho("Runtime warnings:", fg=typer.colors.YELLOW)
            for warning in validation_warnings:
                typer.secho(f"  • {warning}", fg=typer.colors.YELLOW)
            typer.echo()
            if not strict_validation:
                typer.secho(
                    "Tip: Add --strict-validation to treat warnings as errors (recommended for production)",
                    fg=typer.colors.CYAN,
                )

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
        # All ValueErrors in validation context are validation failures
        # (including strict mode validation from Pydantic validators)
        typer.secho(f"Validation failed: {e}", fg=typer.colors.RED, err=True)
        # Intentional: User-friendly validation errors
        raise typer.Exit(ExitCode.VALIDATION_ERROR) from None
    except typer.Exit as e:
        # Re-raise typer.Exit exceptions (like validation errors) without wrapping
        raise
    except Exception as e:
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED, err=True)
        # Design choice: Hide implementation details from end users
        raise typer.Exit(ExitCode.USER_ERROR) from None


def config_list_cmd():  # pragma: no cover
    """List available configuration templates.

    Shows all built-in configuration templates with descriptions.
    Templates are organized by use case and complexity.

    Examples:
        # List all templates
        glassalpha config-list

        # Copy a template
        glassalpha config-template german_credit > audit.yaml

    """
    from pathlib import Path

    # Get configs path from package
    configs_path = Path(__file__).parent.parent.parent / "configs"

    typer.echo("\n📋 Configuration Templates\n")

    # Define templates with descriptions (tiered)
    templates = {
        "📌 Recommended (Start Here)": [
            ("minimal", "8-line minimal config"),
            ("german_credit_simple", "Simple audit example"),
            ("custom_template", "Blank template for your data"),
        ],
        "📚 Examples (Learning)": [
            ("german_credit", "Full-featured German Credit"),
            ("adult_income_simple", "Adult Income fairness"),
            ("adult_income", "Complete Adult Income audit"),
            ("calibration_focused", "Calibration deep-dive"),
            ("fairness_focused", "Fairness analysis"),
            ("reason_codes_german_credit", "ECOA reason codes"),
            ("recourse_german_credit", "Counterfactual recourse"),
            ("quickstart", "Generated by 'glassalpha quickstart'"),
        ],
        "🏢 Compliance (Production)": [
            ("production", "Production-ready with strict mode"),
            ("gdpr_compliance", "GDPR compliance template"),
            ("ccpa_compliance", "CCPA compliance template"),
            ("credit_card_fraud_template", "Fraud detection domain"),
            ("folktables_income_template", "Census data analysis"),
        ],
    }

    for category, items in templates.items():
        typer.secho(f"{category}:", fg=typer.colors.BRIGHT_BLUE, bold=True)
        for name, description in items:
            template_path = configs_path / f"{name}.yaml"
            if template_path.exists():
                typer.echo(f"  • {name:<30} - {description}")
            else:
                typer.echo(f"  • {name:<30} - {description} (not found)")
        typer.echo()

    typer.echo("💡 Usage:")
    typer.echo("  glassalpha config template <name> > my_config.yaml")
    typer.echo("  glassalpha config cheat    # Show common config patterns\n")


def config_template_cmd(  # pragma: no cover
    template: str = typer.Argument(
        ...,
        help="Template name (e.g., 'german_credit', 'minimal', 'custom_template')",
    ),
):
    """Output a configuration template to stdout.

    Prints the specified template to stdout so you can redirect it to a file.
    Use 'glassalpha config-list' to see available templates.

    Examples:
        # Copy template to new file
        glassalpha config-template german_credit > audit.yaml

        # View template
        glassalpha config-template minimal

        # Copy and edit
        glassalpha config-template custom_template > my_config.yaml
        # Then edit my_config.yaml with your data paths

    """
    from pathlib import Path

    # Get configs path from package
    configs_path = Path(__file__).parent.parent.parent / "configs"

    template_path = configs_path / f"{template}.yaml"

    if not template_path.exists():
        typer.secho(f"\n❌ Template not found: {template}", fg=typer.colors.RED, err=True)
        typer.echo("\n💡 See available templates:", err=True)
        typer.echo("  glassalpha config-list\n", err=True)
        raise typer.Exit(ExitCode.USER_ERROR)

    # Output template to stdout
    with open(template_path, encoding="utf-8") as f:
        typer.echo(f.read())


def config_cheat_cmd():  # pragma: no cover
    """Show configuration cheat sheet with common patterns.

    Displays quick reference of common configuration patterns
    without leaving the terminal.

    Examples:
        # View cheat sheet
        glassalpha config-cheat

        # View and search
        glassalpha config-cheat | grep fairness

    """
    cheat_sheet = """
╔══════════════════════════════════════════════════════════════════════════╗
║             GlassAlpha Configuration Cheat Sheet                         ║
╚══════════════════════════════════════════════════════════════════════════╝

┌─ Minimal Config (8 lines) ────────────────────────────────────────────────┐
│ data:                                                                      │
│   dataset: german_credit                                                  │
│   target_column: credit_risk                                              │
│   protected_attributes: [gender]                                          │
│ model:                                                                     │
│   type: logistic_regression                                               │
│ reproducibility:                                                           │
│   random_seed: 42                                                          │
└────────────────────────────────────────────────────────────────────────────┘

┌─ Custom Data ──────────────────────────────────────────────────────────────┐
│ data:                                                                      │
│   path: data/my_data.csv            # Your CSV file                       │
│   target_column: outcome            # Your target column                  │
│   protected_attributes:              # Your sensitive attributes           │
│     - gender                                                               │
│     - age_group                                                            │
│     - race                                                                 │
└────────────────────────────────────────────────────────────────────────────┘

┌─ Add XGBoost Model ────────────────────────────────────────────────────────┐
│ model:                                                                     │
│   type: xgboost                                                            │
│   params:                                                                  │
│     objective: binary:logistic                                             │
│     n_estimators: 100                                                      │
│     max_depth: 5                                                           │
│     learning_rate: 0.1                                                     │
│     random_state: 42                                                       │
└────────────────────────────────────────────────────────────────────────────┘

┌─ Enable Strict Mode (regulatory compliance) ───────────────────────────────┐
│ runtime:                                                                   │
│   strict_mode: true                # Enforces explicit config              │
│   fast_mode: false                 # Full bootstrap samples               │
└────────────────────────────────────────────────────────────────────────────┘

┌─ Enable Fast Mode (development) ───────────────────────────────────────────┐
│ runtime:                                                                   │
│   fast_mode: true                  # 100 bootstrap samples (2-3 seconds)  │
└────────────────────────────────────────────────────────────────────────────┘

┌─ Comprehensive Fairness Analysis ──────────────────────────────────────────┐
│ metrics:                                                                   │
│   compute_fairness: true                                                   │
│   compute_calibration: true                                                │
│   fairness:                                                                │
│     metrics:                                                               │
│       - demographic_parity                                                 │
│       - equal_opportunity                                                  │
│       - equalized_odds                                                     │
│       - predictive_parity                                                  │
│   n_bootstrap: 1000               # Confidence intervals                  │
└────────────────────────────────────────────────────────────────────────────┘

┌─ Preprocessing Artifact Verification ──────────────────────────────────────┐
│ preprocessing:                                                             │
│   mode: artifact                  # Load production preprocessor          │
│   artifact_path: artifacts/preprocessor.joblib                            │
│   expected_file_hash: sha256:abc123...                                    │
│   expected_params_hash: sha256:def456...                                  │
│   fail_on_mismatch: true          # Strict verification                   │
└────────────────────────────────────────────────────────────────────────────┘

┌─ SHAP Explainer Configuration ─────────────────────────────────────────────┐
│ explainers:                                                                │
│   strategy: first_compatible                                               │
│   priority:                                                                │
│     - treeshap                    # Best for XGBoost, LightGBM            │
│     - kernelshap                  # Model-agnostic fallback               │
│   config:                                                                  │
│     treeshap:                                                              │
│       max_samples: 1000           # SHAP computation samples              │
└────────────────────────────────────────────────────────────────────────────┘

💡 Quick Commands:
   glassalpha config-list                    # See all templates
   glassalpha config-template minimal        # View template
   glassalpha config-template german_credit > audit.yaml  # Copy template
   glassalpha validate --config audit.yaml   # Validate config
   glassalpha audit --dry-run                # Test without output

📚 Documentation:
   Configuration Guide:  https://glassalpha.com/getting-started/configuration/
   Quickstart:           https://glassalpha.com/getting-started/quickstart/
   Custom Data:          https://glassalpha.com/getting-started/custom-data/
"""

    typer.echo(cheat_sheet)
