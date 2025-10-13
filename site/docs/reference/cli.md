# CLI Reference

Complete command-line interface reference for GlassAlpha.

## Installation

```bash
pip install glassalpha
```

## Quick Start

```bash
# Get help
glassalpha --help

# Run audit with config
glassalpha audit --config config.yaml --out report.html

# Validate config
glassalpha validate config.yaml
```

## Commands

GlassAlpha provides the following commands:

### `glassalpha audit`

Generate a compliance audit report (HTML/PDF) with optional shift testing.

This is the main command for GlassAlpha. It loads a configuration file,
runs the audit pipeline, and generates a deterministic audit report.

Smart Defaults:
    If no --config is provided, searches for: glassalpha.yaml, audit.yaml, config.yaml
    If no --output is provided, uses {config_name}.html
    Strict mode auto-enables for prod*/production* configs
    Repro mode auto-enables in CI environments

Configuration:
    Runtime options (fast mode, compact report, fallback behavior) are configured
    in the config file under 'runtime:' section. See documentation for details.

Examples:
    # Minimal usage (uses smart defaults)
    glassalpha audit

    # Explicit paths
    glassalpha audit --config audit.yaml --output report.html

    # Strict mode for regulatory compliance
    glassalpha audit --config production.yaml  # Auto-enables strict!

    # Validate configuration without running audit
    glassalpha audit --dry-run

    # Stress test for demographic shifts (E6.5)
    glassalpha audit --check-shift gender:+0.1

    # Multiple shifts with degradation threshold
    glassalpha audit --check-shift gender:+0.1 --check-shift age:-0.05 --fail-on-degradation 0.05

**Options:**

- `--config, -c`: Path to audit configuration YAML file (auto-detects glassalpha.yaml, audit.yaml, config.yaml)
- `--output, -o`: Path for output report (defaults to {config_name}.html)
- `--strict, -s`: Enable strict mode for regulatory compliance (auto-enabled for prod*/production* configs)
- `--profile, -p`: Override audit profile
- `--dry-run`: Validate configuration without generating report (default: `False`)
- `--verbose, -v`: Enable verbose logging (default: `False`)
- `--check-shift`: Test model robustness under demographic shifts (e.g., 'gender:+0.1'). Can specify multiple. (default: `[]`)
- `--fail-on-degradation`: Exit with error if any metric degrades by more than this threshold (e.g., 0.05 for 5pp).

### `glassalpha doctor`

Check environment and optional features.

This command diagnoses the current environment and shows what optional
features are available and how to enable them.

Examples:
    # Basic environment check
    glassalpha doctor

    # Verbose output
    glassalpha doctor --verbose

### `glassalpha list`

List available components with runtime availability status.

Shows registered models, explainers, metrics, and audit profiles.
Indicates which components are available vs require additional dependencies.

Examples:
    # List all components
    glassalpha list

    # List specific type
    glassalpha list models

    # Include enterprise components
    glassalpha list --include-enterprise

**Arguments:**

- `component_type` (text, optional): Component type to list (models, explainers, metrics, profiles)

**Options:**

- `--include-enterprise, -e`: Include enterprise components (default: `False`)
- `--verbose, -v`: Show component details (default: `False`)

### `glassalpha quickstart`

Generate a complete audit project with config and directory structure.

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

**Options:**

- `--output, -o`: Output directory for project scaffold (default: `my-audit-project`)
- `--dataset, -d`: Dataset type (german_credit, adult_income)
- `--model, -m`: Model type (xgboost, lightgbm, logistic_regression)
- `--interactive`: Use interactive mode to customize project (default: `True`)

### `glassalpha validate`

Validate a configuration file.

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

**Arguments:**

- `config_path` (file, optional): Path to configuration file to validate

**Options:**

- `--config, -c`: Path to configuration file to validate (alternative to positional arg)
- `--profile, -p`: Validate against specific profile
- `--strict`: Validate for strict mode compliance (default: `False`)
- `--strict-validation`: Enforce runtime availability checks (recommended for production) (default: `False`)
- `--check-data`: Load and validate actual dataset (checks if target column exists, file is readable) (default: `False`)

## Global Options

These options are available for all commands:

- `--help`: Show help message and exit
- `--version`: Show version and exit

## Exit Codes

GlassAlpha uses standard exit codes:

- `0`: Success
- `1`: Validation failure or policy gate failure
- `2`: Runtime error
- `3`: Configuration error

## Environment Variables

- `PYTHONHASHSEED`: Set for deterministic execution (recommended: `42`)
- `GLASSALPHA_CONFIG_DIR`: Override default config directory
- `GLASSALPHA_CACHE_DIR`: Override default cache directory

---

*This documentation is automatically generated from the CLI code.*
*Last updated: See git history for this file.*
