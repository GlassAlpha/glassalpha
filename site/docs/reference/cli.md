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
    If no --config is provided, searches for: glassalpha.yaml, audit.yaml, audit_config.yaml, config.yaml
    If no --output is provided, uses {config_name}_report.html
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

- `--config, -c`: Path to audit configuration YAML file (auto-detects glassalpha.yaml, audit.yaml, audit_config.yaml, config.yaml)
- `--output, -o`: Path for output report (defaults to {config_name}_report.html)
- `--strict, -s`: Enable strict mode for regulatory compliance (auto-enabled for prod*/production* configs)
- `--profile, -p`: Override audit profile
- `--dry-run`: Validate configuration without generating report (default: `False`)
- `--verbose, -v`: Enable verbose logging (default: `False`)
- `--check-shift`: Test model robustness under demographic shifts (e.g., 'gender:+0.1'). Can specify multiple. (default: `[]`)
- `--fail-on-degradation`: Exit with error if any metric degrades by more than this threshold (e.g., 0.05 for 5pp).

### `glassalpha config`

Configuration templates and helpers.

### `glassalpha config cheat`

Show configuration cheat sheet with common patterns.

Displays quick reference of common configuration patterns
without leaving the terminal.

Examples:
    # View cheat sheet
    glassalpha config-cheat

    # View and search
    glassalpha config-cheat | grep fairness

### `glassalpha config list`

List available configuration templates.

Shows all built-in configuration templates with descriptions.
Templates are organized by use case and complexity.

Examples:
    # List all templates
    glassalpha config-list

    # Copy a template
    glassalpha config-template german_credit > audit.yaml

### `glassalpha config template`

Output a configuration template to stdout.

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

**Arguments:**

- `template` (text, required): Template name (e.g., 'german_credit', 'minimal', 'custom_template')

### `glassalpha docs`

Open documentation in browser.

Opens the GlassAlpha documentation website. You can optionally specify
a topic to jump directly to that section.

Examples:
    # Open docs home
    glassalpha docs

    # Open specific topic
    glassalpha docs model-parameters

    # Just print URL without opening
    glassalpha docs quickstart --no-open

**Arguments:**

- `topic` (text, optional): Documentation topic (e.g., 'model-parameters', 'quickstart', 'cli')

**Options:**

- `--open`: Open in browser (default: `True`)

### `glassalpha doctor`

Check environment and optional features.

This command diagnoses the current environment and shows what optional
features are available and how to enable them.

Examples:
    # Basic environment check
    glassalpha doctor

    # Verbose output with package versions
    glassalpha doctor --verbose

**Options:**

- `--verbose, -v`: Show detailed environment information including package versions and paths (default: `False`)

### `glassalpha export-evidence-pack`

Export evidence pack for audit verification.

Creates tamper-evident ZIP with all audit artifacts, checksums,
and verification instructions for regulatory submission.

The evidence pack includes:
- Audit report (HTML or PDF)
- Provenance manifest (hashes, versions, seeds)
- Policy decision log (stub for v0.3.0)
- Configuration file (if provided)
- SHA256 checksums and verification instructions

Requirements:
    - Completed audit report (HTML or PDF format)
    - Manifest file (auto-generated during audit)

Examples:
    # Basic export with auto-generated name
    glassalpha export-evidence-pack reports/audit_report.html

    # Custom output path
    glassalpha export-evidence-pack audit.pdf --output compliance/evidence_2024.zip

    # Include original config for reproducibility
    glassalpha export-evidence-pack audit.html --config audit_config.yaml

    # Skip badge generation (faster)
    glassalpha export-evidence-pack audit.pdf --no-badge

**Arguments:**

- `report` (path, required): Path to audit report (HTML or PDF)

**Options:**

- `--output`: Output ZIP path (auto-generated if omitted)
- `--config`: Include original config file
- `--no-badge`: Skip badge generation (default: `False`)

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

### `glassalpha publish-check`

Check if package is ready for publication.

Runs pre-publication validation including:
- Version tag matches CHANGELOG
- All tests passing
- Determinism verification
- Example configs valid
- CLI commands documented
- Notebooks executable

Examples:
    # Basic check
    glassalpha publish-check

    # Verbose output
    glassalpha publish-check --verbose

**Options:**

- `--verbose, -v`: Show detailed output from checks (default: `False`)

### `glassalpha quickstart`

Generate a complete audit project with config and directory structure.

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

**Options:**

- `--output, -o`: Output directory for project scaffold (default: `my-audit-project`)
- `--dataset, -d`: Dataset type (german_credit, adult_income)
- `--model, -m`: Model type (xgboost, lightgbm, logistic_regression)

### `glassalpha reasons`

Generate ECOA-compliant reason codes for adverse action notice.

This command extracts top-N negative feature contributions from a trained model
to explain why a specific instance was denied (or approved). Output is formatted
as an ECOA-compliant adverse action notice.

Requirements:
    - Trained model with SHAP-compatible architecture
    - Test dataset with same features as training
    - Instance index to explain

Examples:
    # Generate reason codes for instance 42
    glassalpha reasons \
        --model models/german_credit.pkl \
        --data data/test.csv \
        --instance 42 \
        --output notices/instance_42.txt

    # With custom config
    glassalpha reasons -m model.pkl -d test.csv -i 10 -c config.yaml

    # JSON output
    glassalpha reasons -m model.pkl -d test.csv -i 5 --format json

    # Custom threshold and top-N
    glassalpha reasons -m model.pkl -d test.csv -i 0 --threshold 0.6 --top-n 3

**Options:**

- `--model, -m`: Path to trained model file (.pkl, .joblib). Generate with: glassalpha audit --save-model model.pkl
- `--data, -d`: Path to test data file (CSV). MUST be preprocessed data (e.g., models/test_data.csv from audit).
- `--instance, -i`: Row index of instance to explain (0-based)
- `--config, -c`: Path to reason codes configuration YAML
- `--output, -o`: Path for output notice file (defaults to stdout)
- `--threshold, -t`: Decision threshold for approved/denied (default: `0.5`)
- `--top-n, -n`: Number of reason codes to generate (ECOA typical: 4) (default: `4`)
- `--format, -f`: Output format: 'text' or 'json' (default: `text`)

### `glassalpha recourse`

Generate ECOA-compliant counterfactual recourse recommendations.

This command generates feasible counterfactual recommendations with policy constraints
for individuals receiving adverse decisions. Supports immutable features, monotonic
constraints, and cost-weighted optimization.

Requirements:
    - Trained model with SHAP-compatible architecture
    - Test dataset with same features as training
    - Instance index to explain (must be denied: prediction < threshold)
    - Configuration file with policy constraints (recommended)

Examples:
    # Generate recourse for denied instance
    glassalpha recourse \
        --model models/german_credit.pkl \
        --data data/test.csv \
        --instance 42 \
        --config configs/recourse_german_credit.yaml \
        --output recourse/instance_42.json

    # With custom threshold and top-N
    glassalpha recourse -m model.pkl -d test.csv -i 10 -c config.yaml --top-n 3

    # Output to stdout
    glassalpha recourse -m model.pkl -d test.csv -i 5 -c config.yaml

Configuration File:
    The config file should include:
    - recourse.immutable_features: list of features that cannot be changed
    - recourse.monotonic_constraints: directional constraints (increase_only, decrease_only)
    - recourse.cost_function: cost function for optimization (weighted_l1)
    - data.protected_attributes: list of protected attributes to exclude
    - reproducibility.random_seed: seed for deterministic results

Model Compatibility:
    Recourse works best with sklearn-compatible models:
    ✅ logistic_regression, linear_regression, random_forest (sklearn)
    ⚠️  xgboost, lightgbm (limited support - known issues with feature modification)

    For XGBoost models, consider using 'glassalpha reasons' instead for ECOA-compliant
    adverse action notices. See: https://glassalpha.com/guides/recourse/#known-limitations

**Options:**

- `--model, -m`: Path to trained model file (.pkl, .joblib). Generate with: glassalpha audit --save-model model.pkl
- `--data, -d`: Path to test data file (CSV). Auto-generated if missing using built-in dataset.
- `--instance, -i`: Row index of instance to explain (0-based)
- `--config, -c`: Path to recourse configuration YAML
- `--output, -o`: Path for output recommendations file (JSON, defaults to stdout)
- `--threshold, -t`: Decision threshold for approved/denied (default: `0.5`)
- `--top-n, -n`: Number of counterfactual recommendations to generate (default: `5`)
- `--force-recourse`: Generate recourse recommendations even for approved instances (for testing) (default: `False`)

### `glassalpha setup-env`

Generate environment setup commands for deterministic audits.

This command outputs shell commands to set required environment variables
for byte-identical, reproducible audit generation.

Required for compliance: Regulators require byte-identical outputs for
audit verification. These environment variables ensure deterministic behavior.

Examples:
    # Print commands for current shell
    glassalpha setup-env

    # Save to file and source
    glassalpha setup-env --output .glassalpha-env
    source .glassalpha-env

    # Use in current shell (bash/zsh)
    eval "$(glassalpha setup-env)"

    # Specific shell syntax
    glassalpha setup-env --shell fish > glassalpha.fish
    source glassalpha.fish

**Options:**

- `--shell`: Shell type: bash, zsh, fish. Auto-detects if not specified.
- `--output, -o`: Write environment file instead of printing to stdout

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

### `glassalpha verify-evidence-pack`

Verify evidence pack integrity.

Confirms all checksums match and pack is tamper-free for regulatory
verification. Returns exit code 0 if verified, 1 if verification fails.

The verification checks:
- ZIP file is readable and not corrupted
- SHA256SUMS.txt is present and valid
- All file checksums match
- canonical.jsonl is well-formed
- Required artifacts are present (audit report, manifest)

Requirements:
    - Evidence pack ZIP file created with export-evidence-pack command

Examples:
    # Basic verification
    glassalpha verify-evidence-pack evidence_pack.zip

    # Verbose output with detailed checksums
    glassalpha verify-evidence-pack pack.zip --verbose

    # Verify downloaded pack from regulator
    glassalpha verify-evidence-pack compliance_submission_2024.zip

    # Use in CI/CD pipeline
    glassalpha verify-evidence-pack evidence.zip || exit 1

**Arguments:**

- `pack` (path, required): Evidence pack ZIP to verify

**Options:**

- `--verbose`: Show detailed verification log (default: `False`)

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
