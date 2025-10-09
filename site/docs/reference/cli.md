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
    Repro mode auto-enables in CI environments and for test* configs

Examples:
    # Minimal usage (uses smart defaults)
    glassalpha audit

    # Explicit paths
    glassalpha audit --config audit.yaml --output report.html

    # See what defaults would be used
    glassalpha audit --show-defaults

    # Check output paths before running audit
    glassalpha audit --check-output

    # Strict mode for regulatory compliance
    glassalpha audit --config production.yaml  # Auto-enables strict!

    # Override specific settings
    glassalpha audit -c base.yaml --override custom.yaml

    # Fail if components unavailable (no fallbacks)
    glassalpha audit --no-fallback

    # Stress test for demographic shifts (E6.5)
    glassalpha audit --check-shift gender:+0.1

    # Multiple shifts with degradation threshold
    glassalpha audit --check-shift gender:+0.1 --check-shift age:-0.05 --fail-on-degradation 0.05

**Options:**

- `--config, -c`: Path to audit configuration YAML file (auto-detects glassalpha.yaml, audit.yaml, config.yaml)
- `--output, -o`: Path for output report (defaults to {config_name}.html)
- `--strict, -s`: Enable strict mode for regulatory compliance (auto-enabled for prod*/production* configs). Allows built-in datasets.
- `--strict-full`: Enable full strict mode for maximum regulatory compliance. Requires explicit data schemas and model paths. Disallows built-in datasets. (default: `False`)
- `--repro`: Enable deterministic reproduction mode (auto-enabled in CI and for test* configs)
- `--profile, -p`: Override audit profile
- `--override`: Additional config file to override settings
- `--dry-run`: Validate configuration without generating report (default: `False`)
- `--no-fallback`: Fail if requested components are unavailable (no automatic fallbacks) (default: `False`)
- `--show-defaults`: Show inferred defaults and exit (useful for debugging) (default: `False`)
- `--check-output`: Check output paths are writable and exit (pre-flight validation) (default: `False`)
- `--check-shift`: Test model robustness under demographic shifts (e.g., 'gender:+0.1'). Can specify multiple. (default: `[]`)
- `--fail-on-degradation`: Exit with error if any metric degrades by more than this threshold (e.g., 0.05 for 5pp).
- `--save-model`: Save the trained model to the specified path (e.g., model.pkl). Required for reasons/recourse commands.
- `--fast`: Fast demo mode: reduce bootstrap samples to 100 for lightning-quick audits (~2-3s vs ~5-7s) (default: `False`)
- `--sample`: Sample N rows from dataset for faster iteration (useful for large datasets during development; minimum 100 rows)
- `--compact-report`: Generate compact report (<1MB, default) by excluding individual fairness matched pairs from HTML. Use --full-report for complete data (may be 50-100MB). Full data always saved in manifest.json. (default: `True`)

### `glassalpha datasets`

Dataset management operations

### `glassalpha datasets cache-dir`

Show the directory where datasets are cached.

### `glassalpha datasets fetch`

Fetch and cache a dataset from the registry.

**Arguments:**

- `dataset` (text, required): Dataset key to fetch

**Options:**

- `--force, -f`: Force re-download even if file exists (default: `False`)
- `--dest`: Custom destination path

### `glassalpha datasets info`

Show detailed information about a specific dataset.

**Arguments:**

- `dataset` (text, required): Dataset key to inspect

### `glassalpha datasets list`

List all available datasets in the registry.

### `glassalpha docs`

Open documentation in browser

**Arguments:**

- `topic` (text, optional): Documentation topic (e.g., 'model-parameters', 'quickstart', 'cli')

**Options:**

- `--open`: Open in browser (default: `True`)

### `glassalpha doctor`

Check environment and optional features

### `glassalpha init`

Initialize new audit configuration

**Options:**

- `--output, -o`: Output path for generated configuration file (default: `audit_config.yaml`)
- `--template, -t`: Use a specific template (quickstart, production, development, testing)
- `--interactive`: Use interactive mode to customize configuration (default: `True`)

### `glassalpha list`

List available components

**Arguments:**

- `component_type` (text, optional): Component type to list (models, explainers, metrics, profiles)

**Options:**

- `--include-enterprise, -e`: Include enterprise components (default: `False`)
- `--verbose, -v`: Show component details (default: `False`)

### `glassalpha models`

Show available models.

### `glassalpha prep`

Preprocessing artifact management

### `glassalpha prep hash`

Compute hash(es) for a preprocessing artifact.

This command computes the file hash (SHA256) of a preprocessing artifact.
Optionally, it can also compute the params hash (canonical hash of learned
parameters) by loading and introspecting the artifact.

Examples:
    # Just file hash (fast, no loading)
    glassalpha prep hash artifacts/preprocessor.joblib

    # File and params hash (slower, loads artifact)
    glassalpha prep hash artifacts/preprocessor.joblib --params

**Arguments:**

- `artifact_path` (file, required): Path to preprocessing artifact (.joblib file)

**Options:**

- `--params, -p`: Also compute and show params hash (default: `False`)

### `glassalpha prep inspect`

Inspect a preprocessing artifact and display its manifest.

This command loads the artifact, extracts all learned parameters,
and displays a human-readable summary. Optionally saves the full
manifest to a JSON file.

Examples:
    # Quick inspection
    glassalpha prep inspect artifacts/preprocessor.joblib

    # Detailed inspection
    glassalpha prep inspect artifacts/preprocessor.joblib --verbose

    # Save manifest to file
    glassalpha prep inspect artifacts/preprocessor.joblib --output manifest.json

**Arguments:**

- `artifact_path` (file, required): Path to preprocessing artifact (.joblib file)

**Options:**

- `--output, -o`: Save manifest to JSON file
- `--verbose, -v`: Show detailed component information (default: `False`)

### `glassalpha prep validate`

Validate a preprocessing artifact.

This command performs comprehensive validation of a preprocessing artifact,
including hash verification, class allowlisting, and version compatibility.

Examples:
    # Basic validation (classes + versions)
    glassalpha prep validate artifacts/preprocessor.joblib

    # Validate with expected hashes
    glassalpha prep validate artifacts/preprocessor.joblib \
        --file-hash sha256:abc123... \
        --params-hash sha256:def456...

    # Skip version checks
    glassalpha prep validate artifacts/preprocessor.joblib --no-check-versions

**Arguments:**

- `artifact_path` (file, required): Path to preprocessing artifact (.joblib file)

**Options:**

- `--file-hash`: Expected file hash (sha256:...)
- `--params-hash`: Expected params hash (sha256:...)
- `--check-versions`: Check runtime version compatibility (default: `True`)

### `glassalpha quickstart`

Generate template audit project

**Options:**

- `--output, -o`: Output directory for project scaffold (default: `my-audit-project`)
- `--dataset, -d`: Dataset type (german_credit, adult_income)
- `--model, -m`: Model type (xgboost, lightgbm, logistic_regression)
- `--interactive`: Use interactive mode to customize project (default: `True`)

### `glassalpha reasons`

Generate ECOA-compliant reason codes

**Options:**

- `--model, -m`: Path to trained model file (.pkl, .joblib). Generate with: glassalpha audit --save-model model.pkl
- `--data, -d`: Path to test data file (CSV)
- `--instance, -i`: Row index of instance to explain (0-based)
- `--config, -c`: Path to reason codes configuration YAML
- `--output, -o`: Path for output notice file (defaults to stdout)
- `--threshold, -t`: Decision threshold for approved/denied (default: `0.5`)
- `--top-n, -n`: Number of reason codes to generate (ECOA typical: 4) (default: `4`)
- `--format, -f`: Output format: 'text' or 'json' (default: `text`)

### `glassalpha recourse`

Generate counterfactual recourse recommendations

**Options:**

- `--model, -m`: Path to trained model file (.pkl, .joblib). Generate with: glassalpha audit --save-model model.pkl
- `--data, -d`: Path to test data file (CSV)
- `--instance, -i`: Row index of instance to explain (0-based)
- `--config, -c`: Path to recourse configuration YAML
- `--output, -o`: Path for output recommendations file (JSON, defaults to stdout)
- `--threshold, -t`: Decision threshold for approved/denied (default: `0.5`)
- `--top-n, -n`: Number of counterfactual recommendations to generate (default: `5`)

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

**Arguments:**

- `config_path` (file, optional): Path to configuration file to validate

**Options:**

- `--config, -c`: Path to configuration file to validate (alternative to positional arg)
- `--profile, -p`: Validate against specific profile
- `--strict`: Validate for strict mode compliance (allows built-in datasets) (default: `False`)
- `--strict-full`: Validate for full strict mode compliance (requires explicit schemas, disallows built-in datasets) (default: `False`)
- `--strict-validation`: Enforce runtime availability checks (recommended for production) (default: `False`)

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