# Configuration guide

Complete guide to configuring GlassAlpha for different use cases, models, and compliance requirements.

!!! tip "New to GlassAlpha?"
Start with the [Quick Start Guide](quickstart.md) to generate your first audit, then return here for detailed configuration options.

## Overview

GlassAlpha uses YAML configuration files to define every aspect of the audit process. Configuration files are policy-as-code, enabling version control, review processes, and reproducible audits.

## Progressive Configuration Levels

Choose the complexity level that matches your needs:

### Level 1: Minimal (5 lines) - Getting Started

Perfect for your first audit. Uses built-in datasets and sensible defaults.

```yaml
data: { dataset: german_credit, target_column: credit_risk }
model: { type: logistic_regression }
reproducibility: { random_seed: 42 }
```

**What you get**: Working audit with performance metrics, basic explanations, and reproducibility.

**Use when**: Learning GlassAlpha, quick prototyping, demos.

[See full minimal.yaml →](https://github.com/GlassAlpha/glassalpha/blob/main/configs/minimal.yaml)

??? example "Run this config"
`bash
    glassalpha audit --config minimal.yaml --output audit.html --fast
    `

---

### Level 2: Intermediate (20 lines) - Common Use Cases

Adds fairness analysis and protected attributes. Most teams start here.

```yaml
# Direct configuration (no profiles needed)

data:
  dataset: german_credit
  target_column: credit_risk
  protected_attributes:
    - gender
    - age_group

model:
  type: logistic_regression
  params:
    random_state: 42
    max_iter: 1000

explainers:
  strategy: first_compatible
  priority: [coefficients]

metrics:
  fairness: [demographic_parity, equal_opportunity]

reproducibility:
  random_seed: 42
```

**What you get**: Everything from Level 1 + fairness analysis across protected groups.

**Use when**: Checking for bias, compliance requirements, team reviews.

[See full fairness_focused.yaml →](https://github.com/GlassAlpha/glassalpha/blob/main/configs/fairness_focused.yaml)

---

### Level 3: Production (50+ lines) - Full Compliance

Comprehensive configuration for regulatory submissions.

```yaml
# Direct configuration (no profiles needed)

data:
  dataset: german_credit
  target_column: credit_risk
  protected_attributes:
    - gender
    - age_group
    - foreign_worker

model:
  type: xgboost
  save_path: models/production_model.pkl
  params:
    objective: binary:logistic
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    random_state: 42

explainers:
  strategy: first_compatible
  priority: [treeshap, kernelshap]
  config:
    treeshap:
      max_samples: 1000

metrics:
  performance: [accuracy, precision, recall, f1, auc_roc]
  fairness: [demographic_parity, equal_opportunity, equalized_odds]

reproducibility:
  random_seed: 42
  deterministic: true
  capture_environment: true

report:
  output_format: html
  title: "Production Model Audit Report"

manifest:
  enabled: true
  include_git_info: true
```

**What you get**: Complete audit trail, full metrics, model saving for reasons/recourse.

**Use when**: Regulatory submission, production deployments, audit evidence.

[See full production.yaml →](https://github.com/GlassAlpha/glassalpha/blob/main/configs/production.yaml)

---

## Configuration Templates

Pre-built templates for common scenarios:

| Template                     | Use Case                | Complexity | Key Features                             |
| ---------------------------- | ----------------------- | ---------- | ---------------------------------------- |
| **minimal.yaml**             | Learning, demos         | 5 lines    | Basic audit                              |
| **fairness_focused.yaml**    | Bias detection          | 20 lines   | Protected attributes, fairness metrics   |
| **calibration_focused.yaml** | Probability calibration | 20 lines   | Calibration curves, confidence intervals |
| **production.yaml**          | Regulatory compliance   | 50+ lines  | Full audit trail, all features           |

Copy a template to get started:

```bash
# Copy template to your project
cp configs/fairness_focused.yaml my_audit.yaml

# Customize and run
glassalpha audit --config my_audit.yaml --output audit.html
```

---

## Detailed Configuration Reference

### Basic structure

Every configuration has this structure (minimal → production):

```yaml
# Required: Audit profile determines component selection
# Direct configuration (no profiles needed)

# Required: Reproducibility settings
reproducibility:
  random_seed: 42

# Required: Data configuration
data:
  path: data/my_dataset.csv
  target_column: outcome

# Required: Model configuration
model:
  type: xgboost

# Optional: Additional sections
explainers: { ... }
metrics: { ... }
report: { ... }
```

## Core configuration sections

### Audit profile

The audit profile determines which components are available and what validations are enforced.

```yaml
# Determines the audit context and available components
# Direct configuration (no profiles needed) # Currently supported profile
```

**Available Profiles:**

- `tabular_compliance` - Standard tabular ML compliance audit
- `german_credit_default` - German Credit dataset specific profile

### Reproducibility settings

Ensures deterministic, reproducible audit results.

```yaml
reproducibility:
  # Master random seed (required in strict mode)
  random_seed: 42

  # Optional: Advanced reproducibility settings
  deterministic: true # Enforce deterministic behavior
  capture_environment: true # Capture system information
  validate_determinism: true # Verify reproducibility
```

**Best Practices:**

- Always set `random_seed` for reproducible results
- Use the same seed for comparative audits
- Document seed values in audit reports

### Data configuration

Defines the dataset and feature structure for the audit.

```yaml
data:
  # Required: Path to dataset
  path: data/my_dataset.csv

  # Required: Target column name
  target_column: outcome

  # Optional: Explicit feature columns
  feature_columns:
    - feature1
    - feature2
    - feature3

  # Optional: Protected attributes for fairness analysis
  protected_attributes:
    - gender
    - age_group
    - ethnicity
```

**Supported Data Formats:**

- CSV (`.csv`)
- Parquet (`.parquet`)
- Feather (`.feather`)
- Pickle (`.pkl`)

**Feature Selection:**

- If `feature_columns` not specified, uses all columns except target
- Protected attributes should be included in features for bias analysis
- Features are automatically preprocessed based on data type

**Using your own data?** See [Using Custom Data](custom-data.md) for a complete tutorial on data preparation and requirements.

**Need example datasets?** Browse our [Freely Available Data Sources](data-sources.md) for curated public datasets with example configurations.

### Model configuration

Specifies the ML model to audit and its parameters.

```yaml
model:
  # Required: Model type (triggers appropriate wrapper)
  type: logistic_regression # Default baseline model (always available)

  # Optional: Allow fallback to baseline model if requested model unavailable
  allow_fallback: true # Default: true

  # Optional: Pre-trained model path
  path: models/my_model.pkl

  # Optional: Model parameters (for training)
  params:
    random_state: 42
    max_iter: 1000
```

**Supported Model Types:**

- `logistic_regression` - Scikit-learn LogisticRegression (baseline, always available)
- `xgboost` - XGBoost gradient boosting (optional, requires `pip install 'glassalpha[xgboost]'`)
- `lightgbm` - LightGBM gradient boosting (optional, requires `pip install 'glassalpha[lightgbm]'`)
- `sklearn_generic` - Generic scikit-learn models (baseline, always available)

**Need help choosing a model?** See the [Model Selection Guide](../reference/model-selection.md) for detailed comparisons, performance benchmarks, and use case recommendations.

**Optional Dependencies:**

GlassAlpha uses optional dependencies to keep the core installation lightweight. If you request a model that's not installed, GlassAlpha will:

1. **Automatically fall back** to `logistic_regression` if `allow_fallback: true` (default)
2. **Show clear installation instructions** for the requested model
3. **Fail gracefully** if `allow_fallback: false` and model unavailable

Example fallback behavior:

```bash
glassalpha audit --config my_config.yaml --output report.pdf
# If XGBoost requested but not installed:
# "Model 'xgboost' not available. Falling back to 'logistic_regression'.
#  To enable 'xgboost', run: pip install 'glassalpha[xgboost]'"
```

**Model Loading vs Training:**

- If `path` exists: loads pre-trained model
- If `path` missing: trains new model with `params`
- Parameters are passed to the underlying library

## Explainer configuration

Controls how model predictions are explained and interpreted.

```yaml
explainers:
  # Required: Selection strategy
  strategy: first_compatible

  # Required: Priority order (deterministic selection)
  priority:
    - treeshap # First choice for tree models
    - kernelshap # Fallback for any model

  # Optional: Explainer-specific configuration
  config:
    treeshap:
      max_samples: 1000 # Samples for SHAP computation
      check_additivity: true # Verify SHAP properties

    kernelshap:
      n_samples: 500 # Model evaluations
      background_size: 100 # Background dataset size
```

**Available Explainers:**

- `treeshap` - Exact SHAP values for tree models (XGBoost, LightGBM)
- `kernelshap` - Model-agnostic SHAP approximation
- `coefficients` - Direct coefficients for linear models (LogisticRegression)
- `permutation` - Feature importance via permutation (any model)
- `noop` - No-op placeholder (for testing)

**Selection Strategies:**

- `first_compatible` - Use first explainer compatible with model
- `best_available` - Select highest-priority compatible explainer

!!! tip "Choosing an Explainer"
Not sure which explainer to use? The [Explainer Selection Guide](../reference/explainers.md) provides:

    - Decision trees to choose the right explainer
    - Performance benchmarks and timing comparisons
    - Detailed configuration examples
    - Model compatibility matrix

## Metrics configuration

Defines which performance and fairness metrics to compute.

```yaml
metrics:
  # Performance evaluation metrics
  performance:
    metrics:
      - accuracy
      - precision
      - recall
      - f1
      - auc_roc
      - classification_report

  # Fairness and bias detection metrics
  fairness:
    metrics:
      - demographic_parity
      - equal_opportunity
      - equalized_odds
      - predictive_parity

    # Optional: Bias tolerance thresholds
    config:
      demographic_parity:
        threshold: 0.05 # 5% maximum group difference

  # Optional: Data drift detection metrics
  drift:
    metrics:
      - population_stability_index
      - kl_divergence
      - kolmogorov_smirnov
```

**Performance Metrics:**

- `accuracy` - Overall classification accuracy
- `precision` - Positive predictive value
- `recall` - True positive rate (sensitivity)
- `f1` - Harmonic mean of precision and recall
- `auc_roc` - Area under ROC curve
- `classification_report` - Comprehensive per-class metrics

**Fairness Metrics:**

- `demographic_parity` - Equal positive prediction rates across groups
- `equal_opportunity` - Equal true positive rates across groups
- `equalized_odds` - Equal TPR and FPR across groups
- `predictive_parity` - Equal precision across groups

## Preprocessing configuration

Verifies production preprocessing artifacts for regulatory compliance. See the [Preprocessing Verification Guide](../guides/preprocessing.md) for complete details.

```yaml
preprocessing:
  # Preprocessing mode: 'artifact' (production) or 'auto' (demo only)
  mode: artifact

  # Path to preprocessing artifact (.joblib file)
  artifact_path: artifacts/preprocessor.joblib

  # Expected file hash for integrity verification
  expected_file_hash: "sha256:9373ae67dbcd5c1558fa4ba7a727e05575f9421358a8604a1d0eb0da80385a26"

  # Expected params hash for learned parameters verification
  expected_params_hash: "sha256:cf5e71ee7e1733ca6d857b7f21df94a41f29ebeaaec9c6d46783336757cfcf37"

  # Expected output format (sparse or dense)
  expected_sparse: false

  # Fail on hash mismatch (strict mode)
  fail_on_mismatch: true

  # Version compatibility policy
  version_policy:
    sklearn: exact # exact, patch, or minor
    numpy: patch
    scipy: patch

  # Unknown category detection thresholds
  thresholds:
    warn_unknown_rate: 0.01 # Warn if >1% unknown
    fail_unknown_rate: 0.10 # Fail if >10% unknown
```

!!! warning "Compliance Requirement"
For regulatory audits, always use `mode: artifact` with verified preprocessing artifacts from production. Auto mode is for development/demo only and produces prominent warnings in reports.

!!! tip "Quick Commands"
Generate hashes for your preprocessing artifact:

    ```bash
    glassalpha prep hash artifacts/preprocessor.joblib --params
    ```

    Validate an artifact before use:

    ```bash
    glassalpha prep validate artifacts/preprocessor.joblib \
      --file-hash sha256:abc... \
      --params-hash sha256:def...
    ```

**Preprocessing Modes:**

- `artifact` - Uses verified production artifact (required for compliance)
- `auto` - Automatically fits preprocessing to audit data (demo only, not compliant)

**Key Features:**

- Dual hash verification (file integrity + learned parameters)
- Security class allowlisting (prevents pickle exploits)
- Runtime version compatibility checking
- Unknown category detection and reporting
- Full documentation in audit reports

**See Also:**

- [Preprocessing Verification Guide](../guides/preprocessing.md) - Complete documentation
- [CLI Reference](../reference/cli.md) - Command-line tools

## Report configuration

Controls the format and content of generated audit reports.

```yaml
report:
  # Report template (determines structure and styling)
  template: standard_audit

  # Output format
  output_format: pdf

  # Optional: Report sections to include
  include_sections:
    - executive_summary
    - data_overview
    - preprocessing_verification # NEW: Preprocessing artifact details
    - model_performance
    - global_explanations
    - local_explanations
    - fairness_analysis
    - audit_manifest
    - regulatory_compliance

  # Optional: Report styling
  styling:
    color_scheme: professional
    page_size: A4
    margins: standard
    compliance_statement: true
```

**Available Templates:**

- `standard_audit` - Comprehensive audit report with all sections

**Styling Options:**

- `color_scheme`: professional, minimal, colorful
- `page_size`: A4, Letter, Legal
- `margins`: standard, narrow, wide

## Advanced configuration

### Strict mode

Enforces additional regulatory compliance requirements for production and regulatory use.

```bash
# Enable strict mode via CLI flag (recommended)
glassalpha audit --config my_config.yaml --output audit.pdf --strict
```

**Strict Mode Requirements:**

When strict mode is enabled, the following fields become **required**:

1. **Data path must be explicit**

   ```yaml
   data:
     path: /absolute/path/to/data.csv # Required in strict mode
     # dataset: german_credit won't work in strict mode
   ```

2. **Data schema must be specified**

   ```yaml
   data:
     schema_path: /path/to/schema.yaml # Path to schema file
     # OR provide inline schema (future feature)
   ```

3. **Model path must be specified**

   ```yaml
   model:
     path: /path/to/trained_model.pkl # Pre-trained model required
     # Training from params not allowed in strict mode
   ```

4. **Explicit random seeds (no defaults)**
   ```yaml
   reproducibility:
     random_seed: 42 # Must be explicitly set
   ```

**When to use strict mode:**

- Regulatory submissions (FDA, ECOA, GDPR)
- Production deployments
- Audit documentation for compliance
- When byte-identical reproducibility is critical

**Example strict mode configuration:**

```yaml
# Direct configuration (no profiles needed)

reproducibility:
  random_seed: 42

data:
  path: /var/data/production_dataset.csv
  schema_path: /var/schemas/data_schema.yaml
  target_column: outcome
  protected_attributes:
    - gender
    - race

model:
  type: xgboost
  path: /var/models/prod_model_v1.2.pkl

explainers:
  strategy: first_compatible
  priority: [treeshap]

metrics:
  performance:
    metrics: [accuracy, precision, recall, f1, auc_roc]
  fairness:
    metrics: [demographic_parity, equal_opportunity]
```

**Note:** Strict mode is designed for production use where all artifacts (data, models, schemas) are pre-validated and versioned. For development and testing, use standard mode without the `--strict` flag.

### Manifest configuration

Controls audit trail generation and completeness.

```yaml
manifest:
  enabled: true # Generate audit manifest
  include_git_sha: true # Include Git commit information
  include_config_hash: true # Include configuration integrity hash
  include_data_hash: true # Include dataset integrity hash
  track_component_selection: true # Track selected components
  include_execution_info: true # Include timing and environment
```

### Preprocessing options

Controls data preprocessing before model training/evaluation.

```yaml
preprocessing:
  handle_missing: true # Handle missing values
  missing_strategy: median # median, mode, drop
  scale_features: false # Feature scaling (not needed for trees)
  scaling_method: standard # standard, minmax, robust
  categorical_encoding: label # label, onehot, target
  feature_selection: false # Enable feature selection
  selection_method: mutual_info # mutual_info, correlation
  max_features: 20 # Maximum features to select
```

### Validation configuration

Controls model evaluation and statistical testing.

```yaml
validation:
  cv_folds: 5 # Cross-validation folds
  cv_scoring: roc_auc # Scoring metric for CV
  test_size: 0.2 # Train/test split ratio
  stratify_split: true # Stratify split by target
  bootstrap_samples: 1000 # Bootstrap samples for confidence intervals
  confidence_level: 0.95 # Statistical confidence level
```

### Performance optimization

Controls computational performance and resource usage.

```yaml
performance:
  n_jobs: -1 # Parallel processing (-1 = all cores)
  low_memory_mode: false # Optimize for memory usage
  verbose: true # Enable progress reporting
  progress_bar: true # Show progress bars
```

## Configuration examples

### Basic German Credit audit

```yaml
audit_profile: german_credit_default

reproducibility:
  random_seed: 42

data:
  path: ~/.glassalpha/data/german_credit_processed.csv
  target_column: credit_risk
  protected_attributes:
    - gender
    - age_group

model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 100

explainers:
  strategy: first_compatible
  priority: [treeshap, kernelshap]

metrics:
  performance:
    metrics: [accuracy, precision, recall, f1, auc_roc]
  fairness:
    metrics: [demographic_parity, equal_opportunity]
```

### Enterprise compliance configuration

```yaml
# Direct configuration (no profiles needed)
strict_mode: true

reproducibility:
  random_seed: 42
  deterministic: true
  capture_environment: true

data:
  path: data/production_dataset.csv
  target_column: decision
  feature_columns:
    - income
    - employment_length
    - debt_to_income
    - credit_score
  protected_attributes:
    - race
    - gender
    - age_group

model:
  type: lightgbm
  params:
    objective: binary
    metric: auc
    num_leaves: 31
    feature_fraction: 0.9
    bagging_fraction: 0.8
    bagging_freq: 5

explainers:
  strategy: first_compatible
  priority: [treeshap, kernelshap]
  config:
    treeshap:
      max_samples: 10000
      check_additivity: true

metrics:
  performance:
    metrics: [accuracy, precision, recall, f1, auc_roc, classification_report]
  fairness:
    metrics:
      [demographic_parity, equal_opportunity, equalized_odds, predictive_parity]
    config:
      demographic_parity:
        threshold: 0.02 # Stricter threshold for production
      equal_opportunity:
        threshold: 0.02

manifest:
  enabled: true
  include_git_sha: true
  include_config_hash: true
  include_data_hash: true
  track_component_selection: true

report:
  template: standard_audit
  output_format: pdf
  styling:
    color_scheme: professional
    compliance_statement: true

compliance:
  frameworks: [gdpr, ecoa, fcra]
  fairness_thresholds:
    demographic_parity: 0.02
    equal_opportunity: 0.02
```

### Custom model configuration

```yaml
# Direct configuration (no profiles needed)

reproducibility:
  random_seed: 123

data:
  path: data/custom_dataset.csv
  target_column: target
  feature_columns:
    - numerical_feature_1
    - numerical_feature_2
    - categorical_feature_1
    - categorical_feature_2
  protected_attributes:
    - protected_attribute_1

model:
  type: logistic_regression
  params:
    C: 1.0
    penalty: l2
    solver: lbfgs
    max_iter: 1000

explainers:
  strategy: first_compatible
  priority: [kernelshap] # Use KernelSHAP for linear models
  config:
    kernelshap:
      n_samples: 1000
      background_size: 500

metrics:
  performance:
    metrics: [accuracy, precision, recall, f1]
  fairness:
    metrics: [demographic_parity]

preprocessing:
  handle_missing: true
  missing_strategy: median
  scale_features: true # Important for linear models
  scaling_method: standard
  categorical_encoding: onehot

validation:
  cv_folds: 10
  test_size: 0.3
  stratify_split: true
```

## Configuration best practices

### Reproducibility

1. **Always set random seeds** for deterministic results
2. **Use version control** for configuration files
3. **Document configuration changes** in commit messages
4. **Enable manifest generation** for complete audit trails

### Performance

1. **Use appropriate model types** for your data size and complexity
2. **Adjust sample sizes** for explainers based on dataset size
3. **Enable parallel processing** (`n_jobs: -1`) for faster computation
4. **Use appropriate metrics** - don't compute unnecessary evaluations

### Compliance

1. **Enable strict mode** for regulatory submissions
2. **Set appropriate bias thresholds** for your use case and jurisdiction
3. **Include all relevant protected attributes** in fairness analysis
4. **Document configuration rationale** for audit review

### Security

1. **Use relative paths** or environment variables for file locations
2. **Don't embed sensitive data** in configuration files
3. **Review configurations** before committing to version control
4. **Use appropriate access controls** for configuration repositories

## Troubleshooting configuration issues

## Configuring advanced metrics

### Fairness analysis with statistical confidence

Enable advanced fairness metrics with bootstrap confidence intervals:

```yaml
metrics:
  fairness:
    # Group fairness metrics
    metrics:
      - demographic_parity
      - equal_opportunity
      - predictive_parity

    # Statistical confidence intervals (E10)
    compute_confidence_intervals: true
    n_bootstrap: 1000 # Bootstrap samples (default: 1000)
    confidence_level: 0.95 # 95% CI (default)

    # Individual fairness (E11)
    individual_fairness:
      enabled: true
      distance_metric: euclidean # euclidean or mahalanobis
      similarity_percentile: 90 # Top 10% most similar pairs
      prediction_threshold: 0.10 # 10% difference threshold

# Intersectional analysis (E5.1)
data:
  intersections:
    - "gender*race" # 2-way intersections
    - "age*income"
```

**Learn more**: [Fairness Metrics Reference](../reference/fairness-metrics.md)

### Calibration analysis with confidence intervals

Enable calibration testing with statistical rigor:

```yaml
metrics:
  calibration:
    enabled: true

    # Binning strategy
    n_bins: 10 # Fixed bins (default)
    bin_strategy: fixed # fixed or adaptive

    # Confidence intervals (E10+)
    compute_confidence_intervals: true
    n_bootstrap: 1000
    confidence_level: 0.95

    # Bin-wise error bars
    compute_bin_wise_ci: true
```

**Learn more**: [Calibration Analysis](../reference/calibration.md)

### Robustness testing (adversarial perturbations)

Test model stability under input perturbations:

```yaml
metrics:
  stability:
    enabled: true

    # Epsilon perturbation levels (E6+)
    epsilon_values: [0.01, 0.05, 0.1] # 1%, 5%, 10% noise
    threshold: 0.15 # Max allowed prediction change (15%)
```

**Learn more**: [Robustness Testing](../reference/robustness.md)

### Dataset bias detection

Automatically runs when `protected_attributes` are specified:

```yaml
data:
  protected_attributes:
    - gender
    - race
    - age

  # Configure continuous attribute binning (E12)
  binning:
    age:
      strategy: domain_specific # domain_specific, custom, equal_width, equal_frequency
      bins: [18, 25, 35, 50, 65, 100] # Age range boundaries
```

**Learn more**: [Detecting Dataset Bias](../guides/dataset-bias.md)

### Demographic shift testing

Test robustness under population changes (CLI-only):

```bash
# Simulate demographic shift in CI/CD
glassalpha audit --config audit.yaml \
  --check-shift gender:+0.1 \
  --check-shift age:-0.05 \
  --fail-on-degradation 0.05
```

**Learn more**: [Testing Demographic Shifts](../guides/shift-testing.md)

### Common configuration errors

**Missing Required Fields:**

```yaml
# Error: Missing required field 'data.target_column'
data:
  path: data.csv
  # target_column: missing!
```

**Invalid Model Type:**

```yaml
# Error: Model type 'invalid_model' not found in registry
model:
  type: invalid_model # Should be: xgboost, lightgbm, etc.
```

**Incompatible Components:**

```yaml
# Warning: No compatible explainers for model type
explainers:
  priority: [treeshap] # TreeSHAP only works with tree models
model:
  type: logistic_regression # Linear model - use kernelshap instead
```

### Validation commands

```bash
# Validate configuration before running audit
glassalpha validate --config my_config.yaml

# Check strict mode compliance
glassalpha validate --config my_config.yaml --strict

# List available components
glassalpha list
```

### Configuration schema validation

GlassAlpha uses Pydantic for configuration validation with detailed error messages:

```bash
ValidationError: 2 validation errors for AuditConfig
data.target_column
  field required (type=value_error.missing)
model.type
  ensure this value has at least 1 characters (type=value_error.any_str.min_length; limit_value=1)
```

## Schema reference

For detailed technical information about GlassAlpha's architecture and interfaces, see the [Trust & Deployment Guide](../reference/trust-deployment.md).

This configuration guide provides the foundation for creating effective, compliant audit configurations. Start with the provided examples and customize based on your specific requirements, data characteristics, and regulatory context.
