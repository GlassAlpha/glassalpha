# Quick Reference Card

Fast reference for common GlassAlpha commands, API patterns, and configurations.

## CLI Commands (Most Common)

| Command                    | Description                     | Example                                                           |
| -------------------------- | ------------------------------- | ----------------------------------------------------------------- |
| `glassalpha audit`         | Generate full audit report      | `glassalpha audit --config audit.yaml --output report.pdf --fast` |
| `glassalpha validate`      | Validate config without running | `glassalpha validate --config audit.yaml`                         |
| `glassalpha list`          | List available components       | `glassalpha list`                                                 |
| `glassalpha datasets list` | Show built-in datasets          | `glassalpha datasets list`                                        |
| `glassalpha datasets info` | Dataset details                 | `glassalpha datasets info german_credit`                          |
| `glassalpha models`        | Show available models           | `glassalpha models`                                               |
| `glassalpha --version`     | Check installed version         | `glassalpha --version`                                            |
| `glassalpha --help`        | Show all commands               | `glassalpha --help`                                               |

## Common Flags

| Flag                 | Purpose                         | Usage                                                     |
| -------------------- | ------------------------------- | --------------------------------------------------------- |
| `--fast`             | Quick demo mode (100 bootstrap) | `glassalpha audit --config audit.yaml --fast`             |
| `--strict`           | Enforce regulatory mode         | `glassalpha audit --config audit.yaml --strict`           |
| `--explain-failures` | Verbose error messages          | `glassalpha audit --config audit.yaml --explain-failures` |
| `--dry-run`          | Validate without executing      | `glassalpha audit --config audit.yaml --dry-run`          |
| `--verbose`          | Detailed logging                | `glassalpha --verbose audit --config audit.yaml`          |

## Python API (Notebooks)

### Basic Usage

```python
import glassalpha as ga

# Generate audit from trained model
result = ga.audit.from_model(
    model=model,                    # Your trained model
    X=X_test,                       # Test features
    y=y_test,                       # Test labels
    protected_attributes={          # Protected attributes for fairness
        "gender": X_test["gender"],
        "age": X_test["age"]
    },
    random_seed=42                  # For reproducibility
)

# View inline in Jupyter
result  # Auto-displays HTML summary

# Export to PDF
result.to_pdf("audit.pdf")
```

### All Parameters

```python
result = ga.audit.from_model(
    model,                          # Required: trained model
    X,                              # Required: test features
    y,                              # Required: test labels
    protected_attributes,           # Required: dict of protected attrs
    random_seed=42,                 # Recommended: for reproducibility
    threshold=0.5,                  # Decision threshold
    explainer_samples=1000,         # SHAP background samples (100 for speed)
    fairness_threshold=0.10,        # Max acceptable disparity
    compute_calibration=True,       # Include calibration analysis
    compute_individual_fairness=True,  # Individual consistency checks
    strict_mode=False               # Enforce regulatory requirements
)
```

### Access Results

```python
# Performance metrics
print(f"Accuracy: {result.performance['accuracy']:.3f}")
print(f"AUC: {result.performance['auc_roc']:.3f}")

# Fairness metrics
print(f"Demographic parity: {result.fairness['demographic_parity_difference']:.3f}")
print(f"Equal opportunity: {result.fairness['equal_opportunity_difference']:.3f}")

# Calibration
print(f"ECE: {result.calibration['expected_calibration_error']:.3f}")

# Plot results
result.fairness.plot_group_metrics()
result.calibration.plot()
```

## Configuration Patterns

### 1. Minimal Config (Quick Start)

```yaml
audit_profile: tabular_compliance

reproducibility:
  random_seed: 42

data:
  dataset: german_credit # Use built-in dataset
  target_column: credit_risk
  protected_attributes:
    - gender
    - age_group

model:
  type: logistic_regression
  params:
    random_state: 42
```

### 2. Fairness-Focused Config

```yaml
audit_profile: tabular_compliance

reproducibility:
  random_seed: 42

data:
  path: data/credit_data.csv
  target_column: approved
  protected_attributes:
    - gender
    - race
    - age_group

  # Intersectional analysis
  intersections:
    - "gender*race"

metrics:
  fairness:
    metrics:
      - demographic_parity
      - equal_opportunity
      - predictive_parity

    # Confidence intervals
    compute_confidence_intervals: true
    n_bootstrap: 1000
    confidence_level: 0.95

    # Individual fairness
    individual_fairness:
      enabled: true
      consistency_threshold: 0.05

model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 100
```

### 3. Performance-Focused Config

```yaml
audit_profile: tabular_compliance

reproducibility:
  random_seed: 42

data:
  path: data/large_dataset.csv
  target_column: outcome
  protected_attributes:
    - gender

model:
  type: lightgbm # Fast for large data
  params:
    objective: binary
    n_estimators: 100

explainers:
  strategy: first_compatible
  priority:
    - treeshap
  config:
    treeshap:
      background_samples: 100 # Faster iteration
```

### 4. Strict Mode (Regulatory Compliance)

```yaml
audit_profile: tabular_compliance

reproducibility:
  random_seed: 42

data:
  path: data/production_data.csv
  schema: data/schema.yaml # Required in strict mode
  target_column: decision
  protected_attributes:
    - gender
    - race

model:
  path: models/production_model.pkl
  type: xgboost

# Policy gates
policy_gates: configs/policy/sr_11_7_gates.yaml

# Evidence pack
evidence_pack:
  enabled: true
  output: evidence_pack.zip

strict_mode: true # Enforce all requirements
```

### 5. Custom Data Template

```yaml
audit_profile: tabular_compliance

reproducibility:
  random_seed: 42

data:
  dataset: custom # Important for custom data
  path: /path/to/your/data.csv
  target_column: your_target_column
  protected_attributes:
    - attribute1
    - attribute2

  # Optional: preprocessing artifact
  preprocessing:
    artifact_path: preprocessing_pipeline.joblib

model:
  type: xgboost # or logistic_regression, lightgbm
  params:
    objective: binary:logistic
    n_estimators: 100
    max_depth: 5
    random_state: 42

explainers:
  strategy: first_compatible
  priority:
    - treeshap # For tree models
    - coefficients # For linear models

metrics:
  performance:
    metrics:
      - accuracy
      - precision
      - recall
      - f1
      - auc_roc

  fairness:
    metrics:
      - demographic_parity
      - equal_opportunity
```

## Troubleshooting Quick Checks

### 1. Config Validation Fails

```bash
# Validate before running
glassalpha validate --config audit.yaml

# Check for required fields
# - data.path or data.dataset
# - data.target_column
# - data.protected_attributes
# - model.type
```

### 2. Model Not Found

```bash
# Check available models
glassalpha models

# Install model dependencies
pip install 'glassalpha[explain]'  # For XGBoost, LightGBM, SHAP
```

### 3. Dataset Loading Error

```bash
# Check dataset exists
ls -la data/your_data.csv

# Use built-in datasets for testing
glassalpha datasets list
glassalpha audit --config configs/german_credit_simple.yaml --fast
```

### 4. Slow Audit Performance

```yaml
# In your config, reduce samples:
explainers:
  config:
    treeshap:
      background_samples: 100 # Default 1000

# Or skip slow sections in dev
calibration:
  enabled: false
```

### 5. Protected Attributes Not Found

```python
# Debug in Python:
import pandas as pd
df = pd.read_csv("data.csv")
print(df.columns.tolist())
print(df.dtypes)

# Ensure column names match exactly (case-sensitive)
```

## Next Steps

- **Full guide**: [Quick Start](quickstart.md)
- **Your own data**: [Using Custom Data](custom-data.md)
- **All options**: [Configuration Reference](configuration.md)
- **Troubleshooting**: [Common Issues](../reference/troubleshooting.md)
- **API docs**: [from_model() Reference](../reference/api/api-audit.md)
