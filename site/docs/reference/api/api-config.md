# Configuration API reference

Programmatic API for creating, validating, and managing audit configurations.

## Quick Start

```python
import glassalpha as ga

# Method 1: Load from YAML file
config = ga.config.load("audit_config.yaml", strict=True)

# Method 2: Build programmatically
config = ga.config.AuditConfig(
    model=ga.config.ModelConfig(
        path="model.joblib",
        type="xgboost"
    ),
    data=ga.config.DataConfig(
        test_data="test.csv",
        target_column="approved"
    ),
    random_seed=42
)

# Validate configuration
ga.config.validate(config)

# Run audit with config
result = ga.audit.from_config(config)
```

---

## `glassalpha.config.load()`

Load and validate audit configuration from YAML file.

### Signature

```python
def load(
    config_path: str | Path,
    strict: bool = False,
    model: dict | None = None
) -> AuditConfig
```

### Parameters

| Parameter     | Type            | Description                                                                                                                                                   |
| ------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config_path` | `str` or `Path` | **Required**. Path to YAML configuration file                                                                                                                 |
| `strict`      | `bool`          | **Optional**. Enable strict validation mode (default: `False`). When `True`, enforces regulatory requirements: explicit seeds, schema validation, no defaults |
| `model`       | `dict`          | **Optional**. Override model configuration from config file (e.g., `{"type": "xgboost"}`)                                                                     |

### Returns

**`AuditConfig`** object with validated configuration.

### Examples

```python
# Basic load
config = ga.config.load("audit.yaml")

# Strict mode (for regulatory submission)
config = ga.config.load("audit.yaml", strict=True)

# Load with model override
config = ga.config.load("audit.yaml", model={"type": "lightgbm"})
```

### Raises

| Exception           | When                      | How to Fix                              |
| ------------------- | ------------------------- | --------------------------------------- |
| `FileNotFoundError` | Config file doesn't exist | Check path and ensure file exists       |
| `ValidationError`   | Invalid YAML syntax       | Validate YAML syntax                    |
| `ValidationError`   | Missing required fields   | Add required fields (see error message) |
| `ValidationError`   | Strict mode violations    | Add explicit seed, schema, version pins |

---

## `glassalpha.config.AuditConfig`

Configuration object defining audit parameters.

### Constructor

```python
class AuditConfig:
    def __init__(
        self,
        model: ModelConfig,
        data: DataConfig,
        random_seed: int | None = None,
        explainer: ExplainerConfig | None = None,
        fairness: FairnessConfig | None = None,
        calibration: CalibrationConfig | None = None,
        policy: PolicyConfig | None = None,
        output: OutputConfig | None = None,
        metadata: dict[str, Any] | None = None
    )
```

### Attributes

| Attribute     | Type                | Description                                                                                          |
| ------------- | ------------------- | ---------------------------------------------------------------------------------------------------- |
| `model`       | `ModelConfig`       | **Required**. Model configuration (path, type, parameters)                                           |
| `data`        | `DataConfig`        | **Required**. Data configuration (paths, schema, protected attributes)                               |
| `random_seed` | `int`               | **Optional**. Random seed for reproducibility (default: `None`). Always set for deterministic audits |
| `explainer`   | `ExplainerConfig`   | **Optional**. Explainer settings (strategy, samples, features)                                       |
| `fairness`    | `FairnessConfig`    | **Optional**. Fairness analysis settings (thresholds, metrics)                                       |
| `calibration` | `CalibrationConfig` | **Optional**. Calibration settings (bins, confidence intervals)                                      |
| `policy`      | `PolicyConfig`      | **Optional**. Policy gates and compliance rules                                                      |
| `output`      | `OutputConfig`      | **Optional**. Output format and paths                                                                |
| `metadata`    | `dict`              | **Optional**. Custom metadata (project name, auditor, etc.)                                          |

### Methods

#### `to_yaml(filepath)`

Export configuration to YAML file.

```python
config.to_yaml("audit_config.yaml")
```

#### `to_dict()`

Convert configuration to dictionary.

```python
config_dict = config.to_dict()
```

#### `validate(strict=False)`

Validate configuration completeness and correctness.

```python
config.validate(strict=True)  # Raises ValidationError if invalid
```

### Example

```python
config = ga.config.AuditConfig(
    model=ga.config.ModelConfig(
        path="models/xgb_model.joblib",
        type="xgboost"
    ),
    data=ga.config.DataConfig(
        test_data="data/test.csv",
        target_column="approved",
        protected_attributes=["gender", "race"]
    ),
    random_seed=42,
    policy=ga.config.PolicyConfig(
        gates={
            "min_accuracy": 0.70,
            "max_bias": 0.10
        }
    ),
    metadata={
        "project": "Credit Model Q1 2025",
        "auditor": "Risk Team"
    }
)

config.validate(strict=True)
result = ga.audit.from_config(config)
```

---

## Sub-Configuration Classes

### `ModelConfig`

Model loading and configuration.

**Attributes:**

| Attribute       | Type                  | Description                                                                                     |
| --------------- | --------------------- | ----------------------------------------------------------------------------------------------- |
| `path`          | `str`                 | **Required**. Path to serialized model file (`.joblib`, `.pkl`)                                 |
| `type`          | `str`                 | **Required**. Model type: `"xgboost"`, `"lightgbm"`, `"random_forest"`, `"logistic_regression"` |
| `preprocessing` | `PreprocessingConfig` | **Optional**. Preprocessing pipeline configuration                                              |
| `version`       | `str`                 | **Optional**. Model version identifier                                                          |

**Example:**

```python
model_config = ga.config.ModelConfig(
    path="models/xgb_v2.3.joblib",
    type="xgboost",
    version="2.3.0",
    preprocessing=ga.config.PreprocessingConfig(
        path="preprocessing/pipeline.joblib"
    )
)
```

### `DataConfig`

Data loading and schema configuration.

**Attributes:**

| Attribute              | Type        | Description                                                |
| ---------------------- | ----------- | ---------------------------------------------------------- |
| `test_data`            | `str`       | **Required**. Path to test data (`.csv`, `.parquet`)       |
| `target_column`        | `str`       | **Required**. Name of target column                        |
| `feature_columns`      | `list[str]` | **Optional**. Explicit feature list (if not all columns)   |
| `protected_attributes` | `list[str]` | **Optional**. Columns to use for fairness analysis         |
| `schema`               | `str`       | **Optional**. Path to schema file (`.yaml`) for validation |
| `index_column`         | `str`       | **Optional**. Column to use as index                       |

**Example:**

```python
data_config = ga.config.DataConfig(
    test_data="data/test_set.csv",
    target_column="approved",
    feature_columns=["income", "debt_ratio", "credit_score"],
    protected_attributes=["gender", "race", "age_group"],
    schema="schemas/credit_data.yaml"
)
```

### `ExplainerConfig`

Explainability method configuration.

**Attributes:**

| Attribute             | Type        | Description                                                                           |
| --------------------- | ----------- | ------------------------------------------------------------------------------------- |
| `strategy`            | `str`       | Selection strategy: `"first_compatible"`, `"fastest"` (default: `"first_compatible"`) |
| `priority`            | `list[str]` | Ordered list of explainers to try: `"treeshap"`, `"kernelshap"`, `"linear"`           |
| `max_samples`         | `int`       | Background samples for SHAP (default: `1000`)                                         |
| `features_to_explain` | `int`       | Max features to include in explanations (default: all)                                |

**Example:**

```python
explainer_config = ga.config.ExplainerConfig(
    strategy="first_compatible",
    priority=["treeshap", "kernelshap"],
    max_samples=500,
    features_to_explain=15
)
```

### `FairnessConfig`

Fairness analysis configuration.

**Attributes:**

| Attribute          | Type        | Description                                                                           |
| ------------------ | ----------- | ------------------------------------------------------------------------------------- |
| `threshold`        | `float`     | Decision threshold for fairness metrics (default: `0.5`)                              |
| `tolerance`        | `float`     | Acceptable bias threshold (default: `0.05` = 5%)                                      |
| `min_group_size`   | `int`       | Minimum group size for analysis (default: `30`)                                       |
| `metrics`          | `list[str]` | Metrics to compute: `"demographic_parity"`, `"equal_opportunity"`, `"equalized_odds"` |
| `confidence_level` | `float`     | Confidence level for intervals (default: `0.95`)                                      |

**Example:**

```python
fairness_config = ga.config.FairnessConfig(
    threshold=0.5,
    tolerance=0.10,  # 10% tolerance
    min_group_size=50,
    metrics=["demographic_parity", "equal_opportunity"],
    confidence_level=0.95
)
```

### `CalibrationConfig`

Calibration analysis configuration.

**Attributes:**

| Attribute            | Type    | Description                                                        |
| -------------------- | ------- | ------------------------------------------------------------------ |
| `n_bins`             | `int`   | Number of calibration bins (default: `10`)                         |
| `strategy`           | `str`   | Binning strategy: `"uniform"`, `"quantile"` (default: `"uniform"`) |
| `compute_confidence` | `bool`  | Compute confidence intervals (default: `True`)                     |
| `confidence_level`   | `float` | Confidence level (default: `0.95`)                                 |

**Example:**

```python
calibration_config = ga.config.CalibrationConfig(
    n_bins=10,
    strategy="uniform",
    compute_confidence=True,
    confidence_level=0.95
)
```

### `PolicyConfig`

Policy-as-code gates and compliance rules.

**Attributes:**

| Attribute           | Type               | Description                                               |
| ------------------- | ------------------ | --------------------------------------------------------- |
| `gates`             | `dict[str, float]` | Policy gates as key-value thresholds                      |
| `citations`         | `dict[str, str]`   | Citations for each gate (regulation reference)            |
| `fail_on_violation` | `bool`             | Exit with error code on gate violation (default: `False`) |

**Example:**

```python
policy_config = ga.config.PolicyConfig(
    gates={
        "min_accuracy": 0.70,
        "max_bias": 0.10,
        "max_ece": 0.05,
        "min_auc": 0.75
    },
    citations={
        "min_accuracy": "SR 11-7 Section 4.2",
        "max_bias": "ECOA Regulation B",
        "max_ece": "Internal Policy v2.1"
    },
    fail_on_violation=True
)
```

### `OutputConfig`

Output format and destination configuration.

**Attributes:**

| Attribute       | Type   | Description                                     |
| --------------- | ------ | ----------------------------------------------- |
| `pdf_path`      | `str`  | Path for PDF report output                      |
| `json_path`     | `str`  | Path for JSON metrics output                    |
| `manifest_path` | `str`  | Path for provenance manifest                    |
| `template`      | `str`  | Report template (default: `"standard"`)         |
| `include_plots` | `bool` | Include visualizations in PDF (default: `True`) |

**Example:**

```python
output_config = ga.config.OutputConfig(
    pdf_path="reports/audit_2025-01.pdf",
    json_path="reports/metrics_2025-01.json",
    manifest_path="reports/manifest_2025-01.json",
    template="standard",
    include_plots=True
)
```

---

## Configuration Structure

GlassAlpha uses direct configuration without profiles. Specify model, explainer, and metric settings explicitly in your YAML config file.

### Basic Configuration Structure

```yaml
model:
  type: xgboost # or "lightgbm", "sklearn"
  params:
    objective: binary:logistic
    n_estimators: 100

explainers:
  strategy: first_compatible # or "all", "priority"
  priority: [treeshap, kernelshap]

metrics:
  performance:
    metrics: [accuracy, precision, recall, f1, auc_roc]
  fairness:
    protected_attributes: [gender, race, age]
    metrics: [demographic_parity, equal_opportunity]

reproducibility:
  random_seed: 42
```

### Model Types

| Model Type | Description                   | Use Case                           |
| ---------- | ----------------------------- | ---------------------------------- |
| `xgboost`  | XGBoost with TreeSHAP support | Production ML systems, tree models |
| `lightgbm` | LightGBM integration          | Large datasets, performance        |
| `sklearn`  | Scikit-learn models           | Baseline, linear models            |

---

## Validation

### `glassalpha.config.validate()`

Validate configuration object or file.

```python
def validate(
    config: AuditConfig | str | Path,
    strict: bool = False
) -> bool
```

**Parameters:**

- `config`: `AuditConfig` object or path to YAML file
- `strict`: Enable strict validation (regulatory mode)

**Returns:** `True` if valid

**Raises:** `ValidationError` with detailed message if invalid

### Validation rules

#### Standard mode

- Required fields present (`model`, `data`)
- Valid model type
- Data files exist
- Schema matches data (if schema provided)

#### Strict mode (regulatory)

All standard checks plus:

- Explicit `random_seed` (no default)
- Schema file required
- Model version required
- All package versions pinned
- Git commit tracked
- Protected attributes declared

### Example

```python
# Validate config object
config = ga.config.AuditConfig(...)
ga.config.validate(config, strict=True)

# Validate YAML file
ga.config.validate("audit.yaml", strict=True)

# Catch validation errors
try:
    ga.config.validate(config, strict=True)
except ga.ValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Fix: {e.fix_suggestion}")
```

---

## Full Example: Programmatic Audit

```python
import glassalpha as ga

# Build configuration programmatically
config = ga.config.AuditConfig(
    model=ga.config.ModelConfig(
        path="models/credit_model.joblib",
        type="xgboost",
        version="1.2.0",
        preprocessing=ga.config.PreprocessingConfig(
            path="preprocessing/pipeline.joblib"
        )
    ),
    data=ga.config.DataConfig(
        test_data="data/test_q1_2025.csv",
        target_column="approved",
        feature_columns=[
            "income", "debt_ratio", "credit_score",
            "employment_length", "loan_amount"
        ],
        protected_attributes=["gender", "race", "age_group"],
        schema="schemas/credit_application_v2.yaml"
    ),
    random_seed=42,
    explainer=ga.config.ExplainerConfig(
        strategy="first_compatible",
        priority=["treeshap", "kernelshap"],
        max_samples=1000
    ),
    fairness=ga.config.FairnessConfig(
        threshold=0.5,
        tolerance=0.05,
        min_group_size=30,
        metrics=["demographic_parity", "equal_opportunity"],
        confidence_level=0.95
    ),
    calibration=ga.config.CalibrationConfig(
        n_bins=10,
        strategy="uniform",
        compute_confidence=True
    ),
    policy=ga.config.PolicyConfig(
        gates={
            "min_accuracy": 0.70,
            "max_bias": 0.10,
            "max_ece": 0.05
        },
        citations={
            "min_accuracy": "SR 11-7 Section 4.2",
            "max_bias": "ECOA Regulation B"
        },
        fail_on_violation=True
    ),
    output=ga.config.OutputConfig(
        pdf_path="reports/credit_audit_2025-q1.pdf",
        json_path="reports/metrics_2025-q1.json",
        manifest_path="reports/manifest_2025-q1.json",
        template="financial_services"
    ),
    metadata={
        "project": "Credit Model Q1 2025 Audit",
        "auditor": "Jane Smith",
        "department": "Model Risk Management",
        "regulatory_submission": True
    }
)

# Validate configuration
config.validate(strict=True)

# Export configuration for review
config.to_yaml("audit_config_2025-q1.yaml")

# Run audit
result = ga.audit.from_config(config)

# Check policy gates
if result.policy_decision.failed():
    print("❌ Policy gates failed:")
    for gate, value in result.policy_decision.violations.items():
        print(f"  {gate}: {value}")
    exit(1)

# Export audit report
result.to_pdf("audit_report.pdf")

print("✅ Audit complete and compliant")

# Note: Automated evidence pack export planned for future release
```

---

## Configuration file format (YAML)

### Minimal example

```yaml
# Minimal audit configuration
model:
  path: model.joblib
  type: xgboost

data:
  test_data: test.csv
  target_column: approved

random_seed: 42
```

### Full example

```yaml
# Complete audit configuration
model:
  path: models/credit_model.joblib
  type: xgboost
  version: "1.2.0"
  preprocessing:
    path: preprocessing/pipeline.joblib

data:
  test_data: data/test_q1_2025.csv
  target_column: approved
  feature_columns:
    - income
    - debt_ratio
    - credit_score
    - employment_length
    - loan_amount
  protected_attributes:
    - gender
    - race
    - age_group
  schema: schemas/credit_application_v2.yaml

random_seed: 42

explainer:
  strategy: first_compatible
  priority:
    - treeshap
    - kernelshap
  max_samples: 1000

fairness:
  threshold: 0.5
  tolerance: 0.05
  min_group_size: 30
  metrics:
    - demographic_parity
    - equal_opportunity
  confidence_level: 0.95

calibration:
  n_bins: 10
  strategy: uniform
  compute_confidence: true

policy:
  gates:
    min_accuracy: 0.70
    max_bias: 0.10
    max_ece: 0.05
  citations:
    min_accuracy: "SR 11-7 Section 4.2"
    max_bias: "ECOA Regulation B"
  fail_on_violation: true

report:
  template: standard_audit
  output_format: pdf

manifest:
  enabled: true
  manifest_path: reports/manifest_2025-q1.json
  template: financial_services
  include_plots: true

metadata:
  project: "Credit Model Q1 2025 Audit"
  auditor: "Jane Smith"
  department: "Model Risk Management"
  regulatory_submission: true
```

---

## Related Documentation

- **[Audit API](audit-entry-points.md)** - `from_model()` API for notebooks
- **[Pipeline API](api-pipeline.md)** - Lower-level pipeline interface
- **[CLI Reference](../cli.md)** - Command-line configuration
- **[Configuration Guide](../../getting-started/configuration.md)** - Configuration best practices
- **[ML Engineer Workflow](../../guides/ml-engineer-workflow.md)** - Production configuration

## Support

- **GitHub Issues**: [Report bugs](https://github.com/GlassAlpha/glassalpha/issues)
- **Discussions**: [Ask questions](https://github.com/GlassAlpha/glassalpha/discussions)
- **Email**: [contact@glassalpha.com](mailto:contact@glassalpha.com)
