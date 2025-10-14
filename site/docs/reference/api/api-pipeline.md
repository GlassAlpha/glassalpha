# Pipeline API Reference

Lower-level API for advanced audit pipeline control and customization.

## Quick Start

```python
import glassalpha as ga

# Method 1: High-level from_model() (recommended for notebooks)
result = ga.audit.from_model(model, X_test, y_test, random_seed=42)

# Method 2: Configuration-based (CLI equivalent)
config = ga.config.load("audit.yaml")
result = ga.audit.from_config(config)

# Method 3: Pipeline (advanced customization)
pipeline = ga.AuditPipeline(config)
pipeline.register_hook("pre_explain", my_custom_validator)
result = pipeline.run()
```

**When to use each:**

- **`from_model()`**: Notebooks, quick exploration, simple use cases
- **`audit.run(config)`**: Production, CI/CD, reproducible audits
- **`AuditPipeline`**: Custom hooks, extensions, advanced control

---

## `glassalpha.audit.run()`

Run audit from configuration object or file.

### Signature

```python
def run(config: AuditConfig | str | Path) -> AuditResult
```

### Parameters

| Parameter | Type                             | Description                                                    |
| --------- | -------------------------------- | -------------------------------------------------------------- |
| `config`  | `AuditConfig` or `str` or `Path` | **Required**. `AuditConfig` object or path to YAML config file |

### Returns

**`AuditResult`** object with all audit results.

### Example

```python
import glassalpha as ga

# From config file
result = ga.audit.from_config("audit.yaml")

# From config object
config = ga.config.AuditConfig(...)
result = ga.audit.from_config(config)

# Export results
result.to_pdf("report.pdf")
```

### Raises

| Exception           | When                                      | How to Fix                         |
| ------------------- | ----------------------------------------- | ---------------------------------- |
| `FileNotFoundError` | Config file or model/data files not found | Check paths in configuration       |
| `ValidationError`   | Invalid configuration                     | Fix configuration errors           |
| `ModelLoadError`    | Cannot load model                         | Check model file and compatibility |
| `DataLoadError`     | Cannot load data                          | Check data file format and schema  |

---

## `glassalpha.AuditPipeline`

Advanced audit pipeline with extensibility hooks.

### Constructor

```python
class AuditPipeline:
    def __init__(
        self,
        config: AuditConfig,
        verbose: bool = False,
        progress: bool = True
    )
```

**Parameters:**

- `config` (`AuditConfig`): Configuration object
- `verbose` (`bool`): Enable detailed logging (default: `False`)
- `progress` (`bool`): Show progress bars (default: `True`)

### Methods

#### `run()`

Execute complete audit pipeline.

```python
def run() -> AuditResult
```

**Returns:** `AuditResult` object

**Example:**

```python
config = ga.config.load("audit.yaml")
pipeline = ga.AuditPipeline(config, verbose=True)
result = pipeline.run()
```

#### `register_hook(hook_name, callback)`

Register custom callback for pipeline extension points.

```python
def register_hook(
    hook_name: str,
    callback: Callable
) -> None
```

**Parameters:**

- `hook_name` (`str`): Hook name (see [Available Hooks](#available-hooks))
- `callback` (`Callable`): Function to call at hook point

**Example:**

```python
def validate_predictions(pipeline_state):
    """Custom validation before explanation"""
    predictions = pipeline_state["predictions"]
    assert predictions.min() >= 0, "Negative predictions detected"
    assert predictions.max() <= 1, "Predictions exceed 1.0"

pipeline = ga.AuditPipeline(config)
pipeline.register_hook("pre_explain", validate_predictions)
result = pipeline.run()
```

#### `run_stage(stage_name)`

Execute single pipeline stage (for debugging/testing).

```python
def run_stage(stage_name: str) -> dict
```

**Parameters:**

- `stage_name` (`str`): Stage to run: `"load"`, `"predict"`, `"explain"`, `"fairness"`, `"calibration"`, `"report"`

**Returns:** Dictionary with stage results

**Example:**

```python
pipeline = ga.AuditPipeline(config)

# Run only prediction stage
predictions = pipeline.run_stage("predict")

# Run only explanation stage
explanations = pipeline.run_stage("explain")
```

### Available Hooks

| Hook Name          | When Called                   | Use Case                         |
| ------------------ | ----------------------------- | -------------------------------- |
| `pre_load`         | Before loading model/data     | Custom validation, preprocessing |
| `post_load`        | After loading model/data      | Model compatibility checks       |
| `pre_predict`      | Before generating predictions | Feature validation               |
| `post_predict`     | After generating predictions  | Prediction sanity checks         |
| `pre_explain`      | Before explanation generation | Explainer selection logic        |
| `post_explain`     | After explanation generation  | Custom explanation processing    |
| `pre_fairness`     | Before fairness analysis      | Group validation                 |
| `post_fairness`    | After fairness analysis       | Custom fairness metrics          |
| `pre_calibration`  | Before calibration analysis   | Threshold validation             |
| `post_calibration` | After calibration analysis    | Custom calibration metrics       |
| `pre_report`       | Before report generation      | Custom report sections           |
| `post_report`      | After report generation       | Report post-processing           |

### Hook Signature

All hooks receive `pipeline_state` dictionary:

```python
def my_hook(pipeline_state: dict) -> None:
    """
    pipeline_state contains:
        config: AuditConfig
        model: Loaded model object
        X_test: Test features
        y_test: Test labels
        predictions: Model predictions (if available)
        explanations: Explanation results (if available)
        fairness_results: Fairness metrics (if available)
        calibration_results: Calibration metrics (if available)
    """
    # Your custom logic here
    pass
```

### Example: Custom Fairness Metric

```python
import numpy as np

def compute_custom_metric(pipeline_state):
    """Compute custom fairness metric: max ratio of FPR across groups"""
    fairness = pipeline_state["fairness_results"]

    fprs = [group["fpr"] for group in fairness["groups"].values()]
    max_fpr_ratio = max(fprs) / min(fprs)

    # Add to results
    fairness["custom_metrics"] = {
        "max_fpr_ratio": max_fpr_ratio
    }

    # Validate
    if max_fpr_ratio > 2.0:
        print(f"⚠️ WARNING: Max FPR ratio = {max_fpr_ratio:.2f} (threshold: 2.0)")

pipeline = ga.AuditPipeline(config)
pipeline.register_hook("post_fairness", compute_custom_metric)
result = pipeline.run()
```

### Example: Custom Report Section

```python
def add_executive_summary(pipeline_state):
    """Add custom executive summary section"""
    result = pipeline_state["audit_result"]

    summary = {
        "title": "Executive Summary",
        "content": {
            "model_version": pipeline_state["config"].model.version,
            "test_samples": len(pipeline_state["y_test"]),
            "accuracy": result.performance['accuracy'],
            "bias_detected": result.fairness.has_bias(threshold=0.10),
            "recommendation": "APPROVE" if not result.fairness.has_bias() else "REVIEW"
        }
    }

    # Add to report sections
    pipeline_state["report_sections"].insert(0, summary)

pipeline = ga.AuditPipeline(config)
pipeline.register_hook("pre_report", add_executive_summary)
result = pipeline.run()
```

---

## Advanced Workflows

### Batch Processing

Process multiple models with single configuration.

```python
import glassalpha as ga

models = {
    "LogisticRegression": "models/lr_v1.joblib",
    "RandomForest": "models/rf_v1.joblib",
    "XGBoost": "models/xgb_v1.joblib"
}

base_config = ga.config.load("base_audit.yaml")

results = {}
for name, model_path in models.items():
    # Update config for this model
    config = base_config.copy()
    config.model.path = model_path
    config.model.type = name.lower()
    config.output.pdf_path = f"reports/audit_{name}.pdf"

    # Run audit
    results[name] = ga.audit.from_config(config)

# Compare models
for name, result in results.items():
    print(f"{name}:")
    print(f"  Accuracy: {result.performance['accuracy']:.3f}")
    print(f"  Bias: {result.fairness.demographic_parity_difference:.3f}")
    print(f"  ECE: {result.calibration.expected_calibration_error:.3f}")
```

### Parallel Execution

Run audits in parallel for faster batch processing.

```python
from concurrent.futures import ProcessPoolExecutor
import glassalpha as ga

def run_single_audit(config_path: str) -> dict:
    """Run single audit and return metrics"""
    result = ga.audit.from_config(config_path)
    return {
        "config": config_path,
        "accuracy": result.performance['accuracy'],
        "bias": result.fairness.demographic_parity_difference,
        "ece": result.calibration.expected_calibration_error
    }

# Prepare configs
configs = [
    "audits/model_v1.yaml",
    "audits/model_v2.yaml",
    "audits/model_v3.yaml"
]

# Run in parallel (max 4 workers)
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_single_audit, configs))

# Aggregate results
import pandas as pd
df = pd.DataFrame(results)
print(df.to_string())
```

### Custom Explainer Selection

Override explainer selection logic.

```python
import glassalpha as ga
from glassalpha.explain import TreeSHAPExplainer, KernelSHAPExplainer

def select_explainer(pipeline_state):
    """Custom explainer selection based on model size"""
    model = pipeline_state["model"]
    X_test = pipeline_state["X_test"]

    # Use TreeSHAP for tree models, KernelSHAP for others
    if hasattr(model, "get_booster"):  # XGBoost
        explainer = TreeSHAPExplainer(model, X_test[:500])
    else:
        explainer = KernelSHAPExplainer(model, X_test[:100])

    pipeline_state["explainer"] = explainer
    print(f"Selected: {explainer.__class__.__name__}")

config = ga.config.load("audit.yaml")
pipeline = ga.AuditPipeline(config)
pipeline.register_hook("pre_explain", select_explainer)
result = pipeline.run()
```

### Conditional Fairness Analysis

Skip fairness analysis based on custom logic.

```python
def conditional_fairness(pipeline_state):
    """Only run fairness if protected attributes are well-represented"""
    data_config = pipeline_state["config"].data
    X_test = pipeline_state["X_test"]

    # Check group sizes
    for attr in data_config.protected_attributes:
        values = X_test[attr].value_counts()
        if values.min() < 30:
            print(f"⚠️ Skipping fairness: {attr} has group with n < 30")
            pipeline_state["skip_fairness"] = True
            return

    pipeline_state["skip_fairness"] = False

pipeline = ga.AuditPipeline(config)
pipeline.register_hook("pre_fairness", conditional_fairness)
result = pipeline.run()
```

---

## Error Handling

### Graceful Degradation

Handle missing features gracefully.

```python
import glassalpha as ga

config = ga.config.load("audit.yaml")
pipeline = ga.AuditPipeline(config)

try:
    result = pipeline.run()
except ga.ExplainerNotAvailableError:
    print("⚠️ SHAP not available, running without explanations")
    config.explainer = None
    result = pipeline.run()

# Check what was computed
if result.explanations is None:
    print("Audit completed without explanations")
else:
    print(f"Explanations generated: {result.explanations.feature_importance.head()}")
```

### Retry with Fallback

Retry audit with fallback configuration on failure.

```python
import glassalpha as ga

def run_with_fallback(config_path: str) -> ga.AuditResult:
    """Run audit with automatic fallback on failure"""
    try:
        # Try full audit
        result = ga.audit.from_config(config_path)
        return result
    except ga.ExplainerError:
        # Fallback: disable explanations
        print("⚠️ Explainer failed, retrying without explanations")
        config = ga.config.load(config_path)
        config.explainer = None
        return ga.audit.from_config(config)
    except ga.FairnessError:
        # Fallback: disable fairness
        print("⚠️ Fairness failed, retrying without fairness analysis")
        config = ga.config.load(config_path)
        config.fairness = None
        return ga.audit.from_config(config)

result = run_with_fallback("audit.yaml")
```

---

## Integration Examples

### MLflow Integration

```python
import mlflow
import glassalpha as ga

# Start MLflow run
with mlflow.start_run():
    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Run audit
    config = ga.config.AuditConfig(...)
    result = ga.audit.from_config(config)

    # Log metrics
    mlflow.log_metrics({
        "accuracy": result.performance['accuracy'],
        "auc_roc": result.performance.auc_roc,
        "bias": result.fairness.demographic_parity_difference,
        "ece": result.calibration.expected_calibration_error
    })

    # Log audit artifacts
    result.to_pdf("audit.pdf")
    mlflow.log_artifact("audit.pdf")
    mlflow.log_dict(result.to_json(), "metrics.json")

    # Tag run
    mlflow.set_tag("audit_status", "PASS" if not result.fairness.has_bias() else "REVIEW")
```

### Airflow DAG

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import glassalpha as ga

def run_audit(**context):
    """Airflow task to run audit"""
    config_path = context["dag_run"].conf.get("config", "audit.yaml")
    result = ga.audit.from_config(config_path)

    # Push metrics to XCom
    context["task_instance"].xcom_push(key="accuracy", value=result.performance['accuracy'])
    context["task_instance"].xcom_push(key="bias", value=result.fairness.demographic_parity_difference)

    return "audit_complete"

def check_compliance(**context):
    """Airflow task to check compliance gates"""
    ti = context["task_instance"]
    accuracy = ti.xcom_pull(task_ids="run_audit", key="accuracy")
    bias = ti.xcom_pull(task_ids="run_audit", key="bias")

    if accuracy < 0.70 or bias > 0.10:
        raise ValueError("Compliance gates failed")

    return "compliance_pass"

with DAG("model_audit", start_date=datetime(2025, 1, 1), schedule="@weekly") as dag:
    audit_task = PythonOperator(task_id="run_audit", python_callable=run_audit)
    compliance_task = PythonOperator(task_id="check_compliance", python_callable=check_compliance)

    audit_task >> compliance_task
```

### Kedro Pipeline

```python
from kedro.pipeline import Pipeline, node
import glassalpha as ga

def create_audit_pipeline():
    """Kedro pipeline for ML auditing"""
    return Pipeline([
        node(
            func=lambda model_path, config_path: ga.audit.from_config(config_path),
            inputs=["trained_model", "audit_config"],
            outputs="audit_result",
            name="run_audit"
        ),
        node(
            func=lambda result: result.to_pdf("reports/audit.pdf"),
            inputs="audit_result",
            outputs=None,
            name="export_pdf"
        ),
        node(
            func=lambda result: {
                "accuracy": result.performance['accuracy'],
                "bias": result.fairness.demographic_parity_difference,
                "compliant": not result.fairness.has_bias(threshold=0.10)
            },
            inputs="audit_result",
            outputs="compliance_metrics",
            name="compute_compliance"
        )
    ])
```

---

## Testing Utilities

### Mock Pipeline for Testing

```python
import glassalpha as ga
from unittest.mock import Mock

def test_custom_hook():
    """Test custom pipeline hook"""
    # Create mock config
    config = Mock(spec=ga.config.AuditConfig)

    # Track hook calls
    hook_called = {"count": 0}

    def test_hook(pipeline_state):
        hook_called["count"] += 1
        assert "config" in pipeline_state

    # Register and run
    pipeline = ga.AuditPipeline(config)
    pipeline.register_hook("pre_load", test_hook)

    # Note: Use test mode to avoid actual model loading
    pipeline.run(test_mode=True)

    assert hook_called["count"] == 1
```

### Contract Tests

```python
import glassalpha as ga
import pytest

def test_audit_reproducibility():
    """Verify byte-identical audit results with same seed"""
    config = ga.config.load("tests/fixtures/audit.yaml")

    result1 = ga.audit.from_config(config)
    result2 = ga.audit.from_config(config)

    # Check determinism
    assert result1.performance.accuracy == result2.performance.accuracy
    assert result1.fairness.demographic_parity_difference == result2.fairness.demographic_parity_difference
    assert (result1.explanations.shap_values == result2.explanations.shap_values).all()

def test_strict_mode_enforcement():
    """Verify strict mode catches missing requirements"""
    config = ga.config.AuditConfig(
        model=ga.config.ModelConfig(path="model.joblib", type="xgboost"),
        data=ga.config.DataConfig(test_data="test.csv", target_column="y"),
        random_seed=None  # Missing seed
    )

    with pytest.raises(ga.ValidationError, match="random_seed required in strict mode"):
        ga.config.validate(config, strict=True)
```

---

## Performance Optimization

### Lazy Loading

```python
import glassalpha as ga

# Configure lazy loading for large datasets
config = ga.config.load("audit.yaml")
config.data.lazy_load = True  # Load data in chunks
config.explainer.max_samples = 500  # Reduce SHAP samples

pipeline = ga.AuditPipeline(config)
result = pipeline.run()
```

### Caching

```python
import glassalpha as ga
from functools import lru_cache

@lru_cache(maxsize=10)
def run_cached_audit(config_hash: str, config_path: str) -> ga.AuditResult:
    """Run audit with caching based on config hash"""
    return ga.audit.from_config(config_path)

# Compute config hash
import hashlib
config_content = open("audit.yaml", "rb").read()
config_hash = hashlib.sha256(config_content).hexdigest()

# Run (uses cache on repeated calls)
result = run_cached_audit(config_hash, "audit.yaml")
```

---

## Related Documentation

- **[Audit API](audit-entry-points.md)** - `from_model()` for notebooks
- **[Configuration API](api-config.md)** - Configuration objects and validation
- **[CLI Reference](../cli.md)** - Command-line interface
- **[ML Engineer Workflow](../../guides/ml-engineer-workflow.md)** - Production usage patterns

## Support

- **GitHub Issues**: [Report bugs](https://github.com/GlassAlpha/glassalpha/issues)
- **Discussions**: [Ask questions](https://github.com/GlassAlpha/glassalpha/discussions)
- **Email**: [contact@glassalpha.com](mailto:contact@glassalpha.com)
