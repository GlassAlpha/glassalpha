# API Reference

Complete Python API documentation for programmatic audit generation and configuration.

## Choose Your API Level

GlassAlpha provides three API levels depending on your use case:

| API Level                              | When to Use                                           | Key Features                                |
| -------------------------------------- | ----------------------------------------------------- | ------------------------------------------- |
| **[Audit API](api-audit.md)**          | Notebooks, quick exploration, interactive development | `from_model()`, `AuditResult`, plot methods |
| **[Configuration API](api-config.md)** | Production pipelines, CI/CD, reproducible audits      | Config objects, validation, profiles        |
| **[Pipeline API](api-pipeline.md)**    | Custom extensions, advanced control, hooks            | `AuditPipeline`, extensibility hooks        |

---

## Quick Start by Use Case

### Notebook Development

**Goal**: Generate audit interactively, visualize results inline, iterate quickly.

```python
import glassalpha as ga

# Train your model
model.fit(X_train, y_train)

# Generate audit in one line
result = ga.audit.from_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    protected_attributes={'gender': gender, 'race': race},
    random_seed=42
)

# Display inline (Jupyter)
result

# Explore metrics (dict-based access)
result.performance['accuracy']
result.fairness['demographic_parity_max_diff']
result.calibration['expected_calibration_error']

# Visualization: fairness plots available now
result.fairness.plot_group_metrics()

# Export PDF with all visualizations
result.to_pdf("audit.pdf")
```

**[→ Full Audit API Documentation](api-audit.md)**

---

### Production Pipelines

**Goal**: Reproducible audits from config files, CI/CD integration, deterministic results.

```python
import glassalpha as ga

# Load configuration
config = ga.config.load("audit.yaml", strict=True)

# Run audit
result = ga.audit.from_config(config)

# Export audit report
result.to_pdf("audit.pdf")

# Note: Policy gates and evidence pack export planned for future release
```

**YAML Configuration:**

```yaml
model:
  path: models/credit_model.joblib
  type: xgboost

data:
  test_data: data/test.csv
  target_column: approved
  protected_attributes:
    - gender
    - race

random_seed: 42

policy:
  gates:
    min_accuracy: 0.70
    max_bias: 0.10
  fail_on_violation: true

report:
  template: standard_audit
  output_format: pdf

manifest:
  enabled: true
```

**[→ Full Configuration API Documentation](api-config.md)**

---

### Advanced Customization

**Goal**: Custom hooks, explainer selection, batch processing, framework integration.

```python
import glassalpha as ga

# Load configuration
config = ga.config.load("audit.yaml")

# Create pipeline
pipeline = ga.AuditPipeline(config, verbose=True)

# Register custom hook
def validate_predictions(pipeline_state):
    predictions = pipeline_state["predictions"]
    assert predictions.min() >= 0
    assert predictions.max() <= 1

pipeline.register_hook("post_predict", validate_predictions)

# Run with custom hooks
result = pipeline.run()
```

**[→ Full Pipeline API Documentation](api-pipeline.md)**

---

## API comparison

### When to Use Each API

| Task                         | Audit API | Config API | Pipeline API |
| ---------------------------- | :-------: | :--------: | :----------: |
| Jupyter/Colab notebook       |    ✅     |     ⚠️     |      ❌      |
| Quick model comparison       |    ✅     |     ⚠️     |      ❌      |
| Interactive threshold tuning |    ✅     |     ❌     |      ❌      |
| CI/CD integration            |    ❌     |     ✅     |      ⚠️      |
| Reproducible audits          |    ⚠️     |     ✅     |      ✅      |
| Regulatory submission        |    ❌     |     ✅     |      ✅      |
| Custom explainer selection   |    ❌     |     ⚠️     |      ✅      |
| Batch processing             |    ⚠️     |     ✅     |      ✅      |
| Custom hooks/extensions      |    ❌     |     ❌     |      ✅      |
| MLflow/Airflow integration   |    ⚠️     |     ✅     |      ✅      |

**Legend:**

- ✅ Recommended
- ⚠️ Possible but not ideal
- ❌ Not supported

---

## Common Workflows

### 1. Notebook → Production Pipeline

Start with `from_model()` for exploration, then export config for production:

```python
# Step 1: Explore in notebook
result = ga.audit.from_model(model, X_test, y_test, random_seed=42)

# Step 2: Export configuration
result.to_config("audit_config.yaml")

# Step 3: Use in CI/CD
# $ glassalpha audit --config audit_config.yaml --output report.pdf
```

### 2. Batch Model Comparison

Compare multiple models programmatically:

```python
models = {
    "LogisticRegression": lr_model,
    "RandomForest": rf_model,
    "XGBoost": xgb_model
}

results = {}
for name, model in models.items():
    results[name] = ga.audit.from_model(
        model=model, X_test=X_test, y_test=y_test, random_seed=42
    )

# Compare
for name, result in results.items():
    print(f"{name}: Acc={result.performance['accuracy']:.3f}, "
          f"Bias={result.fairness.demographic_parity_difference:.3f}")
```

### 3. CI/CD Gate Check

Audit model in CI and fail if gates violated:

```python
# ci_audit.py
import glassalpha as ga
import sys

config = ga.config.load("audit.yaml", strict=True)
result = ga.audit.from_config(config)

# Check metrics manually
bias_threshold = 0.10
if result.fairness['demographic_parity_max_diff'] > bias_threshold:
    print(f"❌ Fairness check failed: bias {result.fairness['demographic_parity_max_diff']:.3f} > {bias_threshold}")
    sys.exit(1)

print("✅ All checks passed")
result.to_pdf("audit_report.pdf")

# Note: Automated policy gates planned for future release
```

**GitHub Actions:**

```yaml
- name: Run ML Audit
  run: python ci_audit.py
```

### 4. Custom Fairness Metric

Add custom metric via pipeline hook:

```python
def compute_custom_metric(pipeline_state):
    fairness = pipeline_state["fairness_results"]

    # Custom metric: max ratio of FPR across groups
    fprs = [g["fpr"] for g in fairness["groups"].values()]
    max_fpr_ratio = max(fprs) / min(fprs)

    fairness["custom_metrics"] = {"max_fpr_ratio": max_fpr_ratio}

pipeline = ga.AuditPipeline(config)
pipeline.register_hook("post_fairness", compute_custom_metric)
result = pipeline.run()
```

---

## Migration Guide

### From CLI to Python API

**Before (CLI):**

```bash
glassalpha audit --config audit.yaml --output report.pdf
```

**After (Python):**

```python
import glassalpha as ga

# Method 1: Config file
result = ga.audit.from_config("audit.yaml")
result.to_pdf("report.pdf")

# Method 2: Programmatic config
config = ga.config.AuditConfig(...)
result = ga.audit.from_config(config)
result.to_pdf("report.pdf")
```

### From sklearn to GlassAlpha

**Before (sklearn only):**

```python
from sklearn.metrics import accuracy_score, roc_auc_score

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions)}")
print(f"AUC: {roc_auc_score(y_test, probabilities[:, 1])}")
```

**After (GlassAlpha):**

```python
import glassalpha as ga

# All metrics + fairness + calibration + explanations + PDF report
result = ga.audit.from_model(model, X_test, y_test, random_seed=42)
result  # Display inline
```

---

## Related Documentation

- **[CLI Reference](../cli.md)** - Command-line interface
- **[Quick Start Guide](../../getting-started/quickstart.md)** - First audit tutorial
- **[Data Scientist Workflow](../../guides/data-scientist-workflow.md)** - Notebook development guide
- **[ML Engineer Workflow](../../guides/ml-engineer-workflow.md)** - Production deployment guide
- **[Configuration Guide](../../getting-started/configuration.md)** - YAML configuration format
- **[Quickstart Notebook](https://colab.research.google.com/github/GlassAlpha/glassalpha/blob/main/examples/notebooks/quickstart_colab.ipynb)** - Try in Colab (zero setup)
- **[Example Tutorials](../../examples/index.md)** - Detailed walkthroughs

---

## API details

### [Audit API (from_model)](api-audit.md)

High-level Python API for notebook development and interactive exploration.

**Key functions:**

- `glassalpha.audit.from_model()` - Generate audit from trained model
- `AuditResult` - Container with performance, fairness, calibration, explanations
- Plot methods - Inline visualizations for notebooks

**Use for:**

- Jupyter/Colab notebooks
- Quick model exploration
- Interactive metric visualization
- Threshold tuning

---

### [Configuration API](api-config.md)

Programmatic configuration and validation for production pipelines.

**Key classes:**

- `AuditConfig` - Main configuration object
- `ModelConfig`, `DataConfig`, `FairnessConfig`, etc. - Sub-configurations
- Validation functions - Schema and strict mode validation
- Profiles - Preset configurations by use case

**Use for:**

- Production CI/CD pipelines
- Reproducible audits
- Regulatory compliance
- Batch processing

---

### [Pipeline API (Advanced)](api-pipeline.md)

Lower-level API for custom extensions and advanced control.

**Key features:**

- `AuditPipeline` - Extensible audit pipeline
- Hooks - Custom callbacks at each pipeline stage
- Stage execution - Run individual stages for debugging
- Integration examples - MLflow, Airflow, Kedro

**Use for:**

- Custom explainer selection
- Custom fairness metrics
- Framework integration (MLflow, Airflow)
- Advanced batch processing

---

## Support

- **GitHub Issues**: [Report bugs](https://github.com/GlassAlpha/glassalpha/issues)
- **Discussions**: [Ask questions](https://github.com/GlassAlpha/glassalpha/discussions)
- **Email**: [contact@glassalpha.com](mailto:contact@glassalpha.com)
