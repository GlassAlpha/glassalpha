# Audit API Reference

Primary API for generating ML audits from trained models in notebooks and scripts.

## Quick Start

```python
import glassalpha as ga

# Generate audit from trained model
result = ga.audit.from_model(
    model=my_model,
    X_test=X_test,
    y_test=y_test,
    protected_attributes={'gender': gender_col, 'race': race_col},
    random_seed=42
)

# Display inline (Jupyter)
result

# Access metrics
print(f"Accuracy: {result.performance['accuracy']:.3f}")
print(f"Bias: {result.fairness.demographic_parity_difference:.3f}")

# Export PDF
result.to_pdf("audit_report.pdf")
```

---

## `glassalpha.audit.from_model()`

Generate a comprehensive audit directly from a trained model.

### Signature

```python
def from_model(
    model,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    protected_attributes: dict[str, pd.Series | np.ndarray] | None = None,
    feature_names: list[str] | None = None,
    target_name: str = "target",
    threshold: float = 0.5,
    random_seed: int | None = None,
    explainer_samples: int = 1000,
    include_calibration: bool = True,
    include_fairness: bool = True,
    include_explanations: bool = True
) -> AuditResult
```

### Parameters

| Parameter              | Type                           | Description                                                                                                                                                              |
| ---------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `model`                | **Required**                   | Trained scikit-learn compatible model with `predict()` and `predict_proba()` methods. Supports: RandomForest, XGBoost, LightGBM, LogisticRegression, etc.                |
| `X_test`               | `pd.DataFrame` or `np.ndarray` | **Required**. Test features of shape `(n_samples, n_features)`. If DataFrame, column names used as feature names.                                                        |
| `y_test`               | `pd.Series` or `np.ndarray`    | **Required**. True labels of shape `(n_samples,)`. Binary classification (0/1 or True/False).                                                                            |
| `protected_attributes` | `dict[str, array-like]`        | **Optional**. Dictionary mapping attribute names to arrays of shape `(n_samples,)`. Used for fairness analysis. Example: `{'gender': gender_array, 'race': race_array}`. |
| `feature_names`        | `list[str]`                    | **Optional**. Feature names if `X_test` is ndarray. Inferred from DataFrame columns if not provided.                                                                     |
| `target_name`          | `str`                          | **Optional**. Name of target variable (default: `"target"`). Used in report labels.                                                                                      |
| `threshold`            | `float`                        | **Optional**. Decision threshold for binary classification (default: `0.5`). Range: [0, 1].                                                                              |
| `random_seed`          | `int`                          | **Optional**. Random seed for reproducibility (default: `None`). Recommended: always set for deterministic results.                                                      |
| `explainer_samples`    | `int`                          | **Optional**. Number of background samples for SHAP explainer (default: `1000`). Reduce for faster computation, increase for more accurate explanations.                 |
| `include_calibration`  | `bool`                         | **Optional**. Compute calibration metrics (default: `True`). Set to `False` to skip for speed.                                                                           |
| `include_fairness`     | `bool`                         | **Optional**. Compute fairness metrics (default: `True`). Requires `protected_attributes` to be provided.                                                                |
| `include_explanations` | `bool`                         | **Optional**. Generate model explanations (default: `True`). Set to `False` to skip SHAP computation.                                                                    |

### Returns

**`AuditResult`** object with the following attributes:

- `performance`: Performance metrics (accuracy, precision, recall, F1, AUC)
- `fairness`: Fairness analysis (demographic parity, equal opportunity)
- `calibration`: Calibration metrics (ECE, Brier score, curves)
- `explanations`: Model explanations (SHAP values, feature importance)
- `manifest`: Provenance information (seeds, versions, hashes)

### Examples

#### Minimal Example

```python
import glassalpha as ga
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create and train model
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Generate audit
result = ga.audit.from_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    random_seed=42
)
```

#### With Fairness Analysis

```python
import pandas as pd

# Prepare data with protected attributes
df = pd.read_csv("credit_applications.csv")
X = df.drop(columns=["approved", "gender", "race"])
y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model.fit(X_train, y_train)

# Audit with fairness analysis
result = ga.audit.from_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    protected_attributes={
        'gender': df.loc[X_test.index, 'gender'],
        'race': df.loc[X_test.index, 'race']
    },
    random_seed=42
)

# Check for bias (using custom threshold)
bias_threshold = 0.10  # 10% threshold
if result.fairness['demographic_parity_max_diff'] > bias_threshold:
    print(f"⚠️ Bias detected: {result.fairness['demographic_parity_max_diff']:.3f}")
```

#### Custom Threshold

```python
# Audit at different threshold
result = ga.audit.from_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    threshold=0.3,  # Lower threshold (more predictions as positive)
    random_seed=42
)
```

#### Performance Optimization

```python
# Faster audit (for large datasets or quick iteration)
result = ga.audit.from_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    explainer_samples=100,  # Reduce from default 1000
    include_calibration=False,  # Skip calibration
    random_seed=42
)
```

### Raises

| Exception    | When                                            | How to Fix                                                        |
| ------------ | ----------------------------------------------- | ----------------------------------------------------------------- |
| `TypeError`  | Model doesn't have `predict_proba()`            | Use a probabilistic classifier or set `include_calibration=False` |
| `ValueError` | `X_test` and `y_test` have different lengths    | Ensure test data is aligned                                       |
| `ValueError` | `y_test` is not binary                          | Convert to binary (0/1) or use appropriate model                  |
| `ValueError` | `protected_attributes` arrays have wrong length | Ensure protected attribute arrays match `X_test` length           |
| `ValueError` | `threshold` not in [0, 1]                       | Use valid threshold between 0 and 1                               |

### Notes

**Reproducibility:**

- Always set `random_seed` for deterministic results
- SHAP explainer uses random sampling; same seed = same explanations

**Performance:**

- TreeSHAP (for tree models) is much faster than KernelSHAP
- Reduce `explainer_samples` for faster computation (tradeoff: explanation quality)
- Fairness analysis is fast (no significant overhead)

**Compatibility:**

- Works with any scikit-learn compatible model
- Requires `predict()` for predictions and `predict_proba()` for calibration
- If model doesn't have `predict_proba()`, set `include_calibration=False`

---

## `AuditResult`

Container for audit results with convenience methods for inspection and export.

### Attributes

#### `performance`

Performance metrics object with:

| Attribute          | Type         | Description                                                |
| ------------------ | ------------ | ---------------------------------------------------------- |
| `accuracy`         | `float`      | Overall accuracy (correct predictions / total predictions) |
| `precision`        | `float`      | Precision (true positives / predicted positives)           |
| `recall`           | `float`      | Recall/TPR (true positives / actual positives)             |
| `f1`               | `float`      | F1 score (harmonic mean of precision and recall)           |
| `auc_roc`          | `float`      | Area under ROC curve                                       |
| `confusion_matrix` | `np.ndarray` | 2x2 confusion matrix [[TN, FP], [FN, TP]]                  |

**Methods:**

- `plot_confusion_matrix()`: Display confusion matrix heatmap
- `plot_roc_curve()`: Plot ROC curve with AUC

#### `fairness`

Fairness analysis object (if `protected_attributes` provided):

| Attribute                       | Type        | Description                                    |
| ------------------------------- | ----------- | ---------------------------------------------- |
| `demographic_parity_difference` | `float`     | Max difference in approval rates across groups |
| `equal_opportunity_difference`  | `float`     | Max difference in TPR across groups            |
| `equalized_odds_difference`     | `float`     | Max of TPR and FPR differences                 |
| `groups`                        | `list[str]` | Protected groups analyzed                      |
| `group_metrics`                 | `dict`      | Per-group performance metrics                  |

**Methods:**

- `plot_group_metrics(metric="selection_rate")`: Bar chart comparing metrics across groups _(implemented)_

**Checking for bias:**

```python
# Custom threshold approach (recommended)
bias_threshold = 0.10  # Customize per use case
if result.fairness['demographic_parity_max_diff'] > bias_threshold:
    print("⚠️ Bias detected")
```

#### `calibration`

Calibration metrics object (if `include_calibration=True`):

| Attribute                    | Type         | Description                                        |
| ---------------------------- | ------------ | -------------------------------------------------- |
| `expected_calibration_error` | `float`      | Expected Calibration Error (ECE) - lower is better |
| `brier_score`                | `float`      | Brier score - lower is better                      |
| `calibration_curve_x`        | `np.ndarray` | Predicted probabilities (binned)                   |
| `calibration_curve_y`        | `np.ndarray` | Observed frequencies                               |

**Visualizing calibration:**

```python
# Current approach: Use PDF report
result.to_pdf("audit.pdf")  # Includes calibration curves

# Or access raw data for custom plotting
ece = result.calibration['expected_calibration_error']
brier = result.calibration['brier_score']
```

#### `explanations`

Model explanations object (if `include_explanations=True`):

| Attribute            | Type         | Description                                   |
| -------------------- | ------------ | --------------------------------------------- |
| `feature_importance` | `pd.Series`  | Global feature importance (sorted descending) |
| `shap_values`        | `np.ndarray` | SHAP values for all test instances            |
| `base_value`         | `float`      | Baseline prediction value                     |

**Methods:**

- `plot_importance(top_n=10)`: Bar chart of top-N important features
- `plot_summary()`: SHAP summary plot (beeswarm)
- `plot_force(instance_idx)`: Force plot for specific instance

#### `manifest`

Provenance and reproducibility information:

| Attribute            | Type  | Description                             |
| -------------------- | ----- | --------------------------------------- |
| `random_seed`        | `int` | Random seed used                        |
| `glassalpha_version` | `str` | GlassAlpha version                      |
| `sklearn_version`    | `str` | Scikit-learn version                    |
| `model_type`         | `str` | Model class name                        |
| `n_features`         | `int` | Number of features                      |
| `n_test_samples`     | `int` | Test set size                           |
| `timestamp`          | `str` | Audit generation timestamp (ISO format) |

### Methods

#### `display()`

Display inline HTML summary in Jupyter notebooks.

```python
result.display()
```

**Jupyter Auto-Display:**
When returned as last expression in cell, automatically displays:

```python
result  # Automatically calls display()
```

#### `to_pdf(filepath, template="standard")`

Export audit to professional PDF report.

**Parameters:**

- `filepath` (`str` or `Path`): Output PDF path
- `template` (`str`): Report template (default: `"standard"`)

**Example:**

```python
result.to_pdf("audit_report.pdf")
```

#### `to_json(filepath=None)`

Export all metrics to JSON.

**Parameters:**

- `filepath` (`str` or `Path`, optional): If provided, writes JSON to file. If `None`, returns dict.

**Example:**

```python
# Return as dict
metrics_dict = result.to_json()

# Write to file
result.to_json("metrics.json")
```

#### `to_config(filepath)`

Export audit configuration for reproduction via CLI.

**Parameters:**

- `filepath` (`str` or `Path`): Output YAML config path

**Example:**

```python
result.to_config("audit_config.yaml")
```

Then reproduce with CLI:

```bash
glassalpha audit --config audit_config.yaml --output report.pdf
```

#### `summary()`

Print concise text summary to console.

```python
result.summary()
```

**Output:**

```
Audit Summary
=============
Performance:
  Accuracy: 0.756
  AUC-ROC: 0.821
Fairness:
  Demographic parity: 0.062 (⚠️ WARNING: exceeds 0.05 threshold)
  Equal opportunity: 0.043 (✓ PASS)
Calibration:
  ECE: 0.032 (✓ PASS: < 0.05)
```

### Example Usage

```python
# Generate audit
result = ga.audit.from_model(model, X_test, y_test,
                              protected_attributes={'gender': gender},
                              random_seed=42)

# Inline display in notebook
result

# Access specific metrics
print(f"Model accuracy: {result.performance['accuracy']:.3f}")
print(f"Fairness gap: {result.fairness.demographic_parity_difference:.3f}")
print(f"Calibration: ECE = {result.calibration.expected_calibration_error:.3f}")

# Check for issues
bias_threshold = 0.10
if result.fairness['demographic_parity_max_diff'] > bias_threshold:
    print(f"⚠️ Bias detected above {bias_threshold:.0%} threshold")
    result.fairness.plot_group_metrics()

# Export multiple formats
result.to_pdf("audit_report.pdf")
result.to_json("metrics.json")
result.to_config("audit_config.yaml")

# Plot key results
result.performance.plot_confusion_matrix()
result.calibration.plot()
result.explanations.plot_importance(top_n=15)
```

---

## Plot Methods

All plot methods return `matplotlib` figure and axes objects for customization.

### Performance Plots

#### `result.performance.plot_confusion_matrix()`

Display confusion matrix as heatmap.

**Parameters:**

- `normalize` (`bool`, default `False`): Show proportions instead of counts
- `cmap` (`str`, default `"Blues"`): Matplotlib colormap
- `ax` (`matplotlib.axes.Axes`, optional): Axes to plot on

**Returns:** `(fig, ax)` tuple

**Example:**

```python
fig, ax = result.performance.plot_confusion_matrix(normalize=True)
ax.set_title("Confusion Matrix (Normalized)")
```

#### `result.performance.plot_roc_curve()`

**Status**: Not yet implemented for interactive use.

**Workaround**: ROC curves are included in PDF reports via `result.to_pdf()`. For custom ROC plots, use `sklearn.metrics.roc_curve()` directly with your predictions.

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get predictions (you'll need to store these from your model)
# y_pred_proba = model.predict_proba(X_test)[:, 1]
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
# roc_auc = auc(fpr, tpr)

# Plot
# plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
```

### Calibration Visualization

**Current approach**: Use PDF reports for comprehensive visualizations

```python
# Generate full report with all plots
result.to_pdf("audit_report.pdf")
result.to_html("audit_report.html")
```

**Custom plotting**: Access raw metrics for matplotlib

```python
import matplotlib.pyplot as plt

# Access calibration data
ece = result.calibration['expected_calibration_error']
brier = result.calibration['brier_score']

# Create custom plot
fig, ax = plt.subplots()
ax.bar(['ECE', 'Brier'], [ece, brier])
ax.set_title("Calibration Metrics")
plt.show()
```

### Explanation Plots

#### `result.explanations.plot_importance(top_n=10)`

Bar chart of top-N most important features.

**Parameters:**

- `top_n` (`int`, default `10`): Number of features to show
- `ax` (`matplotlib.axes.Axes`, optional): Axes to plot on

**Returns:** `(fig, ax)` tuple

#### `result.explanations.plot_summary()`

SHAP summary plot (beeswarm) showing feature effects.

**Parameters:**

- `max_display` (`int`, default `20`): Max features to display
- `ax` (`matplotlib.axes.Axes`, optional): Axes to plot on

**Returns:** `(fig, ax)` tuple

#### `result.explanations.plot_force(instance_idx)`

Force plot for specific instance showing feature contributions.

**Parameters:**

- `instance_idx` (`int`): Index of instance to explain

**Returns:** `matplotlib` figure object

**Example:**

```python
# Explain why instance 42 was classified as positive
result.explanations.plot_force(instance_idx=42)
```

---

## Advanced Usage

### Custom Plot Styling

```python
import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8-paper')

# Create custom layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot on custom axes
result.performance.plot_confusion_matrix(ax=ax1)
# Note: plot_roc_curve not yet implemented - use PDF report
result.fairness.plot_group_metrics(ax=ax3)
# Note: calibration.plot() not yet implemented - use PDF report

plt.tight_layout()
plt.savefig("combined_analysis.pdf", dpi=300)
```

### Batch Auditing

```python
models = {
    "LogisticRegression": lr_model,
    "RandomForest": rf_model,
    "XGBoost": xgb_model
}

results = {}
for name, model in models.items():
    results[name] = ga.audit.from_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        protected_attributes={'gender': gender},
        random_seed=42
    )
    results[name].to_pdf(f"audit_{name}.pdf")

# Compare models
for name, result in results.items():
    print(f"{name}: Accuracy={result.performance['accuracy']:.3f}, "
          f"Fairness={result.fairness.demographic_parity_difference:.3f}")
```

### Integration with MLflow

```python
import mlflow

with mlflow.start_run():
    # Train and log model
    mlflow.sklearn.log_model(model, "model")

    # Generate audit
    result = ga.audit.from_model(model, X_test, y_test, random_seed=42)

    # Log metrics
    mlflow.log_metric("accuracy", result.performance['accuracy'])
    mlflow.log_metric("auc_roc", result.performance.auc_roc)
    mlflow.log_metric("fairness_gap", result.fairness.demographic_parity_difference)
    mlflow.log_metric("calibration_ece", result.calibration.expected_calibration_error)

    # Log audit artifacts
    result.to_pdf("audit.pdf")
    mlflow.log_artifact("audit.pdf")
    mlflow.log_dict(result.to_json(), "metrics.json")
```

---

## Related Documentation

- **[Quickstart Notebook](https://colab.research.google.com/github/GlassAlpha/glassalpha/blob/main/examples/notebooks/quickstart_colab.ipynb)** - Interactive examples using `from_model()` API
- **[Example Tutorials](../../examples/index.md)** - Detailed walkthroughs
- **[Data Scientist Workflow](../../guides/data-scientist-workflow.md)** - Notebook-based development
- **[Configuration API](api-config.md)** - Programmatic configuration
- **[Pipeline API](api-pipeline.md)** - Lower-level audit pipeline
- **[CLI Reference](../cli.md)** - Command-line interface

## Support

- **GitHub Issues**: [Report bugs](https://github.com/GlassAlpha/glassalpha/issues)
- **Discussions**: [Ask questions](https://github.com/GlassAlpha/glassalpha/discussions)
- **Email**: [contact@glassalpha.com](mailto:contact@glassalpha.com)
