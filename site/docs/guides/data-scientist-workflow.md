# Data Scientist Workflow

Guide for data scientists and researchers using GlassAlpha for exploratory model analysis, fairness research, and notebook-based development.

## Overview

This guide is for data scientists and researchers who need to:

- Explore model fairness in Jupyter notebooks
- Conduct interactive bias analysis
- Compare multiple models quickly
- Prototype audit configurations
- Research fairness metrics and trade-offs
- Generate publication-ready visualizations

**Not a data scientist?** For production workflows, see [ML Engineer Workflow](ml-engineer-workflow.md). For compliance workflows, see [Compliance Officer Workflow](compliance-workflow.md).

## Key Capabilities

### Notebook-First Development

Interactive exploration with inline results:

- Jupyter/Colab integration with `from_model()` API
- Inline HTML audit summaries
- Interactive plotting with `result.plot()` methods
- No configuration files needed for quick experiments

### Rapid Model Comparison

Compare fairness across models:

- Audit multiple models in same notebook
- Side-by-side metric comparison
- Threshold sweep analysis
- Trade-off visualization (accuracy vs fairness)

### Research-Friendly Features

Support for academic work:

- Statistical confidence intervals for all metrics
- Reproducible experiments (fixed seeds)
- Export metrics as JSON/CSV for papers
- Publication-quality plots

## Typical Workflows

### Workflow 1: Quick Model Fairness Check

**Scenario**: You've trained a model and want to quickly check for bias before diving deeper.

#### Step 1: Train model in notebook

```python
# Notebook cell 1: Train model
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv("data/credit_applications.csv")
X = df.drop(columns=["approved", "gender", "race"])
y = df["approved"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
```

#### Step 2: Audit inline (no config file)

```python
# Notebook cell 2: Quick audit
import glassalpha as ga

# Create audit result directly from model
result = ga.audit.from_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    protected_attributes={
        "gender": df.loc[X_test.index, "gender"],
        "race": df.loc[X_test.index, "race"]
    },
    random_seed=42
)

# Display inline summary
result  # Jupyter automatically displays HTML summary
```

**What you see**: Interactive HTML summary with:

- Performance metrics (accuracy, AUC, F1)
- Fairness metrics by group
- Feature importance
- Warnings if bias detected

#### Step 3: Explore metrics interactively

```python
# Notebook cell 3: Dig into specific metrics
print(f"Overall accuracy: {result.performance['accuracy']:.3f}")
print(f"AUC-ROC: {result.performance.auc_roc:.3f}")

print("\nFairness by gender:")
print(f"  Demographic parity: {result.fairness.demographic_parity_difference:.3f}")
print(f"  Equal opportunity: {result.fairness.equal_opportunity_difference:.3f}")

print("\nFairness by race:")
for group in result.fairness.groups:
    print(f"  {group}: TPR={result.fairness.tpr[group]:.3f}")
```

#### Step 4: Visualize if needed

```python
# Notebook cell 4: Plot key metrics
result.fairness.plot_group_metrics()
result.calibration.plot()
result.performance.plot_confusion_matrix()
```

#### Step 5: Export PDF when satisfied

```python
# Notebook cell 5: Generate full report
result.to_pdf("reports/credit_model_audit.pdf")
```

### Workflow 2: Model Comparison (Fairness vs Accuracy Trade-offs)

**Scenario**: Compare multiple models to find best fairness/accuracy balance.

#### Step 1: Train multiple models

```python
# Notebook cell 1: Train 3 models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import glassalpha as ga

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=50),
    "XGBoost": XGBClassifier(random_state=42, n_estimators=50)
}

# Train all models
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: {model.score(X_test, y_test):.3f} accuracy")
```

#### Step 2: Audit all models

```python
# Notebook cell 2: Audit each model
results = {}
for name, model in models.items():
    results[name] = ga.audit.from_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        protected_attributes={"gender": df.loc[X_test.index, "gender"]},
        random_seed=42
    )
```

#### Step 3: Compare metrics

```python
# Notebook cell 3: Build comparison table
import pandas as pd

comparison = []
for name, result in results.items():
    comparison.append({
        "Model": name,
        "Accuracy": result.performance['accuracy'],
        "AUC": result.performance.auc_roc,
        "Demographic Parity": result.fairness.demographic_parity_difference,
        "Equal Opportunity": result.fairness.equal_opportunity_difference,
        "Fairness Score": result.fairness.overall_fairness_score
    })

comparison_df = pd.DataFrame(comparison)
print(comparison_df.to_string(index=False))
```

#### Step 4: Visualize trade-offs

```python
# Notebook cell 4: Plot accuracy vs fairness
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
    comparison_df["Accuracy"],
    comparison_df["Fairness Score"],
    s=100
)

for idx, row in comparison_df.iterrows():
    ax.annotate(row["Model"], (row["Accuracy"], row["Fairness Score"]))

ax.set_xlabel("Accuracy")
ax.set_ylabel("Fairness Score (higher is better)")
ax.set_title("Model Comparison: Accuracy vs Fairness")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

#### Step 5: Select best model

```python
# Notebook cell 5: Decision logic
# Find model with accuracy >0.75 and best fairness score
threshold_met = comparison_df[comparison_df["Accuracy"] > 0.75]
best_model_name = threshold_met.loc[threshold_met["Fairness Score"].idxmax(), "Model"]

print(f"Best model: {best_model_name}")
print(f"  Accuracy: {threshold_met.loc[threshold_met['Fairness Score'].idxmax(), 'Accuracy']:.3f}")
print(f"  Fairness: {threshold_met.loc[threshold_met['Fairness Score'].idxmax(), 'Fairness Score']:.3f}")

# Generate full audit for best model
best_result = results[best_model_name]
best_result.to_pdf(f"reports/{best_model_name.lower().replace(' ', '_')}_audit.pdf")
```

### Workflow 3: Interactive Threshold Exploration

**Scenario**: Find optimal decision threshold balancing performance and fairness.

#### Step 1: Audit at default threshold

```python
# Notebook cell 1
import glassalpha as ga

result_baseline = ga.audit.from_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    protected_attributes={"gender": df.loc[X_test.index, "gender"]},
    threshold=0.5,  # Default
    random_seed=42
)

print(f"Baseline (threshold=0.5):")
print(f"  Accuracy: {result_baseline.performance.accuracy:.3f}")
print(f"  Demographic parity: {result_baseline.fairness.demographic_parity_difference:.3f}")
```

#### Step 2: Sweep thresholds

```python
# Notebook cell 2: Test multiple thresholds
import numpy as np

thresholds = np.arange(0.3, 0.8, 0.05)
sweep_results = []

for threshold in thresholds:
    result = ga.audit.from_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        protected_attributes={"gender": df.loc[X_test.index, "gender"]},
        threshold=threshold,
        random_seed=42
    )
    sweep_results.append({
        "threshold": threshold,
        "accuracy": result.performance['accuracy'],
        "precision": result.performance.precision,
        "recall": result.performance.recall,
        "dem_parity": result.fairness.demographic_parity_difference,
        "eq_opp": result.fairness.equal_opportunity_difference
    })

sweep_df = pd.DataFrame(sweep_results)
```

#### Step 3: Visualize trade-offs

```python
# Notebook cell 3: Plot threshold sweep
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Performance metrics
ax1.plot(sweep_df["threshold"], sweep_df["accuracy"], label="Accuracy", marker="o")
ax1.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", marker="s")
ax1.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", marker="^")
ax1.set_xlabel("Decision Threshold")
ax1.set_ylabel("Metric Value")
ax1.set_title("Performance vs Threshold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Fairness metrics
ax2.plot(sweep_df["threshold"], sweep_df["dem_parity"], label="Demographic Parity", marker="o")
ax2.plot(sweep_df["threshold"], sweep_df["eq_opp"], label="Equal Opportunity", marker="s")
ax2.axhline(y=0.05, color='r', linestyle='--', label="Tolerance (±0.05)")
ax2.axhline(y=-0.05, color='r', linestyle='--')
ax2.set_xlabel("Decision Threshold")
ax2.set_ylabel("Fairness Gap")
ax2.set_title("Fairness vs Threshold")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### Step 4: Find optimal threshold

```python
# Notebook cell 4: Select optimal threshold
# Find threshold where fairness < 0.05 and accuracy maximized
fair_results = sweep_df[sweep_df["dem_parity"].abs() < 0.05]
optimal_idx = fair_results["accuracy"].idxmax()
optimal_threshold = fair_results.loc[optimal_idx, "threshold"]

print(f"Optimal threshold: {optimal_threshold:.2f}")
print(f"  Accuracy: {fair_results.loc[optimal_idx, 'accuracy']:.3f}")
print(f"  Demographic parity: {fair_results.loc[optimal_idx, 'dem_parity']:.3f}")
```

### Workflow 4: Research Paper Figures

**Scenario**: Generate publication-quality figures for academic papers.

#### Step 1: Run comprehensive audit

```python
# Notebook cell 1
import glassalpha as ga

result = ga.audit.from_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    protected_attributes={
        "gender": df.loc[X_test.index, "gender"],
        "race": df.loc[X_test.index, "race"]
    },
    random_seed=42
)
```

#### Step 2: Export metrics for tables

```python
# Notebook cell 2: Export metrics as CSV
metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "AUC-ROC",
        "Demographic Parity (Gender)",
        "Equal Opportunity (Gender)",
        "Demographic Parity (Race)",
        "Equal Opportunity (Race)"
    ],
    "Value": [
        result.performance['accuracy'],
        result.performance.precision,
        result.performance.recall,
        result.performance.f1,
        result.performance.auc_roc,
        result.fairness.demographic_parity_difference_gender,
        result.fairness.equal_opportunity_difference_gender,
        result.fairness.demographic_parity_difference_race,
        result.fairness.equal_opportunity_difference_race
    ]
})

metrics_df.to_csv("paper/tables/model_metrics.csv", index=False)
```

#### Step 3: Generate publication plots

```python
# Notebook cell 3: High-quality plots for paper
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper')  # Publication style

# Calibration plot
fig, ax = plt.subplots(figsize=(6, 5))
result.calibration.plot(ax=ax, show_confidence=True, style="paper")
plt.savefig("paper/figures/calibration.pdf", bbox_inches='tight', dpi=300)

# Fairness comparison
fig, ax = plt.subplots(figsize=(6, 5))
result.fairness.plot_group_comparison(ax=ax, style="paper")
plt.savefig("paper/figures/fairness_comparison.pdf", bbox_inches='tight', dpi=300)

# Feature importance
fig, ax = plt.subplots(figsize=(6, 5))
result.explanations.plot_importance(ax=ax, top_n=10, style="paper")
plt.savefig("paper/figures/feature_importance.pdf", bbox_inches='tight', dpi=300)
```

#### Step 4: Export for LaTeX tables

```python
# Notebook cell 4: LaTeX-formatted tables
latex_table = metrics_df.to_latex(
    index=False,
    float_format="%.3f",
    caption="Model Performance and Fairness Metrics",
    label="tab:metrics"
)

with open("paper/tables/metrics_table.tex", "w") as f:
    f.write(latex_table)
```

## Best Practices

### Reproducibility

- **Always set seeds**: Use consistent `random_seed` parameter
- **Record environment**: Document package versions (`pip freeze > requirements.txt`)
- **Save configs**: Export audit configurations for later reuse
- **Document experiments**: Use markdown cells to explain each analysis step

### Exploratory Analysis

- **Start simple**: Begin with default settings before customizing
- **Iterate quickly**: Use `from_model()` for fast prototyping
- **Compare baselines**: Always benchmark against simple models
- **Visualize early**: Use plot methods to catch issues quickly

### Model Development

- **Audit during development**: Don't wait until final model
- **Track fairness metrics**: Monitor alongside accuracy during training
- **Test multiple configurations**: Try different thresholds, hyperparameters
- **Document decisions**: Record why you chose specific models/thresholds

### Publication Preparation

- **Use consistent seeds**: Ensure figures are reproducible
- **Export metrics as data**: Don't manually transcribe numbers
- **Version control configs**: Git track audit configurations
- **Archive artifacts**: Save models, data, and audit results together

## Common Analysis Patterns

### Pattern 1: Quick Fairness Check

```python
result = ga.audit.from_model(model, X_test, y_test, protected_attributes={"gender": gender})
if result.fairness.has_bias():
    print("⚠️ Bias detected!")
    result.fairness.plot_group_metrics()
```

### Pattern 2: Metric Extraction

```python
metrics = {
    "accuracy": result.performance['accuracy'],
    "fairness": result.fairness.demographic_parity_difference,
    "calibration": result.calibration.expected_calibration_error
}
```

### Pattern 3: Batch Audit

```python
results = [
    ga.audit.from_model(model, X, y, protected_attributes=attrs, random_seed=seed)
    for seed in range(5)  # Multiple random seeds for robustness
]

# Aggregate results
avg_accuracy = np.mean([r.performance.accuracy for r in results])
```

### Pattern 4: Custom Threshold

```python
result = ga.audit.from_model(
    model, X_test, y_test,
    protected_attributes={"race": race},
    threshold=0.45,  # Custom threshold
    random_seed=42
)
```

## Transitioning to Production

When your exploratory work is ready for production:

### Step 1: Create config file

```python
# Export notebook audit to config file
result.to_config("configs/production_audit.yaml")
```

### Step 2: Validate reproducibility

```bash
# Run from CLI to ensure consistency (use --fast for quick validation)
glassalpha audit --config configs/production_audit.yaml --output prod_audit.pdf --fast

# For final production audit: omit --fast and add --strict
# glassalpha audit --config configs/production_audit.yaml --output prod_audit.pdf --strict
```

### Step 3: Hand off to ML Engineer

Share with ML engineering team:

- Audit configuration file
- Model artifact (`.pkl` file)
- Notebook with analysis
- Requirements file

See [ML Engineer Workflow](ml-engineer-workflow.md) for production integration.

## Troubleshooting

### Issue: Inline display not working

**Symptom**: `result` in notebook doesn't show HTML summary

**Solution**:

```python
# Explicitly display
from IPython.display import display
display(result)

# Or use direct method
result.display()
```

### Issue: Plot methods not found

**Symptom**: `AttributeError: 'AuditResult' has no attribute 'plot'`

**Solution**:

```python
# Update GlassAlpha to latest version
!pip install --upgrade glassalpha

# Or use component-specific plots
result.fairness.plot_group_metrics()
result.calibration.plot()
```

### Issue: Slow audit in notebook

**Symptom**: Audit takes >30 seconds in notebook

**Solution**:

```python
# Reduce explainer samples for faster iteration
result = ga.audit.from_model(
    model, X_test, y_test,
    protected_attributes=attrs,
    explainer_samples=100,  # Default 1000
    random_seed=42
)
```

### Issue: Memory error with large dataset

**Symptom**: Kernel dies during audit

**Solution**:

```python
# Sample data for exploration
X_sample = X_test.sample(n=1000, random_state=42)
y_sample = y_test.loc[X_sample.index]

result = ga.audit.from_model(model, X_sample, y_sample, ...)
```

## Related Resources

### For Researchers

- [Fairness Metrics Reference](../reference/fairness-metrics.md) - Statistical definitions
- [Calibration Analysis](../reference/calibration.md) - Probability calibration

### For Transition to Production

- [ML Engineer Workflow](ml-engineer-workflow.md) - CI/CD integration
- [Configuration Guide](../getting-started/configuration.md) - Full config reference
- [Compliance Officer Workflow](compliance-workflow.md) - Regulatory submission

### Examples

- [German Credit Audit](../examples/german-credit-audit.md) - Complete walkthrough
- [Example Tutorials](../examples/index.md) - Walkthrough guides
- [Interactive Notebooks](https://github.com/GlassAlpha/glassalpha/tree/main/examples/notebooks) - Jupyter notebooks on GitHub

## Support

For research-specific questions:

- GitHub Discussions: [GlassAlpha/glassalpha/discussions](https://github.com/GlassAlpha/glassalpha/discussions)
- Email: [research@glassalpha.com](mailto:research@glassalpha.com)
- Documentation: [glassalpha.com](https://glassalpha.com)
