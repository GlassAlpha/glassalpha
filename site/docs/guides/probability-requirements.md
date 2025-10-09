# Probability Requirements

Guide to when predicted probabilities are required vs optional for audit metrics.

---

## Quick Summary

| Metric Type                         | Requires `y_proba` | Alternative                       |
| ----------------------------------- | ------------------ | --------------------------------- |
| **Accuracy, Precision, Recall, F1** | ❌ No              | Decision labels only              |
| **ROC AUC, PR AUC**                 | ✅ Yes             | Skip or use `decision_function()` |
| **Brier Score, Log Loss**           | ✅ Yes             | Skip                              |
| **Calibration (ECE, curves)**       | ✅ Yes             | Skip calibration analysis         |
| **Fairness (DP, EO)**               | ❌ No              | Works with labels                 |
| **SHAP Explanations**               | ❌ No              | Works with any `predict()`        |

---

## Why Probabilities Matter

**Predicted probabilities** (y_proba) represent model confidence:

- `y_proba[i] = 0.85` → Model is 85% confident this is positive class
- `y_pred[i] = 1` → Binary decision (threshold usually 0.5)

**Why needed for some metrics:**

- **ROC AUC**: Ranks predictions by confidence (requires probabilities)
- **Calibration**: Checks if probabilities match actual rates (requires probabilities)
- **Brier score**: Measures

probability accuracy (requires probabilities)

---

## Metrics by Probability Requirement

### No Probabilities Required

These metrics work with predicted labels only:

```python
import glassalpha as ga

# No predict_proba() needed
result = ga.audit.from_predictions(
    y_true=y_test,
    y_pred=predictions,  # Labels only
    protected_attributes={"gender": gender},
    calibration=False,  # Skip calibration
    random_seed=42
)

# Available metrics
result.performance['accuracy']      # ✅
result.performance.precision     # ✅
result.performance.recall        # ✅
result.performance.f1            # ✅
result.fairness.demographic_parity_diff  # ✅
result.fairness.equalized_odds_diff      # ✅

# Not available (require probabilities)
result.performance.get("roc_auc")      # None
result.performance.get("brier_score")  # None
```

### Probabilities Required

These metrics need predicted probabilities:

```python
# With probabilities
result = ga.audit.from_model(
    model=model,  # Must have predict_proba()
    X=X_test,
    y=y_test,
    random_seed=42
)

# Additional metrics available
result.performance.roc_auc       # ✅ Requires probabilities
result.performance.pr_auc        # ✅ Requires probabilities
result.performance.brier_score   # ✅ Requires probabilities
result.performance.log_loss      # ✅ Requires probabilities
result.calibration.ece           # ✅ Requires probabilities
```

---

## Getting Probabilities from Models

### sklearn Models

Most sklearn classifiers support `predict_proba()`:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import glassalpha as ga

# LogisticRegression (has predict_proba by default)
model = LogisticRegression()
model.fit(X_train, y_train)

result = ga.audit.from_model(model, X_test, y_test, random_seed=42)
print(result.performance.roc_auc)  # ✅ Works

# RandomForest (has predict_proba by default)
model = RandomForestClassifier()
model.fit(X_train, y_train)

result = ga.audit.from_model(model, X_test, y_test, random_seed=42)
print(result.performance.roc_auc)  # ✅ Works
```

### SVM (Special Case)

**LinearSVC**: No `predict_proba()` by default. Use `probability=True`:

```python
from sklearn.svm import SVC

# ❌ No probabilities
model = SVC()
model.fit(X_train, y_train)

result = ga.audit.from_model(model, X_test, y_test, random_seed=42)
print(result.performance.get("roc_auc"))  # None (no probabilities)

# ✅ With probabilities (WARNING: slower!)
model = SVC(probability=True)
model.fit(X_train, y_train)

result = ga.audit.from_model(model, X_test, y_test, random_seed=42)
print(result.performance.roc_auc)  # Works
```

**Performance impact**: `probability=True` requires extra cross-validation during training (~5x slower).

### XGBoost / LightGBM

Both support probabilities by default:

```python
import xgboost as xgb
import lightgbm as lgb
import glassalpha as ga

# XGBoost
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
result = ga.audit.from_model(model, X_test, y_test, random_seed=42)
print(result.performance.roc_auc)  # ✅ Works

# LightGBM
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
result = ga.audit.from_model(model, X_test, y_test, random_seed=42)
print(result.performance.roc_auc)  # ✅ Works
```

---

## Handling Models Without Probabilities

### Option 1: Skip Probability-Based Metrics

```python
import glassalpha as ga

# Model without predict_proba()
result = ga.audit.from_model(
    model=svm_model,  # No predict_proba()
    X=X_test,
    y=y_test,
    calibration=False,  # Skip calibration
    random_seed=42
)

# Still get core metrics
print(f"Accuracy: {result.performance['accuracy']:.3f}")
print(f"F1: {result.performance.f1:.3f}")
print(f"Fairness: {result.fairness.demographic_parity_diff:.3f}")
```

### Option 2: Use decision_function() as Proxy

Some models have `decision_function()` (signed distance to decision boundary):

```python
from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(X_train, y_train)

# Get decision function scores
scores = model.decision_function(X_test)

# Convert to probabilities (sigmoid transformation)
from scipy.special import expit
probabilities = expit(scores)

# Use with from_predictions()
result = ga.audit.from_predictions(
    y_true=y_test,
    y_pred=model.predict(X_test),
    y_proba=probabilities,  # Pseudo-probabilities
    random_seed=42
)

# AUC works (but calibration may be poor)
print(result.performance.roc_auc)  # ✅
print(result.calibration.ece)      # ⚠️ May be miscalibrated
```

**Warning**: `decision_function()` scores are NOT calibrated probabilities. AUC is valid, but calibration metrics may be misleading.

### Option 3: Calibrate Probabilities

If using `decision_function()`, calibrate for better probability estimates:

```python
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import glassalpha as ga

# Train LinearSVC
base_model = LinearSVC()
base_model.fit(X_train, y_train)

# Calibrate probabilities
calibrated_model = CalibratedClassifierCV(base_model, cv=3)
calibrated_model.fit(X_train, y_train)

# Now has predict_proba()
result = ga.audit.from_model(
    model=calibrated_model,
    X=X_test,
    y=y_test,
    random_seed=42
)

# All metrics available
print(result.performance.roc_auc)  # ✅
print(result.calibration.ece)      # ✅ Calibrated
```

---

## Error Handling

### Accessing Metrics Without Probabilities

GlassAlpha raises helpful errors:

```python
from sklearn.svm import LinearSVC
import glassalpha as ga

model = LinearSVC()  # No predict_proba()
model.fit(X_train, y_train)

result = ga.audit.from_model(model, X_test, y_test, random_seed=42)

# Accessing AUC raises error
try:
    print(result.performance.roc_auc)
except ga.AUCWithoutProbaError as e:
    print(e.code)     # "GAE1009"
    print(e.message)  # "Metric 'roc_auc' requires y_proba"
    print(e.fix)      # "Either: (1) Use model with predict_proba()..."
```

### Safe Access

Use `.get()` to avoid errors:

```python
# Returns None if not available
auc = result.performance.get("roc_auc")
if auc is not None:
    print(f"AUC: {auc:.3f}")
else:
    print("AUC not available (no probabilities)")
```

---

## Binary vs Multi-Class Probabilities

### Binary Classification

Probabilities can be 1D (positive class only) or 2D (both classes):

```python
import numpy as np
import glassalpha as ga

# Option 1: 1D array (positive class probabilities)
y_proba = model.predict_proba(X_test)[:, 1]  # Shape: (n_samples,)

result = ga.audit.from_predictions(
    y_true=y_test,
    y_pred=predictions,
    y_proba=y_proba,  # 1D array
    random_seed=42
)

# Option 2: 2D array (both classes)
y_proba_full = model.predict_proba(X_test)  # Shape: (n_samples, 2)

result = ga.audit.from_predictions(
    y_true=y_test,
    y_pred=predictions,
    y_proba=y_proba_full[:, 1],  # Extract positive class
    random_seed=42
)
```

**GlassAlpha convention**: Always use positive class probabilities (class 1) for binary classification.

### Multi-Class (Not Yet Supported)

```python
# Multi-class probabilities (3 classes)
y_proba = model.predict_proba(X_test)  # Shape: (n_samples, 3)

# ❌ Not supported in v0.2
result = ga.audit.from_model(...)
# Raises: GAE1004 NonBinaryClassificationError
```

**Coming in v0.3**: Multi-class support with one-vs-rest fairness analysis.

---

## Best Practices

### 1. Always Use Probabilities for Compliance Audits

**Why**: Regulators may ask for calibration analysis, which requires probabilities.

```python
# ✅ Best practice
model = LogisticRegression()  # Has predict_proba()
model.fit(X_train, y_train)

result = ga.audit.from_model(model, X_test, y_test, random_seed=42)

# Full audit with calibration
result.to_pdf("compliance_audit.pdf")
```

### 2. Document When Probabilities Not Available

If using model without `predict_proba()`, document limitation:

```python
result = ga.audit.from_model(
    model=svm_model,  # No predict_proba()
    X=X_test,
    y=y_test,
    calibration=False,
    random_seed=42
)

# Add note to manifest
result.manifest["limitations"] = "Model does not provide probability estimates. AUC and calibration metrics not available."
```

### 3. Verify Calibration Before Trusting Probabilities

Not all models with `predict_proba()` are well-calibrated:

```python
result = ga.audit.from_model(model, X_test, y_test, random_seed=42)

# Check calibration quality
ece = result.calibration.ece
if ece > 0.10:
    print(f"⚠️ Poor calibration (ECE={ece:.3f}). Consider CalibratedClassifierCV.")
```

**Good calibration**: ECE < 0.05
**Poor calibration**: ECE > 0.10

---

## Checklist

**Before training:**

- [ ] Choose model with `predict_proba()` if possible
- [ ] For SVM, use `probability=True` (if needed)
- [ ] Budget extra training time for probability-enabled models

**During audit:**

- [ ] Verify which metrics are available
- [ ] Use `.get()` for safe access to probability-based metrics
- [ ] Document if probabilities not available

**For compliance:**

- [ ] Always include calibration analysis (requires probabilities)
- [ ] Check ECE < 0.10 for well-calibrated model
- [ ] Document calibration method if using `CalibratedClassifierCV`

---

## Examples

### Example 1: Logistic Regression (Has Probabilities)

```python
from sklearn.linear_model import LogisticRegression
import glassalpha as ga

model = LogisticRegression()
model.fit(X_train, y_train)

result = ga.audit.from_model(
    model=model,
    X=X_test,
    y=y_test,
    protected_attributes={"gender": gender},
    random_seed=42
)

# All metrics available
print(f"Accuracy: {result.performance['accuracy']:.3f}")
print(f"AUC: {result.performance.roc_auc:.3f}")
print(f"Brier: {result.performance.brier_score:.3f}")
print(f"ECE: {result.calibration.ece:.3f}")
```

### Example 2: LinearSVC (No Probabilities)

```python
from sklearn.svm import LinearSVC
import glassalpha as ga

model = LinearSVC()
model.fit(X_train, y_train)

result = ga.audit.from_model(
    model=model,
    X=X_test,
    y=y_test,
    protected_attributes={"gender": gender},
    calibration=False,  # Skip calibration
    random_seed=42
)

# Core metrics available
print(f"Accuracy: {result.performance['accuracy']:.3f}")
print(f"F1: {result.performance.f1:.3f}")

# Probability-based metrics not available
print(result.performance.get("roc_auc"))  # None
```

### Example 3: Calibrating SVM

```python
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import glassalpha as ga

# Train + calibrate
base_model = LinearSVC()
model = CalibratedClassifierCV(base_model, cv=3)
model.fit(X_train, y_train)

result = ga.audit.from_model(
    model=model,
    X=X_test,
    y=y_test,
    protected_attributes={"gender": gender},
    random_seed=42
)

# All metrics available (calibrated probabilities)
print(f"AUC: {result.performance.roc_auc:.3f}")
print(f"ECE: {result.calibration.ece:.3f}")  # Should be low
```

---

## Related

- **[Audit Entry Points](../reference/api/audit-entry-points.md)** - from_model(), from_predictions() API reference
- **[Calibration Analysis](../reference/calibration.md)** - Interpreting calibration metrics
- **[Missing Data Guide](missing-data.md)** - Handling NaN in protected attributes
- **[Supported Models](../reference/supported-models.md)** - Model compatibility matrix

---

## Support

- **GitHub Issues**: [Report bugs](https://github.com/GlassAlpha/glassalpha/issues)
- **Discussions**: [Ask questions](https://github.com/GlassAlpha/glassalpha/discussions)
