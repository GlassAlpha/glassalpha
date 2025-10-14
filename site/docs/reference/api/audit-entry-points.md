# Audit Entry Points

Complete reference for `from_model()`, `from_predictions()`, and `from_config()` - the three ways to generate audits programmatically.

---

## Quick Comparison

| Method                 | Input            | Use Case                                  | Explanations       | Determinism      |
| ---------------------- | ---------------- | ----------------------------------------- | ------------------ | ---------------- |
| **from_model()**       | Model + data     | Notebook exploration, feature engineering | ✅ SHAP + recourse | ⚠️ Requires seed |
| **from_predictions()** | Predictions only | Model deleted, external predictions       | ❌ No              | ✅ Deterministic |
| **from_config()**      | YAML config      | CI/CD, reproducibility verification       | ✅ If enabled      | ✅ Enforced      |

---

## from_model()

Generate audit from trained sklearn-compatible model.

### Signature

```python
glassalpha.audit.from_model(
    model,
    X,
    y,
    *,
    protected_attributes=None,
    sample_weight=None,
    random_seed=42,
    feature_names=None,
    class_names=None,
    explain=True,
    recourse=False,
    calibration=True,
    stability=False
) -> AuditResult
```

### Parameters

**Required:**

- **model** (`Any`) - Fitted sklearn-compatible model with `predict()` method
- **X** (`pd.DataFrame | np.ndarray` of shape `(n_samples, n_features)`) - Test features
- **y** (`pd.Series | np.ndarray` of shape `(n_samples,)`) - True labels (binary: 0/1)

**Optional:**

- **protected_attributes** (`dict[str, pd.Series | np.ndarray]`) - Protected attributes for fairness analysis. Keys are attribute names, values are arrays. Missing values (NaN) mapped to "Unknown" category.

  Example: `{"gender": gender_array, "race": race_array}`

- **sample_weight** (`pd.Series | np.ndarray` of shape `(n_samples,)`) - Sample weights for weighted metrics

- **random_seed** (`int`, default: `42`) - Random seed for deterministic SHAP sampling. Set explicitly for reproducibility.

- **feature_names** (`list[str]`) - Feature names (inferred from DataFrame columns if not provided)

- **class_names** (`list[str]`) - Class names (e.g., `["Denied", "Approved"]`)

- **explain** (`bool`, default: `True`) - Generate SHAP explanations. Set to `False` for faster audits without explanations.

- **recourse** (`bool`, default: `False`) - Generate recourse recommendations (counterfactuals). Slower than explanations.

- **calibration** (`bool`, default: `True`) - Compute calibration metrics. Requires `predict_proba()` method.

- **stability** (`bool`, default: `False`) - Run stability tests (monotonicity, feature importance). Slower.

### Returns

**AuditResult** - Immutable result object with:

- `result.id` - SHA-256 hash (deterministic with same seed)
- `result.performance` - Performance metrics (accuracy, precision, recall, F1, ROC AUC, etc.)
- `result.fairness` - Fairness metrics by protected group (demographic parity, equalized odds, etc.)
- `result.calibration` - Calibration metrics (ECE, Brier score, calibration curve)
- `result.stability` - Stability test results (monotonicity violations, feature importance validation)
- `result.explanations` - SHAP explanation summary (if `explain=True`)
- `result.recourse` - Recourse recommendations (if `recourse=True`)

### Raises

- **GAE1001** (`InvalidProtectedAttributesError`) - Invalid `protected_attributes` format
- **GAE1003** (`LengthMismatchError`) - Length mismatch between X, y, or protected_attributes
- **GAE1004** (`NonBinaryClassificationError`) - Non-binary classification detected (>2 classes)
- **GAE1012** (`MultiIndexNotSupportedError`) - X or y has MultiIndex

### Examples

**Basic audit:**

```python
import glassalpha as ga
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Generate audit
result = ga.audit.from_model(model, X_test, y_test, random_seed=42)

# Explore metrics
print(f"Accuracy: {result.performance['accuracy']:.3f}")
print(f"AUC: {result.performance.roc_auc:.3f}")

# Export PDF
result.to_pdf("audit.pdf")
```

**With protected attributes:**

```python
# Audit fairness across gender and race
result = ga.audit.from_model(
    model=model,
    X=X_test,
    y=y_test,
    protected_attributes={
        "gender": gender,  # pd.Series or np.ndarray
        "race": race
    },
    random_seed=42
)

# Check fairness
print(f"Demographic parity difference: {result.fairness.demographic_parity_max_diff:.3f}")
print(f"Equalized odds difference: {result.fairness.equalized_odds_max_diff:.3f}")
```

**Fast audit (no explanations):**

```python
# Skip SHAP for faster execution
result = ga.audit.from_model(
    model=model,
    X=X_test,
    y=y_test,
    explain=False,
    calibration=False,
    random_seed=42
)
```

**Missing values in protected attributes:**

```python
# NaN values automatically mapped to "Unknown"
gender_with_missing = pd.Series([0, 1, np.nan, 1, 0])

result = ga.audit.from_model(
    model=model,
    X=X_test,
    y=y_test,
    protected_attributes={"gender": gender_with_missing},
    random_seed=42
)
# "Unknown" treated as third category
```

### Notes

- **Determinism**: Results are deterministic when `random_seed` is fixed and model predictions are deterministic
- **Performance**: SHAP explanations take ~80% of execution time. Set `explain=False` for 5x speedup.
- **Probabilities**: AUC, calibration metrics require `predict_proba()`. Models with only `decision_function()` will skip these metrics.
- **Multi-class support**: Supports both binary and multi-class classification with appropriate metrics for each.

### Related

- [AuditResult API](#auditresult-api) - Result object documentation
- [Missing Data Guide](../../guides/missing-data.md) - Handling missing values in protected attributes
- [Probability Requirements](../../guides/probability-requirements.md) - When probabilities are required
- [Example: German Credit](../../examples/german-credit-audit.md) - Full walkthrough

---

## from_predictions()

Generate audit from predictions (no model required).

Use when you have predictions but not the model itself (e.g., model deleted, external predictions, compliance verification).

### Signature

```python
glassalpha.audit.from_predictions(
    y_true,
    y_pred,
    y_proba=None,
    *,
    protected_attributes=None,
    sample_weight=None,
    random_seed=42,
    class_names=None,
    model_fingerprint=None,
    calibration=True
) -> AuditResult
```

### Parameters

**Required:**

- **y_true** (`pd.Series | np.ndarray` of shape `(n_samples,)`) - True labels (binary: 0/1)
- **y_pred** (`pd.Series | np.ndarray` of shape `(n_samples,)`) - Predicted labels (binary: 0/1)

**Optional:**

- **y_proba** (`pd.DataFrame | np.ndarray` of shape `(n_samples, n_classes)` or `(n_samples,)`) - Predicted probabilities. If 2D, uses positive class column (index 1). If 1D, treats as positive class probabilities. Required for AUC, Brier score, calibration metrics.

- **protected_attributes** (`dict[str, pd.Series | np.ndarray]`) - Protected attributes (same as `from_model()`)

- **sample_weight** (`pd.Series | np.ndarray` of shape `(n_samples,)`) - Sample weights

- **random_seed** (`int`, default: `42`) - Random seed for consistency

- **class_names** (`list[str]`) - Class names for report

- **model_fingerprint** (`str`) - Optional model hash for tracking (default: "unknown")

- **calibration** (`bool`, default: `True`) - Compute calibration metrics. Requires `y_proba`.

### Returns

**AuditResult** - Result object (same as `from_model()`) but:

- **No explanations** - Cannot compute SHAP without model
- **No recourse** - Cannot compute counterfactuals without model

### Raises

- **GAE1001** (`InvalidProtectedAttributesError`) - Invalid `protected_attributes` format
- **GAE1003** (`LengthMismatchError`) - Length mismatch between arrays
- **GAE1004** (`NonBinaryClassificationError`) - Non-binary classification

### Examples

**Binary classification with probabilities:**

```python
import glassalpha as ga

# You have predictions from an external model
result = ga.audit.from_predictions(
    y_true=y_test,
    y_pred=predictions,
    y_proba=probabilities[:, 1],  # Positive class probabilities
    protected_attributes={"gender": gender},
    random_seed=42
)

print(f"Accuracy: {result.performance['accuracy']:.3f}")
print(f"AUC: {result.performance.roc_auc:.3f}")
```

**Without probabilities (no AUC/calibration):**

```python
# Only predictions available
result = ga.audit.from_predictions(
    y_true=y_test,
    y_pred=predictions,
    protected_attributes={"gender": gender},
    calibration=False,  # Skip calibration (no probabilities)
    random_seed=42
)

# AUC not available
print(result.performance.get("roc_auc"))  # None
```

**Verify external model predictions:**

```python
# Compliance verification: check if external predictions meet fairness requirements
result = ga.audit.from_predictions(
    y_true=y_test,
    y_pred=external_predictions,
    y_proba=external_probabilities,
    protected_attributes={"race": race, "gender": gender},
    model_fingerprint="external_model_v2.1",
    random_seed=42
)

# Check fairness
if result.fairness.demographic_parity_max_diff > 0.10:
    print("⚠️ Fairness threshold exceeded")
```

### Notes

- **No explanations**: SHAP explanations require model access. Use `from_model()` if you need explanations.
- **Probabilities optional**: Can audit without probabilities, but skips AUC and calibration metrics.
- **Determinism**: Fully deterministic (no model randomness).

### Related

- [from_model()](#from_model) - When you have model access
- [Probability Requirements](../../guides/probability-requirements.md) - When probabilities are needed

---

## from_config()

Generate audit from YAML configuration file.

Use for CI/CD, reproducibility verification, and regulatory compliance.

### Signature

```python
glassalpha.audit.from_config(
    config_path
) -> AuditResult
```

### Parameters

- **config_path** (`str | Path`) - Path to YAML configuration file

### Returns

**AuditResult** - Same as `from_model()`, with additional validation:

- **Data hash verification**: Ensures data matches expected hashes
- **Result ID verification**: Optionally verifies result matches expected ID (for reproducibility)

### Raises

- **GAE2002** (`ResultIDMismatchError`) - Result ID doesn't match `expected_result_id` in config
- **GAE2003** (`DataHashMismatchError`) - Data hash doesn't match expected hash
- **FileNotFoundError** - Config or referenced files not found

### Config Schema

```yaml
model:
  path: "models/xgboost.pkl" # Pickled model
  type: "xgboost.XGBClassifier" # For verification

data:
  X_path: "data/X_test.parquet"
  y_path: "data/y_test.parquet"
  protected_attributes:
    gender: "data/gender.parquet"
    race: "data/race.parquet"
  expected_hashes: # Optional: verify data integrity
    X: "sha256:abc123..."
    y: "sha256:def456..."

audit:
  random_seed: 42
  explain: true
  recourse: false
  calibration: true

validation:
  expected_result_id: "abc123..." # Optional: verify reproducibility
```

### Examples

**Basic usage:**

```python
import glassalpha as ga

# Generate audit from config
result = ga.audit.from_config("audit.yaml")

# Export PDF
result.to_pdf("report.pdf")
```

**CI/CD integration:**

```python
import glassalpha as ga
import sys

try:
    result = ga.audit.from_config("audit.yaml")
except ga.ResultIDMismatchError as e:
    print(f"❌ Reproducibility check failed: {e}")
    sys.exit(1)
except ga.DataHashMismatchError as e:
    print(f"❌ Data integrity check failed: {e}")
    sys.exit(1)

print("✅ Audit generated successfully")
result.to_pdf("report.pdf")
```

**Verify reproducibility:**

```yaml
# audit.yaml
validation:
  expected_result_id: "abc123def456..." # From previous run

# Fails if result ID doesn't match (non-deterministic)
```

### Notes

- **Reproducibility**: Set `expected_result_id` to verify byte-identical results
- **Data integrity**: Set `expected_hashes` to verify data hasn't changed
- **CI/CD**: Combine both for full reproducibility verification

### Related

- [Configuration Guide](../../getting-started/configuration.md) - YAML config format
- [Reproducibility in Configuration](../../getting-started/configuration.md#reproducibility-settings) - Ensuring deterministic results
- [CI/CD Examples](../../guides/ml-engineer-workflow.md#cicd-integration) - GitHub Actions examples

---

## AuditResult API

Result object returned by all audit entry points.

### Attributes

#### Core Identity

- **id** (`str`) - SHA-256 hash of result (64 hex characters). Deterministic with same seed.
- **schema_version** (`str`) - Result schema version (e.g., "0.2.0")
- **manifest** (`dict`) - Provenance metadata (model fingerprint, data hashes, random seed, package versions)

#### Metric Sections

- **performance** (`PerformanceMetrics`) - Accuracy, precision, recall, F1, AUC, Brier score, log loss
- **fairness** (`FairnessMetrics`) - Demographic parity, equalized odds, calibration by group
- **calibration** (`CalibrationMetrics`) - Expected calibration error, Brier score, calibration curve
- **stability** (`StabilityMetrics`) - Monotonicity violations, feature importance validation
- **explanations** (`ExplanationSummary | None`) - SHAP values, feature importance (if `explain=True`)
- **recourse** (`RecourseSummary | None`) - Counterfactual recommendations (if `recourse=True`)

### Methods

#### Metric Access

```python
# Dict-style access
result.performance["accuracy"]  # Raises KeyError if missing

# Attribute-style access
result.performance['accuracy']     # Raises GlassAlphaError if missing

# Safe access with default
result.performance.get("roc_auc", None)
```

#### Comparison

```python
# Strict equality (byte-identical IDs)
result1 == result2
```

#### AuditResult.equals() {#auditresultequals}

```python
result.equals(other, rtol=1e-5, atol=1e-8) -> bool
```

Compare two audit results with numerical tolerance for cross-platform reproducibility.

**Parameters:**

- `other` (AuditResult): Another audit result to compare against
- `rtol` (float, default 1e-5): Relative tolerance for floating point comparisons
- `atol` (float, default 1e-8): Absolute tolerance for floating point comparisons

**Returns:**

- `bool`: True if results are numerically equivalent within tolerance

**Notes:**

- Uses `numpy.allclose()` semantics for metric value comparisons
- Compares all metric sections: performance, fairness, calibration, stability
- Ignores metadata like timestamps but validates core audit identity
- Essential for cross-platform validation and CI/CD pipelines

**Examples:**

```python
# Basic usage
if result1.equals(result2):
    print("Results are numerically equivalent")

# Custom tolerance for high-precision metrics
result1.equals(result2, rtol=1e-7, atol=1e-10)

# Cross-platform validation
linux_result.equals(cloud_result, rtol=1e-4)  # Account for platform differences
```

#### Display

```python
# Jupyter display (HTML table)
result

# Summary dict (for logging)
result.summary()

# String representation
print(result)  # AuditResult(id='abc123...', schema_version='0.2.0')
```

#### Export

```python
# Export to PDF
result.to_pdf("audit.pdf", overwrite=False)

# Export to JSON (canonical format)
result.to_json("audit.json", overwrite=False)

# Generate reproduction config
config = result.to_config()

# Save everything (PDF + JSON + config + manifest)
result.save("output_dir/", overwrite=False)
```

### Examples

**Explore metrics:**

```python
result = ga.audit.from_model(model, X_test, y_test, random_seed=42)

# Performance
print(f"Accuracy: {result.performance['accuracy']:.3f}")
print(f"F1: {result.performance.f1:.3f}")

# Fairness
for group, metrics in result.fairness.items():
    if "group_" in group:
        print(f"{group}: TPR={metrics['tpr']:.3f}")
```

**Compare results:**

```python
result_v1 = ga.audit.from_model(model_v1, X_test, y_test, random_seed=42)
result_v2 = ga.audit.from_model(model_v2, X_test, y_test, random_seed=42)

# Strict comparison (fails if any difference)
if result_v1 == result_v2:
    print("Byte-identical results")

# Tolerance-based (handles floating-point differences)
if result_v1.equals(result_v2, rtol=1e-5):
    print("Functionally equivalent results")
```

**Reproduce audit:**

```python
# Step 1: Generate audit
result = ga.audit.from_model(model, X_test, y_test, random_seed=42)

# Step 2: Export config
config = result.to_config()
# config = {
#     "model": {"fingerprint": "...", "type": "xgboost.XGBClassifier"},
#     "data": {"X_hash": "sha256:...", "y_hash": "sha256:..."},
#     "random_seed": 42,
#     "expected_result_id": "abc123..."
# }

# Step 3: Verify reproducibility later
result2 = ga.audit.from_config("audit.yaml")  # Uses config above
assert result.id == result2.id  # Byte-identical
```

### Notes

- **Immutability**: All metrics are deeply immutable (frozen dataclasses, read-only arrays)
- **Hashability**: Results can be used in sets and as dict keys
- **Picklability**: Results can be pickled for caching

### Related

- [Metric Definitions](../fairness-metrics.md) - Detailed metric explanations
- [Reproducibility in Configuration](../../getting-started/configuration.md#reproducibility-settings) - Ensuring deterministic results

---

## Error Reference

All errors inherit from `GlassAlphaError` and include:

- **code** - Machine-readable error code (e.g., "GAE1001")
- **message** - Human-readable description
- **fix** - Actionable fix suggestion
- **docs** - URL to documentation

### Input Validation Errors

| Code    | Error                           | When Raised                           |
| ------- | ------------------------------- | ------------------------------------- |
| GAE1001 | InvalidProtectedAttributesError | Invalid `protected_attributes` format |
| GAE1003 | LengthMismatchError             | Array length mismatch                 |
| GAE1004 | NonBinaryClassificationError    | >2 classes detected                   |
| GAE1005 | UnsupportedMissingPolicyError   | Invalid `missing_policy` value        |
| GAE1008 | NoPredictProbaError             | Model has no `predict_proba()`        |
| GAE1009 | AUCWithoutProbaError            | Accessing AUC without probabilities   |
| GAE1012 | MultiIndexNotSupportedError     | DataFrame has MultiIndex              |

### Data/Result Validation Errors

| Code    | Error                 | When Raised                      |
| ------- | --------------------- | -------------------------------- |
| GAE2002 | ResultIDMismatchError | Result ID doesn't match expected |
| GAE2003 | DataHashMismatchError | Data hash doesn't match expected |

### File Operation Errors

| Code    | Error           | When Raised                       |
| ------- | --------------- | --------------------------------- |
| GAE4001 | FileExistsError | File exists and `overwrite=False` |

### Example

```python
try:
    result = ga.audit.from_model(model, X_test, y_test)
except ga.InvalidProtectedAttributesError as e:
    print(e.code)     # "GAE1001"
    print(e.message)  # "Invalid protected_attributes format: ..."
    print(e.fix)      # "Use dict with string keys mapping to arrays..."
    print(e.docs)     # "https://glassalpha.com/errors/GAE1001"
```

---

## Related Documentation

- **[Quick Start](../../getting-started/quickstart.md)** - First audit tutorial
- **[Configuration Guide](../../getting-started/configuration.md)** - YAML config format
- **[Missing Data Guide](../../guides/missing-data.md)** - Handling NaN in protected attributes
- **[Probability Requirements](../../guides/probability-requirements.md)** - When probabilities are needed
- **[Reproducibility in Configuration](../../getting-started/configuration.md#reproducibility-settings)** - Ensuring deterministic results
- **[Quickstart Notebook](https://colab.research.google.com/github/GlassAlpha/glassalpha/blob/main/examples/notebooks/quickstart_colab.ipynb)** - Try in Colab
- **[Example Tutorials](../../examples/index.md)** - Detailed walkthroughs
- **[CLI Reference](../cli.md)** - Command-line interface

---

## Support

- **GitHub Issues**: [Report bugs](https://github.com/GlassAlpha/glassalpha/issues)
- **Discussions**: [Ask questions](https://github.com/GlassAlpha/glassalpha/discussions)
- **Email**: [contact@glassalpha.com](mailto:contact@glassalpha.com)
