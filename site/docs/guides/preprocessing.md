# Working with Categorical Data

## Quick Start: Categorical Data

**Problem**: Most real datasets have categorical features (strings, categories), but sklearn models require numeric input only.

**Solution**: Preprocess categorical features before training and auditing.

### Step 1: Identify Categorical Columns

```python
import pandas as pd
import glassalpha as ga

# Load your data
data = ga.datasets.load_german_credit()

# Find categorical columns
cat_cols = data.select_dtypes(include=['object', 'category']).columns
print(f"Categorical columns: {list(cat_cols)}")
# Output: ['checking_account_status', 'credit_history', 'purpose', ...]
```

### Step 2: One-Hot Encode Categorical Features

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Split features and target
X = data.drop('credit_risk', axis=1)
y = data['credit_risk']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)

# Transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

### Step 3: Train Model with Preprocessed Data

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)
model.fit(X_train_processed, y_train)
```

### Step 4: Audit with Preprocessed Data

```python
# Generate audit using preprocessed test data
result = ga.audit.from_model(
    model=model,
    X=X_test_processed,
    y=y_test,
    protected_attributes={'gender': data.loc[X_test.index, 'gender']}
)

# Save report
result.to_html('audit_report.html')
```

### Common Patterns

#### Pattern 1: Mixed Categorical and Numeric

```python
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

numeric_cols = X.select_dtypes(include=['number']).columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first'), cat_cols)
])
```

#### Pattern 2: Handle Missing Values

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, cat_cols)
])
```

#### Pattern 3: German Credit Complete Example

```python
import glassalpha as ga
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
data = ga.datasets.load_german_credit()

# Split features and target
X = data.drop('credit_risk', axis=1)
y = data['credit_risk']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess categorical features
cat_cols = X.select_dtypes(include=['object', 'category']).columns
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
], remainder='passthrough')

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train model
model = LogisticRegression(random_state=42, max_iter=5000)
model.fit(X_train_processed, y_train)

# Generate audit
result = ga.audit.from_model(
    model=model,
    X=X_test_processed,
    y=y_test,
    protected_attributes={
        'gender': data.loc[X_test.index, 'gender'],
        'age_group': data.loc[X_test.index, 'age_group']
    }
)

# Save report
result.to_html('german_credit_audit.html')
print(f"✓ Audit complete! Open german_credit_audit.html to view.")
```

### Next Steps

- **For production audits**: See [Preprocessing Artifact Verification](#overview) below
- **For more examples**: See `examples/notebooks/german_credit_walkthrough.ipynb`
- **For troubleshooting**: See [Common Errors](#troubleshooting)

---

## Preprocessing Artifact Verification

## Overview

Preprocessing artifact verification ensures that your ML audit evaluates the model with the **exact same data transformations** used in production. This is critical for regulatory compliance, as auditors need to verify that the audit results match what the model actually sees in deployment.

## The Problem

In production ML systems, raw data goes through preprocessing (scaling, encoding, imputation) before reaching the model. If your audit uses different preprocessing, it's evaluating a **different system** than what's deployed.

**Without artifact verification:**

```
Production:    Raw Data → [Production Preprocessing] → Model → Predictions
Audit:         Raw Data → [Different Preprocessing]  → Model → ❌ Invalid Results
```

**With artifact verification:**

```
Production:    Raw Data → [Production Preprocessing] → Model → Predictions
Audit:         Raw Data → [Same Preprocessing]       → Model → ✓ Valid Results
```

## Quick Start

### 1. Save Your Production Preprocessing Artifact

When training your model, save the fitted preprocessing pipeline:

```python
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Fit your preprocessing on training data
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
preprocessor.fit(X_train)

# Save it
joblib.dump(preprocessor, 'preprocessing.joblib')
```

### 2. Compute Hashes

Generate verification hashes for your artifact:

```bash
glassalpha prep hash preprocessing.joblib --params
```

Output:

```
✓ File hash:   sha256:abc123...
✓ Params hash: sha256:def456...

Config snippet:
preprocessing:
  mode: artifact
  artifact_path: preprocessing.joblib
  expected_file_hash: 'sha256:abc123...'
  expected_params_hash: 'sha256:def456...'
```

### 3. Configure Your Audit

Add the preprocessing section to your audit config:

```yaml
preprocessing:
  mode: artifact
  artifact_path: preprocessing.joblib
  expected_file_hash: "sha256:abc123..."
  expected_params_hash: "sha256:def456..."
  expected_sparse: false
  fail_on_mismatch: true
```

### 4. Run Your Audit

```bash
glassalpha audit --config audit.yaml --output report.pdf
```

The audit will:

1. ✓ Verify file integrity (SHA256 hash)
2. ✓ Verify learned parameters (params hash)
3. ✓ Validate transformer classes (security)
4. ✓ Check runtime version compatibility
5. ✓ Detect unknown categories in audit data
6. ✓ Transform data using production preprocessing
7. ✓ Document everything in the audit report

## Configuration Reference

### Preprocessing Modes

#### `mode: artifact` (Production/Compliance)

Uses a verified preprocessing artifact from production. **Required for regulatory compliance.**

```yaml
preprocessing:
  mode: artifact
  artifact_path: path/to/preprocessor.joblib
  expected_file_hash: "sha256:..."
  expected_params_hash: "sha256:..."
  expected_sparse: false
  fail_on_mismatch: true
```

**When to use:** Always for production audits and regulatory submissions.

#### `mode: auto` (Development/Demo Only)

Automatically fits preprocessing to the audit data. **NOT suitable for compliance.**

```yaml
preprocessing:
  mode: auto # or omit preprocessing section entirely
```

**When to use:** Early development, demos, quickstarts only.

**Warning:** Auto mode produces a prominent warning in audit reports.

### Version Compatibility Policy

Control how strict version checking is:

```yaml
preprocessing:
  version_policy:
    sklearn: exact # Require exact version match (1.3.2 == 1.3.2)
    numpy: patch # Allow patch differences (1.24.1 → 1.24.3)
    scipy: minor # Allow minor differences (1.10.0 → 1.11.2)
```

**Policies:**

- `exact`: Versions must match exactly (most strict)
- `patch`: Allow patch version drift (e.g., 1.3.1 → 1.3.5)
- `minor`: Allow minor version drift (e.g., 1.3.0 → 1.5.0)

**Recommendation:** Use `exact` for sklearn in strict mode, `patch` for numpy/scipy.

### Unknown Category Thresholds

Configure when to warn/fail on unknown categories:

```yaml
preprocessing:
  thresholds:
    warn_unknown_rate: 0.01 # Warn if >1% unknown
    fail_unknown_rate: 0.10 # Fail if >10% unknown
```

Unknown categories are values in the audit data that weren't seen during training (e.g., new product codes, new geographic regions).

## CLI Commands

### `glassalpha prep hash`

Compute verification hashes for an artifact.

```bash
# Quick file hash only
glassalpha prep hash preprocessing.joblib

# File + params hash (with config snippet)
glassalpha prep hash preprocessing.joblib --params
```

**Use case:** After saving a new preprocessing artifact, generate hashes for your config.

### `glassalpha prep inspect`

Inspect an artifact and view its learned parameters.

```bash
# Basic inspection
glassalpha prep inspect preprocessing.joblib

# Detailed with all parameters
glassalpha prep inspect preprocessing.joblib --verbose

# Save manifest to JSON
glassalpha prep inspect preprocessing.joblib --output manifest.json
```

**Use case:** Understand what transformations and parameters are in an artifact.

### `glassalpha prep validate`

Validate an artifact before using it in an audit.

```bash
# Full validation with expected hashes
glassalpha prep validate preprocessing.joblib \
    --file-hash sha256:abc123... \
    --params-hash sha256:def456...

# Quick validation (classes + versions only)
glassalpha prep validate preprocessing.joblib --no-check-versions
```

**Use case:** Pre-flight check before running an audit to catch issues early.

## Creating Preprocessing Artifacts

### Basic Example

```python
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Define transformations
numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['education', 'occupation', 'marital_status']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Fit on training data
preprocessor.fit(X_train)

# Save
joblib.dump(preprocessor, 'preprocessing.joblib')
```

### German Credit Example

See `scripts/create_preprocessing_artifacts.py` for a complete example that:

- Loads German Credit dataset
- Defines numeric and categorical features
- Creates sklearn Pipeline with proper transformations
- Generates artifact + manifest with hashes

```bash
# From repo root
python scripts/create_preprocessing_artifacts.py german_credit --output-dir artifacts
```

## Audit Report Output

When using artifact mode, the audit report includes a **Preprocessing Verification** section showing:

### Success Banner (Artifact Mode)

```
✓ Production Artifact Verified

This audit used a verified preprocessing artifact from production,
ensuring the model was evaluated with the exact same transformations
used in deployment.
```

### Preprocessing Summary Table

| Property    | Value              | Status      |
| ----------- | ------------------ | ----------- |
| Mode        | `artifact`         | ✓ Compliant |
| File Hash   | `sha256:abc123...` | ✓ Verified  |
| Params Hash | `sha256:def456...` | ✓ Verified  |

### Preprocessing Components

For each component in the pipeline:

- Component name and class
- Configuration (strategy, handle_unknown, drop, etc.)
- Learned parameters (medians, means, scales, encoder categories)
- Applied columns

### Runtime Version Comparison

| Library | Artifact Version | Audit Version | Status  |
| ------- | ---------------- | ------------- | ------- |
| sklearn | 1.3.2            | 1.3.2         | ✓ Match |
| numpy   | 2.0.1            | 2.0.1         | ✓ Match |
| scipy   | 1.11.3           | 1.11.3        | ✓ Match |

### Unknown Category Detection

| Column     | Unknown Rate | Assessment |
| ---------- | ------------ | ---------- |
| occupation | 0.5%         | ✓ Low      |
| education  | 2.3%         | ⚠ Moderate |

### Warning Banner (Auto Mode)

```
⚠️ WARNING: Non-Compliant Preprocessing Mode

This audit used AUTO preprocessing mode, which is NOT suitable
for regulatory compliance. Auto mode dynamically fits preprocessing
transformers to the audit data, creating a different preprocessing
pipeline than production.

For compliance-grade audits:
- Use mode: artifact in preprocessing config
- Provide the exact preprocessing artifact used in production
- Include both expected_file_hash and expected_params_hash
```

## Troubleshooting

### Hash Mismatch

**Error:**

```
Preprocessing artifact file hash mismatch!
  Expected: sha256:abc123...
  Actual:   sha256:xyz789...
```

**Causes:**

- Artifact file was modified or corrupted
- Wrong artifact file specified
- File was regenerated with different random seed

**Solution:**

- Verify artifact file path is correct
- Regenerate hashes: `glassalpha prep hash artifact.joblib --params`
- Update config with new hashes

### Params Hash Mismatch

**Error:**

```
Preprocessing artifact params hash mismatch!
  Expected: sha256:def456...
  Actual:   sha256:uvw123...
```

**Causes:**

- Artifact was retrained with different data
- Different preprocessing configuration
- sklearn version change affecting parameter representation

**Solution:**

- Ensure using the correct artifact from production
- Regenerate params hash
- Review preprocessing configuration changes

### Unsupported Transformer Class

**Error:**

```
Unsupported transformer sklearn.preprocessing._function_transformer.FunctionTransformer
```

**Cause:** The artifact contains a transformer not on the security allowlist.

**Solution:**

- Use supported transformers only (see Supported Classes below)
- If you need custom transformations, implement them as subclasses of supported transformers
- Contact support if you need additional transformers allowlisted

### Version Mismatch Warning

**Warning:**

```
⚠ Version warning: sklearn version mismatch: artifact=1.3.2, audit=1.4.0
```

**Cause:** Artifact was created with different library versions than audit environment.

**Solutions:**

- **Option 1 (Recommended):** Match versions exactly in audit environment
  ```bash
  pip install sklearn==1.3.2 numpy==2.0.1 scipy==1.11.3
  ```
- **Option 2:** Adjust version policy to allow differences
  ```yaml
  preprocessing:
    version_policy:
      sklearn: minor # Allow 1.3.x → 1.4.x
  ```
- **Option 3:** Regenerate artifact in current environment (if acceptable for your use case)

### High Unknown Category Rate

**Warning:**

```
Column 'occupation' has 15.3% unknown categories (threshold: 10.0%)
```

**Cause:** Audit data contains categories not seen during training.

**Implications:**

- Possible data distribution shift
- New categories in production data
- Data quality issues

**Solutions:**

- **If expected:** Increase threshold in config
  ```yaml
  preprocessing:
    thresholds:
      fail_unknown_rate: 0.20 # Allow up to 20%
  ```
- **If unexpected:** Investigate data changes and consider retraining

### Sparse/Dense Mismatch

**Error:**

```
Sparsity mismatch: expected sparse=True, got sparse=False
```

**Cause:** Preprocessor output format doesn't match expectation.

**Solution:**

- Check `sparse_output` setting in encoders
- Update `expected_sparse` in config to match actual output
- Ensure consistency between training and audit environments

## Supported Transformer Classes

For security, only these sklearn classes are allowed in preprocessing artifacts:

**Pipelines & Composition:**

- `sklearn.pipeline.Pipeline`
- `sklearn.compose.ColumnTransformer`

**Imputation:**

- `sklearn.impute.SimpleImputer`

**Encoding:**

- `sklearn.preprocessing.OneHotEncoder`
- `sklearn.preprocessing.OrdinalEncoder`

**Scaling:**

- `sklearn.preprocessing.StandardScaler`
- `sklearn.preprocessing.MinMaxScaler`
- `sklearn.preprocessing.RobustScaler`

**Other Transformations:**

- `sklearn.preprocessing.KBinsDiscretizer`
- `sklearn.preprocessing.PolynomialFeatures`

If you need additional transformers, please submit a feature request with your use case.

## Best Practices

### 1. Save Artifacts During Model Training

```python
# During training
preprocessor.fit(X_train)
joblib.dump(preprocessor, f'preprocessing_v{model_version}.joblib')

# Compute and store hashes
file_hash = compute_file_hash('preprocessing_v1.joblib')
params_hash = compute_params_hash(extract_manifest(preprocessor))

# Store hashes in your version control or config management system
# Example: Save to YAML config
with open('preprocessing_hashes_v1.yaml', 'w') as f:
    yaml.dump({
        'version': 1,
        'file_hash': file_hash,
        'params_hash': params_hash
    }, f)
```

### 2. Version Your Artifacts

Use semantic versioning for preprocessing artifacts:

```
preprocessing_v1.0.0.joblib  # Initial production release
preprocessing_v1.0.1.joblib  # Bug fix (parameter update)
preprocessing_v1.1.0.joblib  # New feature (additional column)
preprocessing_v2.0.0.joblib  # Breaking change (removed columns)
```

### 3. Document Preprocessing Changes

Maintain a changelog for preprocessing updates:

```markdown
## Preprocessing v1.1.0 (2024-01-15)

- Added `new_feature` to numeric features
- Updated StandardScaler with new mean/std from expanded dataset
- File hash: sha256:abc123...
- Params hash: sha256:def456...
```

### 4. Test Artifacts Before Production

```bash
# Validate artifact
glassalpha prep validate preprocessing.joblib \
    --file-hash sha256:abc123... \
    --params-hash sha256:def456...

# Run test audit
glassalpha audit --config test_audit.yaml --output test_report.pdf

# Review report preprocessing section
```

### 5. Store Artifacts Securely

- Use artifact registries (MLflow, DVC, etc.)
- Implement access controls
- Enable audit logging for artifact access
- Back up artifacts with model checkpoints

### 6. Monitor Unknown Category Rates

Track unknown category rates over time to detect:

- Data distribution shifts
- New categorical values in production
- Data quality degradation

Set up alerts when rates exceed thresholds:

```yaml
preprocessing:
  thresholds:
    warn_unknown_rate: 0.01 # Alert at 1%
    fail_unknown_rate: 0.05 # Block audit at 5%
```

## Strict Mode Requirements

When running audits in strict mode (`strict_mode: true`), preprocessing artifact verification has additional requirements:

**Required:**

- `mode: artifact` (auto mode is not allowed)
- `artifact_path` must be specified
- `expected_file_hash` must be provided
- `expected_params_hash` must be provided

**Enforced:**

- Hash mismatches cause audit failure
- Version mismatches are treated as errors (not warnings)
- Unknown categories above `fail_unknown_rate` cause failure

**Example strict mode config:**

```yaml
strict_mode: true

preprocessing:
  mode: artifact
  artifact_path: preprocessing.joblib
  expected_file_hash: "sha256:abc123..."
  expected_params_hash: "sha256:def456..."
  fail_on_mismatch: true
  version_policy:
    sklearn: exact
    numpy: patch
    scipy: patch
```

## FAQ

**Q: Can I use preprocessing artifacts with different ML frameworks (PyTorch, TensorFlow)?**

A: Currently, only sklearn preprocessing is supported. Support for other frameworks is planned. You can use sklearn for preprocessing even if your model is in another framework.

**Q: What if my preprocessing includes custom functions?**

A: Custom functions in preprocessing artifacts are not supported for security reasons. Consider:

1. Using sklearn's built-in transformers
2. Creating a subclass of a supported transformer
3. Pre-processing data before the artifact (with full documentation)

**Q: How do I handle preprocessing that depends on the current date or external data?**

A: For audit reproducibility, preprocessing should be deterministic. Options:

1. Snapshot external data at training time
2. Include date-dependent features in the artifact's learned parameters
3. Document non-deterministic preprocessing in audit notes

**Q: Can I update an artifact without retraining the model?**

A: No. The artifact must match what the model was trained with. If you update preprocessing, you must retrain the model and create a new artifact.

**Q: What's the performance impact of artifact verification?**

A: Minimal (<1 second overhead):

- File hash: <0.1s
- Loading artifact: <0.5s
- Params hash: <0.1s
- Validation: <0.1s
- Total: <1s

The actual transformation time is the same as without verification.

## Support

If you encounter issues with preprocessing artifact verification:

1. Check this guide's troubleshooting section
2. Run `glassalpha prep validate` for detailed diagnostics
3. Review audit logs for specific error messages
4. Open an issue: https://github.com/yourusername/glassalpha/issues

## Determinism {#determinism}

Preprocessing artifacts must be deterministic for reproducible audits. GlassAlpha enforces this through:

### Hash-Based Validation

Every preprocessing artifact includes dual hashes:

- **File hash**: SHA256 of the serialized artifact
- **Params hash**: Canonical hash of learned parameters (immutables, thresholds, etc.)

### Version Compatibility

Artifacts are validated against the current GlassAlpha version. The preprocessing module maintains backward compatibility for minor version changes but requires explicit migration for major version updates.

### Strict Mode Requirements

In strict mode, preprocessing artifacts must be explicitly provided via `mode: artifact`. Auto mode is not permitted for regulatory submissions.

## Related Documentation

- [Configuration Reference](../getting-started/configuration.md)
- [CLI Commands](../reference/cli.md)
- [Troubleshooting](../reference/troubleshooting.md)

## Related Guides

- **[Detecting Dataset Bias](dataset-bias.md)** - Audit data quality before preprocessing
- **[Testing Demographic Shifts](shift-testing.md)** - Validate robustness under population changes
- **[SR 11-7 Compliance](../compliance/sr-11-7-mapping.md)** - Banking regulatory requirements (Section III.C.1)
