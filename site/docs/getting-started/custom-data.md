# Using custom data

This guide shows you how to use your own datasets with GlassAlpha for ML auditing. Whether you have CSV files, database exports, or other tabular data, this tutorial will get you up and running quickly.

## Quick start

The fastest way to use custom data:

**Option 1: Use quickstart generator** (recommended for first-time users)

The easiest way is to use the quickstart command and then modify the generated config:

```bash
# Generate a project with example config
glassalpha quickstart

# Edit the generated config with your data
cd my-audit-project
nano audit_config.yaml
```

Update the key fields in `audit_config.yaml`:

- `data.dataset: custom` → Use "custom" instead of built-in dataset name
- `data.path: /path/to/your/data.csv` → Your dataset path
- `data.target_column: your_target` → Your prediction target column
- `data.protected_attributes: [...]` → Your sensitive features

Then run:

```bash
glassalpha audit
```

**Option 2: Minimal configuration** (if you know what you're doing)

```yaml
# my_audit_config.yaml
# Direct configuration

reproducibility:
  random_seed: 42

data:
  dataset: custom # Important: Use "custom" for your own data
  path: /path/to/your/data.csv
  target_column: your_target_column
  protected_attributes:
    - gender
    - age

model:
  type: logistic_regression
  params:
    random_state: 42
```

Run the audit:

```bash
glassalpha audit --config my_audit_config.yaml --output audit.pdf
```

That's it! GlassAlpha will automatically load your data, train the model, and generate a comprehensive audit report.

## Data requirements

### Minimum requirements

Your dataset must have:

1. **Tabular format**: Rows are observations, columns are features
2. **Target column**: The outcome you're predicting
3. **Feature columns**: Input variables for prediction
4. **Consistent data types**: Each column has one type (numeric, categorical, etc.)

### Supported file formats

| Format  | Extension  | Best For              | Notes                         |
| ------- | ---------- | --------------------- | ----------------------------- |
| CSV     | `.csv`     | Small-medium datasets | Most common, widely supported |
| Parquet | `.parquet` | Large datasets        | Compressed, faster loading    |
| Feather | `.feather` | Fast I/O              | Efficient binary format       |
| Pickle  | `.pkl`     | Python objects        | Python-specific format        |

Format is automatically detected from file extension.

### Recommended data characteristics

For best results:

- **Sample size**: 500+ rows minimum, 5,000+ recommended
- **Features**: 5-100 features (TreeSHAP slows with >1,000 features)
- **Target**: Binary classification (0/1 or True/False)
- **Missing values**: <30% per column
- **Protected attributes**: At least one (race, sex, age, etc.)

## Step-by-step tutorial

### Step 1: Prepare your data

#### Example dataset structure

```csv
age,income,education,credit_score,employment_years,loan_approved,gender,race
35,65000,bachelors,720,8,1,female,white
42,48000,high_school,650,15,0,male,black
28,85000,masters,780,3,1,female,asian
...
```

**Key points:**

- **Target column** (`loan_approved`): What you're predicting (0 or 1)
- **Feature columns**: All others except protected attributes (if desired)
- **Protected attributes** (`gender`, `race`): For fairness analysis
- **Headers**: First row should contain column names
- **No index column**: Remove any row number columns

#### Data cleaning checklist

Before using your data with GlassAlpha:

- [ ] Remove any row index columns
- [ ] Ensure column names have no special characters
- [ ] Convert dates to numeric features if needed
- [ ] Verify target column is binary (0/1)
- [ ] Check for excessive missing values
- [ ] Encode categorical variables if needed (GlassAlpha handles this automatically)

### Step 2: Create configuration file

Create a YAML configuration file specifying your data:

```yaml
# loan_approval_audit.yaml
# Direct configuration

# Essential: Set random seed for reproducibility
reproducibility:
  random_seed: 42

# Data configuration
data:
  dataset: custom # Required for custom data
  path: ~/data/loan_applications.csv # Absolute or home-relative path
  target_column: loan_approved # Column containing 0/1 outcomes

  # Optional: Specify which columns to use as features
  feature_columns:
    - age
    - income
    - education
    - credit_score
    - employment_years
    # Note: protected_attributes are automatically included

  # Required for fairness analysis
  protected_attributes:
    - gender
    - race

# Model configuration
model:
  type: xgboost # Or: logistic_regression, lightgbm
  params:
    objective: binary:logistic
    n_estimators: 100
    max_depth: 5
    random_state: 42

# Explanation configuration
explainers:
  strategy: first_compatible
  priority:
    - treeshap # Use TreeSHAP for XGBoost
    - kernelshap # Fallback for any model

# Metrics to compute
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
      - equalized_odds
    config:
      demographic_parity:
        threshold: 0.05 # Maximum 5% difference between groups
```

### Step 3: Validate configuration

Before running a full audit, validate your configuration:

```bash
# Check for configuration errors
glassalpha validate --config loan_approval_audit.yaml

# Dry run (checks data loading without generating report)
glassalpha audit --config loan_approval_audit.yaml --output test.pdf --dry-run
```

### Step 4: Run the audit

Generate your audit report:

```bash
glassalpha audit \
  --config loan_approval_audit.yaml \
  --output loan_approval_audit.pdf \
  --strict
```

**Flags explained:**

- `--config`: Path to your configuration file
- `--output`: Where to save the PDF report
- `--strict`: Enable regulatory compliance mode (recommended)

**Expected output:**

```
GlassAlpha Audit Generation
========================================
Loading configuration from: loan_approval_audit.yaml
Audit profile: tabular_compliance
Strict mode: ENABLED

Running audit pipeline...
✓ Data loaded: 5,234 samples, 12 features
✓ Model trained: XGBoost (100 estimators)
✓ Explanations generated: TreeSHAP
✓ Fairness metrics computed: 3 groups analyzed

📊 Audit Summary:
  ✅ Performance: 82.3% accuracy, 0.86 AUC
  ⚠️ Bias detected: gender.demographic_parity (7.2% difference)

Generating PDF report...
✓ Report saved: loan_approval_audit.pdf (1.4 MB)

⏱️ Total time: 6.3s
```

## Configuration options

### Data section

#### Basic options

```yaml
data:
  dataset: custom # Required for custom data
  path: /path/to/data.csv # Absolute or ~/relative path
  target_column: outcome # Column name for prediction target
```

#### Feature selection

**Option 1: Use all columns (default)**

```yaml
data:
  dataset: custom
  path: data.csv
  target_column: approved
  # All columns except target and protected become features
```

**Option 2: Explicitly specify features**

```yaml
data:
  dataset: custom
  path: data.csv
  target_column: approved
  feature_columns: # Only these columns used as features
    - age
    - income
    - credit_score
```

#### Protected attributes

Protected attributes are used for fairness analysis:

```yaml
data:
  protected_attributes:
    - gender # Binary or categorical
    - race # Multiple categories
    - age # Can be continuous or binned
```

**Important**: Protected attributes are automatically included in the feature set for model training unless explicitly excluded.

#### File path options

```yaml
# Absolute path (recommended)
data:
  path: /Users/username/data/my_data.csv

# Home directory relative (also good)
data:
  path: ~/data/my_data.csv

# Current directory relative (not recommended - can fail)
data:
  path: data/my_data.csv

# Environment variable
data:
  path: ${DATA_DIR}/my_data.csv
```

!!! warning "Common Mistake: Relative Paths"
Using relative paths like `data/file.csv` can fail depending on where you run the command.

    **Problem**: `FileNotFoundError: Data file not found at data/file.csv`

    **Fix**: Always use absolute paths or home-relative paths:
    ```yaml
    path: /Users/yourname/data/file.csv  # Absolute
    path: ~/data/file.csv                 # Home-relative
    ```

### Model selection

Choose the model type based on your needs:

!!! tip "Pro Tip: Start with LogisticRegression"
XGBoost and LightGBM are powerful but require additional installation (`pip install 'glassalpha[explain]'`).

    **Best practice**: Start with `type: logistic_regression` (always available) to verify your setup works, then upgrade to tree models if you need better performance.

#### Logistic Regression (baseline)

```yaml
model:
  type: logistic_regression
  params:
    random_state: 42
    max_iter: 1000
    C: 1.0 # Regularization strength
```

**When to use:**

- Quick baseline audit
- Linear relationships
- High interpretability needed
- XGBoost/LightGBM not installed

#### XGBoost (recommended)

```yaml
model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 100
    max_depth: 5
    learning_rate: 0.1
    random_state: 42
```

**When to use:**

- Best predictive performance
- TreeSHAP explanations desired
- Non-linear relationships
- 1K-100K samples

#### LightGBM (fast alternative)

```yaml
model:
  type: lightgbm
  params:
    objective: binary
    n_estimators: 100
    num_leaves: 31
    learning_rate: 0.1
    random_state: 42
```

**When to use:**

- Large datasets (>100K samples)
- Need faster training
- Memory constraints
- Many features (>100)

### Preprocessing options

GlassAlpha handles preprocessing automatically, but you can customize:

```yaml
preprocessing:
  handle_missing: true # Automatically handle missing values
  missing_strategy: median # For numeric: median, mean, mode
  scale_features: false # Not needed for tree models
  categorical_encoding: label # label, onehot, target
```

## Domain-specific examples

### Financial services (credit scoring)

```yaml
# Direct configuration

data:
  dataset: custom
  path: ~/data/loan_applications.csv
  target_column: approved
  protected_attributes:
    - gender
    - race
    - age_group

model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 150
    max_depth: 6

metrics:
  fairness:
    config:
      demographic_parity:
        threshold: 0.05 # ECOA compliance
      equal_opportunity:
        threshold: 0.05
```

### Healthcare (treatment outcomes)

```yaml
# Direct configuration

data:
  dataset: custom
  path: ~/data/patient_outcomes.csv
  target_column: treatment_success
  protected_attributes:
    - race
    - gender
    - age
    - disability_status

model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 100

metrics:
  fairness:
    config:
      demographic_parity:
        threshold: 0.03 # Stricter for healthcare
      equal_opportunity:
        threshold: 0.03
```

### Hiring (candidate screening)

```yaml
# Direct configuration

data:
  dataset: custom
  path: ~/data/candidate_screening.csv
  target_column: hired
  protected_attributes:
    - gender
    - race
    - age
    - veteran_status

model:
  type: logistic_regression # More interpretable for HR
  params:
    random_state: 42
    max_iter: 1000

metrics:
  fairness:
    metrics:
      - demographic_parity
      - equal_opportunity
      - predictive_parity
    config:
      demographic_parity:
        threshold: 0.02 # Very strict for hiring
```

### Criminal justice (risk assessment)

```yaml
# Direct configuration

data:
  dataset: custom
  path: ~/data/risk_assessments.csv
  target_column: recidivism
  protected_attributes:
    - race
    - sex
    - age_category

model:
  type: logistic_regression
  params:
    random_state: 42

metrics:
  fairness:
    metrics:
      - demographic_parity
      - equal_opportunity
      - equalized_odds
      - predictive_parity
    config:
      demographic_parity:
        threshold: 0.05
      equal_opportunity:
        threshold: 0.05
```

## Common issues and solutions

### Issue: "Data file not found"

**Problem**:

```
FileNotFoundError: Data file not found at ~/data/my_data.csv
```

**Solutions**:

1. Use absolute paths: `/Users/username/data/my_data.csv`
2. Verify file exists: `ls ~/data/my_data.csv`
3. Check file permissions: `ls -l ~/data/my_data.csv`
4. Use `dataset: custom` in config

### Issue: "Target column not found"

**Problem**:

```
ValueError: Target column 'outcome' not found in dataset
```

**Solutions**:

1. Check exact column name (case-sensitive)
2. Print column names: `import pandas as pd; print(pd.read_csv('data.csv').columns)`
3. Remove any spaces: `outcome` not `outcome `
4. Verify CSV has headers

### Issue: "Protected attributes not detected"

**Problem**: Fairness metrics show errors or no groups

!!! warning "Common Mistake: Missing Protected Attributes"
If you don't specify `protected_attributes`, fairness metrics will fail with errors.

    **Problem**: `ValueError: No protected attributes specified for fairness analysis`

    **Fix**: Always include at least one protected attribute:
    ```yaml
    protected_attributes:
      - gender
      - race
      - age
    ```

**Solutions**:

1. Verify column names match config exactly (case-sensitive!)
2. Check for missing values in protected columns
3. Ensure protected columns are in dataset
4. Review data types: `df['gender'].dtype`

### Issue: "Model training failed"

**Problem**: Error during model fitting

**Solutions**:

1. Check for NaN values: `df.isnull().sum()`
2. Verify target is binary (0/1)
3. Ensure sufficient samples (>100)
4. Try logistic_regression first
5. Check feature data types

### Issue: "SHAP computation too slow"

**Problem**: Audit takes too long on large dataset

**Solutions**:

1. Reduce SHAP samples in config:

   ```yaml
   explainers:
     config:
       treeshap:
         max_samples: 100 # Default: 1000
   ```

2. Use sample of data for testing
3. Enable parallel processing:

   ```yaml
   performance:
     n_jobs: -1
   ```

4. Consider LightGBM (faster than XGBoost)

## Data preparation scripts

### Convert Excel to CSV

```python
import pandas as pd

# Read Excel file
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Save as CSV
df.to_csv('data.csv', index=False)
```

### Clean column names

```python
import pandas as pd

df = pd.read_csv('data.csv')

# Remove special characters and spaces
df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
df.columns = df.columns.str.lower()

df.to_csv('data_cleaned.csv', index=False)
```

### Handle missing values

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')

# Check missing values
print(df.isnull().sum())

# Option 1: Drop rows with missing target
df = df.dropna(subset=['target_column'])

# Option 2: Fill numeric with median
df['age'] = df['age'].fillna(df['age'].median())

# Option 3: Fill categorical with mode
df['category'] = df['category'].fillna(df['category'].mode()[0])

df.to_csv('data_cleaned.csv', index=False)
```

### Create protected attribute bins

```python
import pandas as pd

df = pd.read_csv('data.csv')

# Bin continuous age into categories
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 25, 40, 60, 100],
    labels=['young', 'middle', 'senior', 'elderly']
)

df.to_csv('data_with_bins.csv', index=False)
```

## Best practices

### Data privacy

1. **Remove PII**: Strip names, addresses, SSNs before auditing
2. **Anonymize IDs**: Hash or remove customer IDs
3. **Aggregate when possible**: Use binned age instead of exact
4. **Document data handling**: Record what was removed/changed

### Model selection

1. **Start simple**: Use logistic_regression for baseline
2. **Compare models**: Try XGBoost and LightGBM
3. **Balance performance vs interpretability**: Logistic is more interpretable
4. **Consider domain**: Healthcare may prefer simpler models

### Fairness analysis

1. **Set appropriate thresholds**: Stricter for high-stakes domains
2. **Use multiple metrics**: Demographic parity + equal opportunity
3. **Check intersectionality**: Race × gender interactions
4. **Document trade-offs**: Performance vs fairness

### Reproducibility

1. **Always set random_seed**: `random_seed: 42`
2. **Version control configs**: Use git for YAML files
3. **Document data sources**: Where data came from, when
4. **Save preprocessing steps**: Scripts for data cleaning

## Next steps

You're now ready to audit your own models! Here's what to do:

1. **✅ Try it**: Run your first custom data audit using the [quick start](#quick-start) above
2. **📊 Compare**: [Test with built-in datasets](datasets.md) to benchmark your results
3. **⚙️ Optimize**: Learn about [configuration options](configuration.md) to customize your audits
4. **🎯 Choose wisely**: Pick the best model and explainer for your use case (coming soon)

**Found this helpful?** [Star us on GitHub ⭐](https://github.com/GlassAlpha/glassalpha) to help others discover GlassAlpha!

## Additional resources

- [Configuration Guide](configuration.md) - Full YAML reference
- [Built-in Datasets](datasets.md) - Automatic dataset fetching
- [Model Selection](../reference/faq.md#model-support) - Choosing the right model
- [Troubleshooting](../reference/troubleshooting.md) - Common issues

---

**Questions?** Open an issue on [GitHub](https://github.com/GlassAlpha/glassalpha/issues) or check the [FAQ](../reference/faq.md).
