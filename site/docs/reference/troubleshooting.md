# Troubleshooting guide

Common issues, error messages, and solutions for GlassAlpha. This guide helps diagnose and resolve problems quickly.

## Quick diagnostics

If you're experiencing issues, start with these diagnostic commands:

```bash
# Verify installation (if package is installed)
glassalpha --version

# Alternative: Use module invocation (development/uninstalled)
PYTHONPATH=src python3 -m glassalpha --version

# Check component availability
glassalpha list
# Alternative: PYTHONPATH=src python3 -m glassalpha list

# Validate configuration
glassalpha validate --config german_credit_simple.yaml
# Alternative: PYTHONPATH=src python3 -m glassalpha validate --config german_credit_simple.yaml

# Test with minimal configuration
glassalpha audit --config german_credit_simple.yaml --output test.pdf --dry-run
# Alternative: PYTHONPATH=src python3 -m glassalpha audit --config german_credit_simple.yaml --output test.pdf --dry-run
```

## CLI command issues

### `glassalpha: command not found`

This indicates GlassAlpha isn't properly installed or the CLI entry point isn't available.

**Solutions:**

1. **Install the package** (recommended for users):

   ```bash
   cd glassalpha
   pip install -e .

   # Verify installation
   glassalpha --version
   ```

2. **Use module invocation** (development/troubleshooting):

   ```bash
   cd glassalpha
   PYTHONPATH=src python3 -m glassalpha --version
   PYTHONPATH=src python3 -m glassalpha audit --config german_credit_simple.yaml --output test.pdf --dry-run
   ```

3. **Check your environment**:

   ```bash
   # Check if package is installed
   pip list | grep glassalpha

   # Check Python path
   python3 -c "import sys; print('\n'.join(sys.path))"
   ```

## Installation issues

### Python version compatibility

**Error:**

```
ERROR: Package 'glassalpha' requires a different Python: 3.10.0 not in '>=3.11'
```

**Solution:**

```bash
# Check Python version
python --version

# Upgrade Python (macOS with Homebrew)
brew install python@3.11
python3.11 -m pip install -e .

# Upgrade Python (Linux)
sudo apt update && sudo apt install python3.11

# Create virtual environment with correct Python
python3.11 -m venv venv
source venv/bin/activate
pip install -e .
```

### XGBoost installation issues (macOS)

**Error:**

```
RuntimeError: libomp not found. Install with: brew install libomp
```

**Solution:**

```bash
# Install OpenMP library
brew install libomp

# Reinstall XGBoost
pip uninstall xgboost
pip install xgboost

# Verify installation
python -c "import xgboost; print('XGBoost version:', xgboost.__version__)"
```

### Dependency conflicts

**Error:**

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solution:**

```bash
# Create clean virtual environment
python -m venv clean_env
source clean_env/bin/activate  # On Windows: clean_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install GlassAlpha
pip install -e .

# If conflicts persist, install dependencies individually
pip install pandas numpy scikit-learn
pip install xgboost lightgbm
pip install shap matplotlib seaborn
pip install -e . --no-deps
```

### Missing system dependencies

**Error (Linux):**

```
ImportError: libgomp.so.1: cannot open shared object file
```

**Solution (Ubuntu/Debian):**

```bash
sudo apt update
sudo apt install libgomp1 build-essential
```

**Solution (CentOS/RHEL):**

```bash
sudo yum install libgomp gcc gcc-c++
```

## Configuration errors

### Missing required fields

**Error:**

```
ValidationError: 1 validation error for AuditConfig
data.target_column
  field required (type=value_error.missing)
```

**Solution:**

```yaml
# Add missing required fields
data:
  path: data/dataset.csv
  target_column: outcome # This was missing

model:
  type: xgboost # Ensure model type is specified
```

### Invalid file paths

**Error:**

```
FileNotFoundError: Configuration file 'nonexistent.yaml' not found
```

**Solutions:**

```bash
# Use absolute paths
glassalpha audit --config /full/path/to/config.yaml --output report.pdf

# Config files are packaged with GlassAlpha - no need for configs/ directory
# Just reference by name:
glassalpha audit --config german_credit_simple.yaml --output audit.pdf
```

### Model type not found

**Error:**

```
Warning: Model type 'unknown_model' not found in registry
```

**Solution:**

```bash
# Check available models
glassalpha list models

# Use correct model type
# Valid options: xgboost, lightgbm, logistic_regression, sklearn_generic
```

```yaml
model:
  type: xgboost # Use registered model type
```

### Schema validation errors

**Error:**

```
ValidationError: 2 validation errors for TabularDataSchema
features
  ensure this value has at least 1 characters
target
  field required
```

**Solution:**

```yaml
data:
  path: data/dataset.csv
  target_column: target # Must specify target
  feature_columns: # Either specify features explicitly
    - feature1
    - feature2
  # OR let GlassAlpha auto-detect (remove feature_columns)
```

## Model and dependency issues

### Model not available errors

**Error messages:**

```
Model 'xgboost' not available. Falling back to 'logistic_regression'.
To enable 'xgboost', run: pip install 'glassalpha[xgboost]'
```

Or:

```
Missing optional dependency 'xgboost' for plugin 'xgboost'.
Try: pip install 'glassalpha[xgboost]'
```

**Cause:** GlassAlpha uses optional dependencies to keep the core installation lightweight. XGBoost and LightGBM are not installed by default.

**Solutions:**

1. **Use the baseline model** (recommended for getting started):

   ```yaml
   # In your config.yaml
   model:
     type: logistic_regression # Always available
     params:
       random_state: 42
   ```

2. **Install the features you need**:

   ```bash
   # For SHAP + tree models (includes XGBoost and LightGBM)
   pip install 'glassalpha[explain]'

   # For all features
   pip install 'glassalpha[all]'
   ```

3. **Check what's currently available**:

   ```bash
   glassalpha doctor
   ```

4. **Disable fallbacks for strict requirements**:
   ```yaml
   model:
     type: xgboost
     allow_fallback: false # Fail if XGBoost not available
   ```

### Model loading failures

**Error messages:**

```
ModuleNotFoundError: No module named 'xgboost'
```

**Cause:** The model library isn't installed in your environment.

**Solutions:** Same as above - install the required optional dependency.

### Fallback model not suitable

**Error messages:**

```
Model 'xgboost' not available. Falling back to 'logistic_regression'.
```

**Cause:** The requested model isn't available, and fallback is enabled.

**Solutions:**

1. **Accept the fallback** (LogisticRegression is a solid baseline)
2. **Install the requested model** using the provided installation command
3. **Disable fallbacks** if you need the specific model:
   ```yaml
   model:
     type: xgboost
     allow_fallback: false
   ```

### Explainer errors (SHAP library required)

**Error messages:**

```
SHAP library required for KernelSHAP explainer
```

Or:

```
No explainer from ['treeshap'] is available. Missing dependencies: ['shap'].
Install with: pip install 'glassalpha[explain]'
OR use zero-dependency explainers: ['coefficients', 'permutation']
```

**Cause:** Your configuration requests a SHAP-based explainer (TreeSHAP or KernelSHAP) but the SHAP library isn't installed in your environment.

**Solutions:**

1. **Use zero-dependency explainers** (recommended for getting started):

   ```yaml
   # In your config.yaml
   explainers:
     priority:
       - coefficients # Best for LogisticRegression (no dependencies)
       - permutation # Works with any model (no dependencies)
   ```

2. **Install SHAP for tree model explanations**:

   ```bash
   # Install SHAP + XGBoost + LightGBM
   pip install 'glassalpha[explain]'

   # Verify installation
   glassalpha doctor
   ```

3. **Use the quickstart config** which works without SHAP:

   ```bash
   glassalpha audit --config quickstart.yaml --output audit.html
   ```

**Which explainer should I use?**

| Model Type         | Best Explainer | Requires SHAP? | Notes                              |
| ------------------ | -------------- | -------------- | ---------------------------------- |
| LogisticRegression | `coefficients` | ❌ No          | Zero dependencies, fast            |
| XGBoost/LightGBM   | `treeshap`     | ✅ Yes         | Most accurate for tree models      |
| RandomForest       | `treeshap`     | ✅ Yes         | Handles tree ensembles well        |
| Any model          | `permutation`  | ❌ No          | Universal fallback, model-agnostic |

**Example configs:**

For LogisticRegression (works with base install):

```yaml
model:
  type: logistic_regression
explainers:
  priority:
    - coefficients
```

For XGBoost (requires `pip install 'glassalpha[explain]'`):

```yaml
model:
  type: xgboost
explainers:
  priority:
    - treeshap
    - permutation # Fallback if SHAP fails
```

## Data issues

### Dataset not found

**Error:**

```
FileNotFoundError: Dataset file 'data/missing.csv' not found
```

**Solutions:**

1. **Check file paths (most common issue):**

```bash
# Check file existence
ls -la data/missing.csv

# Check current directory
pwd

# Use absolute paths in config
```

```yaml
data:
  path: /Users/yourname/data/dataset.csv # Absolute path
  # OR
  path: ~/data/dataset.csv # Home directory expansion
```

!!! warning "Common Mistake: Relative Paths"
Relative paths are resolved from where you run the command, not where the config file is located. Use absolute paths or `~` for home directory.

2. **Use built-in datasets for testing:**

```yaml
data:
  source: german_credit # Built-in dataset (auto-downloaded)
  fetch: if_missing
```

3. **Browse available datasets:**

```bash
glassalpha datasets list
```

**Need help with data preparation?** See [Using Custom Data](../getting-started/custom-data.md) for a complete tutorial.

### Column not found

**Error:**

```
KeyError: Column 'target' not found in dataset
```

**Causes:**

- Column name misspelled or incorrect case
- CSV file has no headers
- Column name has spaces or special characters

**Solutions:**

1. **Inspect your data:**

```bash
# Check column names
python -c "import pandas as pd; df = pd.read_csv('data/dataset.csv'); print('Columns:', df.columns.tolist()); print('Shape:', df.shape)"

# Check first few rows
python -c "import pandas as pd; print(pd.read_csv('data/dataset.csv').head())"
```

2. **Update configuration with exact column name:**

```yaml
data:
  path: data/dataset.csv
  target_column: "actual_column_name" # Use quotes if spaces/special chars


  # Check for common issues:
  # - Extra spaces: "target " vs "target"
  # - Case sensitivity: "Target" vs "target"
  # - Special characters: "target_column" vs "target-column"
```

3. **Fix CSV headers:**

```python
import pandas as pd

# Read CSV without headers
df = pd.read_csv('data/dataset.csv', header=None)

# Add column names
df.columns = ['feature1', 'feature2', 'target']

# Save with headers
df.to_csv('data/dataset_with_headers.csv', index=False)
```

### Protected attributes not found

**Error:**

```
KeyError: Protected attribute 'gender' not found in dataset
```

**Solutions:**

1. **Check attribute names:**

```bash
# List all columns
python -c "import pandas as pd; print(pd.read_csv('data/dataset.csv').columns.tolist())"
```

2. **Update config with correct names:**

```yaml
data:
  protected_attributes:
    - sex # Not 'gender'
    - age_group # Not 'age'
```

3. **If attributes are missing from your data:**

```yaml
data:
  protected_attributes: [] # Empty list if no protected attributes available

# Note: Fairness metrics will be limited without protected attributes
```

**For more on protected attributes:** See [Using Custom Data](../getting-started/custom-data.md#protected-attributes).

### Data format issues

**Error:**

```
ParserError: Error tokenizing data. C error: Expected 5 fields, saw 7
```

**Causes:**

- Inconsistent number of columns
- Unescaped commas in text fields
- Corrupted file

**Solutions:**

1. **Check for delimiter issues:**

```python
import pandas as pd

# Try different delimiters
df = pd.read_csv('data/dataset.csv', delimiter='\t')  # Tab-separated
df = pd.read_csv('data/dataset.csv', delimiter=';')   # Semicolon-separated
```

2. **Handle problematic CSV files:**

```python
# Read with error handling
df = pd.read_csv('data/dataset.csv',
                 on_bad_lines='skip',  # Skip bad lines
                 encoding='utf-8')      # Specify encoding
```

3. **Convert to a supported format:**

```python
import pandas as pd

# Read problematic CSV
df = pd.read_csv('data/dataset.csv', on_bad_lines='skip')

# Save as Parquet (more robust)
df.to_parquet('data/dataset.parquet', index=False)
```

```yaml
# Update config to use Parquet
data:
  path: data/dataset.parquet
```

### Data type issues

**Error:**

```
TypeError: Object of type 'datetime64' is not JSON serializable
```

**Causes:**

- Datetime columns not converted
- Complex object types
- Mixed data types in columns

**Solutions:**

1. **Convert datetime columns:**

```python
import pandas as pd

df = pd.read_csv('data/dataset.csv')

# Option 1: Convert to Unix timestamp
df['date_column'] = pd.to_datetime(df['date_column']).astype('int64') // 10**9

# Option 2: Extract useful features
df['year'] = pd.to_datetime(df['date_column']).dt.year
df['month'] = pd.to_datetime(df['date_column']).dt.month
df['day_of_week'] = pd.to_datetime(df['date_column']).dt.dayofweek

# Drop original datetime column
df = df.drop('date_column', axis=1)

df.to_csv('data/processed_dataset.csv', index=False)
```

2. **Handle categorical data:**

```python
# Check data types
print(df.dtypes)

# Convert object columns to category or numeric
df['category_column'] = df['category_column'].astype('category').cat.codes
```

3. **Identify problematic columns:**

```python
# Find columns with mixed types
for col in df.columns:
    unique_types = df[col].apply(type).unique()
    if len(unique_types) > 1:
        print(f"Column '{col}' has mixed types: {unique_types}")
```

### Missing values

**Error:**

```
ValueError: Input contains NaN, infinity or a value too large
```

**Causes:**

- Missing data in dataset
- Infinite values from calculations
- Very large numbers causing overflow

**Solutions:**

1. **Inspect missing data:**

```python
import pandas as pd

df = pd.read_csv('data/dataset.csv')

# Check missing values per column
print(df.isnull().sum())

# Check percentage missing
print(df.isnull().sum() / len(df) * 100)

# Identify rows with missing target
print(df[df['target'].isnull()])
```

2. **Handle missing values:**

```python
# Option 1: Drop rows with missing target (recommended)
df = df.dropna(subset=['target'])

# Option 2: Drop columns with >30% missing
threshold = 0.3
df = df.loc[:, df.isnull().mean() < threshold]

# Option 3: Impute missing values
from sklearn.impute import SimpleImputer

# For numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# For categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

df.to_csv('data/cleaned_dataset.csv', index=False)
```

3. **Check for infinite values:**

```python
import numpy as np

# Replace infinite values
df = df.replace([np.inf, -np.inf], np.nan)

# Then handle as missing values
df = df.dropna()
```

**More data preparation guidance:** See [Using Custom Data - Data Cleaning](../getting-started/custom-data.md#step-1-prepare-your-data).

### Target column issues

**Error:**

```
ValueError: Target column must be binary (0/1) for classification
```

**Causes:**

- Target has more than 2 unique values
- Target is text instead of numeric
- Target has missing values

**Solutions:**

1. **Check target distribution:**

```python
import pandas as pd

df = pd.read_csv('data/dataset.csv')
print("Target values:", df['target'].unique())
print("Target counts:", df['target'].value_counts())
print("Target type:", df['target'].dtype)
```

2. **Convert target to binary:**

```python
# If target is text (e.g., "Yes"/"No")
df['target'] = (df['target'] == 'Yes').astype(int)

# If target is numeric but not 0/1
df['target'] = (df['target'] > threshold).astype(int)

# If target is multi-class, convert to binary
# Example: predict if class is "high risk" vs others
df['target'] = (df['target'] == 'high_risk').astype(int)

df.to_csv('data/dataset_binary.csv', index=False)
```

3. **Remove missing target values:**

```python
df = df.dropna(subset=['target'])
print(f"Rows after removing missing targets: {len(df)}")
```

### Insufficient data

**Error:**

```
ValueError: Insufficient samples for training (found 50, need at least 100)
```

**Causes:**

- Dataset too small
- Too many missing values removed
- Heavy class imbalance with very few positive samples

**Solutions:**

1. **Check dataset size:**

```python
import pandas as pd

df = pd.read_csv('data/dataset.csv')
print(f"Total rows: {len(df)}")
print(f"Class distribution:\n{df['target'].value_counts()}")
print(f"Minimum class size: {df['target'].value_counts().min()}")
```

2. **Use a larger dataset:**

   - Browse [freely available data sources](../getting-started/data-sources.md)
   - Combine multiple datasets if possible
   - Use German Credit (1,000 rows) for testing

3. **If stuck with small data:**

```yaml
# Adjust model parameters for small datasets
model:
  type: logistic_regression # Better for small data than tree models
  params:
    max_iter: 1000
    C: 1.0

# Reduce explainer sample sizes
explainers:
  config:
    treeshap:
      max_samples: 50 # Reduce to match data size
```

### Class imbalance

**Warning:**

```
Warning: Severe class imbalance detected (99.5% negative class)
```

**Solutions:**

1. **Check imbalance ratio:**

```python
import pandas as pd

df = pd.read_csv('data/dataset.csv')
value_counts = df['target'].value_counts()
imbalance_ratio = value_counts.max() / value_counts.min()
print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
print(f"Class distribution:\n{value_counts}")
```

2. **Adjust model for imbalance:**

```yaml
model:
  type: xgboost
  params:
    # Critical for imbalanced data
    scale_pos_weight: 99 # Ratio of negative to positive samples
    objective: binary:logistic
```

3. **Use appropriate metrics:**

```yaml
metrics:
  performance:
    metrics:
      - precision # More important than accuracy for imbalanced data
      - recall
      - f1
      - pr_auc # Better than roc_auc for imbalanced data
```

**Example configurations:** See `credit_card_fraud_template.yaml` (packaged with GlassAlpha) for handling 99.8% class imbalance.

## Runtime errors

### Memory issues

**Error:**

```
MemoryError: Unable to allocate array with shape (10000, 10000)
```

**Solutions:**

```yaml
# Reduce sample sizes for explainers
explainers:
  config:
    treeshap:
      max_samples: 100 # Reduce from default 1000
    kernelshap:
      n_samples: 50 # Reduce from default 500

# Enable low memory mode
performance:
  low_memory_mode: true
  n_jobs: 1 # Reduce parallelism
```

```bash
# Use smaller datasets for testing
head -n 1000 large_dataset.csv > small_dataset.csv
```

### Model training failures

**Error:**

```
XGBoostError: Check failed: labels.Size() == num_row
```

**Solution:**

```python
# Check data consistency
import pandas as pd
df = pd.read_csv('data/dataset.csv')
print(f"Rows: {len(df)}")
print(f"Target values: {df['target'].value_counts()}")
print(f"Missing values: {df.isnull().sum()}")

# Remove rows with missing target values
df = df.dropna(subset=['target'])
df.to_csv('data/cleaned_dataset.csv', index=False)
```

### SHAP computation errors

**Error:**

```
Exception: TreeExplainer only supports the following model types: xgboost.Booster
```

**Solution:**

```yaml
# Use compatible explainer
explainers:
  priority:
    - kernelshap # Model-agnostic alternative
  config:
    kernelshap:
      n_samples: 100
```

### PDF generation issues

**Error:**

```
OSError: cannot load library 'pango-1.0-0'
```

**Solution (Linux):**

```bash
# Install required system libraries
sudo apt install libpango1.0-dev libcairo2-dev libgtk-3-dev
```

**Solution (macOS):**

```bash
# Install with Homebrew
brew install pango cairo gtk+3
```

**Solution (Alternative):**

```bash
# Use HTML output instead
# Modify report configuration:
```

```yaml
report:
  output_format: html # Generate HTML instead of PDF
```

## Performance issues

### Slow execution

**Problem:** Audit takes longer than expected

**Solutions:**

1. **Reduce Sample Sizes:**

```yaml
explainers:
  config:
    treeshap:
      max_samples: 100 # Default: 1000
    kernelshap:
      n_samples: 50 # Default: 500
      background_size: 50 # Default: 100
```

2. **Enable Parallel Processing:**

```yaml
performance:
  n_jobs: -1 # Use all CPU cores
```

3. **Use Simpler Models:**

```yaml
model:
  params:
    n_estimators: 50 # Reduce from 100
    max_depth: 3 # Reduce from 6
```

4. **Skip Expensive Operations:**

```bash
# Use dry run for config validation
glassalpha audit --config config.yaml --output test.pdf --dry-run

# Test with smaller datasets first
head -n 500 large_dataset.csv > test_dataset.csv
```

### High memory usage

**Problem:** System runs out of memory during audit

**Solutions:**

1. **Enable Memory Optimization:**

```yaml
performance:
  low_memory_mode: true
  n_jobs: 1 # Reduce parallelism
```

2. **Process Data in Chunks:**

```python
# Pre-process large datasets
import pandas as pd
df = pd.read_csv('large_dataset.csv')
sample = df.sample(n=1000, random_state=42)  # Use sample for testing
sample.to_csv('sample_dataset.csv', index=False)
```

3. **Optimize Explainer Settings:**

```yaml
explainers:
  config:
    treeshap:
      max_samples: 50 # Very small for memory constraints
    kernelshap:
      n_samples: 20
      background_size: 20
```

### Large PDF files

**Problem:** Generated PDFs are too large

**Solutions:**

1. **Optimize Report Configuration:**

```yaml
report:
  styling:
    optimize_size: true
    compress_images: true
```

2. **Reduce Plot Resolution:**

```yaml
explainers:
  config:
    treeshap:
      plot_dpi: 150 # Default: 300
```

3. **Exclude Optional Sections:**

```yaml
report:
  include_sections:
    - executive_summary
    - model_performance
    # Remove: local_explanations, detailed_plots
```

## Debugging and diagnostics

### Enable verbose logging

```bash
# Enable detailed logging
glassalpha --verbose audit --config config.yaml --output audit.pdf

# Check specific component status
export GLASSALPHA_LOG_LEVEL=DEBUG
glassalpha audit --config config.yaml --output audit.pdf
```

### Component registry debugging

```python
# Debug component registration
python -c "
from glassalpha.core import list_components
import pprint
pprint.pprint(list_components())
"
```

### Configuration debugging

```python
# Debug configuration loading
python -c "
from glassalpha.config import load_config_from_file
config = load_config_from_file('your_config.yaml')
print('Loaded config:', config.model_dump())
"
```

### Data loading debugging

```python
# Debug data issues
python -c "
from glassalpha.data import TabularDataLoader
loader = TabularDataLoader()
data = loader.load('data/dataset.csv')
print('Shape:', data.shape)
print('Columns:', data.columns.tolist())
print('Dtypes:', data.dtypes.to_dict())
print('Missing:', data.isnull().sum().to_dict())
"
```

## Getting help

### Before requesting support

1. **Check this troubleshooting guide**
2. **Run diagnostic commands** shown above
3. **Try with the German Credit example** (known working configuration)
4. **Enable verbose logging** for detailed error information

### Information to include in support requests

```bash
# System information
glassalpha --version
python --version
pip list | grep -E "(glassalpha|xgboost|lightgbm|shap|pandas)"

# Configuration (remove sensitive data)
cat your_config.yaml

# Complete error message
glassalpha --verbose audit --config config.yaml --output audit.pdf 2>&1
```

### Support channels

- **GitHub Issues:** [https://github.com/GlassAlpha/glassalpha/issues](https://github.com/GlassAlpha/glassalpha/issues)
- **Documentation:** [Complete guides](../index.md)
- **Community:** [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)

## Prevention best practices

### Environment setup

```bash
# Always use virtual environments
python -m venv glassalpha-env
source glassalpha-env/bin/activate

# Keep dependencies updated
pip install --upgrade pip
pip install --upgrade glassalpha
```

### Configuration management

```yaml
# Always specify explicit seeds for reproducibility
reproducibility:
  random_seed: 42
# Use version control for configurations
# Git commit: "Update audit configuration for production"

# Validate configurations before use
```

```bash
glassalpha validate --config production.yaml --strict
```

### Testing strategy

```bash
# Test with small datasets first
glassalpha audit --config config.yaml --output test.pdf --dry-run

# Test new configurations incrementally
# Start with minimal config, add features gradually
```

### Monitoring and logging

```bash
# Enable logging for production
export GLASSALPHA_LOG_LEVEL=INFO

# Log all audit runs
glassalpha audit --config config.yaml --output audit.pdf 2>&1 | tee audit.log
```

This troubleshooting guide covers the most common issues encountered with GlassAlpha. If you encounter an issue not covered here, please check the GitHub issues or contact support with the diagnostic information requested above.
