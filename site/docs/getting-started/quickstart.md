# Quick start guide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GlassAlpha/glassalpha/blob/main/examples/notebooks/quickstart_colab.ipynb)

## ⚠️ Requirements

**Python 3.11 or higher required.** Check your version:

```bash
python --version  # Must show 3.11.x, 3.12.x, or 3.13.x
```

If you need to upgrade, see [Installing Python 3.11+](#installing-python-311) below.

## Step 0: Verify Python version (1 minute)

Before proceeding, confirm you have Python 3.11+ installed:

```bash
python --version
```

**Must show**: `3.11.x`, `3.12.x`, or `3.13.x`

**If you see an older version**:

1. See [Installing Python 3.11+](#installing-python-311) below
2. After installation, close and reopen your terminal
3. Re-run `python --version` to verify

**Troubleshooting**:

- If `python: command not found` → Python not in PATH, see installation section below
- If permission errors → Use virtual environment: `python3.11 -m venv ~/glassalpha-env`

**Prefer notebooks?** Try our [interactive Colab notebook](https://colab.research.google.com/github/GlassAlpha/glassalpha/blob/main/examples/notebooks/quickstart_colab.ipynb) - generate your first audit in 8 minutes with zero setup.

## ⚡ Lightning-fast development mode

**For development and testing:** Enable fast mode in your config for instant feedback (2-3 seconds instead of 5-7 seconds):

```yaml
# In your audit_config.yaml
runtime:
  fast_mode: true # Reduces bootstrap samples from 1000 to 100
```

Then run your audit normally:

```bash
glassalpha audit --config your_config.yaml
```

## The 5-minute version

Get your first professional audit report in 5 minutes:

### Using quickstart generator (easiest)

```bash
# 1. Install (1-2 minutes)
pip install "glassalpha[all]"

# 2. Generate project (30 seconds)
glassalpha quickstart

# 3. Run audit (30 seconds)
cd my-audit-project && python run_audit.py

# 4. Done! Open your professional report
open reports/audit_report.html  # macOS
xdg-open reports/audit_report.html  # Linux
start reports/audit_report.html  # Windows

# Optional: Create evidence pack for regulatory submission
glassalpha export-evidence-pack reports/audit_report.html --output evidence.zip
```

[Evidence pack guide →](../guides/evidence-packs.md) - Package audits for regulatory submission

**Note**: Base installation uses LogisticRegression model (fast, zero extra dependencies).
For advanced models only, install with `pip install "glassalpha[explain]"` instead.

### Using example configs

```bash
# 1. Install
pip install glassalpha

# 2. Create project from example config
mkdir my-audit && cd my-audit
glassalpha quickstart --dataset german_credit --model xgboost

# 3. Run audit
glassalpha audit

# 4. Done! Open your professional report
open audit_config.html  # macOS (auto-named from config)
# xdg-open audit_config.html  # Linux
# start audit_config.html  # Windows
```

**Tip**: Example configs are also available in the repository at `src/glassalpha/configs/` if you install from source.

**What you get**: A comprehensive audit report with:

- ✅ Model performance metrics (accuracy, precision, recall, F1, AUC)
- ✅ Fairness analysis (bias detection across demographic groups)
- ✅ Feature importance (coefficient-based explanations showing what drives predictions)
- ✅ Individual explanations (why specific decisions were made)
- ✅ Preprocessing verification (optional, for production artifact validation)
- ✅ Complete audit trail (reproducibility manifest with all seeds and hashes)

**Note:** This quickstart uses LogisticRegression with coefficient-based explanations (zero dependencies). For tree-based models with SHAP explanations, install with `pip install -e ".[explain]"`.

**Next steps**:

- [Use your own data](custom-data.md)
- [Verify preprocessing artifacts](../guides/preprocessing.md) (for production audits)
- [Try other datasets](datasets.md)
- [Understand the configuration](configuration.md)

---

## Python API (Notebooks & Scripts)

**Perfect for**: Jupyter notebooks, interactive exploration, programmatic workflows

Generate audits without YAML files using the `from_model()` API:

```python
import glassalpha as ga
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data (GlassAlpha returns file paths, not DataFrames)
german_path = ga.datasets.fetch('german_credit')
df = pd.read_csv(german_path)

# Encode categorical columns for sklearn compatibility
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    if col != 'credit_risk':  # Don't encode target
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Split features and target
X = df.drop(columns=["credit_risk"])
y = df["credit_risk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Generate audit (3 lines)
result = ga.audit.from_model(
    model=model,
    X=X_test,
    y=y_test,
    protected_attributes={
        "age_years": X_test["age_years"],
        "gender": X_test["gender"]
    },
    random_seed=42
)

# View inline in Jupyter
result  # Auto-displays HTML summary

# Or export PDF
result.to_pdf("audit.pdf")
```

**What you get:**

- ✅ Auto-detection of model type (XGBoost, LightGBM, sklearn)
- ✅ Inline HTML display in Jupyter notebooks
- ✅ Full fairness and performance metrics
- ✅ SHAP explanations (if model supports TreeSHAP)
- ✅ Byte-identical reproducibility with `random_seed`

**Try it now**: [Open our Colab quickstart notebook](https://colab.research.google.com/github/GlassAlpha/glassalpha/blob/main/examples/notebooks/quickstart_colab.ipynb) (zero setup, runs in browser)

**API Reference**: See [`from_model()` documentation](../reference/api/audit-entry-points.md) for all parameters

---

## The 10-minute version

Get up and running with GlassAlpha in less than 10 minutes. This guide will take you from installation to generating your first professional audit PDF.

## Prerequisites

- Python 3.11 or higher
- Git (optional, only needed for manual setup)
- 2GB available disk space
- Command line access

## Step 1: Installation

Choose between the quickstart generator (recommended) or manual setup:

### Option A: Quickstart generator (recommended)

The fastest way to get started. Creates a complete audit project in <60 seconds:

```bash
# Install GlassAlpha (if not already installed)
pip install glassalpha

# Generate a ready-to-run audit project
glassalpha quickstart
```

**What you get:**

- Complete project directory structure (data/, models/, reports/)
- Pre-configured audit configuration file (`audit_config.yaml`)
- Example run script (`run_audit.py`) demonstrating programmatic API
- Project README with next steps and advanced usage
- `.gitignore` tailored for GlassAlpha projects

**Run your first audit:**

```bash
cd my-audit-project
python run_audit.py  # Generates audit report in <5 seconds
```

**With custom options**:

```bash
glassalpha quickstart \
  --dataset german_credit \
  --model xgboost \
  --output my-project
```

**Skip to:** [Step 3: Review your audit report](#step-3-review-your-audit-report) once your report is generated.

### Option B: Manual setup

If you prefer manual setup or want to work from the repository:

### Clone and install

Clone and setup:

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha
```

Python 3.11, 3.12, or 3.13 supported:

```bash
python3 --version   # should show 3.11.x, 3.12.x, or 3.13.x
```

Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install GlassAlpha:

```bash
python -m pip install --upgrade pip

# Option 1: Base install (LogisticRegression only, recommended for getting started)
pip install -e .

# Option 2: With advanced ML libraries (if you need XGBoost/LightGBM)
pip install -e ".[explain]"      # SHAP + XGBoost + LightGBM
pip install -e ".[all]"          # All features

# Option 3: Development install (includes testing tools)
pip install -e ".[dev]"
```

Verify installation:

```bash
glassalpha --help

# Check what models are available
glassalpha list
```

You should see the CLI help message with available commands.

### Installation verification checklist

Run these checks to confirm everything is working:

- [ ] **CLI is accessible**: `glassalpha --help` shows help message
- [ ] **Python version correct**: `python --version` shows 3.11+ (3.11.x, 3.12.x, or 3.13.x)
- [ ] **Base dependencies installed**: `glassalpha list` shows available components
- [ ] **Models available**: `glassalpha list models` shows at least `logistic_regression`
- [ ] **Config validation works**: `glassalpha validate --config german_credit_simple.yaml` passes

**All checks passed?** → Proceed to Step 2

**Some checks failed?** → See [Troubleshooting first-run errors](#troubleshooting-first-run-errors) below

## Step 2: Generate your first audit

GlassAlpha comes with a ready-to-use German Credit dataset example that demonstrates all core capabilities.

### Run the audit command

Generate audit report:

```bash
glassalpha audit \
  --config german_credit_simple.yaml \
  --output my_first_audit.html
```

!!! tip "Fast Mode for Demos"
Enable fast mode in your config (`runtime.fast_mode: true`) to reduce bootstrap samples from 1000 to 100 for lightning-quick demos (~2-3 seconds vs ~5-7 seconds).

**Statistical Impact:**

- **Still Valid**: Results are statistically sound for most practical purposes
- **Precision**: ~0.5-1% wider confidence intervals compared to full mode
- **Use For**: Development, demos, iterative model tuning, CI/CD validation
- **Avoid For**: Final regulatory submissions, high-stakes decisions

**When to Use Full Mode:**

- Production audits requiring maximum precision
- Regulatory compliance where statistical rigor is critical
- Research publications or academic validation
- When fairness metrics need sub-1% precision

!!! info "Timing Expectations"
**With fast mode enabled** (`runtime.fast_mode: true`):

    - German Credit (LogisticRegression): ~2-3 seconds
    - German Credit (XGBoost): ~3-4 seconds
    - Adult Income (LogisticRegression): ~4-5 seconds
    - Adult Income (XGBoost): ~5-6 seconds

    **Production mode** (fast mode disabled):

    - German Credit (LogisticRegression): ~5-7 seconds
    - German Credit (XGBoost): ~7-9 seconds
    - Adult Income (LogisticRegression): ~12-15 seconds
    - Adult Income (XGBoost): ~15-18 seconds

    Times measured on Apple M1 Max (32GB RAM). Your mileage may vary based on hardware and dataset size.

!!! info "Progress Bars"
GlassAlpha shows progress bars for long-running bootstrap operations (calibration and fairness confidence intervals). Progress bars:

    - **Auto-detect environment**: Terminal vs Jupyter notebook
    - **Respect configuration**: Disabled in strict mode (professional audit output)
    - **Can be disabled**: Set `GLASSALPHA_NO_PROGRESS=1` environment variable
    - **Skip fast operations**: Only show for 100+ bootstrap samples

    Progress bars use minimal CPU overhead and provide visual feedback during statistical computations.

**Note:** The simple configuration uses `logistic_regression` model (always available). For advanced models like XGBoost or LightGBM, install with `pip install 'glassalpha[explain]'`.

### What happens

1. **Automatic Dataset Resolution**: Uses built-in German Credit dataset
2. **Model Training**: Trains LogisticRegression classifier (baseline model)
3. **Explanations**: Generates coefficient-based feature importance
4. **Fairness Analysis**: Computes bias metrics for protected attributes (gender, age)
5. **Report Generation**: Creates professional HTML audit report with visualizations

### Expected output

```
Loading data and initializing components...
✓ Audit pipeline completed in 2.34s

📊 Audit Summary:
  ✅ Performance metrics: 8 computed
     ✅ accuracy: 75.4%
  ⚖️ Fairness metrics: 62/62 computed
     ✅ No bias detected
  🔍 Explanations: ✅ Global feature importance
     Most important: purpose_used_car (+1.022)
  📋 Dataset: 1,000 samples, 23 features
  🔧 Components: 2 selected
     Model: logistic_regression
     Explainer: coefficients

Generating PDF report: my_first_audit.pdf
✓ Saved plot to /tmp/plots/shap_importance.png
✓ Saved plot to /tmp/plots/performance_summary.png
✓ Saved plot to /tmp/plots/fairness_analysis.png

🎉 Audit Report Generated Successfully!
==================================================
📁 Output: /path/to/my_first_audit.pdf
📊 Size: 847,329 bytes (827.5 KB)
⏱️ Total time: 3.12s
   • Pipeline: 2.34s
   • PDF generation: 0.78s

The audit report is ready for review and regulatory submission.
```

## Step 3: Review your audit report

Open `my_first_audit.pdf` to see your comprehensive audit report containing:

### Executive summary

- Key findings and compliance status
- Model performance overview
- Bias detection results
- Regulatory assessment

### Model performance analysis

- Accuracy, precision, recall, F1 score, AUC-ROC
- Confusion matrix
- Performance visualizations

### Model explanations

- Global feature importance rankings (coefficient-based for linear models, SHAP for tree models)
- Individual prediction explanations
- Clear visualization of what drives predictions

### Fairness analysis

- Demographic parity assessment
- Equal opportunity analysis
- Bias detection across protected attributes
- Statistical significance testing

### Reproducibility manifest

- Complete audit trail with timestamps
- Dataset fingerprints and model parameters
- Random seeds and component versions
- Git commit information

## Step 4: Understanding the configuration

The `german_credit_simple.yaml` configuration file contains all audit settings (packaged with GlassAlpha):

Direct configuration (no profiles needed):

```yaml
model:
  type: xgboost
explainers:
  strategy: first_compatible
```

Reproducibility settings:

```yaml
reproducibility:
  random_seed: 42
```

Data configuration:

```yaml
data:
  dataset: german_credit # Uses built-in German Credit dataset
  fetch: if_missing # Automatically download if needed
  target_column: credit_risk
  protected_attributes:
    - gender
    - age_group
    - foreign_worker
```

Model configuration:

```yaml
model:
  type: logistic_regression # Baseline model (always available)
  params:
    random_state: 42
    max_iter: 1000
# For advanced models (requires pip install 'glassalpha[explain]'):
# type: xgboost
# params:
#   objective: binary:logistic
#   n_estimators: 100
#   max_depth: 5
```

Explainer selection:

```yaml
explainers:
  strategy: first_compatible
  priority:
    - coefficients # Zero-dependency explainer for linear models
  config:
    coefficients:
      normalize: true
# For tree models with SHAP (requires pip install 'glassalpha[explain]'):
# priority:
#   - treeshap # Best for XGBoost, LightGBM, RandomForest
#   - kernelshap # Model-agnostic SHAP fallback
```

Metrics to compute:

```yaml
metrics:
  performance:
    metrics: [accuracy, precision, recall, f1, auc_roc]
  fairness:
    metrics: [demographic_parity, equal_opportunity]
```

## Common mistakes (and how to fix them)

### Mistake 1: "Directory already exists"

❌ **Error**: `Directory 'my-audit-project' already exists and is not empty`

✅ **Solution**: Use `--output` to specify a different directory:

```bash
glassalpha quickstart --output my-project-2
# Or choose a custom name:
glassalpha quickstart --output credit-audit-2024
```

**Why it happens**: `glassalpha quickstart` defaults to `my-audit-project`. If you've run it before, you need a different name.

### Mistake 2: PDF generation takes too long

❌ **Problem**: PDF generation hangs or takes 5+ minutes

✅ **Solution**: Use HTML format instead (instant, portable):

```bash
glassalpha audit --output audit.html  # Recommended for development
```

**For regulatory submission**: Generate HTML first, then convert if needed:

```bash
glassalpha audit --output audit.html
# Later, if you need PDF:
glassalpha html-to-pdf audit.html
```

**Why**: HTML is instant (<1 second), byte-identical, and works everywhere. PDF generation is slow due to WeasyPrint rendering. Use HTML for development, PDF only for final regulatory submission.

### Mistake 3: Config file not found

❌ **Error**: `Config file not found: audit.yaml`

✅ **Solution**: Run `glassalpha quickstart` first to generate a project:

```bash
glassalpha quickstart --output my-project
cd my-project
python run_audit.py  # Now it will find audit.yaml
```

**Why it happens**: The audit command looks for config files in the current directory. Make sure you're in the project directory created by quickstart.

### Mistake 4: Protected attributes not found

❌ **Error**: `DataSchemaError: Column 'gender' not found`

✅ **Fix**: Check spelling and verify column names match exactly (case-sensitive)

**Debugging steps**:

1. Print column names: `python -c "import pandas as pd; print(pd.read_csv('data.csv').columns)"`
2. Update config to match exact column names
3. Note: Column names are case-sensitive (`Gender` ≠ `gender`)

### Mistake 5: Model type mismatch

❌ **Error**: `ExplainerCompatibilityError: treeshap not compatible with LogisticRegression`

✅ **Fix**: Use `coefficients` explainer for linear models, `treeshap` for tree models

**In your config**:

```yaml
# For LogisticRegression:
explainers:
  priority:
    - coefficients  # Fast, accurate for linear models

# For XGBoost/LightGBM:
explainers:
  priority:
    - treeshap  # Best for tree models
```

**Reference**: See [Model-Explainer Compatibility](../reference/model-explainer-compatibility.md) for full matrix

## Next steps

### Try advanced features

Enable strict mode for regulatory compliance:

```bash
glassalpha audit \
  --config german_credit_simple.yaml \
  --output regulatory_audit.html \
  --strict
```

Use a different model (edit config file: model.type: lightgbm):

```bash
glassalpha audit \
  --config german_credit_simple.yaml \
  --output lightgbm_audit.html
```

### Explore more options

See all available CLI options:

```bash
glassalpha audit --help
```

List available components:

```bash
glassalpha list
```

Validate configuration without running audit:

```bash
glassalpha validate --config german_credit_simple.yaml
```

View available templates and configs:

```bash
glassalpha config-list          # See available config templates
glassalpha config-template german_credit  # View a template
glassalpha list                 # List available models, explainers, metrics
```

### Work with your own data

Ready to audit your own models? We've made it easy:

1. **Follow the tutorial**: See [Using Custom Data](custom-data.md) for step-by-step guidance
2. **Use our template**: The fully-commented configuration template `custom_template.yaml` (packaged with GlassAlpha)
3. **Try public datasets**: Browse [built-in datasets](datasets.md) for testing

**Need to choose a model?** The [Model Selection Guide](../reference/model-selection.md) helps you pick between LogisticRegression, XGBoost, and LightGBM with performance benchmarks.

For detailed customization options, see the [Configuration Guide](configuration.md).

## Common use cases

### Financial services compliance

- Credit scoring model validation
- Fair lending assessments
- Regulatory reporting (ECOA, FCRA)
- Model risk management

### HR and employment

- Hiring algorithm audits
- Promotion decision analysis
- Salary equity assessments
- EEO compliance verification

### Healthcare and insurance

- Risk assessment model validation
- Treatment recommendation audits
- Coverage decision analysis
- Health equity evaluations

## Troubleshooting first-run errors

### Issue: Python version error

**Symptom**: `RuntimeError: GlassAlpha requires Python 3.11 or higher. You have Python 3.9...`

**Cause**: Using Python 3.10 or older

**Solution**: Install Python 3.11.8 (see [Installing Python 3.11+](#installing-python-311) above)

**Quick fix**:

```bash
# Using pyenv (recommended)
curl https://pyenv.run | bash
pyenv install 3.11.8
pyenv global 3.11.8
python --version  # Verify: should show 3.11.8
```

**If pyenv doesn't work**: Download from [python.org/downloads](https://python.org/downloads) and install manually.

### Issue: `glassalpha: command not found`

**Symptom**: After installation, running `glassalpha` results in "command not found"

**Cause**: CLI entry point not in PATH or package not installed

**Solution**:

```bash
# Option 1: Verify installation
pip list | grep glassalpha

# Option 2: Reinstall with pip
pip install -e .

# Option 3: Use module invocation (development)
cd glassalpha
PYTHONPATH=src python3 -m glassalpha --help
```

**Still not working?** Check if you're in the correct virtual environment:

```bash
which python  # Should show your venv path
```

### Issue: Import errors on first audit

**Symptom**: `ModuleNotFoundError: No module named 'sklearn'` or similar

**Cause**: Missing dependencies

**Solution**:

```bash
# Ensure pip is up to date
python -m pip install --upgrade pip

# Reinstall with dependencies
pip install -e ".[all]"

# Verify installation
pip list | grep -E "scikit-learn|pandas|numpy"
```

### Issue: XGBoost/LightGBM not available

**Symptom**: `glassalpha models` only shows `logistic_regression`

**Cause**: Advanced ML libraries not installed (base install only)

**Solution**: This is expected behavior for base install.

```bash
# Install advanced models
pip install -e ".[explain]"

# Verify XGBoost/LightGBM are now available
glassalpha models
```

### Issue: First audit fails with config error

**Symptom**: `ConfigError: missing required field 'data.path'`

**Cause**: Config file format issue or wrong path

**Solution**:

```bash
# Validate config before running audit
glassalpha validate --config german_credit_simple.yaml

# If file doesn't exist, ensure you're in correct directory
cd glassalpha
```

### Issue: Dataset download fails

**Symptom**: `DatasetError: Failed to fetch german_credit dataset`

**Cause**: Network issue or data loading problem

**Solution**:

```bash
# Check if config specifies built-in dataset
# Built-in datasets (german_credit, adult_income) load automatically
# If using custom data, verify the path in your config:
glassalpha validate --config your_config.yaml --check-data

# For offline use, set fetch: never in config:
# data:
#   dataset: german_credit
#   fetch: never
#   offline: true
```

### Issue: Permission errors on macOS

**Symptom**: `PermissionError: [Errno 13] Permission denied`

**Cause**: System Python or restrictive permissions

**Solution**:

```bash
# Use virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Or use user install
pip install --user -e .
```

### Issue: Slow first audit (>30 seconds)

**Symptom**: First audit takes much longer than expected

**Cause**: Cold start (package imports, dataset download)

**Expected**: First run is slower due to:

- One-time dataset download (~1-2MB)
- Python package imports
- Model training

**Solution**: Subsequent runs will be faster (3-5 seconds). If consistently slow:

```bash
# Enable fast mode for faster iterations
# Edit config and add:
# runtime:
#   fast_mode: true  # Reduces to 100 bootstrap samples (2-3s vs 5-7s)

# Or reduce SHAP samples in config:
# explainers:
#   config:
#     treeshap:
#       max_samples: 100  # Fewer background samples
```

### Issue: PDF generation fails

**Symptom**: Audit completes but no PDF created, or `WeasyPrint` errors

**Cause**: HTML mode works, PDF generation has issues

**Solution**:

```bash
# Use HTML output instead (works without WeasyPrint)
glassalpha audit --config german_credit_simple.yaml --output audit.html

# Or install PDF dependencies
pip install -e ".[pdf]"
```

### Still having issues?

1. **Check the full troubleshooting guide**: [Troubleshooting Reference](../reference/troubleshooting.md)
2. **Search existing issues**: [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues)
3. **Ask for help**: [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)

When reporting issues, include:

- Output of `glassalpha --help` (first few lines)
- Output of `python --version`
- Full error message
- Operating system

## Installing Python 3.11+

If you need to install or upgrade Python, here are the recommended methods:

### Using pyenv (recommended for developers)

```bash
# Install pyenv (one-time setup)
curl https://pyenv.run | bash

# Add to your shell (add to ~/.bashrc or ~/.zshrc)
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Install Python 3.11.8 (latest patch version)
pyenv install 3.11.8
pyenv global 3.11.8  # Set as default

# Verify
python --version  # Should show 3.11.8
```

### Using python.org (official installer)

1. Go to [python.org/downloads](https://python.org/downloads)
2. Download Python 3.11.8 (or latest 3.11.x)
3. Run the installer
4. **Important**: Check "Add Python to PATH" during installation
5. Open new terminal and verify:
   ```bash
   python --version  # Should show 3.11.8
   ```

### Using conda/mamba (data science environments)

```bash
# Create new environment with Python 3.11
conda create -n glassalpha python=3.11
conda activate glassalpha

# Or with mamba (faster)
mamba create -n glassalpha python=3.11
mamba activate glassalpha
```

### Troubleshooting Python installation

**Issue**: `python: command not found` after installation

**Solution**: Ensure Python is in your PATH:

```bash
# Check where Python was installed
which python3.11
# If not in PATH, add it:
export PATH="/path/to/python3.11:$PATH"
```

**Issue**: Permission denied (macOS/Linux)

**Solution**: Use virtual environment or user install:

```bash
# Create virtual environment
python3.11 -m venv ~/glassalpha-env
source ~/glassalpha-env/bin/activate

# Or use user install
pip install --user glassalpha
```

## Getting help

- **Documentation**: [Complete Guide](../index.md)
- **Guides**:
  - [Using Custom Data](custom-data.md) - Audit your own models
  - [Preprocessing Verification](../guides/preprocessing.md) - Verify production artifacts
  - [Built-in Datasets](datasets.md) - Automatic dataset fetching and caching
  - [Configuration Reference](configuration.md) - All configuration options
  - [Model Selection Guide](../reference/model-selection.md) - Choose the right model
  - [Explainer Deep Dive](../reference/explainers.md) - Understanding explanations
- **Examples**:
  - [German Credit Deep Dive](../examples/german-credit-audit.md) - Complete audit walkthrough
  - [Healthcare Bias Detection](../examples/healthcare-bias-detection.md) - Medical AI compliance example
  - [Fraud Detection Audit](../examples/fraud-detection-audit.md) - Financial services example
- **Support**:
  - [FAQ](../reference/faq.md) - Frequently asked questions
  - [Troubleshooting Guide](../reference/troubleshooting.md) - Common issues and solutions
  - [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues) - Report bugs or request features

## Summary

You now have GlassAlpha installed and have generated your first audit report. The system provides:

- **Production-ready audit generation** in seconds
- **Professional PDF reports** suitable for regulatory review
- **Comprehensive analysis** covering performance, fairness, and explainability
- **Full reproducibility** with complete audit trails
- **Flexible configuration** for different use cases and models

GlassAlpha transforms complex ML audit requirements into a simple, reliable workflow that meets the highest professional and regulatory standards.

## Next steps

### Dive deeper into core concepts

- **[Understanding Fairness Metrics](../reference/fairness-metrics.md)** - Learn about group, intersectional, and individual fairness with statistical confidence intervals
- **[Detecting Dataset Bias](../guides/dataset-bias.md)** - Catch proxy correlations, drift, and sampling bias before model training
- **[Calibration Analysis](../reference/calibration.md)** - Ensure predicted probabilities match observed outcomes

### Advanced features

- **[Testing Demographic Shifts](../guides/shift-testing.md)** - Validate model robustness under population changes with CI/CD gates
- **[Robustness Testing](../reference/robustness.md)** - Test stability under adversarial perturbations (ε-perturbation sweeps)
- **[Generating Reason Codes](../guides/reason-codes.md)** - ECOA-compliant adverse action notices

### Regulatory compliance

- **[SR 11-7 Compliance Mapping](../compliance/sr-11-7-mapping.md)** - Federal Reserve guidance for banking models (complete clause-to-artifact mapping)
- **[Trust & Deployment](../reference/trust-deployment.md)** - Reproducibility, determinism, and evidence pack export
