# Troubleshooting

Common issues and solutions for GlassAlpha.

## Installation Issues

### "Module not found: glassalpha"

**Problem**: GlassAlpha not installed or not in Python path.

**Solution**:

```bash
# Install with pip
pip install glassalpha[all]

# Or from source
cd packages && pip install -e ".[all]"
```

### "Command not found: glassalpha"

**Problem**: CLI not in PATH (usually pipx issue).

**Solution**:

```bash
# Ensure pipx path is configured
python3 -m pipx ensurepath

# Restart shell
source ~/.bashrc  # or ~/.zshrc
```

## Configuration Errors

### "Target column 'X' not found in data"

**Problem**: Config specifies wrong target column name.

**Solution**: Check actual column names in your dataset:

```bash
# For built-in datasets
glassalpha datasets info german_credit

# For custom data
head -1 your_data.csv
```

**Common fixes**:

- German Credit: Use `target_column: credit_risk`
- Adult Income: Use `target_column: income_over_50k` (not `income`)

### "Protected attributes not found: ['gender']"

**Problem**: Dataset uses different column names for protected attributes.

**Solution**: Check schema for correct names:

```bash
glassalpha datasets info adult_income
```

**Common fixes**:

- Adult Income: Use `sex` instead of `gender`
- Check if dataset uses `age_group` instead of `age`

### "Missing data schema in strict mode"

**Problem**: Strict mode requires explicit schema.

**Solution**: Add schema to config:

```yaml
data:
  schema: schemas/my_dataset.yaml
```

Or disable strict mode for development:

```bash
glassalpha audit --config config.yaml  # no --strict flag
```

## Model Issues

### "Model type not supported by TreeExplainer"

**Problem**: SHAP doesn't recognize wrapped model.

**Solution**: Use `--save-model` with audit, then reasons/recourse will work:

```bash
# Save model during audit
glassalpha audit --config config.yaml --save-model model.pkl

# Use saved model for reasons
glassalpha reasons --model model.pkl --data data.csv --instance 0
```

### "XGBoost not installed"

**Problem**: Tree model extras not installed.

**Solution**:

```bash
pip install "glassalpha[xgboost]"
# or for all features
pip install "glassalpha[all]"
```

## Report Generation Issues

### "Failed to generate PDF: WeasyPrint not installed"

**Problem**: PDF dependencies not installed.

**Solution**:

```bash
pip install "glassalpha[pdf]"
```

On macOS, you may also need:

```bash
brew install cairo pango gdk-pixbuf
```

### "HTML report is 50+ MB"

**Problem**: Large embedded images (usually older matplotlib version).

**Solution**: Update dependencies:

```bash
pip install --upgrade glassalpha pillow matplotlib
```

Or use PDF format instead (much smaller):

```yaml
report:
  output_format: pdf
```

### "Matplotlib warning: unknown argument 'optimize'"

**Problem**: Outdated GlassAlpha version.

**Solution**: Update to latest version:

```bash
pip install --upgrade glassalpha
```

## Performance Issues

### "Audit takes >60 seconds"

**Solution**: Use fast mode for development:

```bash
glassalpha audit --config config.yaml --fast
```

This reduces bootstrap samples from 1000 to 100.

### "Out of memory during audit"

**Solution**: Reduce dataset size or bootstrap samples:

```yaml
# In config.yaml
metrics:
  fairness:
    bootstrap_samples: 100 # Reduce from 1000
```

Or sample the dataset:

```bash
glassalpha audit --config config.yaml --sample 1000
```

### "Sample size must be at least 100 rows"

**Problem**: The `--sample` flag requires at least 100 rows for statistical validity.

**Solution**: Use a larger sample size:

```bash
# For development testing
glassalpha audit --config config.yaml --sample 1000

# For production (full dataset)
glassalpha audit --config config.yaml
```

## QuickStart Issues

### "Directory already exists"

**Problem**: QuickStart won't overwrite existing directory.

**Solution**:

```bash
# Use different name
glassalpha quickstart --output my-audit-v2

# Or remove old directory
rm -rf my-audit-project
glassalpha quickstart
```

### "Adult Income audit fails after quickstart"

**Problem**: This was a bug in versions < 0.2.1.

**Solution**: Update to latest version:

```bash
pip install --upgrade glassalpha
```

## CLI Issues

### "No such option: --out"

**Problem**: Option was renamed to `--output`.

**Solution**: Use `--output` or `-o`:

```bash
glassalpha audit --config config.yaml --output report.html
```

### "No such option: --verbose"

**Problem**: The `--verbose` flag is global, not command-specific.

**Solution**: Use it before the command:

```bash
# Wrong - doesn't work
glassalpha audit --config config.yaml --verbose

# Right - global flag
glassalpha --verbose audit --config config.yaml
# or
glassalpha -v audit --config config.yaml
```

### "Got unexpected extra argument"

**Problem**: Command expects flags, not positional arguments.

**Solution**: Use flags explicitly:

```bash
# Wrong
glassalpha validate config.yaml

# Right
glassalpha validate --config config.yaml
```

## Reasons/Recourse Commands

### "Instance index out of range"

**Problem**: Specified instance doesn't exist in dataset.

**Solution**: Check dataset size:

```python
import pandas as pd
df = pd.read_csv("data.csv")
print(f"Dataset has {len(df)} rows (indices 0-{len(df)-1})")
```

### "Model file not found"

**Problem**: Need to save model first.

**Solution**:

```bash
# Save model during audit
glassalpha audit --config config.yaml --save-model model.pkl

# Then use it
glassalpha reasons --model model.pkl --data data.csv --instance 0
```

## Determinism Issues

### "PDF hash differs across runs"

**Problem**: Environment not deterministic.

**Solution**: Set required environment variables:

```bash
export TZ=UTC
export MPLBACKEND=Agg
export PYTHONHASHSEED=0

glassalpha audit --config config.yaml --output audit.pdf
```

### "Manifest hash verification failed"

**Problem**: File modified after generation.

**Solution**: Regenerate audit with clean state:

```bash
# Ensure clean git state
git status

# Regenerate
glassalpha audit --config config.yaml --output audit.pdf
```

## Getting Help

If your issue isn't covered here:

1. **Check documentation**: https://glassalpha.com/guides/
2. **Search issues**: https://github.com/glassalpha/glassalpha/issues
3. **Ask for help**: https://github.com/glassalpha/glassalpha/discussions
4. **Report bugs**: https://github.com/glassalpha/glassalpha/issues/new

When reporting issues, include:

- GlassAlpha version: `glassalpha --version`
- Python version: `python --version`
- Operating system
- Full error message
- Minimal reproduction steps
