# ML Engineer Workflow

Guide for ML engineers and data scientists implementing GlassAlpha audits in development, CI/CD, and production workflows.

## Overview

This guide is for technical practitioners who need to:

- Integrate audits into local development loops
- Set up CI/CD gates with policy enforcement
- Debug audit failures and configuration issues
- Optimize audit performance for large datasets
- Maintain reproducibility across environments

**Not an ML engineer?** For compliance-focused workflows, see [Compliance Officer Workflow](compliance-workflow.md).

## Key Capabilities

### Local Development Loop

Iterate quickly: Train → Audit → Fix → Reaudit

- Fast feedback (<60 seconds for small datasets)
- Inline HTML display in notebooks
- CLI for scripting and automation

### CI/CD Integration

Automated compliance checks on every commit or PR:

- Pre-commit hooks for local validation
- GitHub Actions for PR gates
- Exit codes for pass/fail decisions

### Debugging Tools

Diagnose audit failures:

- `--explain-failures` flag for verbose error messages
- Manifest inspection for reproducibility issues
- Config validation before running audit

## Typical Workflows

### Workflow 1: Local Development Loop

**Scenario**: You're iterating on a credit scoring model and want to check fairness after each training run.

#### Step 1: Set up config

Create a config file for quick iteration:

```yaml
# dev_audit.yaml
model:
  path: "models/credit_model.pkl"
  type: "xgboost"

data:
  train: "data/train.csv"
  test: "data/test.csv"
  target_column: "default"
  protected_attributes:
    - "gender"
    - "age_group"

audit_profile: "quickstart"
random_seed: 42

# Fast settings for dev
explainer:
  type: "treeshap"
  background_samples: 100 # Use 100 for speed, 1000 for production

fairness:
  metrics: ["demographic_parity", "equalized_odds"]
  threshold: 0.5

# Skip slow sections in dev
calibration:
  enabled: false
recourse:
  enabled: false
```

#### Step 2: Run audit locally

```bash
# Train model
python train_model.py --config model_config.yaml

# Audit immediately
glassalpha audit \
  --config dev_audit.yaml \
  --output dev_audit.pdf

# Check exit code
echo $?  # 0 = pass, 1 = failed gates, 2 = error
```

**Tip**: Use `--no-pdf` flag to skip PDF generation for faster iteration:

```bash
glassalpha audit \
  --config dev_audit.yaml \
  --no-pdf \
  --output metrics.json
```

#### Step 3: Debug failures

If audit fails, use `--explain-failures`:

```bash
glassalpha audit \
  --config dev_audit.yaml \
  --explain-failures \
  | tee audit_debug.log
```

**Common failures and fixes**:

| Error                         | Cause                   | Fix                                                     |
| ----------------------------- | ----------------------- | ------------------------------------------------------- |
| `ModelNotFoundError`          | Wrong path in config    | Check `model.path` value                                |
| `DataSchemaError`             | Missing columns         | Verify `data.protected_attributes` match CSV            |
| `ExplainerCompatibilityError` | Model type mismatch     | Use `treeshap` for tree models, `kernelshap` for others |
| `InsufficientSamplesError`    | Small group size (n<30) | Collect more data or aggregate groups                   |

#### Step 4: Iterate

Make changes and rerun:

```bash
# Fix model (e.g., add fairness constraints)
python train_model.py --fairness-constraint 0.1

# Reaudit
glassalpha audit --config dev_audit.yaml --output dev_audit_v2.pdf

# Compare metrics
glassalpha inspect --audit dev_audit.pdf --output metrics_v1.json
glassalpha inspect --audit dev_audit_v2.pdf --output metrics_v2.json
diff metrics_v1.json metrics_v2.json
```

### Workflow 2: CI/CD Integration with Policy Gates

**Scenario**: Block PR merges if model fails compliance gates.

#### Step 1: Define policy gates

```yaml
# policy/ci_baseline.yaml
policy_name: "CI Baseline Gates"
version: "1.0"

gates:
  - name: "Minimum AUC"
    metric: "roc_auc"
    threshold: 0.75
    comparison: "greater_than"
    severity: "error"

  - name: "Demographic Parity"
    metric: "demographic_parity_difference"
    threshold: 0.10
    comparison: "less_than"
    severity: "error"

  - name: "Calibration Quality"
    metric: "expected_calibration_error"
    threshold: 0.05
    comparison: "less_than"
    severity: "warning"
```

#### Step 2: Create GitHub Action

```yaml
# .github/workflows/model-audit.yml
name: Model Audit

on:
  pull_request:
    paths:
      - "models/**"
      - "data/**"
      - "configs/**"

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install glassalpha[all]

      - name: Run model audit
        run: |
          glassalpha audit \
            --config prod_audit.yaml \
            --policy-gates policy/ci_baseline.yaml \
            --output audit_report.pdf \
            --strict

      - name: Upload audit artifacts
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: audit-report
          path: |
            audit_report.pdf
            audit_report.manifest.json
            policy_decision.json

      - name: Comment PR with results
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '❌ Model audit failed. See artifacts for details.'
            })
```

#### Step 3: Add pre-commit hook (optional)

For faster local feedback before pushing:

```bash
# Install pre-commit hook
cp scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

Pre-commit hook runs lightweight checks:

- Config validation
- Data schema verification
- Quick audit with `--no-pdf`

**Full audit still runs in CI** (pre-commit is for early feedback only).

### Workflow 3: Notebook Development

**Scenario**: Interactive model development with inline audit results.

#### Step 1: Train model in notebook

```python
# notebook cell 1
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import glassalpha as ga

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X_train = train.drop(columns=["target"])
y_train = train["target"]
X_test = test.drop(columns=["target"])
y_test = test["target"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```

#### Step 2: Audit inline

```python
# notebook cell 2
# Create audit result inline (no config file needed)
result = ga.audit.from_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    protected_attributes={"gender": X_test["gender"], "age": X_test["age"]},
    random_seed=42
)

# Display in notebook
result  # Shows HTML summary automatically
```

#### Step 3: Explore metrics interactively

```python
# notebook cell 3
# Access metrics programmatically
print(f"AUC: {result.performance.roc_auc:.3f}")
print(f"Demographic parity: {result.fairness.demographic_parity_difference:.3f}")
print(f"Calibration ECE: {result.calibration.expected_calibration_error:.3f}")

# Plot calibration curve
result.calibration.plot()

# Plot fairness threshold sweep
result.fairness.plot_threshold_sweep()
```

#### Step 4: Export to PDF when ready

```python
# notebook cell 4
# Export full audit report
result.to_pdf("audit_report.pdf")
```

### Workflow 4: Debugging Reproducibility Issues

**Scenario**: Audit results differ between local and CI environments.

#### Step 1: Compare manifests

```bash
# Run audit locally
glassalpha audit --config audit.yaml --output local_audit.pdf

# Run audit in CI (download artifacts)
# Compare manifests
diff local_audit.manifest.json ci_audit.manifest.json
```

**Common differences**:

| Field              | Cause                  | Fix                              |
| ------------------ | ---------------------- | -------------------------------- |
| `data_hash`        | Different data files   | Ensure CI uses same data version |
| `package_versions` | Different environments | Pin versions in requirements.txt |
| `random_seed`      | Missing seed in config | Set `random_seed: 42` explicitly |
| `platform`         | macOS vs Linux         | Use Docker for consistency       |

#### Step 2: Use Docker for reproducibility

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["glassalpha", "audit", "--config", "audit.yaml", "--output", "report.pdf", "--strict"]
```

Run audit in Docker:

```bash
docker build -t audit-runner .
docker run -v $(pwd)/output:/app/output audit-runner
```

**Benefits**: Consistent environment across local, CI, and production.

#### Step 3: Use constraints files

Lock exact package versions:

```bash
# Generate constraints
pip freeze > constraints.txt

# Install with constraints
pip install -r requirements.txt -c constraints.txt
```

GlassAlpha provides platform-specific constraints in `constraints/`.

### Workflow 5: Performance Optimization

**Scenario**: Audit takes too long on large datasets.

#### Optimization strategies

**1. Reduce explainer samples**:

```yaml
explainer:
  type: "treeshap"
  background_samples: 100 # Default 1000, use 100 for 10x speedup
```

**2. Disable slow sections in dev**:

```yaml
calibration:
  enabled: false # Skip in dev, enable in prod
recourse:
  enabled: false # Skip in dev
robustness:
  enabled: false # Skip in dev
```

**3. Use sampling for large test sets**:

```yaml
data:
  test_sample_size: 1000 # Sample 1000 rows from test set
```

**4. Parallelize multiple audits**:

```bash
# Audit multiple models in parallel
for model in models/*.pkl; do
  glassalpha audit --config ${model%.pkl}.yaml &
done
wait
```

**5. Cache explainer results**:

```yaml
explainer:
  cache: true # Reuse SHAP values if data/model unchanged
```

## Best Practices

### Configuration Management

- **Dev vs Prod configs**: Separate configs for speed (dev) vs completeness (prod)
- **Version control**: Commit configs to git
- **Validation**: Run `glassalpha validate-config` before audit

### Reproducibility

- **Always set seeds**: `random_seed: 42` in all configs
- **Pin versions**: Use constraints files or Docker
- **Strict mode**: Enable `--strict` for production audits
- **Document environment**: Include requirements.txt with model

### CI/CD

- **Fast feedback**: Pre-commit hooks for lightweight checks
- **Comprehensive gates**: Full audit in CI before merge
- **Artifact retention**: Upload PDFs and manifests for traceability
- **Clear failures**: Use `--explain-failures` in CI logs

### Debugging

- **Start simple**: Minimal config, add complexity incrementally
- **Check logs**: Use `--verbose` for detailed output
- **Inspect manifests**: Verify data hashes, seeds, versions
- **Test locally first**: Don't debug in CI

## Troubleshooting

### Issue: Audit is slow (>5 minutes)

**Causes**:

- Large background samples for SHAP
- Large test set
- Expensive sections enabled (recourse, robustness)

**Fixes**:

- Reduce `explainer.background_samples` to 100
- Sample test set: `data.test_sample_size: 1000`
- Disable slow sections in dev config

### Issue: Non-deterministic results

**Causes**:

- Missing random seed
- Unseed platform differences (macOS vs Linux)
- Package version differences

**Fixes**:

- Set `random_seed: 42` in config
- Use `--strict` mode (enforces determinism checks)
- Run in Docker for platform consistency

### Issue: Explainer fails with compatibility error

**Causes**:

- Wrong explainer for model type (e.g., `treeshap` for neural network)

**Fixes**:

- Use `treeshap` for tree models (XGBoost, LightGBM, RandomForest)
- Use `kernelshap` for other models (logistic regression, neural nets)
- Check compatibility matrix: [Model-Explainer Compatibility](../reference/model-explainer-compatibility.md)

### Issue: CI fails but local passes

**Causes**:

- Different environments (package versions, platform)
- Different data (cached locally, fresh in CI)

**Fixes**:

- Compare manifests: `diff local.manifest.json ci.manifest.json`
- Use same constraints file in CI and local
- Run audit in Docker locally to match CI environment

## Related Resources

- [Quickstart Guide](../getting-started/quickstart.md) - 60-second first audit
- [Configuration Reference](../getting-started/configuration.md) - All config options
- [CLI Reference](../reference/cli.md) - All CLI commands
- [Troubleshooting](../reference/troubleshooting.md) - Common issues
- [Compliance Officer Workflow](compliance-workflow.md) - Evidence packs, regulator communication
- [Model Validator Workflow](validator-workflow.md) - Independent verification

## Support

For technical questions:

- GitHub Discussions: [GlassAlpha/glassalpha/discussions](https://github.com/GlassAlpha/glassalpha/discussions)
- Email: [contact@glassalpha.com](mailto:contact@glassalpha.com)
- Documentation: [glassalpha.com](https://glassalpha.com)
