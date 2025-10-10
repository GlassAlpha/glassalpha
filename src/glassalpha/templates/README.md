# GlassAlpha Configuration Templates

This directory contains minimal configuration templates for common use cases.

## Available Templates

### `quickstart.yaml` - Quick Testing

**Use when**: First time using GlassAlpha, exploring features
**Time to audit**: ~10 seconds
**Features**:

- Built-in dataset (no file needed)
- Fast logistic regression
- HTML output
- Basic metrics

**Usage**:

```bash
glassalpha audit --config templates/quickstart.yaml --output report.html
```

---

### `production.yaml` - Regulatory Compliance

**Use when**: Regulatory submission, production deployment
**Time to audit**: ~30-60 seconds
**Features**:

- Custom data with explicit schema
- Preprocessing artifact verification
- Comprehensive metrics
- PDF output
- Strict mode enabled
- Full audit trail

**Setup Required**:

1. Update data paths
2. Generate preprocessing hashes: `glassalpha prep hash`
3. List all features explicitly
4. Enable strict mode

**Usage**:

```bash
# First, prepare artifacts
glassalpha prep hash models/preprocessor.joblib

# Then run audit
glassalpha audit --config production.yaml --output audit.pdf
```

---

### `development.yaml` - Model Development

**Use when**: Iterating on models, testing different approaches
**Time to audit**: ~15-20 seconds
**Features**:

- Flexible data source
- Train model from config
- Auto preprocessing
- HTML output
- Balanced speed/insight

**Usage**:

```bash
glassalpha audit --config templates/development.yaml --output dev_report.html
```

---

### `testing.yaml` - CI/CD Integration

**Use when**: Automated testing, continuous integration
**Time to audit**: ~8 seconds
**Features**:

- Built-in dataset for reproducibility
- Fast algorithms
- Minimal metrics
- Strict mode for CI/CD
- Full determinism

**Usage**:

```bash
# In CI/CD pipeline
glassalpha audit --config templates/testing.yaml --output ci_report.html
if [ $? -ne 0 ]; then
  echo "Audit failed!"
  exit 1
fi
```

---

## Customizing Templates

### Quick Customization

```bash
# Copy template and modify
cp templates/quickstart.yaml my_config.yaml

# Edit for your needs
# - Change dataset
# - Adjust model parameters
# - Add/remove metrics
```

### Interactive Generation

```bash
# Use the init wizard (coming soon)
glassalpha init --output my_config.yaml
```

---

## Template Comparison

| Feature           | Quickstart  | Development | Testing     | Production |
| ----------------- | ----------- | ----------- | ----------- | ---------- |
| **Speed**         | ⚡⚡⚡ Fast | ⚡⚡ Medium | ⚡⚡⚡ Fast | ⚡ Slow    |
| **Strictness**    | Relaxed     | Relaxed     | Strict      | Strict     |
| **Data**          | Built-in    | Flexible    | Built-in    | Custom     |
| **Preprocessing** | Auto        | Auto        | Auto        | Artifact   |
| **Output**        | HTML        | HTML        | HTML        | PDF        |
| **Metrics**       | Basic       | Balanced    | Minimal     | Complete   |
| **Compliance**    | No          | No          | Partial     | Full       |

---

## When to Use Each Template

```
┌─────────────────────────────────────────────┐
│                                             │
│  First Time Using GlassAlpha?              │
│  ├─ Use: quickstart.yaml                   │
│  └─ Time: 10 seconds                       │
│                                             │
│  Developing a Model?                       │
│  ├─ Use: development.yaml                  │
│  └─ Time: 15-20 seconds                    │
│                                             │
│  Setting Up CI/CD?                         │
│  ├─ Use: testing.yaml                      │
│  └─ Time: 8 seconds                        │
│                                             │
│  Regulatory Submission?                    │
│  ├─ Use: production.yaml                   │
│  └─ Time: 30-60 seconds                    │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Common Modifications

### Change Dataset

```yaml
# From built-in
data:
  dataset: german_credit

# To custom
data:
  path: data/my_data.csv
  target_column: my_target
```

### Change Model

```yaml
# From logistic regression
model:
  type: logistic_regression

# To XGBoost
model:
  type: xgboost
  params:
    n_estimators: 100
```

### Add More Metrics

```yaml
metrics:
  performance:
    - accuracy
    - precision_weighted # Add these
    - recall_weighted
    - roc_auc
```

---

## Getting Help

- Documentation: https://glassalpha.com/guides/configuration/
- Examples: https://glassalpha.com/examples/
- CLI Help: `glassalpha audit --help`

---

**Tip**: Start with `quickstart.yaml` to understand the structure, then customize for your needs!
