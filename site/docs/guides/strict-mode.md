# Strict Mode

Strict mode enforces regulatory compliance requirements for production audits.

## Overview

GlassAlpha offers two levels of strict mode:

- **Quick Strict Mode** (`--strict`): Suitable for development and testing
- **Full Strict Mode**: Required for production audits and regulatory submission

## Quick Strict Mode

Enable with the `--strict` flag:

```bash
glassalpha audit --config audit.yaml --output report.html --strict
```

**What it enforces:**

- Configuration validation against audit profile
- Random seed must be explicitly set
- All warnings treated as errors

**What it allows:**

- AUTO preprocessing (no artifact required)
- Missing data schema (inferred from dataset)
- Default configuration values

**When to use:**

- Development and testing
- Iterating on model configurations
- Quick validation before production

## Full Strict Mode

Required for production audits and regulatory submission.

**Requirements:**

1. **Preprocessing Artifact**: Saved preprocessing pipeline with hash verification

   ```yaml
   preprocessing:
     artifact: path/to/preprocessor.joblib
   ```

2. **Data Schema**: Explicit schema file (no inference)

   ```yaml
   data:
     schema: path/to/schema.yaml
   ```

3. **Explicit Seeds**: All random operations seeded

   ```yaml
   reproducibility:
     random_seed: 42
   ```

4. **Profile Specification**: Explicit audit profile

   ```yaml
   # Direct configuration
   ```

5. **Version Pinning**: Use `constraints.txt` for dependencies

**Enable full strict mode:**

```bash
# Set preprocessing artifact and schema in config
glassalpha audit --config production.yaml --output report.html --strict
```

**When to use:**

- Production audits for regulatory review
- Compliance documentation
- Audit trail for governance
- When byte-identical reproducibility is required

## Comparison

| Feature         | Quick Strict      | Full Strict       |
| --------------- | ----------------- | ----------------- |
| Preprocessing   | AUTO allowed      | Artifact required |
| Data schema     | Inferred OK       | Explicit required |
| Seeds           | Must be set       | Must be set       |
| Warnings        | Treated as errors | Treated as errors |
| Defaults        | Allowed           | Must be explicit  |
| Reproducibility | Single run        | Long-term         |

## Auto-Enabling Strict Mode

Strict mode automatically enables when:

- Config filename matches `prod*.yaml` or `production*.yaml`
- Running in CI environment (GitHub Actions, GitLab CI, etc.)
- `GLASSALPHA_STRICT=1` environment variable set

## Transition Path

### Step 1: Development (No Strict Mode)

```bash
glassalpha audit --config dev.yaml --output dev_audit.html --fast
```

### Step 2: Testing (Quick Strict)

```bash
glassalpha audit --config test.yaml --output test_audit.html --strict
```

### Step 3: Production (Full Strict)

```bash
# 1. Create preprocessing artifact
python scripts/create_preprocessing_artifacts.py

# 2. Create data schema
glassalpha data validate --create-schema

# 3. Run audit with full strict mode
glassalpha audit --config production.yaml --output audit.html --strict
```

## Troubleshooting

### "Quick strict mode enabled - NOT for production audits"

**Problem**: You enabled `--strict` but haven't met full strict requirements.

**Solution**: Add preprocessing artifact and data schema to your config.

### "Missing data schema in strict mode"

**Problem**: Full strict mode requires explicit schema.

**Solution**: Add `data.schema` to your config:

```yaml
data:
  schema: schemas/my_dataset.yaml
```

### "AUTO preprocessing not allowed in strict mode"

**Problem**: Full strict mode requires saved preprocessing artifact.

**Solution**: Save your preprocessing pipeline:

```bash
python scripts/create_preprocessing_artifacts.py
```

Then reference it in config:

```yaml
preprocessing:
  artifact: artifacts/preprocessor.joblib
```

## See Also

- [Preprocessing Guide](preprocessing.md)
- [Configuration Guide](../getting-started/configuration.md)
- [Regulatory Compliance](../compliance/index.md)
