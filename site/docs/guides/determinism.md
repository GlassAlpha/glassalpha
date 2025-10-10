# Determinism Guide

**Why determinism matters**: Byte-identical audit outputs are critical for regulatory compliance and reproducible research. This guide explains how to achieve deterministic audits and troubleshoot non-determinism.

## Contents

- [Why Determinism Matters](#why-determinism-matters)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Validation](#validation)
- [Common Pitfalls](#common-pitfalls)
- [Debugging Non-Determinism](#debugging-non-determinism)
- [CI/CD Integration](#cicd-integration)
- [Advanced Topics](#advanced-topics)

---

## Why Determinism Matters

### Regulatory Compliance

Regulators require **byte-identical** audit outputs to verify:

- No hidden randomness
- No tampering
- Reproducible results across time and environments

**Example scenario**: A bank submits an audit in January. In March, a regulator asks to reproduce it. The audit must produce the **exact same PDF hash** or the submission is invalid.

### Scientific Reproducibility

Researchers need deterministic outputs to:

- Verify published results
- Compare methods fairly
- Build on prior work with confidence

### Production Debugging

Teams need deterministic outputs to:

- Reproduce bugs in production
- Verify fixes work correctly
- Compare model versions fairly

---

## Quick Start

### Minimal Example

```python
from glassalpha import audit_from_config
from pathlib import Path

# Run audit with explicit seed
result = audit_from_config(
    config_path="audit_config.yaml",
    output_path="audit.pdf",
    seed=42  # Explicit seed ensures determinism
)

# Verify determinism with validation tool
from glassalpha.utils.determinism_validator import validate_audit_determinism

report = validate_audit_determinism(
    config_path=Path("audit_config.yaml"),
    runs=3,
    seed=42,
)

print(f"Deterministic: {report.is_deterministic}")
print(f"Summary: {report.summary}")
```

### CLI Example

```bash
# Set deterministic environment
export SOURCE_DATE_EPOCH=1577836800
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run audit with seed
glassalpha audit \
  --config audit_config.yaml \
  --out audit.pdf \
  --seed 42

# Verify hash matches
sha256sum audit.pdf
```

---

## Environment Setup

### Required Environment Variables

```bash
# Core determinism (required)
export SOURCE_DATE_EPOCH=1577836800  # Fixed timestamp (Jan 1, 2020 UTC)
export PYTHONHASHSEED=42              # Dict/set ordering

# Numerical determinism (required)
export OMP_NUM_THREADS=1              # OpenMP threading
export OPENBLAS_NUM_THREADS=1         # OpenBLAS threading
export MKL_NUM_THREADS=1              # MKL threading

# Optional (recommended)
export TZ=UTC                         # Fixed timezone
export MPLBACKEND=Agg                 # Non-interactive matplotlib
export GLASSALPHA_DETERMINISTIC=1     # Strict mode
```

### Why Each Variable Matters

**SOURCE_DATE_EPOCH**: Overrides timestamps in PDF metadata and manifests. Without this, every run embeds the current time.

**PYTHONHASHSEED**: Controls Python's hash algorithm. Without this, dict ordering and set iteration are non-deterministic.

**Threading controls**: Disable BLAS/LAPACK threading. Multi-threaded operations have non-deterministic floating point behavior due to thread scheduling.

**TZ**: Fixes timezone for all datetime operations. Without this, timestamps vary by machine location.

**MPLBACKEND**: Forces non-interactive matplotlib backend. Interactive backends can have display-dependent rendering.

### Python Setup

```python
# Set seeds for all random number generators
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Use GlassAlpha's seed manager for reproducibility
from glassalpha.utils.seeds import SeedManager

seed_manager = SeedManager(master_seed=SEED)
seed_manager.set_global_seeds()
```

---

## Validation

### Validate Audit Determinism

```python
from glassalpha.utils.determinism_validator import DeterminismValidator
from pathlib import Path

validator = DeterminismValidator()

# Run audit 3 times and verify byte-identical outputs
report = validator.validate_audit_determinism(
    config_path=Path("audit_config.yaml"),
    runs=3,
    seed=42,
    check_shap=True,  # Verify SHAP available for deterministic explainer selection
)

if report.is_deterministic:
    print(f"✅ Deterministic: {report.summary}")
else:
    print(f"❌ Non-deterministic: {report.summary}")
    print(f"Sources: {report.non_determinism_sources}")
```

### Manual Validation

```bash
# Run audit twice
glassalpha audit -c config.yaml -o audit1.pdf --seed 42
glassalpha audit -c config.yaml -o audit2.pdf --seed 42

# Compare hashes (should be identical)
sha256sum audit1.pdf audit2.pdf

# Binary comparison (should show no differences)
diff audit1.pdf audit2.pdf && echo "✅ Byte-identical" || echo "❌ Different"
```

---

## Common Pitfalls

### 1. Missing SHAP Installation

**Problem**: Explainer fallback is non-deterministic when SHAP unavailable.

**Solution**: Always install SHAP for production audits:

```bash
pip install 'glassalpha[shap]'
```

**Why**: Without SHAP, the explainer selection falls back to coefficients or permutation explainers non-deterministically.

### 2. Unseeded Bootstrap Operations

**Problem**: Confidence intervals vary across runs.

**Solution**: Always pass `seed` parameter to bootstrap functions:

```python
# BAD - no seed
result = compute_fairness_metrics(y_true, y_pred, sensitive)

# GOOD - explicit seed
result = compute_fairness_metrics(
    y_true, y_pred, sensitive,
    seed=42,
    n_bootstrap=1000
)
```

### 3. Dict Serialization Without Sorting

**Problem**: JSON hashes vary due to dict ordering.

**Solution**: Always use `sort_keys=True`:

```python
import json

# BAD
json.dumps(data)

# GOOD
json.dumps(data, sort_keys=True)
```

### 4. Platform-Specific Behavior

**Problem**: Results differ between Linux/macOS/Windows.

**Solution**: Use consistent environment and test cross-platform:

```bash
# Test on multiple platforms
docker run --rm -v $(pwd):/work python:3.11 bash -c "
  cd /work &&
  pip install -c constraints.txt . &&
  glassalpha audit -c config.yaml -o audit.pdf --seed 42
"
```

### 5. Timestamp Pollution

**Problem**: PDF metadata includes volatile timestamps.

**Solution**: Set `SOURCE_DATE_EPOCH` before running:

```bash
export SOURCE_DATE_EPOCH=1577836800
glassalpha audit -c config.yaml -o audit.pdf --seed 42
```

---

## Debugging Non-Determinism

### Step 1: Isolate the Component

Run components individually to find the source:

```bash
# Test data loading
glassalpha data validate --contract schema.yaml --data data.csv

# Test model predictions (should be deterministic)
python -c "
from glassalpha.models import load_model
model = load_model('model.pkl')
predictions = model.predict(X_test)
print(hash(predictions.tobytes()))  # Should be same every time
"

# Test explainer (most common source of non-determinism)
glassalpha explain --model model.pkl --data test.csv --seed 42
```

### Step 2: Check Environment

Verify all determinism controls are set:

```python
import os

required_vars = {
    'SOURCE_DATE_EPOCH': '1577836800',
    'PYTHONHASHSEED': '42',
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
}

missing = []
wrong_value = []

for key, expected in required_vars.items():
    actual = os.environ.get(key)
    if actual is None:
        missing.append(key)
    elif actual != expected:
        wrong_value.append(f"{key}={actual} (expected {expected})")

if missing:
    print(f"❌ Missing: {missing}")
if wrong_value:
    print(f"⚠️  Wrong value: {wrong_value}")
if not (missing or wrong_value):
    print("✅ Environment configured correctly")
```

### Step 3: Validate Dependencies

Check that dependencies match expected versions:

```bash
# Generate dependency manifest
pip freeze > current_env.txt

# Compare to constraints
diff current_env.txt constraints.txt
```

### Step 4: Use Validation Tool

```python
from glassalpha.utils.determinism_validator import DeterminismValidator

validator = DeterminismValidator()

# Run validation with detailed output
report = validator.validate_audit_determinism(
    config_path="audit_config.yaml",
    runs=5,  # More runs for better confidence
    seed=42,
)

print(f"Is deterministic: {report.is_deterministic}")
print(f"Unique hashes: {len(set(report.hashes))}")
print(f"Run times: {report.run_times}")

if not report.is_deterministic:
    print("\nNon-determinism sources:")
    for source in report.non_determinism_sources:
        print(f"  - {source}")
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Determinism Validation

on: [push, pull_request]

jobs:
  validate-determinism:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v5

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -c constraints.txt -e ".[all]"
          pip install shap  # Required for deterministic explainer selection

      - name: Set deterministic environment
        run: |
          echo "SOURCE_DATE_EPOCH=1577836800" >> $GITHUB_ENV
          echo "PYTHONHASHSEED=42" >> $GITHUB_ENV
          echo "OMP_NUM_THREADS=1" >> $GITHUB_ENV
          echo "OPENBLAS_NUM_THREADS=1" >> $GITHUB_ENV
          echo "MKL_NUM_THREADS=1" >> $GITHUB_ENV

      - name: Run determinism test
        run: |
          pytest tests/test_critical_regression_guards.py::TestCriticalRegressions::test_cli_determinism_regression_guard -v

      - name: Validate cross-platform determinism
        run: |
          # Run audit twice
          glassalpha audit -c examples/german_credit/config.yaml -o audit1.pdf --seed 42
          glassalpha audit -c examples/german_credit/config.yaml -o audit2.pdf --seed 42

          # Compare hashes
          sha256sum audit1.pdf audit2.pdf

          # Verify identical
          diff audit1.pdf audit2.pdf || exit 1
```

### Docker Example

```dockerfile
FROM python:3.11-slim

# Set deterministic environment
ENV SOURCE_DATE_EPOCH=1577836800
ENV PYTHONHASHSEED=42
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TZ=UTC
ENV MPLBACKEND=Agg

# Install dependencies
COPY constraints.txt .
RUN pip install -c constraints.txt glassalpha[all]

# Run audit
WORKDIR /work
CMD ["glassalpha", "audit", "--config", "config.yaml", "--out", "audit.pdf", "--seed", "42"]
```

---

## Advanced Topics

### Deterministic Explainer Selection

Use strict mode to fail fast when preferred explainers unavailable:

```python
from glassalpha.explain.registry import select_explainer_deterministic

# Strict mode - fails if SHAP unavailable
explainer_name, explainer = select_explainer_deterministic(
    model=model,
    X_test=X_test,
    priority=['treeshap', 'kernelshap'],
    strict=True,  # Fail if none available
)
```

### Test Isolation

Use fixtures to ensure clean state between tests:

```python
import pytest
import numpy as np
import random

@pytest.fixture(autouse=True)
def isolate_determinism_state():
    """Reset random state before each test."""
    # Save original state
    orig_np_state = np.random.get_state()
    orig_py_state = random.getstate()

    # Reset to known state
    np.random.seed(42)
    random.seed(42)

    yield

    # Restore (best effort)
    np.random.set_state(orig_np_state)
    random.setstate(orig_py_state)
```

### Cross-Platform Validation

Test on multiple platforms before releasing:

```bash
# Linux
docker run --rm -v $(pwd):/work python:3.11 bash -c "
  cd /work && pip install -c constraints.txt . && glassalpha audit -c config.yaml -o linux.pdf --seed 42
"

# macOS
glassalpha audit -c config.yaml -o macos.pdf --seed 42

# Compare hashes
sha256sum linux.pdf macos.pdf
```

---

## Related Documentation

- [API Reference: DeterminismValidator](../reference/utils/determinism-validator.md)
- [API Reference: Deterministic Explainer Selection](../reference/explain/registry.md)
- [Troubleshooting Guide](../reference/troubleshooting.md)

---

## Summary

**Key takeaways**:

1. Set all environment variables before running audits
2. Always use explicit seeds for reproducibility
3. Install SHAP for deterministic explainer selection
4. Validate determinism with `DeterminismValidator`
5. Test cross-platform before production deployment

**Success criteria**:

- Same config + same data + same environment = byte-identical PDF
- Hash verification works offline
- Auditors can reproduce results years later
- No platform-specific differences
