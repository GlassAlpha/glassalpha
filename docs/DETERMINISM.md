# Determinism Implementation Guide for Contributors

> **User-facing documentation**: See https://glassalpha.com/guides/determinism/
>
> This document explains determinism implementation details for GlassAlpha contributors and CI maintainers.

## Overview

GlassAlpha is a compliance tool that must produce reproducible outputs for regulatory verification. This document covers the technical implementation details, CI validation strategy, and cross-platform behavior that contributors need to understand.

## Determinism Guarantees

### ✅ Within-Combo Determinism (GUARANTEED)

**Same platform + same Python version + same config → byte-identical output**

This is our core guarantee. If you run the same audit twice on the same machine with the same Python version, you will get **exactly the same output**.

**Example**:

```bash
# Ubuntu 22.04, Python 3.11
glassalpha audit -c config.yaml -o audit1.pdf
glassalpha audit -c config.yaml -o audit2.pdf

sha256sum audit1.pdf audit2.pdf
# Both produce: 986411a8e24939c4731d5d2d4644b1cca1248d84d8041d8dca2b33566ce077e0
```

**Why this matters**: Auditors can verify your results by re-running your audit. Regulators can confirm no tampering occurred.

### ⚠️ Cross-Platform Differences (EXPECTED)

**Different platforms produce different outputs (but each is internally consistent)**

Running the same audit on Ubuntu vs macOS will produce **different hashes**, but each platform is deterministic within itself.

**Why**: Platform-specific differences in:

- **Font rendering**: macOS and Ubuntu use different font engines (CoreText vs FreeType)
- **Image processing**: Different compression algorithms and anti-aliasing
- **PDF generation**: WeasyPrint interacts with system libraries differently

**Example**:

```bash
# Ubuntu 22.04, Python 3.11
glassalpha audit -c config.yaml -o audit.pdf
sha256sum audit.pdf
# 986411a8...  (119KB file)

# macOS 14, Python 3.11 (same config, same data)
glassalpha audit -c config.yaml -o audit.pdf
sha256sum audit.pdf
# f11ff1c8...  (272KB file)
```

**This is acceptable**: Both audits contain the same _information_ (metrics, decisions, explanations), just rendered differently. For compliance purposes, what matters is that each platform is reproducible.

### ⚠️ Python Version Differences (EXPECTED)

**Different Python versions produce different outputs**

Python 3.11 and 3.12 have different implementations that affect deterministic outputs.

**Why**: Python version changes affect:

- **Hash algorithms**: `hash()` behavior changed between versions
- **Dict ordering**: Subtle differences in dict iteration
- **NumPy/BLAS**: Different versions interact with numerical libraries differently
- **Pickle format**: Serialization differences

**Example**:

```bash
# Ubuntu 22.04, Python 3.11
sha256sum audit.pdf
# 986411a8...

# Ubuntu 22.04, Python 3.12 (same platform, same config)
sha256sum audit.pdf
# 428d926d...  (different!)
```

**This is acceptable**: Pin your Python version in production and CI. Use the same version for verification.

## How We Achieve Determinism

### Environment Controls

GlassAlpha enforces determinism through strict environment controls:

```bash
# Core determinism controls (automatically set when strict: true)
export PYTHONHASHSEED=0              # Fixed Python hash seed
export TZ=UTC                        # Fixed timezone
export MPLBACKEND=Agg                # Headless matplotlib
export SOURCE_DATE_EPOCH=1577836800  # Fixed timestamp

# Single-threaded execution (prevents race conditions)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# Locale control
export LC_ALL=C
```

### Configuration

Enable strict determinism in your config:

```yaml
reproducibility:
  random_seed: 42
  strict: true # Enforces all determinism controls
  thread_control: true # Single-threaded execution
```

### Random Seed Management

All randomness in GlassAlpha is seeded:

- Model training: `model.params.random_state`
- Data splitting: `reproducibility.random_seed`
- SHAP sampling: Seeded via `thread_control`
- Bootstrap resampling: Seeded via `random_seed`

## Verification

### Local Verification

Check that your local setup produces deterministic outputs:

```bash
# Set up deterministic environment
source scripts/setup-determinism-env.sh

# Run quick determinism check (3 audits)
./scripts/test_determinism.sh quick

# Expected output:
# ✅ SUCCESS: All runs produced identical output
```

### CI Verification

Our CI validates within-combo determinism on every commit:

- **Ubuntu 22.04** + **Python 3.11**: `audit1.pdf` == `audit2.pdf` ✅
- **Ubuntu 22.04** + **Python 3.12**: `audit1.pdf` == `audit2.pdf` ✅
- **macOS 14** + **Python 3.11**: `audit1.pdf` == `audit2.pdf` ✅
- **macOS 14** + **Python 3.12**: `audit1.pdf` == `audit2.pdf` ✅

Cross-platform hashes may differ, but each combo is internally consistent.

### Production Verification

For regulatory verification:

1. **Document your environment**:

   - Operating system (Ubuntu 22.04, macOS 14, etc.)
   - Python version (3.11.4)
   - GlassAlpha version (0.2.0)
   - System dependencies (see `glassalpha doctor`)

2. **Provide verification script**:

   ```bash
   #!/bin/bash
   # Reproduce audit exactly
   pip install glassalpha==0.2.0
   source scripts/setup-determinism-env.sh
   glassalpha audit -c config.yaml -o audit.pdf

   # Verify hash
   echo "Expected: 986411a8e24939c4731d5d2d4644b1cca1248d84d8041d8dca2b33566ce077e0"
   sha256sum audit.pdf
   ```

3. **Include in audit manifest**:
   - Environment hash
   - System platform
   - Python version
   - All dependency versions

## Common Determinism Issues

### Issue 1: Hash Mismatch on Same Machine

**Symptom**: Two runs on same machine produce different hashes

**Likely causes**:

- `strict: true` not set in config
- `thread_control: false` (allows parallel execution)
- Environment variables not set
- Parallel testing (pytest-xdist)

**Fix**:

```yaml
# config.yaml
reproducibility:
  strict: true
  thread_control: true
```

### Issue 2: Hash Mismatch Between Platforms

**Symptom**: Ubuntu and macOS produce different hashes

**This is expected**: See "Cross-Platform Differences" above.

**If you need cross-platform consistency**: Consider using HTML output instead of PDF. HTML rendering is more consistent across platforms (though still not byte-identical).

### Issue 3: Hash Changes After Dependency Update

**Symptom**: Same config produces different hash after `pip install -U`

**Likely cause**: Dependency version change (numpy, matplotlib, weasyprint)

**Fix**: Pin dependencies using `constraints.txt`:

```bash
pip freeze > constraints.txt
pip install -c constraints.txt glassalpha
```

## Development Guidelines

### Writing Deterministic Code

When contributing to GlassAlpha:

1. **Always seed randomness**:

   ```python
   # ❌ Don't do this
   np.random.shuffle(data)

   # ✅ Do this
   from glassalpha.utils.seeds import get_rng
   rng = get_rng("my_operation")
   rng.shuffle(data)
   ```

2. **Avoid system time**:

   ```python
   # ❌ Don't do this
   timestamp = datetime.now()

   # ✅ Do this
   from glassalpha.utils.determinism import get_deterministic_timestamp
   timestamp = get_deterministic_timestamp()
   ```

3. **Control threading**:

   ```python
   # ❌ Don't do this
   import multiprocessing as mp
   pool = mp.Pool()

   # ✅ Do this
   # Single-threaded only (or check config.reproducibility.thread_control)
   ```

4. **Sort non-deterministic collections**:

   ```python
   # ❌ Don't do this (dict iteration order can vary)
   for key in some_dict:
       ...

   # ✅ Do this
   for key in sorted(some_dict.keys()):
       ...
   ```

### Testing Determinism

Every PR must pass determinism checks:

```bash
# Local check before pushing
source scripts/setup-determinism-env.sh
./scripts/test_determinism.sh quick

# CI check (automatic)
# Validates within-combo determinism on 4 platform+Python combos
```

## FAQ

### Q: Why do Ubuntu and macOS produce different PDFs?

**A**: Font rendering and image processing differ between platforms. WeasyPrint (our PDF library) uses system fonts and libraries, which vary by OS. Each platform is deterministic within itself, which is sufficient for compliance.

### Q: Can I get byte-identical PDFs across platforms?

**A**: Not currently. PDF rendering is inherently platform-dependent. However, the **information content** (metrics, decisions, explanations) is identical. For cross-platform verification, compare the audit manifest JSON (which is platform-independent).

### Q: Why do Python 3.11 and 3.12 produce different outputs?

**A**: Python versions have different hash algorithms, dict behaviors, and interact with NumPy differently. Pin your Python version in production to ensure reproducibility.

### Q: How do I verify an audit from 6 months ago?

**A**: Use the exact environment documented in the audit manifest:

1. Same Python version
2. Same GlassAlpha version
3. Same platform (or accept cross-platform differences)
4. Same dependencies (use `constraints.txt` from audit package)

### Q: What if I need to run audits on different platforms?

**A**: Each platform will produce different PDFs, but each is reproducible. Document the platform used for the official audit, and note that verification must use the same platform.

### Q: Is determinism required for all GlassAlpha features?

**A**: Yes. Determinism is a core compliance requirement. All features that produce outputs (audits, reports, metrics) must be deterministic when `strict: true`.

## Summary for Contributors

### What's Guaranteed ✅

- Same platform + same Python + same config = **byte-identical output**
- Within-combo determinism validated in CI on every commit
- All randomness is seeded and reproducible

### What's Expected ⚠️

- Different platforms produce different PDFs (but each is internally consistent)
- Different Python versions produce different outputs
- Dependency updates may change outputs (use `constraints.txt`)

### Best Practices for Contributors 📋

1. Always seed randomness using `glassalpha.utils.seeds.get_rng()`
2. Sort dict keys when serializing JSON
3. Avoid system time (use `get_deterministic_timestamp()`)
4. Test determinism locally before pushing
5. Run `./scripts/test_determinism.sh quick` before submitting PRs
6. Update constraints.txt when adding new dependencies

### Implementation Reference

- Environment setup: `scripts/setup-determinism-env.sh`
- Determinism testing: `scripts/check-determinism-quick.sh`
- CI validation: `.github/workflows/determinism.yml`
- User guide: https://glassalpha.com/guides/determinism/

### For Users

See the comprehensive user guide for:

- Setting up deterministic environments
- Troubleshooting non-determinism
- Production deployment practices
- Regulatory verification workflows

📖 **User Guide**: https://glassalpha.com/guides/determinism/
