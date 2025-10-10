# GlassAlpha Determinism Strategy

> **Long-term strategy for maintaining byte-identical reproducibility across runs, platforms, and time.**
> Critical for regulatory compliance and audit verification.

## Overview

GlassAlpha implements comprehensive determinism enforcement to ensure that identical inputs produce byte-identical outputs across:

- Multiple runs with the same seed
- Different platforms (Linux, macOS, Windows)
- Different Python versions (3.11.4+)
- Different environments (CI, development, production)

This is a **regulatory requirement** for audit verification and compliance. Non-deterministic outputs raise red flags even if differences are cosmetic.

## Core Principles

### 1. **Three Levels of Determinism**

**Level 1: Same Machine, Same Run** (Easy)

- Two consecutive runs produce identical output
- Use case: Development and testing
- Implementation: Fixed seeds

**Level 2: Same Machine, Different Time** (Medium)

- Run today produces same output as run last month
- Use case: Auditor re-running old audit
- Implementation: Dependency pinning + environment control

**Level 3: Different Machines, Same Everything Else** (Hard)

- Run on macOS produces same output as Linux
- Use case: CI verification, multi-auditor verification
- Implementation: Environment control + platform normalization

**GlassAlpha targets Level 3** for all core audit outputs.

### 2. **Defense in Depth**

Determinism is enforced at multiple layers:

1. **Environment Layer**: PYTHONHASHSEED, BLAS threading, SOURCE_DATE_EPOCH
2. **Runtime Layer**: Centralized seed management, deterministic context managers
3. **Data Layer**: Canonical JSON serialization, fixed precision
4. **Output Layer**: PDF metadata normalization, timestamp control

### 3. **Fail Fast vs Fail Graceful**

- **Strict Mode**: Fail immediately on any non-deterministic condition
- **Development Mode**: Warn about potential issues but continue
- **Regulatory Mode**: Byte-identical outputs required

## Implementation Architecture

### 1. Deterministic Context Manager

```python
# Core determinism enforcement
with deterministic(seed=42, strict=True):
    # All operations are deterministic
    audit_results = run_audit(config)
    pdf = generate_report(audit_results)
```

**Controls**:

- Random number generators (Python, NumPy, ML frameworks)
- BLAS/LAPACK threading (OpenBLAS, MKL, OpenMP)
- Hash randomization (PYTHONHASHSEED)
- NumExpr parallelism

### 2. Timestamp Management

**Priority Order**:

1. `SOURCE_DATE_EPOCH` environment variable (CI/Docker builds)
2. Seed-derived deterministic timestamp (local reproducibility)
3. Current time (development only)

```python
# Get deterministic timestamp
timestamp = get_deterministic_timestamp(seed=42)
# Returns: datetime(2020, 1, 1, 0, 0, 42, tzinfo=timezone.utc)
```

### 3. PDF Metadata Normalization

PDFs include volatile metadata that breaks byte-identical comparisons:

```python
# Normalize PDF metadata for determinism
normalize_pdf_metadata(pdf_path)
# Sets fixed: CreationDate, ModDate, Producer, Creator
# Removes: ID, DocumentID, InstanceID
```

### 4. Canonical JSON Serialization

All JSON artifacts use canonical serialization:

```python
# Deterministic JSON output
json_str = canonicalize_json(data)
# Features: sorted keys, rounded floats (6 decimals), no NaN/Inf
```

## Environment Setup

### Required Environment Variables

```bash
# Core determinism variables (always set)
export PYTHONHASHSEED=42                    # Fixed hash seed
export OMP_NUM_THREADS=1                    # Single-threaded BLAS
export OPENBLAS_NUM_THREADS=1               # Single-threaded OpenBLAS
export MKL_NUM_THREADS=1                    # Single-threaded MKL

# Optional: Fixed timestamp for reproducible builds
export SOURCE_DATE_EPOCH=1577836800         # 2020-01-01 00:00:00 UTC

# Optional: Matplotlib backend for deterministic plots
export MPLBACKEND=Agg                       # Non-interactive backend
```

### Dependency Pinning Strategy

Use `constraints.txt` with exact versions:

```
# GlassAlpha Deterministic Environment
numpy==1.24.3
scipy==1.10.1
pandas==2.0.3
scikit-learn==1.3.0
xgboost==2.0.0
lightgbm==4.1.0
shap==0.43.0
matplotlib==3.7.1
weasyprint==60.0
pypdf==3.15.0
```

**When to Update**:

- Security patches only
- Known determinism bugs in dependencies
- **Never** update without regenerating golden fixtures

## Testing Strategy

### 1. Integration Tests

```python
@pytest.mark.integration
@pytest.mark.slow
def test_cli_respects_source_date_epoch(tmp_path):
    """Test CLI uses SOURCE_DATE_EPOCH for deterministic timestamps."""
    # Run audit 3 times with same environment
    # Verify all outputs are byte-identical
    # Test both HTML and PDF formats
```

### 2. Cross-Platform Tests

```python
@pytest.mark.integration
def test_model_predictions_are_platform_independent():
    """Test model predictions are same across platforms."""
    # Run in CI matrix: Linux, macOS
    # Verify predictions hash identically
```

### 3. Golden Fixtures

Pre-computed, byte-identical outputs checked into git:

```
examples/german_credit/
├── golden_audit.pdf              # Full audit PDF
├── golden_audit.manifest.json    # Manifest with hashes
├── golden_audit.sha256           # Just the hash
├── config.yaml                    # Exact config used
└── data.csv                       # Exact data
```

### 4. Determinism Validation Function

```python
# Comprehensive validation
results = validate_audit_determinism(
    config_path="audit.yaml",
    output_path="report.html",
    seed=42,
    runs=3,
    strict=True,
)
assert results["success"], f"Non-deterministic: {results['warnings']}"
```

## Maintenance Workflow

### 1. **Pre-Release Validation**

Before any release:

```bash
# 1. Regenerate golden fixtures
export SOURCE_DATE_EPOCH=1577836800
glassalpha audit --config examples/german_credit/config.yaml --out examples/german_credit/golden_audit.pdf

# 2. Run full determinism test suite
pytest tests/integration/test_determinism.py -v

# 3. Cross-platform validation (CI matrix)
# Linux + macOS must produce identical outputs

# 4. Update constraints.txt if needed
pip freeze > constraints.txt
```

### 2. **Golden Fixture Management**

**When to Update Golden Fixtures**:

- Dependency version changes
- Algorithm improvements that change outputs
- New features that affect report content

**Update Process**:

1. Generate new golden outputs with fixed environment
2. Compute hashes: `sha256sum examples/*/golden_audit.pdf`
3. Commit constraints.txt + golden fixtures together
4. Tag release with version

### 3. **CI/CD Integration**

**Required CI Checks**:

- Determinism tests pass on Linux + macOS matrix
- Golden fixture validation
- Cross-version compatibility (new version reproduces old golden PDFs)

**CI Environment**:

```yaml
env:
  SOURCE_DATE_EPOCH: "1577836800"
  PYTHONHASHSEED: "42"
  OMP_NUM_THREADS: "1"
  MPLBACKEND: "Agg"
```

### 4. **Debugging Non-Determinism**

**Step 1: Isolate the Component**

```bash
# Test each component separately
glassalpha audit --config config.yaml --skip-explanations
glassalpha audit --config config.yaml --skip-fairness
glassalpha audit --config config.yaml --skip-calibration
```

**Step 2: Binary Search**

- Disable half features → still broken?
- Find minimal config that breaks determinism

**Step 3: Hash Intermediate Outputs**

```python
# Add hash checkpoints in pipeline
from glassalpha.utils.hashing import hash_data

shap_hash = hash_data(shap_values)
fairness_hash = hash_data(fairness_metrics)
```

**Step 4: Compare Intermediate JSONs**

```bash
# Run twice, capture intermediates
glassalpha audit --config config.yaml --debug-dump /tmp/run1/
glassalpha audit --config config.yaml --debug-dump /tmp/run2/

# Compare all intermediate JSONs
diff -r /tmp/run1/ /tmp/run2/
```

## Common Pitfalls & Solutions

### 1. **Unseeded Randomness**

**Symptom**: Different SHAP values on same data
**Cause**: Random sampling without fixed seed
**Fix**: All randomness through `glassalpha.utils.seeds`

```python
# BAD
np.random.shuffle(samples)

# GOOD
from glassalpha.utils.seeds import get_rng
rng = get_rng("sample_selection")
rng.shuffle(samples)
```

### 2. **Dictionary Iteration Order**

**Symptom**: Different JSON hash across Python versions
**Cause**: Python dict ordering changes
**Fix**: Canonical JSON with `sort_keys=True`

### 3. **Floating Point Precision**

**Symptom**: Hash differs after recompute
**Cause**: Different FP precision across platforms
**Fix**: Round to 6 decimal places in canonical JSON

### 4. **Timestamps in Output**

**Symptom**: PDF hash changes even with same content
**Cause**: Embedded timestamps (creation time, modification time)
**Fix**: Use `SOURCE_DATE_EPOCH` + PDF metadata sanitization

### 5. **System Fonts**

**Symptom**: PDF looks different on different machines
**Cause**: System font availability varies
**Fix**: Embed fonts in PDF (WeasyPrint handles this)

### 6. **Parallel Processing**

**Symptom**: Results differ when using multiprocessing
**Cause**: Race conditions, non-deterministic order
**Fix**: Deterministic ordering or avoid parallelism

## Enterprise Extensions

### 1. **KMS/Timestamping Integration**

```python
# Enterprise: Cryptographic signing
from glassalpha.enterprise.signing import sign_pdf_with_kms

# Sign PDF with HSM/KMS
signed_pdf = sign_pdf_with_kms(
    pdf_path,
    kms_key_id="arn:aws:kms:us-east-1:123456789:key/xxx",
    timestamp_authority="http://tsa.example.com",
)
```

### 2. **Advanced Metadata**

```python
# Enterprise: Enhanced provenance
provenance = {
    "signed_at": "2025-01-01T12:00:00Z",
    "signer_certificate": "sha256:abc123...",
    "kms_key_id": "arn:aws:kms:us-east-1:123456789:key/xxx",
    "regulatory_framework": "EU AI Act",
    "audit_scope": "Article 16 compliance",
}
```

### 3. **Batch Processing**

```python
# Enterprise: Batch audit validation
from glassalpha.enterprise.batch import BatchAuditValidator

validator = BatchAuditValidator(
    config_template="audit.yaml",
    output_dir="audits/",
    concurrency=4,  # Parallel but deterministic
)

results = validator.run_batch(dataset_paths)
```

## Monitoring & Alerting

### 1. **Determinism Health Checks**

```python
# Health check endpoint
@app.get("/health/determinism")
def check_determinism():
    results = validate_audit_determinism(
        config_path="health_check_config.yaml",
        output_path="/tmp/health_audit.html",
        runs=2,
    )
    return {"status": "healthy" if results["success"] else "degraded"}
```

### 2. **Regression Detection**

```python
# Monitor for determinism regressions
def detect_determinism_regression():
    # Compare current outputs with golden fixtures
    # Alert if hash mismatch detected
    # Rollback if regression confirmed
```

### 3. **Performance Monitoring**

```python
# Track determinism overhead
with Timer() as timer:
    with deterministic(seed=42):
        run_audit()

# Log: determinism overhead < 5% of total time
```

## Success Metrics

### 1. **Technical Metrics**

- ✅ Byte-identical PDFs across platforms
- ✅ <2 second audit generation on German Credit
- ✅ Zero P0 determinism bugs in production
- ✅ 100% determinism test pass rate

### 2. **Regulatory Metrics**

- ✅ Auditors can reproduce results years later
- ✅ Hash verification works offline
- ✅ No platform-specific differences
- ✅ Complete audit trail for compliance

### 3. **User Experience Metrics**

- ✅ 60-second quickstart on standard datasets
- ✅ Zero network dependencies for core functionality
- ✅ Clear error messages for determinism failures
- ✅ Comprehensive documentation for reproducibility

## Risk Mitigation

### 1. **Backward Compatibility**

- New features must not break existing determinism
- Version bumps require golden fixture updates
- Migration guides for any breaking changes

### 2. **Platform Support**

- Primary: Linux (Ubuntu LTS), macOS (Intel/Apple Silicon)
- Secondary: Windows (with documented limitations)
- Tertiary: Docker containers (deterministic base images)

### 3. **Dependency Management**

- Pin all dependencies to exact versions
- Security patches only (no minor version bumps)
- Regular audit of dependency determinism properties

### 4. **Team Training**

- All developers must understand determinism requirements
- Code review checklist includes determinism validation
- Documentation updated for any changes

## Future Enhancements

### 1. **Advanced Validation**

- Statistical tests for determinism (bootstrap confidence intervals)
- Performance regression detection
- Automated golden fixture validation

### 2. **Enhanced Metadata**

- Blockchain-based audit trails
- Zero-knowledge proofs for verification
- Cryptographic signatures for regulatory submission

### 3. **Performance Optimization**

- Lazy determinism enforcement (only when needed)
- Caching of deterministic computations
- Parallel processing with deterministic ordering

## Conclusion

Determinism is not a feature—it's a foundational requirement for regulatory compliance and audit verification. This strategy ensures GlassAlpha maintains byte-identical reproducibility across all supported environments and use cases.

**Key Success Factors**:

1. **Comprehensive testing** at all layers
2. **Defense in depth** across environment, runtime, data, and output layers
3. **Clear maintenance processes** for golden fixtures and dependency management
4. **Regulatory focus** on audit verification and compliance requirements

This strategy will scale as GlassAlpha grows, ensuring that regulatory trust and audit credibility remain core differentiators.
