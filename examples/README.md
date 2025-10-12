# GlassAlpha Examples

This directory contains runnable Jupyter notebooks demonstrating GlassAlpha features.

## Structure

Each feature has a corresponding notebook:

- `{feature-slug}.ipynb` - Demonstrates the feature with runnable code

## Requirements

Every notebook must:

- Include a smoke cell (version, seed, platform)
- Run end-to-end without manual edits
- Set all random seeds explicitly
- Use repo data or tiny synthetic datasets (no downloads)
- Demonstrate CLI equivalency
- Save at least one deterministic artifact
- Link to API docs in first cell

## Running Examples

```bash
# Install dependencies
pip install glassalpha jupyter

# Launch notebook
jupyter notebook examples/{feature-slug}.ipynb
```

## Validation

Before contributing a new notebook:

1. Run it twice in the same environment
2. Verify artifacts are byte-identical
3. Check all cross-links resolve
4. Keep total cells under 30 (prefer 10-20)

See `.cursor/rules/docs.mdc` for complete notebook standards.

## 🎯 Golden Audit Packages

Pre-generated, byte-identical audit packages for trust-building and verification:

### German Credit Audit (`german_credit_golden/`)

**Complete audit package for credit risk assessment:**

- **`audit_report.html`** - Full audit report with performance, fairness, and explanations
- **`manifest.json`** - Complete provenance metadata with hashes and lineage
- **`config.yaml`** - Exact configuration used for reproducibility
- **`SHA256SUMS.txt`** - Checksums for verification
- **`generate_golden_audit.py`** - Script to reproduce the audit

**Dataset**: German Credit (1,000 samples, 20 features)
**Model**: XGBoost classifier (97.7% test accuracy)
**Focus**: Credit risk assessment with fairness analysis

### Adult Income Audit (`adult_income_golden/`)

**Complete audit package for income prediction:**

- **`audit_report.html`** - Full audit report with performance, fairness, and explanations
- **`manifest.json`** - Complete provenance metadata with hashes and lineage
- **`config.yaml`** - Exact configuration used for reproducibility
- **`SHA256SUMS.txt`** - Checksums for verification
- **`generate_golden_audit.py`** - Script to reproduce the audit

**Dataset**: Adult Income (48,842 samples, 14 features)
**Model**: XGBoost classifier (98.0% test accuracy)
**Focus**: Income prediction with protected attribute fairness

## 🔐 Verification

### Verify Golden Packages

To verify the integrity of these golden packages:

```bash
# Generate fresh audit from the config
cd german_credit_golden/
python3 generate_golden_audit.py

# Compare checksums
sha256sum -c SHA256SUMS.txt

# Should show all files match (except timestamps in manifest)
```

### Golden Fixture Contents

Each golden package contains:

- **`audit_report.html`** - Byte-identical reference output
- **`manifest.json`** - Full lineage and provenance
- **`config.yaml`** - Exact config (with `strict: true` for determinism)
- **`SHA256SUMS.txt`** - Hash for verification

**Critical for determinism**: All configs include `reproducibility.strict: true` and `thread_control: true` to ensure byte-identical outputs across runs.

## 🚀 Usage

1. **For learning**: Run the notebooks to understand GlassAlpha workflows
2. **For verification**: Use golden packages to verify your GlassAlpha installation
3. **For trust-building**: Share golden packages with auditors for reproducibility verification
4. **For development**: Use as reference for creating your own audit packages

## 📋 Key Features Demonstrated

- ✅ **Deterministic audits** - Same config + same data = identical results
- ✅ **Complete provenance** - Full audit trail in manifest.json
- ✅ **Fairness analysis** - Protected attribute bias detection
- ✅ **Performance metrics** - Comprehensive model evaluation
- ✅ **SHAP explanations** - Feature importance and individual predictions
- ✅ **Regulatory compliance** - Audit-ready reports and manifests

These examples showcase GlassAlpha's core capabilities for transparent, auditable, and regulator-ready ML model evaluation.
