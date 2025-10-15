# GlassAlpha

[![PyPI version](https://img.shields.io/pypi/v/glassalpha.svg)](https://pypi.org/project/glassalpha/)
[![License](https://img.shields.io/pypi/l/glassalpha.svg)](https://github.com/glassalpha/glassalpha/blob/main/LICENSE)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/glassalpha/glassalpha/badge)](https://securityscorecards.dev/viewer/?uri=github.com/glassalpha/glassalpha)
[![Python versions](https://img.shields.io/pypi/pyversions/glassalpha.svg)](https://pypi.org/project/glassalpha/)

[![Test & Build](https://github.com/glassalpha/glassalpha/workflows/Test%20&%20Build/badge.svg)](https://github.com/glassalpha/glassalpha/actions/workflows/test-and-build.yml)
[![Determinism](https://github.com/glassalpha/glassalpha/workflows/Determinism%20Validation/badge.svg)](https://github.com/glassalpha/glassalpha/actions/workflows/determinism.yml)
[![Security](https://github.com/glassalpha/glassalpha/workflows/OpenSSF%20Scorecard/badge.svg)](https://github.com/glassalpha/glassalpha/actions/workflows/scorecards.yml)

**Ever tried explaining your ML model to a regulator?**

GlassAlpha is an ([open source](https://glassalpha.com/reference/trust-deployment/#licensing-dependencies)) ML compliance toolkit that makes tabular models **transparent, auditable, and regulator-ready**.

Generate deterministic audit reports with statistical confidence intervals, fairness analysis, and deployment gates for CI/CD. HTML by default (fast, portable), PDF export available for regulatory submissions. No dashboards. No black boxes. Byte-stable outputs for regulatory reproducibility.

_Note: GlassAlpha is currently in beta (v0.2.0). Core functionality is stable with 1000+ passing tests and comprehensive documentation. Breaking API changes may occur before v1.0. First stable release expected Q1 2025._

## Quick Start (5 minutes)

> **Requirements**: Python 3.11+ and binary classification models only (multi-class not yet supported)

### 1. Install

```bash
# Create virtual environment (recommended)
python3 -m venv glassalpha-env
source glassalpha-env/bin/activate  # Windows: glassalpha-env\Scripts\activate

# Install GlassAlpha
pip install glassalpha
```

> **Note**: Base install includes HTML reports + LogisticRegression (fast, minimal dependencies). For full features (tree models, SHAP, PDF export): `pip install "glassalpha[all]"`

### 2. Generate Your First Audit

```bash
# Create project (uses German Credit dataset by default)
glassalpha quickstart
cd my-audit-project

# Run audit
glassalpha audit
```

### 3. View Results

```bash
# Open the report
open reports/audit_report.html  # macOS
xdg-open reports/audit_report.html  # Linux
```

**That's it!** You now have a professional compliance audit with performance metrics, fairness analysis, and feature explanations.

---

## Advanced Usage

### Test Demographic Shifts (Robustness Testing)

```bash
glassalpha audit --check-shift gender:+0.1
```

Tests if fairness metrics degrade when gender distribution changes by 10 percentage points.

### Generate Reason Codes (ECOA Compliance)

```bash
glassalpha reasons -m models/model.pkl -d models/test_data.csv -i 0
```

Generates adverse action notices with excluded protected attributes.

### Export Evidence Pack (Regulatory Submission)

```bash
glassalpha export-evidence-pack audit_report.html --output evidence.zip
```

Creates verifiable audit package with checksums and manifest for regulators.

---

## Optional Features

Install full feature set for production use:

```bash
pip install "glassalpha[all]"  # Tree models, SHAP, PDF export (recommended)
```

### One-Time Setup for Reproducibility

For byte-identical audit reports (required for regulatory compliance):

```bash
eval "$(glassalpha setup-env)"
```

Or add to your shell profile for permanent setup:

```bash
glassalpha setup-env >> ~/.bashrc  # or ~/.zshrc
```

---

## Installation (Full Details)

### From PyPI (Recommended)

```bash
# Base installation (HTML reports + LogisticRegression)
pip install glassalpha

# Full feature set (tree models, SHAP, PDF export)
pip install "glassalpha[all]"

# Individual feature groups:
# pip install "glassalpha[tree_models]"  # XGBoost + LightGBM
# pip install "glassalpha[shap]"         # SHAP explainers
# pip install "glassalpha[pdf]"          # PDF export
```

### From Source (Development)

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha
pip install -e ".[all]"
```

> **Troubleshooting**: If you see `externally-managed-environment` error, use a virtual environment (Python 3.11+ requirement).

**Tip:** Run `glassalpha doctor` anytime to check what features are available and see installation options.

**More details:** See the [full installation guide](https://glassalpha.com/getting-started/installation/) and [German Credit tutorial](https://glassalpha.com/examples/german-credit-audit/) to see what's in the report.

## Development Setup

To ensure your local environment matches CI:

```bash
# Clone repository
git clone https://github.com/GlassAlpha/glassalpha.git
cd glassalpha

# Set up determinism environment
source scripts/setup-determinism-env.sh

# Install with dev dependencies
pip install -e ".[dev,all]"

# Verify determinism
./scripts/test_determinism.sh quick
```

### Local vs CI Environment

GlassAlpha requires deterministic outputs for compliance. Our CI enforces:

- Single-threaded execution (no pytest-xdist)
- Fixed random seeds (PYTHONHASHSEED=0)
- UTC timezone (TZ=UTC)
- Headless matplotlib (MPLBACKEND=Agg)

Use `source scripts/setup-determinism-env.sh` to match CI environment locally.

## Repository Structure

- **`src/glassalpha/`** - Main Python package source code
  - **`src/glassalpha/configs/`** - Example audit configurations (packaged with install)
- **`tests/`** - Test suite
- **`site/`** - User documentation and tutorials ([glassalpha.com](https://glassalpha.com/))
- **`examples/`** - Jupyter notebooks and tutorials
- **`scripts/`** - Development and build scripts

## What Makes GlassAlpha Different

**CI/CD deployment gates, not dashboards.** Use shift testing to block deployments if models degrade under demographic changes.

```bash
# Block deployment if fairness degrades under demographic shifts
glassalpha audit --config audit.yaml \
  --check-shift gender:+0.1 \
  --fail-on-degradation 0.05
# Exit code 1 if degradation exceeds 5 percentage points
```

**Byte-identical reproducibility.** Same audit config → byte-identical HTML reports every time (on same platform+Python). HTML is the primary format for speed and determinism. PDF export available as optional feature for print-ready documents. SHA256 hashes in manifests for verification.

**Statistical rigor.** Not just point estimates—95% confidence intervals on fairness and calibration metrics with bootstrap resampling.

**Note**: Evidence pack export (E3) is available in v0.2.1 via `glassalpha export-evidence-pack`. Policy-as-code gates (E1) planned for v0.3.0. Current version supports shift testing gates for CI/CD.

## Core Capabilities

### Supported Models

**Binary classification only** (multi-class support planned for v0.3.0)

**Every audit includes:**

- **Group fairness** with 95% confidence intervals (demographic parity, equal opportunity, predictive parity)
- **Intersectional fairness** for bias at demographic intersections (e.g., gender×race, age×income)
- **Individual fairness** with consistency testing, matched pairs analysis, and counterfactual flip tests
- **Dataset bias detection** before model training (proxy correlations, distribution drift, sampling power analysis)
- **Calibration analysis** with confidence intervals (ECE, Brier score, bin-wise calibration curves)
- **Robustness testing** via adversarial perturbations (ε-sweeps) and demographic shift simulation
- **Feature importance** (coefficient-based for linear models, TreeSHAP for gradient boosting)
- **Individual explanations** via SHAP values for specific predictions
- **Preprocessing verification** with dual hash system (file + params integrity)
- **Complete audit trail** with reproducibility manifest (seeds, versions, git SHA, hashes)

**Separate commands available:**

- **Reason codes** (E2): Generate ECOA-compliant adverse action notices via `glassalpha reasons`
- **Actionable recourse** (E2.5): Generate counterfactual recommendations via `glassalpha recourse` (works best with sklearn models)

### Regulatory Compliance

- **[SR 11-7 Mapping](site/docs/compliance/sr-11-7-mapping.md)**: Complete Federal Reserve guidance coverage (banking)
- **Evidence Packs** (available in v0.2.1): SHA256-verified bundles for regulatory submission via `glassalpha export-evidence-pack`
- **Reproducibility**: Deterministic execution, version pinning, byte-identical HTML reports (same platform+Python)
- **CI/CD Gates**: Shift testing with `--fail-on-degradation` blocks deployments on metric degradation

- XGBoost, LightGBM, Logistic Regression (binary classification only; multi-class in v0.3.0)
- **Everything runs locally** - your data never leaves your machine

All Apache 2.0 licensed.

### Quick Features

- **30-second setup**: Interactive `glassalpha quickstart` wizard
- **Smart defaults**: Auto-detects config files, infers output paths
- **Built-in datasets**: German Credit and Adult Income for quick testing
- **Self-diagnosable errors**: Clear What/Why/Fix error messages
- **Automation support**: `--json-errors` flag for CI/CD pipelines

## Workflow Overview

```
Push to main
├─> Test & Build        (test-and-build.yml) - Always runs
│   ├─> Unit Tests      (fast contract tests)
│   ├─> Integration Tests (medium integration tests)
│   └─> Smoke Tests     (notebooks, security, linting - optional)
├─> Determinism         (determinism.yml) - Always runs
├─> CLI Docs Check      (cli-docs.yml) - If CLI changed
└─> OpenSSF Scorecard   (scorecards.yml) - Weekly + on push

Release published
└─> Publish to PyPI     (release.yml) - Only on release
    └─ Uses artifacts from Test & Build? No, rebuilds
```

## CI/CD Integration

GlassAlpha is designed for automation with deployment gates and standardized exit codes:

```bash
# Block deployment if model degrades under demographic shifts
glassalpha audit --config audit.yaml \
  --check-shift gender:+0.1 \
  --check-shift age:-0.05 \
  --fail-on-degradation 0.05

# Exit codes for scripting
# 0 = Success (all gates pass)
# 1 = Validation error (degradation exceeds threshold, compliance failures)
# 2 = User error (bad config, missing files)
# 3 = System error (permissions, resources)
```

**Auto-detection**: JSON errors automatically enable in GitHub Actions, GitLab CI, CircleCI, Jenkins, and Travis.

**Environment variable**: Set `GLASSALPHA_JSON_ERRORS=1` to enable JSON output.

Example JSON error output:

```json
{
  "status": "error",
  "exit_code": 1,
  "error": {
    "type": "VALIDATION",
    "message": "Shift test failed: degradation exceeds threshold",
    "details": { "max_degradation": 0.072, "threshold": 0.05 },
    "context": { "shift": "gender:+0.1" }
  },
  "timestamp": "2025-10-07T12:00:00Z"
}
```

**Deployment gates in action:**

```yaml
# .github/workflows/model-validation.yml
- name: Validate model before deployment
  run: |
    glassalpha audit --config prod.yaml \
      --check-shift gender:+0.1 \
      --fail-on-degradation 0.05
    # Blocks merge if fairness degrades >5pp under demographic shift
```

## Learn more

- **[Documentation](https://glassalpha.com/)** - User guides, API reference, and tutorials
- **[Contributing Guide](https://glassalpha.com/reference/contributing/)** - How to contribute to the project
- **[German Credit Tutorial](https://glassalpha.com/examples/german-credit-audit/)** - Step-by-step walkthrough with a real dataset
- **[About GlassAlpha](https://glassalpha.com/about/)** - Who, what & why

## Contributing

I'm a one man band, so quality contributions are welcome.

Found a bug? Want to add a model type? PRs welcome! Check the [contributing guide](https://glassalpha.com/reference/contributing/) for dev setup.

The architecture is designed to be extensible. Adding new models, explainers, or metrics shouldn't require touching core code.

### Testing Determinism Locally

Before pushing changes that affect audit generation, verify determinism:

```bash
# Quick check (30 seconds) - run 3 audits and verify identical hashes
./scripts/test_determinism.sh quick

# Full CI mirror (5 minutes) - comprehensive determinism suite
./scripts/test_determinism.sh full
```

If determinism breaks, check:

1. All configs have `reproducibility.strict: true`
2. SHAP operations use single threading
3. Models trained with `random_state` parameter
4. Environment variables set: `OMP_NUM_THREADS=1`, `TZ=UTC`, `MPLBACKEND=Agg`

## Supply Chain Security

Every GlassAlpha release includes verifiable supply chain artifacts for enterprise and regulatory use:

### CycloneDX Software Bill of Materials (SBOM)

- Complete inventory of all dependencies and transitive dependencies
- Machine-readable format for automated compliance checks
- Generated with each release and signed for authenticity

### Sigstore Keyless Signing

- Cryptographic signatures using Sigstore (keyless, certificate-based)
- Verifiable without managing keys or certificates
- Links to GitHub commit provenance

### Verification Commands

Verify a downloaded wheel before installation:

```bash
# Download wheel and signature
wget https://github.com/GlassAlpha/glassalpha/releases/download/v0.2.0/glassalpha-0.2.0-py3-none-any.whl
wget https://github.com/GlassAlpha/glassalpha/releases/download/v0.2.0/sbom.json

# Verify signature
cosign verify-blob --signature glassalpha-0.2.0-py3-none-any.whl.sig glassalpha-0.2.0-py3-none-any.whl

# Inspect SBOM
cat sbom.json | jq '.components | length'  # Count dependencies

# Check for vulnerabilities
pip-audit --desc
```

For automated verification in CI/CD:

```bash
curl -s https://raw.githubusercontent.com/GlassAlpha/glassalpha/main/scripts/verify_dist.sh | bash -s glassalpha-0.2.0-py3-none-any.whl
```

### Security Scanning

- **pip-audit**: Automated vulnerability scanning in CI
- **Dependency pinning**: All dependencies locked to specific versions
- **Supply chain monitoring**: GitHub Dependabot alerts for CVEs

---

## License

The core library is Apache 2.0. See [LICENSE](LICENSE) for the legal stuff.

Enterprise features/support may be added separately if there's demand for more advanced/custom functionality, but the core will always remain open and free. The name "GlassAlpha" is trademarked to keep things unambiguous. Details in [TRADEMARK.md](TRADEMARK.md).

For dependency licenses and third-party components, check the [detailed licensing info](https://glassalpha.com/reference/trust-deployment/#licensing-dependencies).
