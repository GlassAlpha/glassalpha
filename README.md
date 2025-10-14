# GlassAlpha

**Ever tried explaining your ML model to a regulator?**

GlassAlpha is an ([open source](https://glassalpha.com/reference/trust-deployment/#licensing-dependencies)) ML compliance toolkit that makes tabular models **transparent, auditable, and regulator-ready**.

Generate deterministic PDF audit reports with statistical confidence intervals, fairness analysis, and deployment gates for CI/CD. No dashboards. No black boxes. Byte-stable outputs for regulatory reproducibility.

_Note: GlassAlpha is currently in beta (v0.2.0). Core functionality is stable with 1000+ passing tests and comprehensive documentation. Breaking API changes may occur before v1.0. First stable release expected Q1 2025._

## Installation

### Quick install

**Recommended: Use a virtual environment to avoid conflicts**

```bash
# Step 1: Create and activate virtual environment
python3 -m venv glassalpha-env
source glassalpha-env/bin/activate  # On Windows: glassalpha-env\Scripts\activate

# Step 2: Install GlassAlpha with all features
pip install "glassalpha[all]"

# Step 3: Verify installation
glassalpha --version
glassalpha doctor
```

> **💡 Troubleshooting**: If you see `externally-managed-environment` error, you must use a virtual environment (Step 1 above). This is required on Python 3.11+ systems with PEP 668.

**Base installation** includes LogisticRegression model with coefficient-based explanations (fast, zero extra dependencies).

**Optional features:**

```bash
pip install "glassalpha[explain]"  # SHAP + XGBoost + LightGBM
pip install "glassalpha[viz]"      # Enhanced visualizations
pip install "glassalpha[all]"      # All features (recommended)
```

### Install from source

For development or latest features:

```bash
# Create virtual environment
python3 -m venv glassalpha-env
source glassalpha-env/bin/activate

# Clone and install
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha
pip install -e ".[all]"
```

## Development Setup

To ensure your local environment matches CI:

```bash
# Set up determinism environment (required for compliance)
source scripts/setup-determinism-env.sh

# Verify determinism works
./scripts/check-determinism-quick.sh
```

### Local vs CI Environment

GlassAlpha requires deterministic outputs for compliance. Our CI enforces:

- Single-threaded execution (no pytest-xdist)
- Fixed random seeds (PYTHONHASHSEED=0)
- UTC timezone (TZ=UTC)
- Headless matplotlib (MPLBACKEND=Agg)

Use `source scripts/setup-determinism-env.sh` to match CI environment locally.

Create a project with configuration (interactive wizard)

```bash
glassalpha quickstart
```

Generate an audit report

```bash
glassalpha audit
```

**Tip**: For faster iteration during development, enable fast mode in your config:

```yaml
runtime:
  fast_mode: true # Reduces bootstrap samples (2-3s vs 5-7s)
```

That's it. You now have a complete audit report with model performance, explanations, and fairness metrics.

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
./scripts/check-determinism-quick.sh
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
  - **`src/glassalpha/data/configs/`** - Example audit configurations (packaged with install)
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

**Byte-identical reproducibility.** Same audit config → byte-identical HTML reports every time (on same platform+Python). PDFs are suitable for human review but may have minor layout variations due to rendering engine limitations. SHA256 hashes in manifests for verification.

**Statistical rigor.** Not just point estimates—95% confidence intervals on fairness and calibration metrics with bootstrap resampling.

**Note**: Evidence pack export (E3) is available in v0.2.1 via `glassalpha export-evidence-pack`. Policy-as-code gates (E1) planned for v0.3.0. Current version supports shift testing gates for CI/CD.

## Core Capabilities

### Supported Models

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
- **Reproducibility**: Deterministic execution, version pinning, byte-identical PDFs (same platform+Python)
- **CI/CD Gates**: Shift testing with `--fail-on-degradation` blocks deployments on metric degradation

- XGBoost, LightGBM, Logistic Regression (more coming)
- **Everything runs locally** - your data never leaves your machine

All Apache 2.0 licensed.

### Quick Features

- **30-second setup**: Interactive `glassalpha quickstart` wizard
- **Smart defaults**: Auto-detects config files, infers output paths
- **Built-in datasets**: German Credit and Adult Income for quick testing
- **Self-diagnosable errors**: Clear What/Why/Fix error messages
- **Automation support**: `--json-errors` flag for CI/CD pipelines

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
./scripts/check-determinism-quick.sh

# Full CI mirror (5 minutes) - comprehensive determinism suite
./scripts/test_determinism_local.sh
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
