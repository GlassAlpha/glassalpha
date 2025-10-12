# GlassAlpha

**Ever tried explaining your ML model to a regulator?**

GlassAlpha is an ([open source](https://glassalpha.com/reference/trust-deployment/#licensing-dependencies)) ML compliance toolkit that makes tabular models **transparent, auditable, and regulator-ready**.

Generate deterministic PDF audit reports with statistical confidence intervals, fairness analysis, and policy-as-code compliance gates. No dashboards. No black boxes. Just byte-stable evidence packs you can submit to regulators.

_Note: GlassAlpha is currently pre-alpha while I'm actively developing. The audits work and tests pass, so feel free to try it out—feedback welcome! First stable release coming soon._

## Installation

### Option 1: Install from PyPI (easiest)

```bash
# Install with pipx (recommended for CLI tools)
pipx install glassalpha

# Or with pip
pip install glassalpha
```

**Base installation** includes LogisticRegression model with coefficient-based explanations (fast, zero extra dependencies).

**For advanced models** (XGBoost/LightGBM with SHAP):

```bash
pip install 'glassalpha[explain]'  # Adds 5-10 minutes to initial setup
```

### Option 2: Install from source (for development)

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha
pip install -e ".[all]"  # Install with all optional features
```

Create a configuration (interactive wizard)

```bash
glassalpha init
```

Generate an audit report

```bash
# Lightning-fast development mode (2-3 seconds)
glassalpha audit --fast

# Or full audit with all features (5-7 seconds)
glassalpha audit
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

**Policy-as-code, not dashboards.** Define compliance rules in YAML, get PASS/FAIL gates automatically.

```yaml
# policy.yaml
immutables: [age, race, gender] # Can't change
monotone:
  debt_to_income: increase_only # Fairness constraint
degradation_threshold: 0.05 # Max 5pp metric drop under demographic shifts
```

**Byte-identical reproducibility.** Same audit config → same PDF, every time. SHA256-verified evidence packs for regulatory submission.

**Statistical rigor.** Not just point estimates—95% confidence intervals on everything (fairness, calibration, performance).

## Core Capabilities

### Supported Models

### Compliance & Fairness

- **Group Fairness** (E5): Demographic parity, TPR/FPR, with [statistical confidence intervals](site/docs/reference/fairness-metrics.md)
- **Intersectional Fairness** (E5.1): Hidden bias detection in demographic combinations (e.g., race×gender)
- **Individual Fairness** (E11): [Consistency score](site/docs/reference/fairness-metrics.md#individual-fairness)—similar applicants get similar decisions
- **[Dataset Bias Audit](site/docs/guides/dataset-bias.md)** (E12): Proxy feature detection, distribution drift, sampling bias power
- **Statistical Confidence** (E10): Bootstrap CIs for all fairness metrics, sample size warnings

### Explainability & Outcomes

- **TreeSHAP Explanations**: Feature importance with individual prediction breakdowns
- **Reason Codes** (E2): ECOA-compliant adverse action notices
- **Actionable Recourse** (E2.5): "Change X to improve outcome" recommendations with policy constraints

### Robustness & Stability

- **[Calibration Analysis](site/docs/reference/calibration.md)** (E10+): ECE with confidence intervals, bin-wise calibration curves
- **[Adversarial Perturbation](site/docs/reference/robustness.md)** (E6+): ε-perturbation sweeps, robustness score
- **[Demographic Shift Testing](site/docs/guides/shift-testing.md)** (E6.5): Simulate population changes, detect degradation before deployment

### Regulatory Compliance

- **[SR 11-7 Mapping](site/docs/compliance/sr-11-7-mapping.md)**: Complete Federal Reserve guidance coverage (banking)
- **Evidence Packs**: SHA256-verified bundles (PDF + manifest + gates + policy)
- **Reproducibility**: Deterministic execution, version pinning, byte-identical PDFs
- **CI/CD Gates**: Exit code 1 if compliance fails, JSON output for automation

- XGBoost, LightGBM, Logistic Regression (more coming)
- **Everything runs locally** - your data never leaves your machine

All Apache 2.0 licensed.

### Quick Features

- **30-second setup**: Interactive `glassalpha init` wizard
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
