# GlassAlpha

**Ever tried explaining your ML model to a regulator?**

GlassAlpha is an ([open source](https://glassalpha.com/reference/trust-deployment/#licensing-dependencies)) ML compliance toolkit that makes tabular models **transparent, auditable, and regulator-ready**.

Generate deterministic PDF audit reports with statistical confidence intervals, fairness analysis, and policy-as-code compliance gates. No dashboards. No black boxes. Just byte-stable evidence packs you can submit to regulators.

_Note: GlassAlpha is currently pre-alpha while I'm actively developing. The audits work and tests pass, so feel free to try it out—feedback welcome! First stable release coming soon._

## Get started

### Run your first audit in 30 seconds

Clone and install

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha/packages
pip install -e ".[all]"  # Install with all optional features
```

Create a configuration (interactive wizard)

```bash
glassalpha init
```

Generate an audit report

```bash
glassalpha audit  # Uses smart defaults - no flags needed!
```

That's it. You now have a complete audit report with model performance, SHAP explanations, and fairness metrics.

**Tip:** Run `glassalpha doctor` anytime to check what features are available and see installation options.

**More details:** See the [full installation guide](packages/README.md#installation) and [German Credit tutorial](https://glassalpha.com/examples/german-credit-audit/) to see what's in the report.

## Structure

- **`packages/`** - The actual Python package ([dev docs here](packages/README.md))
- **`site/`** - User documentation and tutorials. The docs site is at [glassalpha.com](https://glassalpha.com/)
- **`configs/`** - Example audit configs you can copy and modify

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

### Compliance & Fairness

- **Group Fairness** (E5): Demographic parity, TPR/FPR, with statistical confidence intervals
- **Intersectional Fairness** (E5.1): Hidden bias detection in demographic combinations (e.g., race×gender)
- **Individual Fairness** (E11): Consistency score—similar applicants get similar decisions
- **Dataset Bias Audit** (E12): Proxy feature detection, distribution drift, sampling bias power
- **Statistical Confidence** (E10): Bootstrap CIs for all fairness metrics, sample size warnings

### Explainability & Outcomes

- **TreeSHAP Explanations**: Feature importance with individual prediction breakdowns
- **Reason Codes** (E2): ECOA-compliant adverse action notices
- **Actionable Recourse** (E2.5): "Change X to improve outcome" recommendations with policy constraints

### Robustness & Stability

- **Calibration Analysis** (E10+): ECE with confidence intervals, bin-wise calibration curves
- **Adversarial Perturbation** (E6+): ε-perturbation sweeps, robustness score
- **Demographic Shift Testing** (E6.5): Simulate population changes, detect degradation before deployment

### Regulatory Compliance

- **SR 11-7 Mapping**: Complete Federal Reserve guidance coverage (banking)
- **Evidence Packs**: SHA256-verified bundles (PDF + manifest + gates + policy)
- **Reproducibility**: Deterministic execution, version pinning, byte-identical PDFs
- **CI/CD Gates**: Exit code 1 if compliance fails, JSON output for automation

### Supported Models

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
- **[Developer guide](packages/README.md)** - Architecture deep-dive and contribution guide
- **[German credit tutorial](https://glassalpha.com/examples/german-credit-audit/)** - Step-by-step walkthrough with a real dataset
- **[About GlassAlpha](https://glassalpha.com/about/)** - Who, what & why.

## Contributing

I'm a one man band, so quality contributions are welcome.

Found a bug? Want to add a model type? PRs welcome! Check the [contributing guide](https://glassalpha.com/reference/contributing/) for dev setup.

The architecture is designed to be extensible. Adding new models, explainers, or metrics shouldn't require touching core code.

## License

The core library is Apache 2.0. See [LICENSE](LICENSE) for the legal stuff.

Enterprise features/support may be added separately if there's demand for more advanced/custom functionality, but the core will always remain open and free. The name "GlassAlpha" is trademarked to keep things unambiguous. Details in [TRADEMARK.md](TRADEMARK.md).

For dependency licenses and third-party components, check the [detailed licensing info](packages/README.md#license--dependencies).
