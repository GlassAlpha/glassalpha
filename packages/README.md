# GlassAlpha

**AI Compliance Toolkit** - transparent, auditable, regulator-ready ML audits.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/glassalpha.svg)](https://pypi.org/project/glassalpha/)

## Installation

### For CLI Users (Recommended)

Install with pipx for isolated environment and global command availability:

```bash
# Install pipx (if not already installed)
python3 -m pip install --user pipx
python3 -m pipx ensurepath

# Install GlassAlpha
pipx install glassalpha

# With all optional features
pipx install "glassalpha[all]"
```

### For Python Projects

Install with pip in a virtual environment (recommended to avoid system package conflicts):

```bash
# Create virtual environment (required on macOS/Linux with system Python)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Basic install (includes scikit-learn models and HTML reports)
pip install glassalpha

# With PDF generation support
pip install "glassalpha[pdf]"

# With SHAP explainers
pip install "glassalpha[shap]"

# With tree model support (XGBoost, LightGBM)
pip install "glassalpha[xgboost,lightgbm]"

# Everything (all optional features)
pip install "glassalpha[all]"
```

### For Development

Clone and install in editable mode:

```bash
git clone https://github.com/glassalpha/glassalpha.git
cd glassalpha/packages
pip install -e ".[dev,all]"
```

**Note**: Docker support is planned for a future release. For containerized environments, please use the development installation method above for now.

## Quick Start

```bash
# Generate first audit in 2-3 seconds (with --fast)
glassalpha audit --config quickstart.yaml --output audit.html --fast

# Validate configuration
glassalpha validate config.yaml

# Generate reason codes for adverse action notices
glassalpha reasons --model model.pkl --data test.csv --instance 123 --output reasons.json

# Generate recourse recommendations
glassalpha recourse --model model.pkl --data test.csv --instance 123 --output recourse.json
```

## Features

### Core Capabilities (Open Source)

- **Explainability**: TreeSHAP, KernelSHAP, permutation importance
- **Recourse**: Feasible counterfactuals with policy constraints
- **Fairness**: Demographic parity, equalized odds, calibration by group
- **Reason Codes**: ECOA-compliant adverse action notices
- **Audit Reports**: Deterministic HTML/PDF with full lineage
- **Preprocessing Verification**: Hash-based artifact integrity checks

### What Makes It Different

- **Byte-identical reproducibility**: Same config → same SHA256
- **Policy-as-code**: Immutables, monotonicity, bounds, costs
- **Offline-first**: No network dependencies, air-gap compatible
- **Regulator-ready**: Manifest captures seeds, versions, hashes

## Documentation

- [Installation Guide](https://glassalpha.com/getting-started/installation/)
- [Quick Start](https://glassalpha.com/getting-started/quickstart/)
- [CLI Reference](https://glassalpha.com/reference/cli/)
- [Configuration](https://glassalpha.com/getting-started/configuration/)
- [Examples](https://glassalpha.com/examples/)

## Requirements

- Python 3.11+
- scikit-learn (included in base install)
- Optional: SHAP, XGBoost, LightGBM, WeasyPrint

## Development

```bash
# Clone repository
git clone https://github.com/glassalpha/glassalpha.git
cd glassalpha/packages

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check .

# Format code
black .
```

## License

Apache License 2.0. See [LICENSE](../LICENSE) for details.

## Security

See [SECURITY.md](../SECURITY.md) for security policy and reporting vulnerabilities.

## Trademark

"GlassAlpha" is a trademark. See [TRADEMARK.md](../TRADEMARK.md) for usage guidelines.

## Support

- Documentation: https://glassalpha.com
- Issues: https://github.com/glassalpha/glassalpha/issues
- Discussions: https://github.com/glassalpha/glassalpha/discussions

---

**Note**: This is the OSS core. Enterprise features (dashboards, SSO, regulator-specific templates) are available separately.
