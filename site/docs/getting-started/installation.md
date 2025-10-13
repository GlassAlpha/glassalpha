# Installation guide

Complete installation instructions for GlassAlpha on different platforms and environments.

## System requirements

### Minimum requirements

- **Python**: 3.11 or higher
- **Memory**: 2GB RAM available
- **Storage**: 1GB disk space for installation and temporary files
- **OS**: macOS 10.15+, Linux (Ubuntu 20.04+), Windows 10+ (WSL2 recommended)

### Recommended environment

- **Python**: 3.11+
- **Memory**: 8GB+ RAM for large datasets
- **Storage**: SSD for better performance
- **CPU**: Multi-core processor for parallel processing

### Supported platforms

- **macOS**: Intel and Apple Silicon (M1/M2/M3)
- **Linux**: Ubuntu 20.04+, CentOS 8+, RHEL 8+, and most modern distributions
- **Windows**: 10/11 (native support, WSL2 recommended for best experience)

## Installation methods

### Quick install (recommended)

The fastest way to get started with GlassAlpha:

```bash
# Install with all features
pip install "glassalpha[all]"

# Verify installation
glassalpha --version
glassalpha doctor
```

**For isolated environments:** Create a virtual environment first:

```bash
python3 -m venv ~/.glassalpha-env
source ~/.glassalpha-env/bin/activate  # On Windows: .glassalpha-env\Scripts\activate
pip install "glassalpha[all]"
```

### Install from source

For development or latest features:

```bash
# Clone repository
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha

# Install with all features
pip install -e ".[all]"

# Verify installation
glassalpha --version
glassalpha doctor
```

### Feature matrix

Choose your installation based on your needs:

| Goal                          | Command                             | What's Included                                            | Use Case                                          |
| ----------------------------- | ----------------------------------- | ---------------------------------------------------------- | ------------------------------------------------- |
| **Minimal quickstart (HTML)** | `pip install glassalpha`            | Core functionality, LogisticRegression model, HTML reports | Quick audits, development, lightweight deployment |
| **Generate PDFs**             | `pip install "glassalpha[docs]"`    | PDF report generation (`jinja2`, `weasyprint`)             | Professional reports, regulatory submissions      |
| **SHAP + tree models**        | `pip install "glassalpha[explain]"` | SHAP + XGBoost + LightGBM models                           | Production ML systems, complex models             |
| **Visualization**             | `pip install "glassalpha[viz]"`     | Matplotlib + Seaborn for enhanced plots                    | Better visualizations in reports                  |
| **Full installation**         | `pip install "glassalpha[all]"`     | All optional features and models                           | Complete toolkit, maximum compatibility           |
| **Development**               | `pip install "glassalpha[dev,all]"` | Testing, linting, documentation tools + all features       | Contributors, CI/CD environments                  |

**Tip:** Run `glassalpha doctor` after installation to check your environment and see what features are available.

**Why HTML by default?** Lighter, portable, works anywhere. PDF available via optional `docs` extra for professional reports.

### Quick start (recommended for new users)

Install GlassAlpha with just the essential dependencies for immediate use:

```bash
# Clone and setup
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha

# Python 3.11, 3.12, or 3.13 supported
python3 --version   # should show 3.11.x, 3.12.x, or 3.13.x

# (Recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core package (includes LogisticRegression baseline)
python -m pip install --upgrade pip
pip install -e .

# Verify installation works immediately
glassalpha validate --config quickstart.yaml
```

This installation provides:

- ✅ LogisticRegression model (always available)
- ✅ Permutation explainer (always available)
- ✅ Full audit pipeline functionality
- ✅ PDF report generation

### Advanced installation (for SHAP + tree models)

For XGBoost, LightGBM, and SHAP explainers:

```bash
# Install with SHAP + tree models (includes XGBoost and LightGBM)
pip install -e ".[explain]"

# Check what's available
glassalpha doctor
```

### Development installation

For contributors or those who need development tools:

**Quick setup (recommended):**

```bash
# One-command setup: installs everything, sets up hooks, runs doctor
make dev-setup
```

**Manual setup:**

```bash
# Install everything including dev tools and all features
pip install -e ".[dev,all]"

# Verify installation
glassalpha doctor
```

Development dependencies include:

- **Testing**: pytest, pytest-cov
- **Code Quality**: ruff, mypy, black, pre-commit
- **All Features**: SHAP, XGBoost, LightGBM, matplotlib, WeasyPrint

## Model selection and fallbacks

GlassAlpha uses explicit dispatch to load ML models based on your configuration.

### Available models

| Model                  | Installation                         | Description                      | Use Case                            |
| ---------------------- | ------------------------------------ | -------------------------------- | ----------------------------------- |
| **LogisticRegression** | `pip install glassalpha`             | Baseline model, always available | Quick audits, regulatory compliance |
| **XGBoost**            | `pip install 'glassalpha[xgboost]'`  | Advanced tree model              | Production ML systems               |
| **LightGBM**           | `pip install 'glassalpha[lightgbm]'` | Microsoft's tree model           | Large datasets, performance         |

**Need help choosing?** See the [Model Selection Guide](../reference/model-selection.md) for performance benchmarks and detailed comparisons.

### Automatic fallbacks

If you request a model that isn't installed, GlassAlpha automatically falls back to LogisticRegression:

```bash
# This works even without XGBoost installed (use quickstart-generated config or example config)
glassalpha audit --config src/glassalpha/data/configs/german_credit_simple.yaml --output audit.pdf
# → Falls back to LogisticRegression with clear message
```

To disable fallbacks and require specific models:

```yaml
# In your config file
model:
  type: xgboost
  allow_fallback: false # Strict mode - fails if XGBoost unavailable
```

### Installation recommendations

**For new users:**

```bash
pip install glassalpha  # Start with LogisticRegression
```

**For production use:**

```bash
pip install 'glassalpha[all]'  # All features
```

**For specific needs:**

```bash
pip install 'glassalpha[explain]'   # SHAP + XGBoost + LightGBM
pip install 'glassalpha[viz]'       # Matplotlib + Seaborn
pip install 'glassalpha[docs]'      # PDF generation
```

## Platform-specific installation

### macOS installation

**Prerequisites:**

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11+ (if not already available)
brew install python@3.11

# Install OpenMP for XGBoost support
brew install libomp
```

**Installation:**

```bash
# Clone and setup
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha

# Python 3.11, 3.12, or 3.13 supported
python3 --version   # should show 3.11.x, 3.12.x, or 3.13.x

# (Recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install in editable mode
python -m pip install --upgrade pip
pip install -e ".[dev]"

# Verify installation
glassalpha --help
```

**Common macOS Issues:**

- **XGBoost libomp error**: Install with `brew install libomp`
- **Python version conflicts**: Use `python3.11` explicitly
- **Permission issues**: Avoid `sudo pip install`, use virtual environments

### Linux installation (Ubuntu/Debian)

**Prerequisites:**

```bash
# Update package list
sudo apt update

# Install Python 3.11 and development tools
sudo apt install python3.11 python3.11-venv python3.11-dev
sudo apt install build-essential git

# Install system libraries for PDF generation
sudo apt install libpango1.0-dev libcairo2-dev libgtk-3-dev
```

**Installation:**

```bash
# Clone and setup
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha

# Python 3.11, 3.12, or 3.13 supported
python3 --version   # should show 3.11.x, 3.12.x, or 3.13.x

# (Recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install in editable mode
python -m pip install --upgrade pip
pip install -e ".[dev]"

# Verify installation
glassalpha --help
```

### Linux installation (CentOS/RHEL)

**Prerequisites:**

```bash
# Install EPEL repository
sudo yum install epel-release

# Install Python 3.11 and development tools
sudo yum install python3.11 python3.11-devel python3.11-pip
sudo yum install gcc gcc-c++ make git

# Install system libraries
sudo yum install pango-devel cairo-devel gtk3-devel
```

**Installation:**

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha

python3.11 -m venv glassalpha-env
source glassalpha-env/bin/activate
pip install --upgrade pip
pip install -e .
```

### Windows installation

**Option 1: Windows Subsystem for Linux (Recommended)**

```bash
# Install WSL2 and Ubuntu
wsl --install Ubuntu

# In WSL2, follow Ubuntu installation instructions
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha
python3.11 -m venv glassalpha-env
source glassalpha-env/bin/activate
pip install -e .
```

**Option 2: Native Windows**

```powershell
# Install Python 3.11+ from python.org
# Install Git from git-scm.com

# Clone and install
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha

python -m venv glassalpha-env
glassalpha-env\Scripts\activate
pip install --upgrade pip
pip install -e .
```

**Windows notes:**

- WSL2 provides better compatibility and performance
- Native Windows may have PDF generation limitations
- Use PowerShell or Command Prompt for native installation

## Verification and testing

### Basic verification

After installation, verify GlassAlpha is working correctly:

```bash
# Check version
glassalpha --version

# Verify CLI is working
glassalpha --help

# List available components
glassalpha list
```

Expected output:

```
GlassAlpha version 0.2.0

Available Components
==================
MODELS:
  - xgboost
  - lightgbm
  - logistic_regression
  - sklearn_generic
  - passthrough

EXPLAINERS:
  - treeshap
  - kernelshap
  - noop

METRICS:
  - accuracy
  - precision
  - recall
  - f1
  - auc_roc
  - demographic_parity
  - equal_opportunity
  ...
```

### Python API verification

Test the Python API:

```python
# Verify imports work
from glassalpha.models import load_model
from glassalpha.explain import select_explainer
from glassalpha.config import AuditConfig

# Check explicit dispatch
print("Available models: xgboost, lightgbm, sklearn")
print("Available explainers: treeshap, kernelshap, permutation")

# Verify configuration loading
print("Configuration system working!")
```

### End-to-end test

Run a complete audit to verify all components:

```bash
# Quick smoke test with German Credit dataset
glassalpha audit \
  --config german_credit_simple.yaml \
  --output test_audit.html \
  --dry-run

# If dry-run passes, run actual audit (add [docs] for PDF)
glassalpha audit \
  --config german_credit_simple.yaml \
  --output test_audit.html
```

Successful execution should:

- Complete in under 30 seconds
- Generate a PDF report (~500KB+)
- Show no error messages
- Display audit summary statistics

## Dependencies

### Core dependencies (always installed)

GlassAlpha installs these essential dependencies automatically:

**Data Processing:**

- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning utilities (includes LogisticRegression)

**Configuration & CLI:**

- `pydantic>=2.0.0` - Configuration validation
- `typer>=0.9.0` - Command-line interface
- `pyyaml>=6.0` - YAML configuration files

**Visualization & Reporting:**

- `matplotlib>=3.7.0` - Plotting and visualization
- `seaborn>=0.12.0` - Statistical visualization
- `weasyprint>=59.0` - PDF generation

### Optional dependencies

**Machine Learning Models** (install with specific extras):

- `xgboost>=1.7.0` - Gradient boosting framework (`[xgboost]` extra)
- `lightgbm>=3.3.0` - Microsoft's gradient boosting (`[lightgbm]` extra)
- `shap>=0.42.0` - Model explanations (`[xgboost]` and `[tabular]` extras)

**Development Tools** (with `[dev]` install):

- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Test coverage
- `ruff>=0.0.280` - Code linting and formatting
- `mypy>=1.5.0` - Type checking
- `pre-commit>=3.3.0` - Git hooks

### Version compatibility

GlassAlpha is tested with:

- **Python**: 3.11, 3.12, 3.13
- **scikit-learn**: 1.3.x, 1.4.x
- **XGBoost**: 1.7.x, 2.0.x
- **LightGBM**: 3.3.x, 4.0.x
- **SHAP**: 0.42.x, 0.43.x

## Environment management

### Using virtual environments

**Why Virtual Environments?**

- Isolate dependencies from system Python
- Avoid version conflicts
- Enable project-specific configurations
- Facilitate deployment and distribution

**Creating Virtual Environments:**

```bash
# Using venv (built-in)
python -m venv glassalpha-env
source glassalpha-env/bin/activate  # Linux/macOS
# glassalpha-env\Scripts\activate   # Windows

# Using conda
conda create -n glassalpha python=3.11
conda activate glassalpha

# Using poetry
poetry init
poetry add glassalpha
poetry shell
```

### Environment variables

**Configuration:**

- `GLASSALPHA_CONFIG_DIR` - Default configuration directory
- `GLASSALPHA_DATA_DIR` - Default data directory
- `GLASSALPHA_LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

**Example:**

```bash
export GLASSALPHA_LOG_LEVEL=DEBUG
export GLASSALPHA_CONFIG_DIR=~/.config/glassalpha
glassalpha audit --config my_config.yaml --output audit.pdf
```

## Troubleshooting installation

### Common installation issues

**Python Version Issues:**

```bash
# Check Python version
python --version

# If version is too old, install Python 3.11+
# Use python3.11 explicitly if multiple versions installed
python3.11 -m venv glassalpha-env
```

**Permission Errors:**

```bash
# Use virtual environment instead of system-wide installation
python -m venv glassalpha-env
source glassalpha-env/bin/activate
pip install -e .

# Or install for user only (not recommended)
pip install --user -e .
```

**Dependency Conflicts:**

```bash
# Create clean environment
rm -rf glassalpha-env
python -m venv glassalpha-env
source glassalpha-env/bin/activate
pip install --upgrade pip
pip install -e .
```

**XGBoost Issues (macOS):**

```bash
# Install OpenMP library
brew install libomp

# Reinstall XGBoost
pip uninstall xgboost
pip install xgboost

# Verify installation
python -c "import xgboost; print('XGBoost version:', xgboost.__version__)"
```

**PDF Generation Issues (Linux):**

```bash
# Install system libraries
sudo apt install libpango1.0-dev libcairo2-dev libgtk-3-dev  # Ubuntu/Debian
sudo yum install pango-devel cairo-devel gtk3-devel          # CentOS/RHEL

# Reinstall WeasyPrint
pip uninstall weasyprint
pip install weasyprint
```

### Memory and performance issues

**Insufficient Memory:**

```bash
# Monitor memory usage during installation
pip install -e . --verbose

# If installation fails due to memory, increase swap space (Linux)
sudo swapon --show
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Slow Installation:**

```bash
# Use faster package index
pip install -e . --index-url https://pypi.org/simple/

# Install binary wheels when available
pip install -e . --only-binary=all --prefer-binary
```

### Verification failures

**CLI Command Not Found:**

```bash
# Verify installation
pip list | grep glassalpha

# Check PATH contains virtual environment
which glassalpha
echo $PATH

# Reinstall if necessary
pip install --force-reinstall -e .
```

**Import Errors:**

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Verify installation location
pip show glassalpha

# Test imports individually
python -c "import pandas; print('pandas OK')"
python -c "import xgboost; print('xgboost OK')"
python -c "import glassalpha; print('glassalpha OK')"
```

## Docker installation (alternative)

For containerized environments:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libpango1.0-dev \
    libcairo2-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and install GlassAlpha
RUN git clone https://github.com/GlassAlpha/glassalpha.git
WORKDIR /glassalpha
RUN pip install -e .

# Set entry point
ENTRYPOINT ["glassalpha"]
```

**Usage:**

```bash
# Build image
docker build -t glassalpha .

# Run audit
docker run -v $(pwd):/data glassalpha \
  audit --config /data/config.yaml --output /data/audit.pdf
```

## Understanding configuration paths

Throughout the documentation, you'll see two types of config path references:

**Repository examples** (when running from cloned repo):

```bash
glassalpha audit --config src/glassalpha/data/configs/german_credit_simple.yaml --output audit.html
```

**Custom configs** (when using your own):

```bash
glassalpha audit --config my_config.yaml --output audit.html
```

**Tip**: After cloning the repository, example configs are in the `configs/` directory at the repo root. For your own projects, create configs anywhere and reference them with relative or absolute paths.

## Next steps

After successful installation:

1. **Try the Quick Start** - [Generate your first audit](quickstart.md)
2. **Use Your Own Data** - [Custom data tutorial](custom-data.md) with step-by-step guidance
3. **Browse Example Datasets** - [Freely available data sources](data-sources.md) for testing
4. **Explore Examples** - [German Credit tutorial](../examples/german-credit-audit.md)
5. **Read Configuration Guide** - [Understand all options](configuration.md)
6. **Choose Your Model** - [Model selection guide](../reference/model-selection.md) with benchmarks
7. **Trust & Deployment** - [Architecture, licensing, security, and compliance](../reference/trust-deployment.md) for regulated environments
8. **Join the Community** - [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)

## Getting help

If you encounter installation issues not covered here:

1. **Check the [Troubleshooting Guide](../reference/troubleshooting.md)**
2. **Search [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues)**
3. **Ask in [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)**
4. **Review the [FAQ](../reference/faq.md)**

For enterprise support and custom installation assistance, contact: enterprise@glassalpha.com

This installation guide ensures you have a working GlassAlpha environment ready for professional ML auditing and compliance analysis.
