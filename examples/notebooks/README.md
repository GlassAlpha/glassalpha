# GlassAlpha Example Notebooks

Interactive Jupyter notebooks demonstrating GlassAlpha's audit capabilities.

> **Note**: All examples use binary classification (2 classes). Multi-class models not yet supported (planned for v0.3.0).

## Quick Start Notebooks

### 1. Quickstart (8 minutes)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GlassAlpha/glassalpha/blob/main/examples/notebooks/quickstart_colab.ipynb)

**File**: `quickstart_colab.ipynb`

Generate your first ML audit in under 8 minutes:

- Train a credit scoring model
- Run comprehensive audit with `from_model()` API
- View inline results
- Export professional PDF

**Perfect for**: First-time users, quick demos

### 2. German Credit Walkthrough (30 minutes)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GlassAlpha/glassalpha/blob/main/examples/notebooks/german_credit_walkthrough.ipynb)

**File**: `german_credit_walkthrough.ipynb`

Complete audit walkthrough with detailed explanations:

- Data exploration and preprocessing
- Model training and evaluation
- Fairness analysis with statistical testing
- Calibration testing
- Regulatory compliance mapping

**Perfect for**: Learning all features, understanding audit components

### 3. Custom Data Template (15 minutes)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GlassAlpha/glassalpha/blob/main/examples/notebooks/custom_data_template.ipynb)

**File**: `custom_data_template.ipynb`

Template for auditing your own models:

- Upload your CSV data
- Configure protected attributes
- Run audit with custom settings
- Interpret results for your use case

**Perfect for**: Auditing your own models, production workflows

## Running Locally

### Option 1: Jupyter Lab

```bash
cd examples/notebooks
jupyter lab
```

### Option 2: VS Code

Open notebooks directly in VS Code with the Jupyter extension.

### Option 3: Colab

Click the "Open in Colab" badges above for zero-setup cloud execution.

## Requirements

Install GlassAlpha with notebook dependencies:

```bash
pip install glassalpha[explain]
```

Or with all features:

```bash
pip install glassalpha[all]
```

## Notebook Structure

All notebooks follow this pattern:

1. **Environment Setup** - Install dependencies, set seeds
2. **Data Loading** - Load or upload data
3. **Model Training** - Train or load model
4. **Audit Generation** - Run `from_model()` API
5. **Results Exploration** - Interactive analysis
6. **Export** - Generate PDF reports

## Related Documentation

- [Quick Start Guide](https://glassalpha.com/getting-started/quickstart/)
- [Custom Data Guide](https://glassalpha.com/getting-started/custom-data/)
- [Data Scientist Workflow](https://glassalpha.com/guides/data-scientist-workflow/)
- [API Reference](https://glassalpha.com/reference/api/)

## Support

- **Documentation**: [glassalpha.com](https://glassalpha.com)
- **GitHub Issues**: [Report bugs or request features](https://github.com/GlassAlpha/glassalpha/issues)
- **Discussions**: [Ask questions](https://github.com/GlassAlpha/glassalpha/discussions)
