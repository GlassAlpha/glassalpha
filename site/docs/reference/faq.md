# Frequently asked questions

Common questions about GlassAlpha capabilities, usage, and integration.

## General questions

### What is GlassAlpha?

GlassAlpha is an open-source AI compliance toolkit that generates comprehensive audit reports for machine learning models. It provides:

- **Automated bias detection** and fairness analysis
- **Model explanations** using SHAP and other interpretability methods
- **Reason codes** for ECOA-compliant adverse action notices
- **Professional PDF reports** suitable for regulatory review
- **Complete reproducibility** with audit trails and manifests
- **Regulatory compliance** support for GDPR, ECOA, FCRA, and other frameworks

### Who should use GlassAlpha?

GlassAlpha is designed for:

- **Data scientists** who need to audit ML models for bias and fairness
- **Compliance teams** ensuring regulatory adherence for algorithmic decisions
- **Risk management** professionals assessing model risks
- **Legal teams** preparing for regulatory review or litigation
- **Academic researchers** studying algorithmic fairness and interpretability

### How does GlassAlpha ensure audit quality?

GlassAlpha maintains audit quality through:

- **Deterministic execution** with fixed random seeds for reproducible results
- **Complete audit trails** tracking all decisions and configurations
- **Statistical rigor** with confidence intervals and significance testing
- **Professional reporting** with publication-quality visualizations
- **Regulatory alignment** with established compliance frameworks

For detailed information about system design and quality assurance, see the [Trust & Deployment Guide](../reference/trust-deployment.md).

## Installation & setup

### What are the system requirements?

**Minimum Requirements:**

- Python 3.11 or higher
- 2GB available RAM
- 1GB disk space for installation and temporary files

**Recommended:**

- Python 3.11+
- 8GB+ RAM for large datasets
- SSD storage for better performance
- Multi-core CPU for parallel processing

### Which operating systems are supported?

GlassAlpha is tested and supported on:

- **macOS** 10.15+ (Intel and Apple Silicon)
- **Linux** (Ubuntu 20.04+, CentOS 8+, and most modern distributions)
- **Windows** 10/11 (via WSL2 recommended)

### How do I install GlassAlpha?

**Standard Installation:**

Clone and setup:

```bash
git clone https://github.com/GlassAlpha/glassalpha
cd glassalpha
```

Python 3.11, 3.12, or 3.13 supported:

```bash
python3 --version   # should show 3.11.x, 3.12.x, or 3.13.x
```

Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip and install in editable mode:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Verify installation:

```bash
glassalpha --help
```

See the [Quick Start Guide](../getting-started/quickstart.md) for detailed instructions.

### What dependencies does GlassAlpha require?

**Core Dependencies (base install):**

- pandas, numpy (data processing)
- scikit-learn (machine learning and LogisticRegression)
- Matplotlib, Seaborn (visualizations)
- Jinja2, WeasyPrint (HTML/PDF generation)

**Optional Dependencies (explain install):**

- XGBoost, LightGBM (gradient boosting models)
- SHAP (TreeSHAP and KernelSHAP explanations)

Install options:

```bash
pip install -e .              # Base install (LogisticRegression only)
pip install -e ".[explain]"   # Adds SHAP, XGBoost, LightGBM
pip install -e ".[dev]"       # Development tools
```

## Usage & configuration

### How do I generate my first audit?

1. **Follow the [Quick Start guide](../getting-started/quickstart.md) for a 5-minute introduction**

2. **Use the German Credit example:**

```bash
glassalpha audit \
  --config configs/german_credit_simple.yaml \
  --output my_audit.pdf
```

3. **For your own data:**
   - See [Using Custom Data](../getting-started/custom-data.md) for a complete tutorial
   - Use our configuration template in `packages/configs/custom_template.yaml` with detailed comments

```yaml
audit_profile: tabular_compliance
reproducibility:
  random_seed: 42
data:
  path: your_data.csv
  target_column: outcome
model:
  type: xgboost
```

4. **Run the audit:**

```bash
glassalpha audit --config your_config.yaml --output audit.pdf
```

**Want more examples?** Browse our [freely available data sources](../getting-started/data-sources.md) for curated public datasets.

### What file formats are supported for data?

GlassAlpha supports:

- **CSV** (most common)
- **Parquet** (recommended for large datasets)
- **Feather** (fast binary format)
- **Pickle** (Python objects)

The format is automatically detected from the file extension.

### How do I handle missing data?

Configure preprocessing in your audit configuration:

```yaml
preprocessing:
  handle_missing: true
  missing_strategy:
    median # For numeric: median, mean, mode
    # For categorical: mode, drop
```

GlassAlpha automatically handles most missing value scenarios.

### Can I use my own pre-trained model?

Yes, specify the model path in your configuration:

```yaml
model:
  type: xgboost
  path: models/my_trained_model.pkl
```

GlassAlpha supports models saved with:

- **Pickle** (most scikit-learn models)
- **Joblib** (scikit-learn and XGBoost)
- **Native formats** (XGBoost `.model`, LightGBM `.txt`)

## Model support

### Which machine learning models are supported?

**Currently Supported:**

- **XGBoost** - Gradient boosting with TreeSHAP explanations
- **LightGBM** - Microsoft's gradient boosting framework
- **Logistic Regression** - Linear classification models
- **Generic Scikit-learn** - Most scikit-learn classifiers

**Explanation Support:**

- **TreeSHAP** - Exact SHAP values for tree-based models (XGBoost, LightGBM)
- **KernelSHAP** - Model-agnostic explanations for any model type

**Need help choosing?**

- See the [Model Selection Guide](model-selection.md) for performance benchmarks and use case recommendations
- See the [Explainer Deep Dive](explainers.md) for guidance on explanation methods

### How do I add support for a new model type?

Implement the `ModelInterface` protocol:

```python
from glassalpha.core import ModelRegistry

@ModelRegistry.register("my_model")
class MyModel:
    capabilities = {"supports_shap": True}
    version = "1.0.0"

    def predict(self, X):
        # Implementation
        pass

    def predict_proba(self, X):
        # Implementation
        pass
```

See the [Trust & Deployment Guide](../reference/trust-deployment.md) for technical details.

### Can I use deep learning models?

Deep learning models can be supported through the generic model interface, but:

- **TreeSHAP won't work** (only for tree models)
- **Use KernelSHAP** for model-agnostic explanations
- **Performance may be slower** for explanation generation
- **Consider gradient-based explanations** for better deep learning support

### What about time series or text models?

GlassAlpha currently focuses on **tabular data** for classification tasks. Additional data modalities may be supported in potential future versions based on user demand and community contributions:

- **Time Series**: Under consideration for potential future releases
- **Text/NLP**: Under consideration for potential future releases
- **Computer Vision**: Under consideration for potential future releases

## Compliance & regulatory

### Which regulations does GlassAlpha address?

**Directly Supported:**

- **GDPR** (EU) - Right to explanation, automated decision-making
- **ECOA** (US) - Fair lending, non-discrimination in credit (including reason codes for adverse actions)
- **FCRA** (US) - Accuracy and fairness in credit reporting
- **EU AI Act** - High-risk AI system requirements

**Partially Supported:**

- **Fair Housing Act** (US) - Housing discrimination
- **Employment Standards** (EEOC, various) - Hiring discrimination

### How do I generate ECOA-compliant reason codes?

For credit decisions, ECOA requires providing specific reasons for adverse actions. GlassAlpha provides the `reasons` command:

```bash
glassalpha reasons \
  --model models/credit_model.pkl \
  --data data/test.csv \
  --instance 42 \
  --output notices/instance_42.txt
```

**What you get:**

- Top-N negative feature contributions (ECOA typical: 4 reasons)
- Automatic exclusion of protected attributes (age, gender, race)
- ECOA-compliant adverse action notice template
- Deterministic, reproducible output

**Example output:**

```
ADVERSE ACTION NOTICE
DECISION: DENIED

PRINCIPAL REASONS FOR ADVERSE ACTION:
1. Debt: Value of 5000 negatively impacted the decision
2. Duration: Value of 24 negatively impacted the decision
3. Credit History: Value of 2 negatively impacted the decision
4. Savings: Value of 1000 negatively impacted the decision

IMPORTANT RIGHTS UNDER FEDERAL LAW:
[ECOA disclosure text...]
```

See the [Reason Codes Guide](../guides/reason-codes.md) for complete documentation.

### What's the difference between audit reports and reason codes?

**Audit Reports** (`glassalpha audit`):

- Comprehensive PDF reports for model validation
- Performance metrics, fairness analysis, feature importance
- Used for internal compliance and regulatory submission
- Analyzes the entire model's behavior

**Reason Codes** (`glassalpha reasons`):

- Individual explanations for specific decisions
- ECOA-compliant adverse action notices
- Sent to applicants who were denied
- Explains one instance at a time

Use **audit reports** for model governance and **reason codes** for applicant communication.

See the [Trust & Deployment Guide](../reference/trust-deployment.md) for compliance framework information.

### Can GlassAlpha reports be submitted to regulators?

Yes, GlassAlpha reports are designed for regulatory submission:

- **Professional formatting** suitable for legal and regulatory review
- **Complete audit trails** with reproducibility manifests
- **Statistical rigor** with confidence intervals and significance testing
- **Standardized metrics** aligned with regulatory expectations
- **Comprehensive documentation** covering methodology and limitations

### How does GlassAlpha handle protected attributes?

Protected attributes (race, gender, age) are used for:

1. **Fairness Analysis** - Bias detection across demographic groups
2. **Statistical Testing** - Demographic parity and equal opportunity
3. **Report Generation** - Group-specific performance metrics

**Important:** Protected attributes are used for analysis only, not model training (unless explicitly configured).

### What audit evidence does GlassAlpha provide?

**Generated Evidence:**

- **PDF Audit Reports** - Comprehensive analysis with visualizations
- **Audit Manifests** - Complete execution metadata in JSON format
- **Configuration Records** - All settings and parameters used
- **Individual Explanations** - SHAP-based decision explanations
- **Statistical Analysis** - Bias testing with confidence intervals
- **Reproducibility Data** - Seeds, hashes, and version information

## Performance & limitations

### How fast is GlassAlpha?

**Typical Performance:**

- **Small datasets** (< 1,000 rows): 1-3 seconds
- **Medium datasets** (1,000-10,000 rows): 3-15 seconds
- **Large datasets** (10,000-100,000 rows): 15-60 seconds

**Performance Factors:**

- **Model complexity** (tree depth, number of estimators)
- **Explanation method** (TreeSHAP is faster than KernelSHAP)
- **Number of features** affects SHAP computation time
- **Hardware** (CPU cores, memory) impacts parallel processing

### What are the dataset size limits?

**Practical Limits:**

- **Rows**: No hard limit, tested up to 1M+ rows
- **Features**: Up to ~1,000 features (SHAP computation becomes slow beyond this)
- **Memory**: Depends on available RAM (8GB recommended for 100K+ rows)

**Optimization Options:**

```yaml
# For large datasets
explainers:
  config:
    treeshap:
      max_samples: 100 # Reduce from default 1000
    kernelshap:
      n_samples: 50 # Reduce from default 500

performance:
  low_memory_mode: true
  n_jobs: -1 # Use all CPU cores
```

### Can I run GlassAlpha in production environments?

Yes, GlassAlpha is designed for production use:

**Production Features:**

- **Deterministic execution** for consistent results
- **Configuration management** with version control
- **Audit trails** for compliance and debugging
- **Error handling** with clear error messages
- **Security considerations** (no external network calls)

**Integration Options:**

- **CLI automation** for scheduled audits
- **Python API** for programmatic integration
- **Configuration files** for different environments
- **Enterprise deployment** with RBAC and monitoring

For comprehensive production deployment guidance, see the [Trust & Deployment Guide](../reference/trust-deployment.md).

### Are there any limitations I should know about?

**Current Limitations:**

- **Tabular data only** - No text, image, or time series support yet
- **Classification focus** - Limited regression support
- **English documentation** - Additional languages may be supported based on demand
- **Single machine** - No distributed computing support

**Model Limitations:**

- **TreeSHAP** only works with tree-based models (XGBoost, LightGBM)
- **KernelSHAP** can be slow for complex models or large datasets
- **Fairness metrics** require protected attribute data

## Enterprise features

### What enterprise features are available?

GlassAlpha is organized to support potential future features for enterprise needs. If interested contact: enterprise@glassalpha.com

## Integration & workflow

### How do I integrate GlassAlpha with my existing ML pipeline?

**Python API Integration:**

Load configuration:

```python
from glassalpha.pipeline import AuditPipeline
from glassalpha.config import AuditConfig

config = AuditConfig.from_yaml("audit_config.yaml")
```

Run audit:

```python
pipeline = AuditPipeline(config)
results = pipeline.run()
```

Check results:

```python
if results.success:
    print(f"Audit completed: {results.model_performance}")
else:
    print(f"Audit failed: {results.error_message}")
```

**CLI Integration:**

In CI/CD pipeline:

```bash
glassalpha validate --config production_config.yaml --strict
glassalpha audit --config production_config.yaml --output audit_report.pdf
```

### Can I customize the audit reports?

**Current Customization:**

- **Report sections** can be included/excluded
- **Color schemes** and styling options
- **Company branding** (logo, contact information)
- **Compliance statements** for specific regulations

**Configuration Example:**

```yaml
report:
  template: standard_audit
  styling:
    color_scheme: professional
    company_name: "Your Company"
    logo_path: "assets/logo.png"
  include_sections:
    - executive_summary
    - model_performance
    - fairness_analysis
```

### How do I handle multiple models or environments?

**Multiple Configurations:**

```bash
# Development environment
glassalpha audit --config configs/german_credit_simple.yaml --output dev_audit.pdf

# Production environment
glassalpha audit --config configs/gdpr_compliance.yaml --output prod_audit.pdf --strict
```

**Configuration Overrides:**

```bash
# Base configuration with environment-specific overrides
glassalpha audit \
  --config base_config.yaml \
  --override prod_overrides.yaml \
  --output prod_audit.pdf
```

## Development & extension

### How do I contribute to GlassAlpha?

**Ways to Contribute:**

- **Bug reports** and feature requests via GitHub Issues
- **Code contributions** following our development guidelines
- **Documentation improvements** and examples
- **Testing** on different platforms and use cases

See the [Contributing Guide](../reference/contributing.md) for detailed instructions.

### Can I build custom metrics?

Yes, implement the `MetricInterface`:

```python
from glassalpha.core import MetricRegistry

@MetricRegistry.register("my_metric")
class MyCustomMetric:
    metric_type = "performance"

    def compute(self, y_true, y_pred, **kwargs):
        # Custom calculation
        return {"value": result, "interpretation": "higher_is_better"}
```

### How do I add custom explainers?

Implement the `ExplainerInterface`:

```python
from glassalpha.core import ExplainerRegistry

@ExplainerRegistry.register("my_explainer", priority=75)
class MyExplainer:
    def explain(self, model, X, y=None):
        # Custom explanation logic
        return {"explanations": explanations}

    def supports_model(self, model):
        return model.get_model_type() in ["my_model_type"]
```

## Troubleshooting

### Common installation issues?

**Python Version:**

- Ensure Python 3.11+ (check with `python --version`)
- Use virtual environments to avoid conflicts

**XGBoost on macOS:**

```bash
# If you see libomp errors
brew install libomp
pip uninstall xgboost && pip install xgboost
```

**Memory Issues:**

- Reduce `max_samples` in explainer configuration
- Enable `low_memory_mode` in performance settings
- Use smaller datasets for initial testing

### Where can I get help?

**Support Channels:**

- **Documentation**: Comprehensive guides and examples
- **GitHub Issues**: Bug reports and feature requests
- **Community Discussions**: User questions and sharing
- **Troubleshooting Guide**: Common issues and solutions

For immediate help, check the [Troubleshooting Guide](../reference/troubleshooting.md) or search existing GitHub Issues.

## Getting started

### What's the fastest way to start using GlassAlpha?

1. **Follow the Quick Start Guide** - Get running in under 10 minutes
2. **Try the German Credit example** - See all features working
3. **Adapt the configuration** - Modify for your specific use case
4. **Read the documentation** - Understand advanced features

**Essential Resources:**

- [Quick Start Guide](../getting-started/quickstart.md) - 5-minute introduction
- [Using Custom Data](../getting-started/custom-data.md) - Audit your own models
- [Freely Available Data Sources](../getting-started/data-sources.md) - Public datasets for testing
- [Configuration Guide](../getting-started/configuration.md) - Complete configuration reference
- [Model Selection Guide](model-selection.md) - Choose the right model
- [Explainer Deep Dive](explainers.md) - Understanding explanations
- [German Credit Tutorial](../examples/german-credit-audit.md) - Complete walkthrough
- [CLI Reference](cli.md) - Command-line interface

### I'm new to ML auditing. Where should I start?

**Learning Path:**

1. **Understand the basics** - What is algorithmic bias and fairness?
2. **Review regulations** - What compliance frameworks apply to your use case?
3. **Try GlassAlpha** - Generate an audit report with sample data
4. **Interpret results** - Learn to read audit reports and metrics
5. **Plan implementation** - Design your audit workflow and processes

**Recommended Reading:**

- [Trust & Deployment Guide](../reference/trust-deployment.md) - Architecture, licensing, security, and compliance
- [German Credit Tutorial](../examples/german-credit-audit.md) (detailed interpretation)
- Industry guides on algorithmic fairness and bias

This FAQ covers the most common questions about GlassAlpha. If you don't find your answer here, please check our other documentation or reach out via GitHub Issues.
