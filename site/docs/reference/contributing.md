# Contributing to GlassAlpha

Thank you for your interest in contributing to GlassAlpha! This guide will help you understand how to effectively contribute to our professional ML auditing toolkit.

## Project philosophy

GlassAlpha follows an **audit-first** approach, prioritizing regulatory compliance and professional quality:

- **Quality over features** - Better to have fewer capabilities that work perfectly
- **Determinism over performance** - Reproducible results matter more than speed
- **User value focus** - Every change should improve audit quality or usability
- **Professional standards** - Code quality suitable for regulated industries

## Development setup

### Prerequisites

- **Python 3.11+** (required for type hints and modern features)
- **Git** for version control
- **Virtual environment** tool (venv, conda, or poetry)

### Quick setup

1. **Fork and clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/glassalpha.git
cd glassalpha
```

2. **Create and activate virtual environment:**

```bash
python -m venv glassalpha-dev
source glassalpha-dev/bin/activate  # Windows: glassalpha-dev\Scripts\activate
```

3. **Install in development mode:**

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

4. **Set up development tools:**

```bash
# Install pre-commit hooks
pre-commit install

# Verify installation
glassalpha --version
glassalpha list
```

5. **Run tests to verify setup:**

```bash
pytest
```

### Development dependencies

The `[dev]` installation includes:

- **Testing**: pytest, pytest-cov
- **Code Quality**: ruff (linter), black (formatter), mypy (type checker)
- **Pre-commit**: Automated quality checks
- **Documentation**: Tools for docs development

## Project structure

Understanding the codebase structure helps target contributions effectively:

```
glassalpha/
├── src/glassalpha/           # Main package
│   ├── core/                # Interfaces and registries
│   ├── models/              # Model wrappers (XGBoost, LightGBM, etc.)
│   ├── explain/             # Explainers (TreeSHAP, KernelSHAP)
│   ├── metrics/             # Performance, fairness, drift metrics
│   ├── data/                # Data loading and processing
│   ├── pipeline/            # Audit pipeline orchestration
│   ├── report/              # PDF generation and templates
│   ├── config/              # Configuration management
│   ├── cli/                 # Command-line interface
│   ├── profiles/            # Audit profiles
│   └── utils/               # Utilities (seeds, hashing, etc.)
├── tests/                   # Test suite
├── configs/                 # Example configurations
├── dev/                     # Development resources (internal)
└── pyproject.toml          # Package configuration
```

**Key Extension Points:**

- Add new models by implementing `ModelInterface`
- Add new explainers by implementing `ExplainerInterface`
- Add new metrics by implementing `MetricInterface`
- Add new audit profiles for specific compliance needs

## Code quality standards

### Type safety

All code must include type hints and pass `mypy --strict`:

```python
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.DataFrame,
    metrics: List[str]
) -> Dict[str, float]:
    """Compute fairness metrics across demographic groups."""
    ...
```

### Code formatting and linting

We use automated tools to ensure consistent code quality:

```bash
# Check code formatting
black --check src/

# Check linting
ruff check src/

# Check type hints
mypy src/

# Or use the convenience script
./lint-and-fix.sh
```

### Testing requirements

**Coverage Target**: 50%+ for core modules

**Test Categories:**

- **Unit tests** - Individual component testing
- **Integration tests** - Component interaction testing
- **Determinism tests** - Reproducibility verification
- **End-to-end tests** - Full audit pipeline

**Example Test:**

```python
def test_xgboost_wrapper_deterministic():
    """Test that XGBoost produces identical results with same seed."""
    # Arrange
    X, y = make_classification(n_samples=100, random_state=42)
    model1 = XGBoostWrapper(random_state=42)
    model2 = XGBoostWrapper(random_state=42)

    # Act
    model1.fit(X, y)
    model2.fit(X, y)
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)

    # Assert
    np.testing.assert_array_equal(pred1, pred2)
```

### Performance testing

CLI performance is critical for user experience. We maintain automated performance regression tests.

**Performance Expectations:**

| Command                    | Target | Current Baseline |
| -------------------------- | ------ | ---------------- |
| `glassalpha --help`        | <300ms | ~108ms           |
| `glassalpha --version`     | <100ms | ~46ms            |
| `glassalpha datasets list` | <500ms | ~248ms           |
| Full audit (german_credit) | <60s   | ~5s              |

**Run Performance Tests:**

```bash
# Run all performance tests
pytest tests/test_cli_performance.py -v

# Skip slow tests during rapid iteration
pytest -m "not slow" -v

# Test specific performance aspects
pytest tests/test_cli_performance.py::TestCLIPerformance::test_help_command_fast -v
```

**Performance Test Categories:**

1. **CLI Speed Tests** - Ensure commands stay responsive
2. **Import Cleanness Tests** - Verify no heavy imports at module level
3. **Audit Smoke Tests** - Validate end-to-end audit completion

**Maintaining Performance:**

When adding new features, follow lazy loading patterns:

```python
# ❌ BAD: Eager import at module level
from sklearn.model_selection import train_test_split

# ✅ GOOD: Lazy import within function
def split_data(data):
    from sklearn.model_selection import train_test_split
    return train_test_split(data)
```

**Check for Regressions:**

If your changes affect CLI startup or command performance:

```bash
# Measure CLI help time
time glassalpha --help

# Profile import time
python -X importtime -c "import glassalpha.__main__" 2> importtime.txt
cat importtime.txt

# Run full performance suite
pytest tests/test_cli_performance.py -v -s
```

Performance tests run automatically in CI and will fail if:

- `--help` exceeds 300ms (currently 108ms with 3x safety margin)
- Heavy ML libraries (pandas, sklearn, xgboost) load during `--help`
- Full audits take longer than 60s

## Contribution workflow

### 1. Choose contribution type

**High-Value Contributions:**

- **Bug fixes** - Especially for determinism or audit quality issues
- **Performance improvements** - Faster audit generation
- **New model support** - Additional ML library wrappers
- **Enhanced explanations** - Better SHAP integration or visualizations
- **Improved error handling** - Clearer error messages and recovery
- **Documentation** - Examples, guides, API documentation

**Lower Priority:**

- Complex features without clear audit benefit
- Breaking API changes
- Features requiring significant maintenance overhead

### 2. Create feature branch

```bash
git checkout -b feature/descriptive-name
# or
git checkout -b fix/issue-description
```

### 3. Development process

**Before coding:**

1. Check existing issues and discussions
2. Create an issue for significant changes
3. Discuss architecture for major features

**While coding:**

1. Write tests first (TDD approach recommended)
2. Follow existing patterns and interfaces
3. Maintain deterministic behavior
4. Add comprehensive docstrings

**Code example pattern:**

```python
@ModelRegistry.register("new_model")
class NewModelWrapper:
    """Wrapper for NewML library following GlassAlpha patterns."""

    capabilities = {
        "supports_shap": True,
        "data_modality": "tabular"
    }
    version = "1.0.0"

    def __init__(self, **kwargs):
        """Initialize with deterministic defaults."""
        self._set_deterministic_params(kwargs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions with type safety."""
        # Implementation
        ...
```

### 4. Testing your changes

```bash
# Run specific test categories
pytest tests/test_models/          # Model tests
pytest tests/test_explainers/      # Explainer tests
pytest tests/test_integration/     # Integration tests

# Run all tests with coverage
pytest --cov=src/glassalpha --cov-report=html

# Test determinism (crucial for audit reproducibility)
pytest -k deterministic

# Test end-to-end workflow
glassalpha audit --config configs/german_credit_simple.yaml --output test.pdf
```

### 5. Quality checks

```bash
# Comprehensive quality check
./lint-and-fix.sh

# Manual checks if needed
black src/
ruff check --fix src/
mypy src/
```

### 6. Commit your changes

Use conventional commit format:

```bash
git add .
git commit -m "feat(models): add RandomForest wrapper with TreeSHAP support

- Implement RandomForestWrapper following ModelInterface
- Add comprehensive test suite with determinism checks
- Include capability declarations for SHAP compatibility
- Update model registry and documentation

Closes #123"
```

**Commit Types:**

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation updates
- `test:` Test improvements
- `refactor:` Code restructuring
- `perf:` Performance improvements

## Pull request process

### Before submitting

1. **Rebase on main** to ensure clean history
2. **Run full test suite** and ensure all pass
3. **Check coverage** hasn't decreased significantly
4. **Update documentation** for any user-visible changes
5. **Add changelog entry** if user-facing

### PR description template

```markdown
## Description

Brief description of changes and motivation.

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Determinism verified (if applicable)

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or clearly documented)
```

### Review process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by maintainer
3. **Testing verification** in review environment
4. **Documentation review** for user-facing changes
5. **Merge** after approval

## Architecture guidelines

### Plugin system

GlassAlpha uses a registry-based plugin architecture. Follow these patterns:

**Adding New Components:**

```python
# 1. Implement the interface
class MyExplainer:
    capabilities = {"model_types": ["xgboost"]}

    def explain(self, model, X, y=None):
        # Implementation
        ...

# 2. Register the component
@ExplainerRegistry.register("my_explainer", priority=75)
class MyExplainer:
    ...

# 3. Add tests
def test_my_explainer_registration():
    assert "my_explainer" in ExplainerRegistry.get_all()
```

### Enterprise/OSS separation

Maintain clear boundaries between open source and enterprise features:

```python
# OSS implementation - always available
@ModelRegistry.register("basic_model")
class BasicModel:
    ...

# Enterprise feature - gated behind license check
@check_feature("advanced_models")
def create_advanced_model():
    if not is_enterprise():
        raise FeatureNotAvailable("Requires enterprise license")
    # Enterprise implementation
    ...
```

### Deterministic design

All operations must be reproducible:

```python
def deterministic_operation(data, random_state=None):
    """Ensure operation can be reproduced exactly."""
    if random_state is not None:
        np.random.seed(random_state)

    # Deterministic processing
    result = process_data(data)

    # Include randomness source in output for audit trail
    return {
        "result": result,
        "random_state": random_state,
        "timestamp": datetime.utcnow(),
        "version": __version__
    }
```

## Testing guidelines

### Test organization

```
tests/
├── test_core/              # Core architecture tests
├── test_models/            # Model wrapper tests
├── test_explainers/        # Explainer tests
├── test_metrics/           # Metrics tests
├── test_integration/       # Component integration
├── test_cli/               # CLI interface tests
├── test_deterministic/     # Reproducibility tests
└── test_end_to_end/       # Full pipeline tests
```

### Writing effective tests

**Test Structure (Arrange-Act-Assert):**

```python
def test_specific_behavior():
    """Test description explaining what behavior is verified."""
    # Arrange - Set up test data and conditions
    data = create_test_data()
    config = create_test_config()

    # Act - Execute the operation being tested
    result = perform_operation(data, config)

    # Assert - Verify expected outcomes
    assert result.success
    assert len(result.explanations) > 0
    assert result.manifest["random_seed"] == config.random_seed
```

**Testing Patterns:**

1. **Determinism Tests:**

```python
def test_audit_determinism():
    """Verify identical configs produce identical results."""
    config = load_config("test_config.yaml")

    result1 = run_audit(config)
    result2 = run_audit(config)

    # Critical for regulatory compliance
    assert result1.manifest == result2.manifest
    np.testing.assert_array_equal(result1.shap_values, result2.shap_values)
```

2. **Error Handling Tests:**

```python
def test_graceful_error_handling():
    """Verify clear error messages for common failures."""
    invalid_config = {"model": {"type": "nonexistent"}}

    with pytest.raises(ComponentNotFoundError) as exc_info:
        run_audit(invalid_config)

    assert "nonexistent" in str(exc_info.value)
    assert "available models" in str(exc_info.value)
```

## Documentation standards

### Docstring format

Use Google-style docstrings:

```python
def explain_model_decision(
    model: ModelInterface,
    instance: pd.Series,
    background_data: Optional[pd.DataFrame] = None,
    explainer_type: str = "auto"
) -> Dict[str, Any]:
    """Generate explanation for a single model decision.

    This function provides detailed explanations for individual predictions,
    helping users understand which features contributed to the model's decision
    and by how much.

    Args:
        model: Trained model implementing ModelInterface protocol
        instance: Single data instance to explain (must match model input schema)
        background_data: Reference data for SHAP baseline. If None, uses
            model training data when available
        explainer_type: Type of explainer to use ("auto", "shap", "lime").
            "auto" selects best explainer based on model capabilities

    Returns:
        Dictionary containing:
            - "explanations": Feature importance scores
            - "baseline": Reference prediction value
            - "prediction": Model prediction for this instance
            - "confidence": Prediction confidence if available
            - "metadata": Explainer type, version, parameters used

    Raises:
        ValueError: If instance shape doesn't match model input requirements
        ExplainerNotSupportedError: If no compatible explainer found for model
        DataValidationError: If background_data format is incompatible

    Example:
        >>> model = XGBoostWrapper()
        >>> model.fit(X_train, y_train)
        >>> explanation = explain_model_decision(model, X_test.iloc[0])
        >>> print(explanation["explanations"])
        {'feature1': 0.23, 'feature2': -0.15, ...}

        Generate explanation plot:
        >>> plot_explanation_waterfall(explanation)

    Note:
        Explanations are computed using the model's most compatible explainer.
        For tree-based models (XGBoost, LightGBM), TreeSHAP provides exact
        Shapley values. For other models, KernelSHAP provides approximations.
    """
```

### Example updates

When adding new features, update relevant examples:

1. **Configuration examples** - Show how to use new features
2. **Tutorial updates** - Integrate new capabilities into user journey
3. **API documentation** - Document new interfaces and parameters

## Security and privacy guidelines

### Data handling

- **No PII in logs** - Never log personally identifiable information
- **Sanitize inputs** - Validate and clean all user inputs
- **Hash sensitive data** - Use SHA-256 for any identifier hashing
- **Local processing** - Core library must work completely offline

### Example safe logging

```python
import logging
from glassalpha.utils import hash_dataframe

logger = logging.getLogger(__name__)

def process_audit_data(data: pd.DataFrame):
    """Process audit data safely."""
    data_hash = hash_dataframe(data)

    # Safe: Log hash and metadata, never actual data
    logger.info(f"Processing dataset: hash={data_hash}, shape={data.shape}")

    # NEVER do this:
    # logger.info(f"Processing data: {data.to_dict()}")  # Could contain PII

    return process_data(data)
```

## Getting help

### Communication channels

- **GitHub Issues** - Bug reports, feature requests
- **GitHub Discussions** - General questions, usage help
- **Code Review** - Feedback on pull requests

### Before asking questions

1. **Search existing issues** and discussions
2. **Check documentation** including FAQ and troubleshooting
3. **Try the quickstart tutorial** to understand basic usage
4. **Look at existing code examples** for implementation patterns

### How to ask effective questions

**Good Question Format:**

```
**What I'm trying to do:** Add support for CatBoost models

**What I've tried:**
- Implemented CatBoostWrapper following ModelInterface
- Added basic tests following XGBoost example
- Getting error: "TreeSHAP not compatible with CatBoost"

**Expected behavior:** TreeSHAP should work with CatBoost like other tree models

**Environment:**
- GlassAlpha version: 0.1.0
- Python version: 3.11.5
- CatBoost version: 1.2.0

**Code snippet:** [minimal reproducing example]
```

## Recognition and credits

Contributors are recognized in several ways:

- **Changelog entries** for significant contributions
- **GitHub contributor list** automatically maintained
- **Documentation credits** for major documentation contributions
- **Release notes** highlighting key contributions

## Development roadmap

Understanding our direction helps target valuable contributions:

**Current Focus Areas:**

- Enhanced model support (additional ML libraries)
- Improved explanation quality and performance
- Better error handling and user experience
- Comprehensive documentation and examples

**Potential Future Considerations** (community-driven):

- Additional data modalities based on demand
- Extended compliance framework support
- Enhanced integration capabilities
- Advanced visualization options

---

## Quick reference

**Essential Commands:**

```bash
# Setup
pip install -e ".[dev]" && pre-commit install

# Development
./lint-and-fix.sh              # Code quality checks
pytest                         # Run tests
glassalpha audit --config configs/german_credit_simple.yaml --output test.pdf

# Quality gates
black --check src/             # Formatting
ruff check src/                # Linting
mypy src/                      # Type checking
pytest --cov=src/glassalpha   # Coverage
```

**Getting Unstuck:**

1. Check [Trust & Deployment Guide](trust-deployment.md) for system design
2. Look at existing implementations for patterns
3. Search [GitHub Issues](https://github.com/GlassAlpha/glassalpha/issues)
4. Ask in [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions)

Thank you for contributing to GlassAlpha! Your contributions help make ML auditing more transparent, reliable, and accessible for regulated industries.
