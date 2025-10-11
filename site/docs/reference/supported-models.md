# Supported Models

GlassAlpha currently supports the following model types with production-ready integrations.

## Model Compatibility Matrix

| Model Type          | Status     | Explainer    | Notes                              |
| ------------------- | ---------- | ------------ | ---------------------------------- |
| XGBoost             | Production | TreeSHAP     | Optimized integration, recommended |
| LightGBM            | Production | TreeSHAP     | Native integration available       |
| Logistic Regression | Production | Coefficients | Full scikit-learn compatibility    |
| Random Forest       | Testing    | TreeSHAP     | scikit-learn integration           |
| Gradient Boosting   | Testing    | TreeSHAP     | scikit-learn integration           |

## Installation Requirements

### Basic Installation (LogisticRegression)

```bash
pip install glassalpha
```

Zero-dependency coefficient explanations for linear models.

### Tree models (XGBoost, LightGBM)

```bash
pip install 'glassalpha[explain]'  # Includes SHAP
pip install 'glassalpha[xgboost]'  # Includes XGBoost
pip install 'glassalpha[lightgbm]' # Includes LightGBM
```

Or install all optional dependencies:

```bash
pip install 'glassalpha[all]'
```

## Model-Specific Details

### XGBoost

**Status**: Production-ready

**Explainer**: TreeSHAP (optimized for tree ensembles)

**Configuration**:

```yaml
model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 100
    max_depth: 5
    learning_rate: 0.1
    random_state: 42

explainers:
  strategy: first_compatible
  priority:
    - treeshap
```

**Notes**:

- Best performance for tabular data
- TreeSHAP integration is highly optimized
- Supports all XGBoost objectives for classification
- Deterministic with `random_state` set

### LightGBM

**Status**: Production-ready

**Explainer**: TreeSHAP (native integration)

**Configuration**:

```yaml
model:
  type: lightgbm
  params:
    objective: binary
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 100
    random_state: 42

explainers:
  strategy: first_compatible
  priority:
    - treeshap
```

**Notes**:

- Fast training on large datasets
- Memory-efficient
- Native TreeSHAP support
- Deterministic with `random_state` set

### Logistic Regression

**Status**: Production-ready

**Explainer**: Coefficients (zero dependencies)

**Configuration**:

```yaml
model:
  type: sklearn
  class: LogisticRegression
  params:
    penalty: l2
    C: 1.0
    solver: lbfgs
    max_iter: 100
    random_state: 42

explainers:
  strategy: first_compatible
  priority:
    - coefficients
```

**Notes**:

- No SHAP dependency required
- Coefficient-based explanations
- Fast inference
- Deterministic with `random_state` set
- Good baseline for comparison

## Choosing a Model

### For Banking/Credit

**Recommended**: XGBoost or Logistic Regression

- XGBoost: Best accuracy, TreeSHAP explanations
- Logistic: Simpler to explain to regulators, coefficient-based

### For Healthcare

**Recommended**: Logistic Regression or LightGBM

- Logistic: Maximum interpretability
- LightGBM: Better performance with large datasets

### For Fraud Detection

**Recommended**: XGBoost or LightGBM

- High-dimensional feature spaces
- Imbalanced classes handled well
- Fast inference required

[See model selection guide →](model-selection.md)

## Model-Explainer Compatibility

Not all explainers work with all models. See the compatibility matrix:

[Model-Explainer Compatibility →](model-explainer-compatibility.md)

## Adding Custom Models

GlassAlpha supports custom model implementations through the model interface protocol:

```python
from glassalpha.models.base import ModelInterface
from glassalpha.models import load_model

# Add to models/__init__.py load_model() function
class CustomModel(ModelInterface):
    """Custom model implementation"""

    def predict(self, X):
        # Your prediction logic
        pass

    def predict_proba(self, X):
        # Your probability prediction logic
        pass

    @property
    def capabilities(self):
        return {
            "supports_shap": True,
            "supports_coefficients": False,
            "data_modality": "tabular"
        }
```

[See contributing guide for details →](contributing.md)

## Roadmap

Future model support planned:

- Neural networks (PyTorch, TensorFlow)
- Time series models
- Text models (transformers)
- Image models (CNNs)

[See Phase 2 priorities for timeline →](https://github.com/GlassAlpha/glassalpha/discussions)

## Questions?

- [FAQ](faq.md) - Common questions
- [Troubleshooting](troubleshooting.md) - Debugging help
- [GitHub Discussions](https://github.com/GlassAlpha/glassalpha/discussions) - Ask the community
