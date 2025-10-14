# Explainer Selection

**Module**: `glassalpha.explain`

Explicit dispatch for selecting explainers based on model compatibility and availability.

---

## Overview

GlassAlpha uses explicit dispatch to select explainers based on model type and availability. This approach provides deterministic explainer selection with clear fallback chains for reproducible audits.

**Key features**:

- Model-type based explainer selection
- Deterministic fallback chain (TreeSHAP → KernelSHAP → Permutation)
- Clear error messages when explainers are unavailable
- No dynamic registries - all logic is explicit and traceable

---

## Quick Start

```python
from glassalpha.explain import select_explainer

# Explicit explainer selection based on model type
try:
    explainer = select_explainer("xgboost")  # Returns TreeSHAP for tree models
    explanations = explainer.explain(model, X_test)
except ImportError as e:
    print(f"Explainer not available: {e}")
    print("Install with: pip install 'glassalpha[explain]'")
```

---

## API Reference

### select_explainer

```python
def select_explainer(model_type: str, config: dict) -> ExplainerInterface:
    """Select explainer with guaranteed deterministic behavior.

    Args:
        model_type: Type of model ("xgboost", "lightgbm", "sklearn")
        config: Configuration dictionary with explainer preferences

    Returns:
        ExplainerInterface instance compatible with the model type

    Raises:
        ImportError: If required explainer dependencies are not installed
        ValueError: If model_type is not supported

    Example:
        >>> from glassalpha.explain import select_explainer
        >>> explainer = select_explainer("xgboost")
        >>> explanations = explainer.explain(model, X_test)
    """
```

## Explainer Types

GlassAlpha provides these explainer types through explicit dispatch:

| Explainer        | Model Types       | Description                          | Installation                        |
| ---------------- | ----------------- | ------------------------------------ | ----------------------------------- |
| **TreeSHAP**     | XGBoost, LightGBM | Exact Shapley values for tree models | `pip install 'glassalpha[explain]'` |
| **KernelSHAP**   | Any model         | Approximation for non-tree models    | `pip install 'glassalpha[explain]'` |
| **Permutation**  | Any model         | Model-agnostic feature importance    | Built-in (no extra dependencies)    |
| **Coefficients** | Linear models     | Direct coefficient analysis          | Built-in (no extra dependencies)    |

### Explainer Selection Logic

The explicit dispatch follows this deterministic priority chain:

1. **TreeSHAP** for tree-based models (XGBoost, LightGBM) - most accurate
2. **KernelSHAP** for any model - good approximation when TreeSHAP unavailable
3. **Permutation** explainer - always available, model-agnostic
4. **Coefficients** for linear models - direct interpretation

Example selection:

```python
from glassalpha.explain import select_explainer

# XGBoost model gets TreeSHAP (if available) or KernelSHAP fallback
explainer = select_explainer("xgboost")

# Linear model gets Coefficients explainer
explainer = select_explainer("sklearn")
```

---

## Configuration-Based Selection

GlassAlpha uses configuration-driven explainer selection rather than priority lists:

```yaml
explainers:
  strategy: first_compatible # or "all", "priority"
  priority: [treeshap, kernelshap, permutation]
```

### Explainer Strategies

**first_compatible**: Use first available explainer in priority order
**all**: Try all explainers, return results from all that work
**priority**: Use explicit priority list from config

### Configuration Example

```yaml
explainers:
  strategy: first_compatible
  priority:
    - treeshap # Best for tree models
    - kernelshap # Good approximation
    - permutation # Always available

model:
  type: xgboost
```

### Code Example

```python
from glassalpha.explain import select_explainer

config = {
    "explainers": {
        "strategy": "first_compatible",
        "priority": ["treeshap", "kernelshap", "permutation"]
    }
}

explainer = select_explainer("xgboost", config)
```

try:
explainer = select_explainer("xgboost")
explanations = explainer.explain(model, X_test)
except ImportError as e:
print(f"❌ Explainer unavailable: {e}")
print("Fix: pip install 'glassalpha[explain]'")
raise

````

---

## Error Handling

### Common Errors and Solutions

**ImportError: No module named 'shap'**

```python
try:
    explainer = select_explainer("xgboost")
except ImportError as e:
    print(f"SHAP not installed: {e}")
    print("Install with: pip install 'glassalpha[explain]'")
````

**ValueError: Unknown model_type**

```python
try:
    explainer = select_explainer("unknown_model")
except ValueError as e:
    print(f"Unsupported model type: {e}")
    print("Supported types: xgboost, lightgbm, sklearn")
```

---

## Usage Patterns

### CI/CD Validation

```python
from glassalpha.explain import select_explainer

def validate_explainer_availability(model_type: str):
    """Validate that required explainer is available."""

    try:
        explainer = select_explainer(model_type)
        print(f"✅ {model_type} explainer available")
        return True
    except ImportError as e:
        print(f"❌ Explainer unavailable: {e}")
        return False

# Example usage
validate_explainer_availability("xgboost")
```

### Configuration-Based Selection

````python
from glassalpha.explain import select_explainer

def get_explainer_from_config(config: dict):
    """Get explainer based on configuration."""

    model_type = config["model"]["type"]
    explainer_config = config.get("explainers", {})

    explainer = select_explainer(model_type, explainer_config)
    print(f"Selected explainer for {model_type}")

    return explainer

### Testing Explainer Compatibility

```python
import pytest
from glassalpha.explain import select_explainer

@pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "sklearn"])
def test_explainer_compatibility(model_type):
    """Test that explainer selection works for model type."""

    try:
        explainer = select_explainer(model_type)

        # Test that we got an explainer
        assert explainer is not None
        print(f"✅ {model_type} explainer: {explainer.__class__.__name__}")

    except (ImportError, ValueError) as e:
        pytest.skip(f"Explainer unavailable for {model_type}: {e}")
````

---

## Determinism Guarantees

### What is Deterministic

✅ **Explainer selection**: Same model_type → same explainer selected

✅ **Import checking**: Deterministic dependency availability checks

✅ **Configuration processing**: Same config → same explainer selection

### What Requires Additional Steps

⚠️ **SHAP values**: Require explicit seed for determinism (handled by GlassAlpha)

⚠️ **Background data**: Must be deterministically sampled (handled by explainer)

---

## Supported Explainers

### TreeSHAP

**Name**: `treeshap`

**Requirements**: `shap` package

**Best for**: XGBoost, LightGBM, RandomForest

**Speed**: Fast (tree-optimized)

**Usage**: Automatically selected for tree models

```python
explainer = select_explainer("xgboost")  # Gets TreeSHAP if available
```

### KernelSHAP

**Name**: `kernelshap`

**Requirements**: `shap` package

**Best for**: Any model type

**Speed**: Slow (model-agnostic)

**Usage**: Fallback for non-tree models or when TreeSHAP unavailable

```python
explainer = select_explainer("sklearn")  # Gets KernelSHAP if available
```

### Coefficients

**Name**: `coefficients`

**Requirements**: None (built-in)

**Best for**: Linear models

**Speed**: Instant

**Usage**: Primary explainer for linear models

```python
explainer = select_explainer("sklearn")  # Gets Coefficients for linear models
```

### Permutation

**Name**: `permutation`

**Requirements**: None (built-in)

**Best for**: Any model type, fallback

**Speed**: Medium

**Usage**: Always available fallback explainer

```python
explainer = select_explainer("xgboost")  # Falls back to Permutation if TreeSHAP unavailable
```

---

## Related Documentation

- [User Guide: Determinism](../../guides/determinism.md)
- [Explainer Selection Guide](../../reference/explainers.md)
- [Troubleshooting Guide](../../reference/troubleshooting.md)

---

## Source

**Module path**: `src/glassalpha/explain/__init__.py`

**Example notebook**: `examples/explainer_selection.ipynb`
