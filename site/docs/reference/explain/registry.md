# Explainer Registry

**Module**: `glassalpha.explain.registry`

Registry for managing explainer selection and availability with deterministic behavior.

---

## Overview

The explainer registry provides deterministic explainer selection with fallback handling. Critical for reproducible audits.

**Key features**:

- Priority-based explainer selection
- Deterministic fallback chain
- Availability checking
- Strict mode for production

---

## Quick Start

```python
from glassalpha.explain.registry import select_explainer_deterministic

# Deterministic explainer selection with priority list
explainer_name, explainer = select_explainer_deterministic(
    model=model,
    X_test=X_test,
    priority=['treeshap', 'kernelshap', 'coefficients'],
    strict=False,  # Use fallbacks if preferred unavailable
)

print(f"Selected: {explainer_name}")
```

---

## API Reference

### select_explainer_deterministic

```python
def select_explainer_deterministic(
    model,
    X_test: pd.DataFrame,
    *,
    priority: list[str] | None = None,
    strict: bool = False,
) -> tuple[str, ExplainerInterface]:
    """Select explainer with guaranteed deterministic behavior.

    Args:
        model: Trained model implementing predict/predict_proba
        X_test: Test dataset for validation
        priority: Ordered list of explainer names to try (default: ['treeshap', 'kernelshap', 'coefficients'])
        strict: If True, fail if preferred explainer unavailable. If False, use deterministic fallback chain.

    Returns:
        Tuple of (explainer_name, explainer_instance)

    Raises:
        ExplainerUnavailableError: If no compatible explainer in strict mode

    Example:
        >>> from glassalpha.explain.registry import select_explainer_deterministic
        >>> explainer_name, explainer = select_explainer_deterministic(
        ...     model=xgb_model,
        ...     X_test=X_test,
        ...     priority=['treeshap', 'kernelshap'],
        ...     strict=True,
        ... )
        >>> print(f"Using: {explainer_name}")
    """
```

### \_available_explainers

```python
def _available_explainers() -> dict[str, type[ExplainerInterface]]:
    """Get all currently available explainers.

    Checks which explainers can be imported with current dependencies.

    Returns:
        Dict mapping explainer name to explainer class

    Example:
        >>> from glassalpha.explain.registry import _available_explainers
        >>> available = _available_explainers()
        >>> print(f"Available: {list(available.keys())}")
        ['coefficients', 'permutation', 'treeshap', 'kernelshap']
    """
```

---

## Explainer Priority Lists

### Recommended Priorities by Model Type

**Tree-based models** (XGBoost, LightGBM, RandomForest):

```python
priority = ['treeshap', 'kernelshap', 'permutation', 'coefficients']
```

**Linear models** (LogisticRegression, LinearRegression):

```python
priority = ['coefficients', 'permutation', 'kernelshap']
```

**General fallback** (works for any model):

```python
priority = ['permutation', 'coefficients']
```

---

## Strict Mode

### When to Use Strict Mode

**Use strict=True** when:

- Running production audits
- Regulatory submissions
- Reproducibility is critical
- You've verified dependencies

**Use strict=False** when:

- Exploring new models
- Development/testing
- Dependency availability uncertain

### Strict Mode Example

```python
from glassalpha.explain.registry import select_explainer_deterministic, ExplainerUnavailableError

try:
    explainer_name, explainer = select_explainer_deterministic(
        model=model,
        X_test=X_test,
        priority=['treeshap'],
        strict=True,
    )
except ExplainerUnavailableError as e:
    print(f"❌ Required explainer unavailable: {e}")
    print("Fix: pip install shap")
    raise
```

---

## Exceptions

### ExplainerUnavailableError

```python
class ExplainerUnavailableError(RuntimeError):
    """Raised when requested explainer is not available in strict mode."""
```

**Example**:

```python
from glassalpha.explain.registry import ExplainerUnavailableError

try:
    explainer_name, explainer = select_explainer_deterministic(
        model=model,
        X_test=X_test,
        priority=['treeshap'],
        strict=True,
    )
except ExplainerUnavailableError as e:
    # Handle missing dependency
    print(f"Required explainer unavailable: {e}")
    print("Available explainers:", e.available_explainers)  # If provided
```

---

## Usage Patterns

### CI/CD Pre-Deployment Check

```python
from glassalpha.explain.registry import select_explainer_deterministic, _available_explainers

def validate_explainer_dependencies(model, X_test, required_explainer: str):
    """Validate that required explainer is available."""

    # Check availability
    available = _available_explainers()
    if required_explainer not in available:
        raise RuntimeError(
            f"Required explainer '{required_explainer}' not available. "
            f"Available: {list(available.keys())}"
        )

    # Validate selection works
    try:
        explainer_name, explainer = select_explainer_deterministic(
            model=model,
            X_test=X_test,
            priority=[required_explainer],
            strict=True,
        )
        print(f"✅ {required_explainer} available and working")
        return True
    except Exception as e:
        print(f"❌ {required_explainer} failed: {e}")
        return False
```

### Fallback Strategy

```python
from glassalpha.explain.registry import select_explainer_deterministic

def get_explainer_with_logging(model, X_test):
    """Get explainer with detailed logging."""

    priority = ['treeshap', 'kernelshap', 'permutation', 'coefficients']

    explainer_name, explainer = select_explainer_deterministic(
        model=model,
        X_test=X_test,
        priority=priority,
        strict=False,  # Allow fallbacks
    )

    print(f"Selected explainer: {explainer_name}")

    # Warn if not first choice
    if explainer_name != priority[0]:
        print(f"⚠️  Using fallback explainer. Preferred '{priority[0]}' unavailable.")
        print(f"   Install with: pip install glassalpha[shap]")

    return explainer_name, explainer
```

### Testing Explainer Compatibility

```python
import pytest
from glassalpha.explain.registry import select_explainer_deterministic

@pytest.mark.parametrize("explainer_name", ["treeshap", "kernelshap", "coefficients"])
def test_explainer_compatibility(explainer_name, model, X_test):
    """Test that explainer works with model."""

    try:
        name, explainer = select_explainer_deterministic(
            model=model,
            X_test=X_test,
            priority=[explainer_name],
            strict=True,
        )

        # Test explain works
        explanations = explainer.explain(X_test.iloc[0:1])
        assert explanations is not None

    except ExplainerUnavailableError:
        pytest.skip(f"{explainer_name} not available in test environment")
```

---

## Determinism Guarantees

### What is Deterministic

✅ **Explainer selection order**: Same priority → same explainer selected

✅ **Availability checking**: Deterministic import checks

✅ **Fallback chain**: Deterministic fallback order

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

```python
priority = ['treeshap']  # For tree-based models
```

### KernelSHAP

**Name**: `kernelshap`

**Requirements**: `shap` package

**Best for**: Any model type

**Speed**: Slow (model-agnostic)

```python
priority = ['kernelshap']  # For general models
```

### Coefficients

**Name**: `coefficients`

**Requirements**: None (built-in)

**Best for**: Linear models

**Speed**: Instant

```python
priority = ['coefficients']  # For linear models
```

### Permutation

**Name**: `permutation`

**Requirements**: None (built-in)

**Best for**: Any model type, fallback

**Speed**: Medium

```python
priority = ['permutation']  # Reliable fallback
```

---

## Related Documentation

- [User Guide: Determinism](../../guides/determinism.md)
- [Explainer Selection Guide](../explainers.md)
- [Troubleshooting Guide](../troubleshooting.md)

---

## Source

**Module path**: `src/glassalpha/explain/registry.py`

**Example notebook**: `examples/explainer_selection.ipynb`
