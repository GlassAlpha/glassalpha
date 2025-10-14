# Model-Explainer Compatibility Matrix

This guide helps you choose the right explainer for your model type, ensuring optimal performance and accuracy.

## Quick Reference

| Model Type              | Compatible Explainers                 | Recommended      | Speed   | Accuracy    | Notes                                      |
| ----------------------- | ------------------------------------- | ---------------- | ------- | ----------- | ------------------------------------------ |
| **xgboost**             | treeshap, permutation, kernelshap     | **treeshap**     | Fast    | Exact       | TreeSHAP provides exact Shapley values     |
| **lightgbm**            | treeshap, permutation, kernelshap     | **treeshap**     | Fast    | Exact       | TreeSHAP provides exact Shapley values     |
| **random_forest**       | treeshap, permutation, kernelshap     | **treeshap**     | Medium  | Exact       | Slower with many trees                     |
| **logistic_regression** | coefficients, permutation, kernelshap | **coefficients** | Instant | Exact       | Coefficients are native feature importance |
| **linear_regression**   | coefficients, permutation, kernelshap | **coefficients** | Instant | Exact       | Coefficients are native feature importance |
| **svm**                 | permutation, kernelshap               | **permutation**  | Medium  | Approximate | No native importance method                |
| **neural_network**      | permutation, kernelshap               | **permutation**  | Slow    | Approximate | Consider gradient-based methods            |
| **custom**              | permutation                           | **permutation**  | Medium  | Approximate | Universal fallback                         |

## Installation Requirements

### For Tree Models (treeshap)

```bash
pip install 'glassalpha[explain]'
```

Includes: SHAP library, XGBoost, LightGBM

### For Basic Models (coefficients, permutation)

```bash
pip install glassalpha
```

No additional dependencies needed. Included in base install.

## Detailed Compatibility

### Tree-Based Models

#### XGBoost

**Best Explainer**: `treeshap`

**Why**: TreeSHAP computes exact Shapley values by traversing the tree structure. No approximation needed.

**Configuration**:

```yaml
model:
  type: xgboost

explainers:
  priority:
    - treeshap # Primary choice
    - permutation # Fallback if SHAP not installed
```

**Performance**:

- **Speed**: Fast (~100ms for typical models)
- **Accuracy**: Exact Shapley values
- **Memory**: Efficient tree traversal

**When to Use Alternatives**:

- `permutation`: When SHAP not available
- `kernelshap`: For model-agnostic comparison

---

#### LightGBM

**Best Explainer**: `treeshap`

**Why**: Same as XGBoost - exact Shapley values via tree traversal.

**Configuration**:

```yaml
model:
  type: lightgbm

explainers:
  priority:
    - treeshap
    - permutation
```

**Performance**:

- **Speed**: Fast (~100ms)
- **Accuracy**: Exact
- **Memory**: Very efficient (LightGBM optimized structure)

**Special Considerations**:

- LightGBM's leaf-wise growth makes TreeSHAP particularly efficient
- Supports categorical features natively

---

#### Random Forest (scikit-learn)

**Best Explainer**: `treeshap`

**Why**: Exact Shapley values, but slower with many trees.

**Configuration**:

```yaml
model:
  type: random_forest

explainers:
  priority:
    - treeshap
  config:
    treeshap:
      max_samples: 500 # Limit for performance
```

**Performance**:

- **Speed**: Medium-Slow (scales with tree count)
- **Accuracy**: Exact
- **Memory**: Increases with forest size

**Optimization Tips**:

```yaml
explainers:
  config:
    treeshap:
      max_samples: 500 # Limit computation
      check_additivity: false # Skip validation for speed
```

---

### Linear Models

#### Logistic Regression

**Best Explainer**: `coefficients`

**Why**: Coefficients ARE the feature importance. Instant and exact.

**Configuration**:

```yaml
model:
  type: logistic_regression

explainers:
  priority:
    - coefficients # Instant, exact
    - permutation # Fallback
```

**Performance**:

- **Speed**: Instant (<1ms)
- **Accuracy**: Exact (coefficients = importance)
- **Memory**: Minimal

**Why Not SHAP**?:
While TreeSHAP/KernelSHAP work, they're overkill for linear models. Coefficients provide the same information instantly.

**Interpretation**:

```python
# Positive coefficient → feature increases probability
# Negative coefficient → feature decreases probability
# Magnitude → strength of effect
```

---

#### Linear Regression

**Best Explainer**: `coefficients`

**Why**: Same as logistic regression - coefficients are the feature importance.

**Configuration**:

```yaml
model:
  type: linear_regression

explainers:
  priority:
    - coefficients
```

---

### Other Models

#### Support Vector Machines (SVM)

**Best Explainer**: `permutation`

**Why**: SVMs don't have native importance. Permutation is fast and model-agnostic.

**Configuration**:

```yaml
model:
  type: svm

explainers:
  priority:
    - permutation
    - kernelshap # More accurate but slower
  config:
    permutation:
      n_repeats: 10
      random_state: 42
```

**Performance**:

- **Speed**: Medium (requires predictions)
- **Accuracy**: Good approximation
- **Memory**: Minimal

**Considerations**:

- Non-linear kernels make interpretation harder
- KernelSHAP provides better local explanations

---

#### Neural Networks

**Best Explainer**: `permutation`

**Why**: Universal, no assumptions about model internals.

**Configuration**:

```yaml
model:
  type: neural_network

explainers:
  priority:
    - permutation
  config:
    permutation:
      n_repeats: 20 # More repeats for stability
```

**Performance**:

- **Speed**: Slow (many forward passes)
- **Accuracy**: Approximation
- **Memory**: Depends on batch size

**Future**: Gradient-based methods may be added for deeper insights.

---

## Explainer Details

### TreeSHAP

**How It Works**: Traverses tree structure to compute exact Shapley values.

**Pros**:

- ✅ Exact Shapley values (not approximate)
- ✅ Fast for tree models
- ✅ Handles feature interactions
- ✅ Consistent and stable

**Cons**:

- ❌ Only works with tree models
- ❌ Requires SHAP library (~50MB)
- ❌ Slower with many trees

**Best For**: XGBoost, LightGBM, Random Forest

**Configuration Options**:

```yaml
explainers:
  config:
    treeshap:
      max_samples: 1000 # Max samples to explain
      check_additivity: true # Verify Shapley properties
      feature_perturbation: "interventional" # or "tree_path_dependent"
```

---

### Coefficients

**How It Works**: Uses model's native coefficients as feature importance.

**Pros**:

- ✅ Instant (no computation)
- ✅ Exact for linear models
- ✅ No dependencies
- ✅ Interpretable

**Cons**:

- ❌ Only works with linear models
- ❌ Assumes features are scaled comparably

**Best For**: Logistic Regression, Linear Regression

**Configuration Options**:

```yaml
explainers:
  config:
    coefficients:
      scale_by_std: false # Scale by feature std for comparability
```

---

### Permutation Importance

**How It Works**: Measures importance by randomly shuffling each feature and measuring prediction change.

**Pros**:

- ✅ Model-agnostic (works with any model)
- ✅ No dependencies
- ✅ Intuitive interpretation
- ✅ Captures feature interactions

**Cons**:

- ❌ Approximate (depends on n_repeats)
- ❌ Can be slow (many predictions)
- ❌ Sensitive to correlated features

**Best For**: Universal fallback, SVMs, Neural Networks

**Configuration Options**:

```yaml
explainers:
  config:
    permutation:
      n_repeats: 10 # More = more stable but slower
      random_state: 42 # Reproducibility
      scoring: "accuracy" # or "f1", "roc_auc"
```

**Tips**:

- Use `n_repeats: 5` for quick exploration
- Use `n_repeats: 20+` for production audits
- Watch for correlated features (can inflate importance)

---

### KernelSHAP

**How It Works**: Model-agnostic approximation using weighted linear regression.

**Pros**:

- ✅ Model-agnostic
- ✅ Shapley value guarantees
- ✅ Handles feature interactions

**Cons**:

- ❌ Very slow (many model evaluations)
- ❌ Approximate (sample-based)
- ❌ Requires SHAP library
- ❌ High variance with small samples

**Best For**: Model comparisons, when TreeSHAP unavailable

**Configuration Options**:

```yaml
explainers:
  config:
    kernelshap:
      n_samples: 1000 # More = more accurate but slower
      background_size: 100 # Background dataset size
```

**Warning**: Very computationally expensive. Prefer TreeSHAP for tree models.

---

## Configuration Examples

### Production XGBoost Setup

```yaml
model:
  type: xgboost
  path: models/production_model.pkl

explainers:
  strategy: first_compatible
  priority:
    - treeshap # Primary
    - permutation # Fallback
  config:
    treeshap:
      max_samples: 1000
      check_additivity: true
    permutation:
      n_repeats: 10
      random_state: 42
```

### Quick Exploration (Any Model)

```yaml
explainers:
  priority:
    - permutation # Fast, universal
  config:
    permutation:
      n_repeats: 5 # Quick results
```

### Maximum Accuracy

```yaml
model:
  type: xgboost

explainers:
  priority:
    - treeshap
  config:
    treeshap:
      max_samples: 5000 # Explain more samples
      check_additivity: true # Verify correctness
      feature_perturbation: "interventional"
```

---

## Troubleshooting

### "TreeSHAP requested but model is not a tree model"

**Problem**: TreeSHAP only works with tree-based models.

**Solution**:

```yaml
# For logistic regression
explainers:
  priority:
    - coefficients  # Use native importance

# For other models
explainers:
  priority:
    - permutation  # Universal fallback
```

### "No explainer available"

**Problem**: SHAP not installed and only SHAP explainers requested.

**Solution**:

```bash
# Install SHAP
pip install 'glassalpha[explain]'

# Or use permutation (always available)
```

### "KernelSHAP is very slow"

**Problem**: KernelSHAP requires many model evaluations.

**Solutions**:

```yaml
# Reduce samples
explainers:
  config:
    kernelshap:
      n_samples: 100      # Down from 1000
      background_size: 50 # Down from 100

# Or use faster alternative
explainers:
  priority:
    - treeshap      # If tree model
    - permutation   # Otherwise
```

### "Permutation importance seems random"

**Problem**: Not enough repeats for stability.

**Solution**:

```yaml
explainers:
  config:
    permutation:
      n_repeats: 20 # Up from default 5
      random_state: 42
```

---

## Validation

GlassAlpha validates explainer compatibility automatically:

```bash
# Check compatibility before running
glassalpha validate --config audit.yaml

# Will warn:
⚠ Runtime warnings:
  • TreeSHAP requested but model type 'logistic_regression' is not a tree model
  • Consider using 'coefficients' (for linear) or 'permutation' (universal)
```

**Production validation**:

```bash
glassalpha validate --config audit.yaml --strict-validation

# Treats warnings as errors for CI/CD
```

---

## Performance Comparison

### Speed Benchmarks (German Credit Dataset, 1000 samples)

| Explainer           | Model Type                | Time  | Samples/sec |
| ------------------- | ------------------------- | ----- | ----------- |
| coefficients        | Logistic                  | <1ms  | Instant     |
| treeshap            | XGBoost                   | 120ms | ~8000       |
| treeshap            | LightGBM                  | 80ms  | ~12000      |
| treeshap            | Random Forest (100 trees) | 2.5s  | ~400        |
| permutation (n=10)  | XGBoost                   | 450ms | ~2200       |
| kernelshap (n=1000) | XGBoost                   | 45s   | ~22         |

**Takeaway**: TreeSHAP is 100x faster than KernelSHAP for tree models.

---

## Best Practices

### 1. Match Explainer to Model

```yaml
# Good
model:
  type: xgboost
explainers:
  priority: [treeshap]

# Avoid
model:
  type: xgboost
explainers:
  priority: [kernelshap]  # Much slower, no benefit
```

### 2. Provide Fallbacks

```yaml
# Good - production-ready
explainers:
  priority:
    - treeshap
    - permutation  # Works if SHAP not installed

# Risky - no fallback
explainers:
  priority:
    - treeshap  # Fails if SHAP missing
```

### 3. Use --no-fallback in Production

```bash
# Development - allow fallbacks
glassalpha audit --config config.yaml --output report.pdf

# Production - exact components required
glassalpha audit --config config.yaml --output report.pdf --no-fallback
```

### 4. Validate Before Running

```bash
# Check compatibility
glassalpha validate --config config.yaml

# Production validation
glassalpha validate --config config.yaml --strict-validation
```

---

## Summary

**Quick Decision Guide**:

1. **Tree model?** → Use `treeshap`
2. **Linear model?** → Use `coefficients`
3. **Other model?** → Use `permutation`
4. **Need exact Shapley values?** → Use `treeshap` or `coefficients`
5. **Speed critical?** → Use `coefficients` or `permutation`
6. **Universal solution?** → Use `permutation`

**Default Recommendation**:

```yaml
explainers:
  priority:
    - treeshap # For tree models
    - coefficients # For linear models
    - permutation # Universal fallback
```

This configuration works for all model types and provides optimal performance.

---

## Further Reading

- [SHAP Documentation](https://shap.readthedocs.io/)
- [TreeSHAP Paper](https://arxiv.org/abs/1802.03888)
- [Shapley Values Explained](https://christophm.github.io/interpretable-ml-book/shapley.html)
- [GlassAlpha Configuration Guide](../getting-started/configuration.md)

---

**Questions?** See our [FAQ](../reference/faq.md) or [open a discussion](https://github.com/GlassAlpha/glassalpha/discussions).
