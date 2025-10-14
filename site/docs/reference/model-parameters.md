# Model Parameter Reference

This guide provides comprehensive parameter references for all supported models in GlassAlpha.

!!! info "Quick Links" - **Choosing a model?** → Start with [Model Selection Guide](../reference/model-selection.md) - **Need configuration help?** → See [Configuration Guide](../getting-started/configuration.md) - **Getting started?** → Try the [Quick Start Guide](../getting-started/quickstart.md) first

## Understanding Parameter Validation

GlassAlpha passes model parameters directly to the underlying ML libraries (scikit-learn, XGBoost, LightGBM). These libraries have different behaviors:

- **scikit-learn**: Ignores unknown parameters (no error, may log warning)
- **XGBoost**: Warns about unknown parameters in logs
- **LightGBM**: Warns about unknown parameters in logs

!!! tip "Check Your Logs"
If you see warnings in your audit output about unknown parameters, double-check your parameter names against this reference.

---

## LogisticRegression

Logistic Regression is the baseline model for binary and multiclass classification. It's interpretable, fast, and works well for linearly separable data.

### Common Parameters

| Parameter       | Type     | Default | Valid Range                               | Description                                                     |
| --------------- | -------- | ------- | ----------------------------------------- | --------------------------------------------------------------- |
| `random_state`  | int      | None    | 0-2³¹                                     | Random seed for reproducibility                                 |
| `max_iter`      | int      | 100     | 1-10000                                   | Maximum iterations for solver                                   |
| `C`             | float    | 1.0     | 0.001-100.0                               | Inverse regularization strength (smaller = more regularization) |
| `penalty`       | str      | 'l2'    | 'l1', 'l2', 'elasticnet', 'none'          | Regularization type                                             |
| `solver`        | str      | 'lbfgs' | 'lbfgs', 'liblinear', 'saga', 'newton-cg' | Optimization algorithm                                          |
| `tol`           | float    | 0.0001  | 1e-6 - 0.01                               | Tolerance for stopping criteria                                 |
| `fit_intercept` | bool     | True    | True, False                               | Whether to calculate intercept                                  |
| `class_weight`  | str/dict | None    | 'balanced', dict                          | Weights for imbalanced classes                                  |

### Usage Example

```yaml
model:
  type: logistic_regression
  params:
    random_state: 42
    max_iter: 1000
    C: 1.0
    penalty: "l2"
    solver: "lbfgs"
```

### Common Mistakes

❌ **Wrong:**

```yaml
model:
  type: logistic_regression
  params:
    max_iter: -1 # ❌ Negative value causes error
    regul: 1.0 # ❌ Typo - should be 'C'
    iterations: 1000 # ❌ Wrong name - should be 'max_iter'
    C: 0 # ❌ Zero is invalid (must be > 0)
```

✅ **Correct:**

```yaml
model:
  type: logistic_regression
  params:
    random_state: 42
    max_iter: 1000
    C: 1.0
    penalty: "l2"
```

### Solver Compatibility

| Solver      | l1  | l2  | elasticnet | none | Multiclass       |
| ----------- | --- | --- | ---------- | ---- | ---------------- |
| `lbfgs`     | ❌  | ✅  | ❌         | ✅   | ✅               |
| `liblinear` | ✅  | ✅  | ❌         | ❌   | ❌ (binary only) |
| `saga`      | ✅  | ✅  | ✅         | ✅   | ✅               |
| `newton-cg` | ❌  | ✅  | ❌         | ✅   | ✅               |

### Full Reference

See [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

---

## XGBoost

XGBoost is a powerful gradient boosting library that works well for structured/tabular data.

### Common Parameters

| Parameter          | Type  | Default | Valid Range | Description                             |
| ------------------ | ----- | ------- | ----------- | --------------------------------------- |
| `random_state`     | int   | 0       | 0-2³¹       | Random seed                             |
| `n_estimators`     | int   | 100     | 10-1000     | Number of boosting trees                |
| `max_depth`        | int   | 6       | 2-15        | Maximum tree depth                      |
| `learning_rate`    | float | 0.3     | 0.01-0.3    | Step size shrinkage (eta)               |
| `subsample`        | float | 1.0     | 0.5-1.0     | Fraction of samples for each tree       |
| `colsample_bytree` | float | 1.0     | 0.5-1.0     | Fraction of features for each tree      |
| `min_child_weight` | int   | 1       | 1-10        | Minimum sum of instance weight in child |
| `gamma`            | float | 0       | 0-10        | Minimum loss reduction for split        |
| `reg_alpha`        | float | 0       | 0-10        | L1 regularization                       |
| `reg_lambda`       | float | 1       | 0-10        | L2 regularization                       |

### Usage Example

```yaml
model:
  type: xgboost
  params:
    random_state: 42
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
```

### Performance Tuning

**Fast training (less accuracy):**

```yaml
params:
  n_estimators: 50
  max_depth: 3
  learning_rate: 0.3
```

**Better accuracy (slower):**

```yaml
params:
  n_estimators: 500
  max_depth: 8
  learning_rate: 0.01
  subsample: 0.8
  colsample_bytree: 0.8
```

**Prevent overfitting:**

```yaml
params:
  max_depth: 3 # Shallower trees
  min_child_weight: 5 # Require more samples per leaf
  gamma: 1 # Higher split threshold
  subsample: 0.7 # Use less data per tree
  reg_alpha: 1 # L1 regularization
  reg_lambda: 1 # L2 regularization
```

### Common Mistakes

❌ **Wrong:**

```yaml
model:
  type: xgboost
  params:
    n_trees: 100 # ❌ Wrong name - should be 'n_estimators'
    depth: 6 # ❌ Wrong name - should be 'max_depth'
    learning_rate: 5.0 # ❌ Too high (typically 0.01-0.3)
    max_depth: -1 # ❌ Negative depth invalid for XGBoost
```

✅ **Correct:**

```yaml
model:
  type: xgboost
  params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
```

### Full Reference

See [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)

---

## LightGBM

LightGBM is a fast gradient boosting framework that uses tree-based learning algorithms.

### Common Parameters

| Parameter           | Type  | Default | Valid Range | Description                        |
| ------------------- | ----- | ------- | ----------- | ---------------------------------- |
| `random_state`      | int   | None    | 0-2³¹       | Random seed                        |
| `n_estimators`      | int   | 100     | 10-1000     | Number of boosting trees           |
| `max_depth`         | int   | -1      | -1, 2-15    | Maximum tree depth (-1 = no limit) |
| `learning_rate`     | float | 0.1     | 0.01-0.3    | Boosting learning rate             |
| `num_leaves`        | int   | 31      | 2-131072    | Maximum leaves in one tree         |
| `subsample`         | float | 1.0     | 0.5-1.0     | Fraction of data for training      |
| `colsample_bytree`  | float | 1.0     | 0.5-1.0     | Fraction of features for each tree |
| `min_child_samples` | int   | 20      | 1-100       | Minimum samples in one leaf        |
| `reg_alpha`         | float | 0.0     | 0-10        | L1 regularization                  |
| `reg_lambda`        | float | 0.0     | 0-10        | L2 regularization                  |

### Usage Example

```yaml
model:
  type: lightgbm
  params:
    random_state: 42
    n_estimators: 100
    num_leaves: 31
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
```

### Performance Tuning

**Fast training:**

```yaml
params:
  n_estimators: 50
  num_leaves: 15
  learning_rate: 0.1
```

**Better accuracy:**

```yaml
params:
  n_estimators: 500
  num_leaves: 63
  learning_rate: 0.01
  subsample: 0.8
  min_child_samples: 10
```

**Prevent overfitting:**

```yaml
params:
  num_leaves: 15 # Fewer leaves
  min_child_samples: 50 # More samples per leaf
  subsample: 0.7 # Use less data
  reg_alpha: 1 # L1 regularization
  reg_lambda: 1 # L2 regularization
```

### Important Notes

!!! warning "max_depth = -1"
In LightGBM, `max_depth: -1` means **no limit** on tree depth. This is different from XGBoost where negative values are invalid.

!!! warning "num_leaves vs max_depth"
LightGBM grows trees leaf-wise (best-first) rather than level-wise (depth-first). You typically control tree complexity via `num_leaves` rather than `max_depth`.

### Common Mistakes

❌ **Wrong:**

```yaml
model:
  type: lightgbm
  params:
    n_trees: 100 # ❌ Wrong name - should be 'n_estimators'
    max_leaves: 31 # ❌ Wrong name - should be 'num_leaves'
    learning_rate: -0.1 # ❌ Negative learning rate
```

✅ **Correct:**

```yaml
model:
  type: lightgbm
  params:
    n_estimators: 100
    num_leaves: 31
    learning_rate: 0.1
```

### Full Reference

See [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

---

## Parameter Validation Tips

### Check Your Config

Use `glassalpha validate` to check your configuration:

```bash
glassalpha validate --config my_config.yaml
```

### Enable Verbose Logging

See parameter warnings in your logs:

```yaml
logging:
  level: INFO # Or DEBUG for more detail
```

### Common Parameter Mistakes

1. **Typos in parameter names** - Double-check spelling
2. **Using wrong library's parameter names** - Each library has different conventions
3. **Invalid value ranges** - Check the valid range column in tables above
4. **Negative values where not allowed** - Most counts/sizes must be positive
5. **Wrong types** - Use int for counts, float for rates

### Getting Help

If you're unsure about a parameter:

1. Check this reference guide
2. Run `glassalpha validate --config your_config.yaml`
3. Check official library documentation
4. See our [FAQ](../reference/faq.md) for common issues

---

## See Also

- [Model Selection Guide](../reference/model-selection.md) - Choosing the right model
- [Configuration Guide](../getting-started/configuration.md) - Full config reference
- [Troubleshooting](../reference/troubleshooting.md) - Common issues and solutions
