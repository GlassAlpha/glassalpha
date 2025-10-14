# Explainer selection guide

Choose the right explainer to understand what drives your model's predictions.

!!! info "Quick Links" - **Haven't chosen a model yet?** → Start with [Model Selection Guide](../reference/model-selection.md) - **Ready to configure?** → See [Configuration Guide](../getting-started/configuration.md) for explainer YAML setup - **Getting started?** → Try the [Quick Start Guide](../getting-started/quickstart.md) first

## Quick Decision Tree

```
                    Start Here
                        ↓
           Is your model tree-based?
         (XGBoost, LightGBM, RandomForest)
                   ↙        ↘
                 YES         NO
                  ↓           ↓
         Is SHAP installed?  Is it LogisticRegression?
            ↙        ↘           ↙        ↘
          YES        NO        YES        NO
           ↓          ↓         ↓          ↓
     Use TreeSHAP  Use      Use         Use
     (exact,      Permutation Coefficients KernelSHAP
      fast)       (fast)     (instant)    (slower)
```

**TL;DR**:

- **Tree models + SHAP?** → TreeSHAP (best choice)
- **Tree models, no SHAP?** → Permutation
- **LogisticRegression?** → Coefficients (fastest)
- **Other models?** → KernelSHAP (universal fallback)

## Explainer Comparison

### Performance Benchmarks

Based on German Credit dataset (1,000 rows, 20 features):

| Explainer    | Time to Explain | Accuracy            | Memory | Dependencies |
| ------------ | --------------- | ------------------- | ------ | ------------ |
| TreeSHAP     | 2.3s            | Exact               | 150MB  | SHAP library |
| KernelSHAP   | 12.5s           | Approximate         | 80MB   | SHAP library |
| Permutation  | 1.8s            | Feature-level only  | 50MB   | None         |
| Coefficients | 0.1s            | Exact (linear only) | 10MB   | None         |

**Key takeaway**: TreeSHAP is 5x faster than KernelSHAP and provides exact values.

### Feature Comparison

| Feature                  | TreeSHAP | KernelSHAP | Permutation | Coefficients |
| ------------------------ | -------- | ---------- | ----------- | ------------ |
| **Model Compatibility**  | Trees    | Any        | Any         | Linear only  |
| **Accuracy**             | Exact    | Approx     | Global only | Exact        |
| **Speed**                | Fast     | Slow       | Fast        | Instant      |
| **Individual Samples**   | ✅ Yes   | ✅ Yes     | ❌ No       | ✅ Yes       |
| **Feature Interactions** | ✅ Yes   | ✅ Yes     | ❌ No       | ❌ No        |
| **Installation**         | Optional | Optional   | Built-in    | Built-in     |
| **Interpretability**     | High     | High       | Medium      | Very High    |

## Detailed Explainer Profiles

### TreeSHAP

**Best for**: Tree-based models (XGBoost, LightGBM, RandomForest)

#### When to Choose TreeSHAP

**✅ Choose if**:

- Your model is XGBoost, LightGBM, or RandomForest
- You have SHAP installed
- You want exact SHAP values (not approximations)
- You need fast explanation generation
- You want individual prediction explanations

**❌ Avoid if**:

- Your model is LogisticRegression or neural network
- You can't install the SHAP library
- You don't have tree-based models

#### Installation

```bash
# Install SHAP support
pip install 'glassalpha[explain]'

# Or just SHAP
pip install shap
```

#### Configuration Example

```yaml
explainers:
  strategy: first_compatible
  priority:
    - treeshap
  config:
    treeshap:
      max_samples: 1000 # Samples for background dataset
      check_additivity: true # Verify SHAP properties
```

#### How It Works

TreeSHAP computes exact Shapley values by leveraging the tree structure:

1. **Traverses tree paths**: Uses the model's internal tree structure
2. **Computes contributions**: Calculates exact feature contributions
3. **No approximation**: Unlike KernelSHAP, provides exact values
4. **Polynomial time**: Much faster than exponential KernelSHAP

**Mathematical guarantee**: TreeSHAP values satisfy all SHAP properties exactly.

#### Performance Characteristics

**Speed scaling**:

- **1K samples**: ~2 seconds
- **10K samples**: ~8 seconds
- **100K samples**: ~45 seconds

**Memory scaling**:

- Roughly 150MB per 1K samples
- Can be optimized with `max_samples` parameter

#### Strengths

- **Exact values**: No approximation error
- **Fast**: 5-10x faster than KernelSHAP
- **Individual explanations**: Explains specific predictions
- **Feature interactions**: Captures interaction effects
- **Well-established**: Proven in production use
- **Additive**: Values sum to prediction difference

#### Limitations

- **Tree models only**: Won't work with LogisticRegression or neural networks
- **Requires SHAP**: Optional dependency
- **Memory intensive**: Large models need more memory
- **Less interpretable**: Not as clear as coefficients

#### Real-World Use Cases

**Credit scoring**: Explain why a loan was denied with exact feature contributions
**Fraud detection**: Show which transaction features triggered fraud alert
**Risk assessment**: Break down risk score into individual factor contributions

---

### KernelSHAP

**Best for**: Non-tree models, universal fallback

#### When to Choose KernelSHAP

**✅ Choose if**:

- Your model is not tree-based
- You need SHAP explanations for any model type
- You have SHAP installed
- Explanation time is not critical

**❌ Avoid if**:

- You have a tree-based model (use TreeSHAP instead)
- You need fast explanations (use Permutation)
- You can't install the SHAP library

#### Installation

```bash
# Install SHAP support
pip install 'glassalpha[explain]'
```

#### Configuration Example

```yaml
explainers:
  strategy: first_compatible
  priority:
    - kernelshap
  config:
    kernelshap:
      n_samples: 500 # More samples = better accuracy
      background_size: 100 # Background dataset size
```

#### How It Works

KernelSHAP approximates Shapley values through sampling:

1. **Creates coalitions**: Samples feature combinations
2. **Evaluates model**: Runs model on each coalition
3. **Fits regression**: Learns contribution weights
4. **Approximates**: Estimates Shapley values

**Trade-off**: More samples = better accuracy but slower

#### Performance Characteristics

**Speed scaling**:

- **1K samples, n_samples=100**: ~3 seconds
- **1K samples, n_samples=500**: ~12 seconds
- **10K samples, n_samples=500**: ~45 seconds

**Accuracy vs Speed**:

- **n_samples=100**: Fast but less accurate
- **n_samples=500**: Good balance (default)
- **n_samples=1000+**: Most accurate but slow

#### Strengths

- **Model-agnostic**: Works with any model type
- **Individual explanations**: Explains specific predictions
- **Feature interactions**: Captures interaction effects
- **SHAP guarantees**: Satisfies SHAP axioms (approximately)
- **Flexible**: Can tune accuracy vs speed

#### Limitations

- **Very slow**: 5-10x slower than TreeSHAP
- **Approximate**: Not exact like TreeSHAP
- **Requires SHAP**: Optional dependency
- **Memory intensive**: Stores many model evaluations
- **Hyperparameter sensitive**: n_samples affects results

#### Real-World Use Cases

**Neural networks**: Explain deep learning predictions
**Ensemble models**: Explain stacked or blended models
**Custom models**: Explain any black-box model

---

### Permutation

**Best for**: Quick feature importance without SHAP

#### When to Choose Permutation

**✅ Choose if**:

- You want global feature importance (not individual explanations)
- You don't have SHAP installed
- You need fast explanations
- You want a simple, interpretable approach

**❌ Avoid if**:

- You need individual prediction explanations
- You want to capture feature interactions
- You need exact Shapley values

#### Configuration Example

```yaml
explainers:
  strategy: first_compatible
  priority:
    - permutation
  config:
    permutation:
      n_repeats: 10 # More repeats = more stable
      random_state: 42
```

#### How It Works

Permutation importance measures feature importance by:

1. **Baseline score**: Evaluate model on original data
2. **Shuffle feature**: Randomly permute one feature
3. **New score**: Evaluate model with shuffled feature
4. **Importance**: Difference in scores
5. **Repeat**: Do this for each feature

**Intuition**: Important features cause big score drops when shuffled

#### Performance Characteristics

**Speed scaling**:

- **1K samples**: ~1.8 seconds
- **10K samples**: ~8 seconds
- **100K samples**: ~35 seconds

**Stability**: More repeats = more stable but slower

#### Strengths

- **Fast**: Comparable to TreeSHAP
- **No dependencies**: Built-in to GlassAlpha
- **Model-agnostic**: Works with any model
- **Simple interpretation**: Easy to explain
- **Global view**: Shows overall feature importance

#### Limitations

- **No individual explanations**: Only global importance
- **No interactions**: Doesn't capture feature interactions
- **Less precise**: More variance than SHAP methods
- **Not additive**: Doesn't satisfy SHAP axioms

#### Real-World Use Cases

**Quick audits**: Fast feature importance for exploratory analysis
**No SHAP available**: When you can't install additional dependencies
**Global understanding**: When you only need overall feature rankings

---

### Coefficients

**Best for**: LogisticRegression and linear models

#### When to Choose Coefficients

**✅ Choose if**:

- Your model is LogisticRegression or linear
- You want instant explanations
- You need maximum interpretability
- You want exact feature contributions

**❌ Avoid if**:

- Your model is not linear (XGBoost, neural networks, etc.)

#### Configuration Example

```yaml
model:
  type: logistic_regression

explainers:
  strategy: first_compatible
  priority:
    - coefficients # Automatic for LogisticRegression
```

#### How It Works

For LogisticRegression, coefficients directly show feature importance:

1. **Extract coefficients**: Get model weights
2. **Scale by feature**: Multiply by feature values
3. **Direct contribution**: Each coefficient is an exact contribution

**Mathematical clarity**: β₁x₁ + β₂x₂ + ... = prediction

#### Performance Characteristics

**Speed**: Instant (<0.1 seconds)
**Memory**: Minimal (~10MB)

#### Strengths

- **Instant**: No computation needed
- **Exact**: Not approximate
- **Highly interpretable**: Coefficients are clear
- **Individual explanations**: Shows contribution per sample
- **No dependencies**: Built-in
- **Additive**: Sums to prediction

#### Limitations

- **Linear models only**: Won't work with tree models or neural networks
- **No interactions**: Must manually create interaction terms
- **Feature scaling matters**: Coefficients depend on scale

#### Real-World Use Cases

**Regulatory compliance**: Maximum transparency for regulators
**Legal explanations**: Clear, defensible explanations
**Baseline models**: Quick interpretable baseline

---

## Choose Your Own Adventure

### I have XGBoost and SHAP installed...

**Use**: TreeSHAP

**Why**:

- Exact SHAP values
- Fast computation
- Individual explanations
- Industry standard

**Configuration**:

```yaml
explainers:
  strategy: first_compatible
  priority:
    - treeshap
  config:
    treeshap:
      max_samples: 1000
      check_additivity: true
```

**Expected time**: 2-5 seconds for 1K samples

---

### I have XGBoost but no SHAP...

**Use**: Permutation

**Why**:

- No dependencies required
- Still reasonably fast
- Model-agnostic
- Good for global importance

**Configuration**:

```yaml
explainers:
  strategy: first_compatible
  priority:
    - permutation
  config:
    permutation:
      n_repeats: 10
      random_state: 42
```

**Expected time**: 2-8 seconds for 1K samples

---

### I'm using LogisticRegression...

**Use**: Coefficients

**Why**:

- Instant results
- Maximum interpretability
- Exact contributions
- No dependencies

**Configuration**:

```yaml
model:
  type: logistic_regression

explainers:
  strategy: first_compatible
  priority:
    - coefficients
```

**Expected time**: <0.1 seconds

---

### I have a custom/unusual model...

**Use**: KernelSHAP or Permutation

**Why**:

- Works with any model
- KernelSHAP for individual explanations
- Permutation for global importance

**Configuration**:

```yaml
explainers:
  strategy: first_compatible
  priority:
    - kernelshap # If SHAP installed
    - permutation # Fallback
  config:
    kernelshap:
      n_samples: 500
```

**Expected time**: 10-30 seconds for 1K samples

---

## Explainer Selection by Use Case

### Credit Scoring (XGBoost)

**Recommended**: TreeSHAP

**Why**: Regulators want exact, defensible explanations

**Configuration**:

```yaml
explainers:
  strategy: first_compatible
  priority:
    - treeshap
  config:
    treeshap:
      max_samples: 1000
      check_additivity: true # Verify exactness
```

---

### Fraud Detection (Large Scale)

**Recommended**: Permutation

**Why**: Speed is critical, global importance sufficient

**Configuration**:

```yaml
explainers:
  strategy: first_compatible
  priority:
    - permutation
  config:
    permutation:
      n_repeats: 5 # Reduce for speed
```

---

### Healthcare (High Stakes)

**Recommended**: TreeSHAP or Coefficients

**Why**: Maximum transparency and exactness required

**Configuration**:

```yaml
# For tree models:
explainers:
  priority:
    - treeshap
  config:
    treeshap:
      check_additivity: true

# For linear models:
model:
  type: logistic_regression
explainers:
  priority:
    - coefficients
```

---

### Hiring/HR

**Recommended**: Coefficients (LogisticRegression)

**Why**: Maximum interpretability for fairness audits

**Configuration**:

```yaml
model:
  type: logistic_regression

explainers:
  strategy: first_compatible
  priority:
    - coefficients
```

---

## Common Questions

### What's the difference between TreeSHAP and KernelSHAP?

**TreeSHAP**:

- **Exact** Shapley values
- **Fast** (uses tree structure)
- **Tree models only**

**KernelSHAP**:

- **Approximate** Shapley values
- **Slower** (samples coalitions)
- **Any model type**

**Rule**: Use TreeSHAP for tree models, KernelSHAP for everything else

### Do I need SHAP for good explanations?

**No!** GlassAlpha provides great explanations without SHAP:

- **LogisticRegression**: Use coefficients (even better than SHAP)
- **XGBoost/LightGBM**: Use permutation importance
- **Any model**: Permutation works universally

SHAP is better for individual predictions, but not required.

### How much does SHAP really improve explanations?

**For tree models**: Significant improvement

- TreeSHAP: Individual explanations, exact values
- Permutation: Only global importance, less precise

**For linear models**: Minimal improvement

- Coefficients: Already exact and interpretable
- SHAP: Adds complexity without benefit

**Bottom line**: SHAP is worth it for tree models, less so for linear models

### Can I use multiple explainers?

Yes! GlassAlpha will use the first compatible explainer from your priority list:

```yaml
explainers:
  priority:
    - treeshap # Try first
    - kernelshap # Fallback if treeshap not available
    - permutation # Universal fallback
```

### What if explanation is too slow?

**For TreeSHAP**:

```yaml
config:
  treeshap:
    max_samples: 100 # Reduce from 1000
```

**For KernelSHAP**:

```yaml
config:
  kernelshap:
    n_samples: 100 # Reduce from 500
```

**Or switch to Permutation**:

```yaml
priority:
  - permutation # Much faster
```

## Performance Tuning

### Optimizing TreeSHAP

**For speed**:

```yaml
treeshap:
  max_samples: 100 # Fewer samples
  check_additivity: false # Skip verification
```

**For accuracy**:

```yaml
treeshap:
  max_samples: 2000 # More samples
  check_additivity: true # Verify exactness
```

### Optimizing KernelSHAP

**Fast but less accurate**:

```yaml
kernelshap:
  n_samples: 100
  background_size: 50
```

**Slow but more accurate**:

```yaml
kernelshap:
  n_samples: 1000
  background_size: 500
```

### Optimizing Permutation

**Faster**:

```yaml
permutation:
  n_repeats: 5 # Reduce from 10
```

**More stable**:

```yaml
permutation:
  n_repeats: 20 # Increase from 10
```

## Understanding Explainer Output

### TreeSHAP / KernelSHAP Output

**Global feature importance**:

```
Feature                 | Mean |SHAP|
------------------------|------------
checking_account_status | 0.234
credit_history          | 0.187
duration_months         | -0.156
credit_amount           | -0.142
```

**Individual prediction**:

```
Base value: 0.70 (70% approval rate)

Contributions:
+ checking_account (positive): +0.15
+ credit_history (good):       +0.12
+ age (35):                    +0.04
- duration_months (36):        -0.08
- credit_amount (5000):        -0.05
= Final prediction: 0.88 (88% approval)
```

### Permutation Output

**Feature importance ranking**:

```
Feature                 | Importance
------------------------|------------
checking_account_status | 0.089
credit_history          | 0.076
duration_months         | 0.054
credit_amount           | 0.048
```

**Interpretation**: Shuffling `checking_account_status` drops accuracy by 8.9%

### Coefficients Output

**Feature contributions**:

```
Feature                 | Coefficient | Contribution
------------------------|-------------|-------------
checking_account_status | 1.2         | +0.84
credit_history          | 0.8         | +0.56
duration_months         | -0.02       | -0.72
credit_amount           | -0.0001     | -0.50
Intercept               | -0.5        | -0.50
                                     = 0.68 (68%)
```

## Technical Deep Dive

### Why TreeSHAP is Faster

Traditional SHAP requires 2^n model evaluations for n features.

TreeSHAP exploits tree structure:

- Polynomial time instead of exponential
- Leverages decision tree paths
- No sampling required
- Mathematically equivalent results

**Result**: 100x+ speedup with exact values

### SHAP Axioms Explained

SHAP values satisfy important properties:

1. **Additivity**: Values sum to prediction difference
2. **Symmetry**: Similar features get similar values
3. **Dummy**: Irrelevant features get zero value
4. **Efficiency**: Values sum exactly to output

These properties make SHAP values trustworthy for explanations.

### When SHAP Can Be Misleading

**Correlated features**:

- SHAP distributes credit among correlated features
- Can underestimate individual feature importance
- Solution: Consider feature groups

**High-cardinality categoricals**:

- Many categories can lead to unstable SHAP values
- Solution: Group rare categories

**Extreme values**:

- Outliers can dominate SHAP values
- Solution: Examine distribution of SHAP values

## Next Steps

Now that you've chosen your explainer:

1. **✅ Configure**: Use the examples above to set up your explainer
2. **📊 Run audit**: Generate explanations in your audit report
3. **🔍 Interpret**: Understand what drives your model's decisions
4. **⚙️ Optimize**: Tune parameters for your speed/accuracy needs

## Additional Resources

- [Model Selection](../reference/model-selection.md) - Choose the right model first
- [Using Custom Data](../getting-started/custom-data.md) - Prepare your data
- [Configuration Guide](../getting-started/configuration.md) - Full configuration reference
- [FAQ](../reference/faq.md) - Common explainer questions

---

**Questions?** Open an issue on [GitHub](https://github.com/GlassAlpha/glassalpha/issues) or check the [FAQ](../reference/faq.md).
