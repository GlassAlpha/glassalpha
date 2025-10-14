# Model selection guide

Choose the right model for your ML audit based on your data, requirements, and constraints.

!!! info "Quick Links"

- **Already chosen a model?** → [Model Parameters Reference](../reference/model-parameters.md) for detailed parameter documentation
- **Need explainer info?** → [Explainer Selection Guide](../reference/explainers.md) to pair your model with the right explainer
- **Ready to configure?** → [Configuration Guide](../getting-started/configuration.md) for YAML setup

## Quick Decision: Which Model Should I Use?

```mermaid
graph TB
    Start[Choose Model]
    Start --> Size{Dataset size?}

    Size -->|< 1K rows| Simple[LogisticRegression]
    Size -->|1K-100K| Med{Need speed?}
    Size -->|> 100K| Large[LightGBM]

    Med -->|Yes fast| Fast[LightGBM]
    Med -->|Best accuracy| Acc[XGBoost]

    Simple --> Check1[Quick baseline]
    Fast --> Check2[Fast training]
    Acc --> Check3[Best performance]
    Large --> Check4[Large scale]

    style Simple fill:#e1f5ff
    style Fast fill:#fff3cd
    style Acc fill:#d4edda
    style Large fill:#d4edda
```

**Rule of Thumb**:

- **Starting out?** → LogisticRegression (always available, fast, interpretable)
- **Production system?** → XGBoost (best accuracy-interpretability balance)
- **Very large data?** → LightGBM (handles 1M+ rows efficiently)

**Quick selection**:

- **Just testing?** → LogisticRegression
- **Small-medium dataset (<100K)?** → XGBoost
- **Large dataset (>100K)?** → LightGBM
- **Maximum interpretability?** → LogisticRegression

## Model Comparison

### Performance Benchmarks

Based on real performance with German Credit dataset (1,000 rows, 20 features):

| Metric                         | LogisticRegression | XGBoost  | LightGBM |
| ------------------------------ | ------------------ | -------- | -------- |
| **Training Speed (1K rows)**   | 0.1s               | 0.5s     | 0.3s     |
| **Training Speed (100K rows)** | 2s                 | 45s      | 25s      |
| **Typical Accuracy (German)**  | 74%                | 77%      | 76%      |
| **Memory Usage (100K rows)**   | 50MB               | 300MB    | 200MB    |
| **Interpretability**           | ★★★★★              | ★★★☆☆    | ★★★☆☆    |
| **Installation**               | Built-in           | Optional | Optional |
| **Best Explainer**             | Coefficients       | TreeSHAP | TreeSHAP |

### Feature Comparison

| Feature                    | LogisticRegression | XGBoost     | LightGBM    |
| -------------------------- | ------------------ | ----------- | ----------- |
| **Always Available**       | ✅ Yes             | ⚠️ Optional | ⚠️ Optional |
| **Non-linear Patterns**    | ❌ No              | ✅ Yes      | ✅ Yes      |
| **Handles Missing Values** | ❌ No              | ✅ Yes      | ✅ Yes      |
| **Feature Interactions**   | ❌ Manual          | ✅ Auto     | ✅ Auto     |
| **Regularization**         | ✅ Yes             | ✅ Yes      | ✅ Yes      |
| **Multiclass Support**     | ✅ Yes             | ✅ Yes      | ✅ Yes      |
| **GPU Acceleration**       | ❌ No              | ✅ Yes      | ✅ Yes      |

## Detailed Model Profiles

### LogisticRegression

**Best for**: Quick baselines, linear relationships, maximum interpretability

#### When to Choose LogisticRegression

**✅ Choose if**:

- You're just getting started with GlassAlpha
- You need to verify your setup works
- Your data has linear/simple relationships
- You need maximum model interpretability
- You don't want to install extra dependencies
- You're doing a quick exploratory audit

**❌ Avoid if**:

- Your data has complex non-linear patterns
- You need maximum predictive accuracy
- Your features have important interactions

#### Configuration Example

```yaml
model:
  type: logistic_regression
  params:
    random_state: 42
    max_iter: 1000
    C: 1.0 # Regularization (lower = more regularization)
    penalty: l2 # l1, l2, elasticnet
    solver: lbfgs # lbfgs, saga, liblinear
```

#### Parameter Tuning Tips

**C (Regularization strength)**:

- **Higher C (10, 100)**: Less regularization, may overfit
- **Lower C (0.01, 0.1)**: More regularization, may underfit
- **Default (1.0)**: Good starting point

**Penalty**:

- **l2**: Ridge regularization (default, good for most cases)
- **l1**: Lasso regularization (feature selection)
- **elasticnet**: Combination of l1 and l2

**Solver**:

- **lbfgs**: Fast for small-medium datasets (default)
- **saga**: Good for large datasets
- **liblinear**: Good for small datasets

#### Strengths

- **Extremely fast**: Sub-second training on most datasets
- **Highly interpretable**: Coefficients show feature importance directly
- **No dependencies**: Always available, no extra installation
- **Well understood**: Decades of theoretical backing
- **Stable**: Deterministic, reproducible results
- **Linear explanations**: Easy to explain to stakeholders

#### Limitations

- **Linear only**: Cannot capture complex non-linear patterns
- **Manual feature engineering**: Need to create interaction terms manually
- **Sensitive to scaling**: Features should be normalized
- **No automatic missing value handling**: Must preprocess data

#### Real-World Use Cases

**Financial services**: Baseline credit scoring models for regulatory comparison
**Healthcare**: Patient risk stratification where interpretability is critical
**Legal/Compliance**: Models that need to explain every decision clearly

---

### XGBoost

**Best for**: Maximum accuracy, industry standard, most common production model

#### When to Choose XGBoost

**✅ Choose if**:

- You need the best predictive performance
- Your dataset is small-to-medium (<100K rows)
- You have SHAP installed for TreeSHAP explanations
- You want the most battle-tested tree model
- You're comfortable with additional installation
- You need to handle non-linear relationships

**❌ Avoid if**:

- You have very large datasets (>100K rows) - consider LightGBM
- You want the absolute fastest training time
- You need maximum interpretability - consider LogisticRegression
- You can't install additional dependencies

#### Installation

```bash
# Install XGBoost with SHAP support
pip install 'glassalpha[explain]'

# Or just XGBoost
pip install xgboost
```

#### Configuration Example

```yaml
model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 100 # Number of trees
    max_depth: 5 # Tree depth
    learning_rate: 0.1 # Step size
    subsample: 0.8 # Row sampling
    colsample_bytree: 0.8 # Column sampling
    random_state: 42
```

#### Parameter Tuning Tips

**n_estimators (Number of trees)**:

- **50-100**: Quick baseline
- **100-300**: Good performance
- **300-1000+**: Maximum accuracy (slower)

**max_depth (Tree depth)**:

- **3-5**: Prevents overfitting, faster
- **6-10**: More complex patterns, may overfit
- **10+**: Risk of overfitting

**learning_rate (Step size)**:

- **0.01-0.05**: Slower but more accurate
- **0.1**: Good default balance
- **0.3+**: Faster but may underfit

**subsample & colsample_bytree (Sampling)**:

- **0.8**: Good default (prevents overfitting)
- **1.0**: Use all data (may overfit)
- **0.5-0.7**: More aggressive regularization

#### Strengths

- **Highest accuracy**: Typically 2-5% better than LogisticRegression
- **Non-linear patterns**: Automatically captures complex relationships
- **Feature interactions**: No manual engineering needed
- **Handles missing values**: Built-in missing value support
- **TreeSHAP support**: Exact, fast SHAP explanations
- **Industry standard**: Used by most Kaggle winners
- **Well documented**: Extensive community support

#### Limitations

- **Slower training**: 5-10x slower than LogisticRegression
- **More memory**: 3-5x more memory than LogisticRegression
- **Optional dependency**: Requires installation
- **More hyperparameters**: More tuning needed
- **Less interpretable**: Not as clear as linear models
- **Can overfit**: Requires careful regularization

#### Real-World Use Cases

**Credit scoring**: Production credit risk models with maximum accuracy
**Fraud detection**: Real-time fraud detection with complex patterns
**Customer churn**: Predict customer churn with many interaction effects
**Risk assessment**: Any high-stakes decision requiring best accuracy

---

### LightGBM

**Best for**: Large datasets, faster training, lower memory usage

#### When to Choose LightGBM

**✅ Choose if**:

- You have large datasets (>100K rows)
- Training time is critical
- Memory usage is a constraint
- You need similar accuracy to XGBoost but faster
- You want efficient GPU utilization

**❌ Avoid if**:

- You have very small datasets (<1K rows)
- You need maximum accuracy at any cost
- You prefer the most battle-tested option (XGBoost)
- You can't install additional dependencies

#### Installation

```bash
# Install LightGBM with SHAP support
pip install 'glassalpha[explain]'

# Or just LightGBM
pip install lightgbm
```

#### Configuration Example

```yaml
model:
  type: lightgbm
  params:
    objective: binary
    n_estimators: 100 # Number of trees
    num_leaves: 31 # Max leaves per tree
    learning_rate: 0.1 # Step size
    feature_fraction: 0.9 # Column sampling
    bagging_fraction: 0.8 # Row sampling
    bagging_freq: 5 # Bagging frequency
    random_state: 42
```

#### Parameter Tuning Tips

**n_estimators (Number of trees)**:

- Similar to XGBoost: 100-300 is typical

**num_leaves (Max leaves)**:

- **15-31**: Good default
- **31-63**: More complex patterns
- **63+**: Risk of overfitting

**learning_rate (Step size)**:

- **0.01-0.05**: More accurate, slower
- **0.1**: Good default
- **0.3**: Faster, may underfit

**feature_fraction & bagging_fraction (Sampling)**:

- **0.8-0.9**: Good defaults
- **1.0**: Use all data
- **0.5-0.7**: More regularization

#### Strengths

- **Fastest training**: 2-3x faster than XGBoost
- **Lower memory**: 30-50% less memory than XGBoost
- **Similar accuracy**: Often within 1% of XGBoost
- **Large dataset efficiency**: Scales to millions of rows
- **TreeSHAP support**: Exact, fast SHAP explanations
- **GPU support**: Excellent GPU acceleration
- **Leaf-wise growth**: More efficient tree building

#### Limitations

- **Less battle-tested**: Newer than XGBoost
- **Can overfit easily**: Leaf-wise growth needs careful tuning
- **Small dataset performance**: Not optimized for <1K rows
- **Optional dependency**: Requires installation
- **Different hyperparameters**: Learning curve from XGBoost

#### Real-World Use Cases

**Large-scale fraud detection**: Millions of transactions
**Recommendation systems**: Large user-item matrices
**Click-through prediction**: Large advertising datasets
**Time series at scale**: Many time series to model

---

## Choose Your Own Adventure

### I'm just getting started...

**Use**: LogisticRegression

**Why**:

- Zero setup friction
- Fast feedback loop
- Easy to understand results
- Perfect for learning GlassAlpha

**Example**:

```yaml
model:
  type: logistic_regression
  params:
    random_state: 42
    max_iter: 1000
```

**Next step**: Once comfortable, try [XGBoost for better accuracy](#xgboost)

---

### I need maximum accuracy and have SHAP installed...

**Use**: XGBoost

**Why**:

- Best predictive performance
- TreeSHAP provides exact explanations
- Industry standard for production
- Proven in thousands of deployments

**Example**:

```yaml
model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    random_state: 42
```

**Next step**: [Fine-tune hyperparameters](#xgboost) for your specific data

---

### I have large datasets (>100K rows)...

**Use**: LightGBM

**Why**:

- 2-3x faster than XGBoost
- Lower memory footprint
- Handles large datasets efficiently
- Still gets TreeSHAP benefits

**Example**:

```yaml
model:
  type: lightgbm
  params:
    objective: binary
    n_estimators: 100
    num_leaves: 31
    learning_rate: 0.1
    random_state: 42
```

**Next step**: Monitor training time and [adjust parameters](#lightgbm) if needed

---

### I need maximum interpretability for regulators...

**Use**: LogisticRegression

**Why**:

- Crystal clear feature importance
- Coefficients are explanatory
- Well-understood by regulators
- Easy to audit and explain

**Example**:

```yaml
model:
  type: logistic_regression
  params:
    random_state: 42
    max_iter: 1000
    C: 1.0
```

**Next step**: Consider [feature engineering](../getting-started/custom-data.md#data-preparation-scripts) to improve linear model performance

---

## Model Selection by Use Case

### Credit Scoring

**Recommended**: XGBoost

**Why**: Balance of accuracy and interpretability through SHAP

**Configuration**:

```yaml
model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 150
    max_depth: 5 # Shallower for interpretability
    learning_rate: 0.05 # Slower for stability
    random_state: 42
```

---

### Fraud Detection

**Recommended**: XGBoost or LightGBM (depending on scale)

**Why**: Handles imbalanced data, captures complex fraud patterns

**Configuration**:

```yaml
model:
  type: xgboost
  params:
    objective: binary:logistic
    scale_pos_weight: 99 # Handle 1% fraud rate
    n_estimators: 200
    max_depth: 7 # Deeper for complex patterns
    random_state: 42
```

---

### Healthcare Outcomes

**Recommended**: LogisticRegression or XGBoost

**Why**: LogisticRegression for high interpretability, XGBoost for accuracy

**Configuration**:

```yaml
model:
  type: logistic_regression # For transparency
  params:
    random_state: 42
    max_iter: 1000
    C: 1.0
# OR for better accuracy:
# type: xgboost
# params:
#   objective: binary:logistic
#   n_estimators: 100
#   max_depth: 4  # Shallower for clinical interpretability
```

---

### Hiring/HR

**Recommended**: LogisticRegression

**Why**: Maximum transparency for fairness audits, regulatory requirements

**Configuration**:

```yaml
model:
  type: logistic_regression
  params:
    random_state: 42
    max_iter: 1000
    penalty: l2
```

---

## Common Questions

### Can I use a pre-trained model?

Yes! Specify the model path:

```yaml
model:
  type: xgboost
  path: models/my_pretrained_model.pkl
```

GlassAlpha will load your model and generate the audit without retraining.

### How do I know which model my data needs?

**Start simple**:

1. Try LogisticRegression first (fast baseline)
2. Check the accuracy
3. If accuracy is insufficient, try XGBoost
4. Compare the accuracy improvement vs training time

**Rule of thumb**:

- Accuracy difference <2%: Stick with LogisticRegression
- Accuracy difference 2-5%: Consider XGBoost
- Accuracy difference >5%: Use XGBoost or LightGBM

### What if I don't have XGBoost/LightGBM installed?

GlassAlpha will automatically fall back to LogisticRegression with a helpful message:

```
Model 'xgboost' not available. Falling back to 'logistic_regression'.
To enable 'xgboost', run: pip install 'glassalpha[explain]'
```

### How much do these models actually differ?

**German Credit dataset example** (1,000 rows):

- LogisticRegression: 74% accuracy
- XGBoost: 77% accuracy
- LightGBM: 76% accuracy

**Difference**: 3% accuracy gain for 5x training time

**Is it worth it?** Depends on your use case:

- **High-stakes decisions**: Yes, 3% matters
- **Exploratory analysis**: No, stick with LogisticRegression
- **Production deployment**: Yes, accuracy is critical

### Can I use other scikit-learn models?

Yes! GlassAlpha supports most scikit-learn classifiers:

```yaml
model:
  type: sklearn_generic
  params:
    model_class: RandomForestClassifier
    n_estimators: 100
    random_state: 42
```

However, LogisticRegression, XGBoost, and LightGBM are optimized and recommended.

## Performance Tuning

### If Training is Too Slow

**XGBoost**:

```yaml
model:
  type: xgboost
  params:
    n_estimators: 50 # Reduce from 100
    learning_rate: 0.3 # Increase from 0.1
    tree_method: hist # Faster algorithm
```

**LightGBM**:

```yaml
model:
  type: lightgbm
  params:
    n_estimators: 50 # Reduce from 100
    num_leaves: 15 # Reduce from 31
```

### If Memory Usage is Too High

**XGBoost**:

```yaml
model:
  type: xgboost
  params:
    max_depth: 4 # Reduce from 6
    subsample: 0.5 # Reduce from 0.8
```

**Switch to LightGBM**:

```yaml
model:
  type: lightgbm # 30-50% less memory than XGBoost
```

### If Accuracy is Too Low

**Try XGBoost with more trees**:

```yaml
model:
  type: xgboost
  params:
    n_estimators: 300 # Increase from 100
    learning_rate: 0.05 # Decrease from 0.1
    max_depth: 7 # Increase from 5
```

**Feature engineering** for LogisticRegression:

- Create interaction terms
- Add polynomial features
- Normalize features
- Handle missing values carefully

## Next Steps

Now that you've chosen your model:

1. **✅ Configure**: Use the examples above to set up your model
2. **📊 Run audit**: Generate your first audit report
3. **🔍 Choose explainer**: Learn about [TreeSHAP vs KernelSHAP](../reference/explainers.md)
4. **⚙️ Tune**: Optimize hyperparameters for your specific data

## Additional Resources

- [Using Custom Data](../getting-started/custom-data.md) - How to prepare your data
- [Configuration Guide](../getting-started/configuration.md) - Full configuration reference
- [Model Parameters Reference](../reference/model-parameters.md) - Complete parameter documentation for all models
- [Explainer Selection](../reference/explainers.md) - Choose the right explainer for your model
- [FAQ](../reference/faq.md#model-support) - Common model questions

---

**Questions?** Open an issue on [GitHub](https://github.com/GlassAlpha/glassalpha/issues) or check the [FAQ](../reference/faq.md).
