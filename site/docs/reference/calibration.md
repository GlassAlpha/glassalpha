# Calibration Analysis

!!! info "Part of Understanding Section"
Learn about calibration metrics with statistical confidence intervals. Related topics:

    - **[Fairness Metrics](fairness-metrics.md)** - Statistical rigor for fairness analysis
    - **[Configuration Guide](../getting-started/configuration.md)** - How to enable calibration analysis
    - **[SR 11-7 §III.B.2](../compliance/sr-11-7-mapping.md)** - Validation testing requirements

# Calibration Analysis

Model calibration measures whether predicted probabilities match observed outcomes. A well-calibrated model predicting 70% confidence should be correct 70% of the time.

## Why Calibration Matters

**Poor calibration misleads decision-makers** even when classification accuracy is high.

**Example: Loan Approval**

- Model predicts 90% approval probability
- **Well-calibrated**: 90% of these applicants would repay
- **Poorly calibrated**: Only 70% would repay
- **Impact**: Bank loses money on 20% unexpected defaults

**Regulatory requirement**: SR 11-7 Section III.B.1 requires validation testing including probability accuracy.

## Calibration with Confidence Intervals (E10+)

GlassAlpha provides statistical rigor for calibration analysis:

1. **Expected Calibration Error (ECE)** with 95% confidence intervals
2. **Brier Score** with 95% confidence intervals
3. **Bin-wise calibration curves** with error bars
4. **Deterministic bootstrap** for reproducibility

## Metrics

### Expected Calibration Error (ECE)

**Definition**: Average absolute difference between predicted probability and observed frequency.

**Formula**:

```
ECE = Σ (n_b / n) * |accuracy_b - confidence_b|
```

Where:

- `n_b` = number of samples in bin b
- `n` = total samples
- `accuracy_b` = observed accuracy in bin b
- `confidence_b` = mean predicted probability in bin b

**Interpretation**:

| ECE       | Interpretation    | Status       |
| --------- | ----------------- | ------------ |
| < 0.05    | Well calibrated   | ✅ Excellent |
| 0.05-0.10 | Acceptable        | ⚠️ Fair      |
| > 0.10    | Poorly calibrated | 🔴 Poor      |

**Example**:

- ECE = 0.03: Predictions are off by 3% on average (good)
- ECE = 0.15: Predictions are off by 15% on average (poor)

### Brier Score

**Definition**: Mean squared difference between predicted probability and actual outcome.

**Formula**:

```
Brier = (1/n) * Σ (p_i - y_i)²
```

Where:

- `p_i` = predicted probability for sample i
- `y_i` = actual outcome (0 or 1)

**Interpretation**:

| Brier Score | Interpretation |
| ----------- | -------------- |
| < 0.10      | Excellent      |
| 0.10-0.20   | Good           |
| 0.20-0.30   | Fair           |
| > 0.30      | Poor           |

**Properties**:

- Lower is better (0 = perfect)
- Combines calibration and discrimination
- Sensitive to both over/under-confidence

## Confidence Intervals

### Why CIs Matter

Point estimates can be misleading with small samples:

**Example**:

- ECE = 0.08 (seems acceptable)
- But with 95% CI = [0.02, 0.18]
- **Interpretation**: Could be excellent (0.02) or poor (0.18)
- **Action**: Collect more data for precise estimate

### Bootstrap Method

**Process**:

1. Resample data with replacement (1000 times by default)
2. Compute ECE/Brier for each resample
3. CI bounds = 2.5th and 97.5th percentiles (95% CI)

**Deterministic**: Seeded random sampling ensures byte-identical results.

### Configuration

```yaml
metrics:
  calibration:
    enabled: true

    # Binning strategy
    n_bins: 10 # Fixed bins (default)
    bin_strategy: fixed # fixed or adaptive

    # Confidence intervals
    compute_confidence_intervals: true # Default: true
    n_bootstrap: 1000 # Bootstrap samples (default: 1000)
    confidence_level: 0.95 # CI level (default: 0.95)

    # Bin-wise CIs
    compute_bin_wise_ci: true # Error bars for calibration curve
```

## Calibration Curves

### What They Show

**Calibration curve** plots predicted probability vs observed frequency:

- **X-axis**: Predicted probability bin (e.g., 0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
- **Y-axis**: Observed frequency (fraction of positives in bin)
- **Diagonal**: Perfect calibration line

**Well-calibrated model**: Points cluster near diagonal
**Poorly calibrated model**: Points far from diagonal

### Bin-Wise Confidence Intervals

Error bars show uncertainty in observed frequency for each bin:

```
Bin         | Mean Pred | Observed | 95% CI        | Samples
------------|-----------|----------|---------------|--------
[0.0, 0.1)  | 0.05      | 0.08     | [0.02, 0.14]  | 25
[0.1, 0.2)  | 0.15      | 0.12     | [0.06, 0.18]  | 30
[0.2, 0.3)  | 0.25      | 0.24     | [0.18, 0.30]  | 42
...
[0.9, 1.0)  | 0.95      | 0.91     | [0.82, 1.00]  | 18
```

**Wide CI**: Small sample in bin, uncertain estimate
**Narrow CI**: Large sample, precise estimate

**Skipped bins**: Bins with <10 samples are skipped (insufficient for bootstrap)

### PDF Visualization

Calibration curves in audit PDF include:

- Scatter plot of predicted vs observed
- Perfect calibration diagonal (reference line)
- Error bars (bin-wise 95% CIs)
- ECE annotation
- Bin sample sizes

## PDF Output

Calibration analysis appears as a dedicated section in the audit PDF:

### Calibration Metrics Table

```
Calibration Analysis with Confidence Intervals

Metric                   | Value  | 95% CI        | Interpretation
-------------------------|--------|---------------|----------------
Expected Calibration     | 0.042  | [0.028, 0.058]| Well Calibrated
Error (ECE)              |        |               |
Brier Score              | 0.156  | [0.142, 0.171]| Good

Sample size: 200 | Bins: 10 | Bootstrap samples: 1,000
```

### Bin-Wise Calibration Error

```
Bin         | Mean Pred | Observed | 95% CI        | |Pred - Obs| | Samples
------------|-----------|----------|---------------|-------------|--------
[0.0, 0.1)  | 0.05      | 0.08     | [0.02, 0.14]  | 0.03        | 25
[0.1, 0.2)  | 0.15      | 0.12     | [0.06, 0.18]  | 0.03        | 30
[0.2, 0.3)  | 0.25      | 0.24     | [0.18, 0.30]  | 0.01        | 42
[0.3, 0.4)  | 0.35      | 0.40     | [0.31, 0.49]  | 0.05        | 28
...
```

### Calibration Curve Plot

Visual display with:

- Scatter points (predicted vs observed by bin)
- Diagonal reference line (y = x)
- Error bars (95% CIs)
- ECE value annotated

## JSON Export

All calibration results in audit manifest:

```json
{
  "calibration_ci": {
    "ece": 0.042,
    "ece_ci": {
      "ci_lower": 0.028,
      "ci_upper": 0.058,
      "confidence_level": 0.95,
      "n_bootstrap": 1000
    },
    "brier_score": 0.156,
    "brier_ci": {
      "ci_lower": 0.142,
      "ci_upper": 0.171,
      "confidence_level": 0.95,
      "n_bootstrap": 1000
    },
    "bin_calibration": [
      {
        "bin_range": [0.0, 0.1],
        "mean_predicted": 0.05,
        "observed_frequency": 0.08,
        "ci_lower": 0.02,
        "ci_upper": 0.14,
        "n_samples": 25
      },
      ...
    ],
    "n_bins": 10,
    "n_samples": 200
  }
}
```

## Binning Strategies

### Fixed Bins (Default)

Divide probability range [0, 1] into N equal-width bins:

```yaml
metrics:
  calibration:
    n_bins: 10 # Default
    bin_strategy: fixed
```

**Bins**: [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]

**Pros**: Simple, consistent across models
**Cons**: Empty bins possible if predictions concentrate in narrow range

### Adaptive Bins

Adjust bin edges based on prediction distribution:

```yaml
metrics:
  calibration:
    n_bins: 10
    bin_strategy: adaptive # Quantile-based bins
```

**Method**: Bins contain equal number of samples (quantiles)

**Pros**: All bins have samples, robust to skewed predictions
**Cons**: Bin edges differ across models (harder to compare)

## Interpreting Results

### Well-Calibrated Model

```
ECE: 0.035 [0.022, 0.048]
Brier: 0.142 [0.128, 0.156]

Calibration curve: Points cluster near diagonal
All bin CIs include diagonal
```

**Interpretation**: Predictions are reliable probability estimates. Can be used directly for decision-making.

**Action**: Proceed with deployment.

### Poorly Calibrated Model

```
ECE: 0.18 [0.14, 0.22]
Brier: 0.34 [0.30, 0.38]

Calibration curve: Points systematically below diagonal (underconfident)
Or: Points above diagonal (overconfident)
```

**Interpretation**: Predictions are unreliable. Model may be accurate (good AUC) but probabilities are miscalibrated.

**Actions**:

1. Apply calibration (Platt scaling, isotonic regression)
2. Retrain with better probability estimates
3. Do NOT use raw probabilities for decision-making

### Overconfident Model

**Pattern**: Observed frequency < predicted probability

```
Bin [0.8, 0.9): Predicted 0.85, Observed 0.65
```

**Example**: Model predicts 85% approval, but only 65% actually approved.

**Risk**: False confidence → poor decisions

**Causes**:

- Overfitting
- Class imbalance
- Uncalibrated algorithms (e.g., tree ensembles)

### Underconfident Model

**Pattern**: Observed frequency > predicted probability

```
Bin [0.6, 0.7): Predicted 0.65, Observed 0.82
```

**Example**: Model predicts 65% approval, but 82% actually approved.

**Risk**: Missed opportunities (reject good applicants)

**Causes**:

- Overregularization
- Conservative probability estimates

## Improving Calibration

### 1. Platt Scaling (Logistic Calibration)

Fit logistic regression on model outputs:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(
    base_model,
    method='sigmoid',  # Platt scaling
    cv='prefit'
)
calibrated_model.fit(X_val, y_val)
```

**Best for**: Overconfident models (e.g., XGBoost, Random Forest)

### 2. Isotonic Regression

Non-parametric calibration (piecewise constant):

```python
calibrated_model = CalibratedClassifierCV(
    base_model,
    method='isotonic',  # Isotonic regression
    cv='prefit'
)
calibrated_model.fit(X_val, y_val)
```

**Best for**: Non-monotonic miscalibration

### 3. Temperature Scaling

Scale logits before softmax:

```python
# T = temperature parameter (learned on validation set)
calibrated_probs = softmax(logits / T)
```

**Best for**: Deep learning models

### After Calibration

Re-run GlassAlpha audit to verify improvement:

```bash
glassalpha audit --config calibrated_model_config.yaml --output audit_v2.pdf
```

Compare ECE/Brier before and after calibration.

## Deterministic Execution

All calibration metrics are fully deterministic with explicit seeds:

```yaml
reproducibility:
  random_seed: 42 # Required for reproducible bootstrap
```

**What's deterministic**:

- Bootstrap sample selection
- Bin assignment
- CI computation

**Guarantee**: Same config + same seed = byte-identical results

## Complete Example

### Configuration

```yaml
# audit_config.yaml
# Direct configuration

reproducibility:
  random_seed: 42

data:
  dataset: german_credit
  target_column: credit_risk

model:
  type: xgboost
  params:
    n_estimators: 100
    random_state: 42

metrics:
  calibration:
    enabled: true
    n_bins: 10
    bin_strategy: fixed
    compute_confidence_intervals: true
    n_bootstrap: 1000
    confidence_level: 0.95
    compute_bin_wise_ci: true
```

### Run Audit

```bash
glassalpha audit --config audit_config.yaml --output audit.pdf
```

### Expected Output

```
Running calibration analysis...
  ✓ Computed probabilities for 200 samples
  ✓ ECE: 0.042 [0.028, 0.058] (Well Calibrated)
  ✓ Brier: 0.156 [0.142, 0.171] (Good)
  ✓ Bin-wise CIs: 10 bins (8 with n≥10)
  ✓ Bootstrap: 1,000 samples

Calibration Summary:
  Status: PASS (ECE < 0.05)
  Confidence: High (narrow CIs)
```

## Troubleshooting

### "Calibration section missing from PDF"

**Cause**: Calibration metrics not enabled or model doesn't output probabilities.

**Fixes**:

1. Enable: `metrics.calibration.enabled: true`
2. Check model supports `predict_proba()` (not just `predict()`)

### "Wide confidence intervals"

**Cause**: Small sample size or high variance.

**Fixes**:

1. Collect more data (preferred)
2. Increase bootstrap samples: `n_bootstrap: 5000`
3. Use fewer bins: `n_bins: 5` (more samples per bin)

### "Many bins skipped (n<10)"

**Cause**: Predictions concentrated in narrow probability range.

**Fixes**:

1. Use adaptive bins: `bin_strategy: adaptive`
2. Reduce number of bins: `n_bins: 5`
3. Check if model is overly confident (all predictions near 0 or 1)

### "ECE and Brier disagree"

**Example**: ECE low (0.04) but Brier high (0.28)

**Explanation**:

- **ECE**: Measures calibration only
- **Brier**: Measures calibration + discrimination (accuracy)

**Interpretation**: Model is well-calibrated but has poor discrimination (low AUC). Probabilities are reliable but model can't separate classes well.

### "Calibration looks good but fairness fails"

**Possible issue**: Calibration may differ across protected groups.

**Solution**: Check group-specific calibration:

```yaml
metrics:
  calibration:
    compute_by_group: true # Coming in future release
```

Currently: Export predictions and compute group-specific ECE manually.

## Best Practices

### 1. Always Check Calibration for Probability-Based Decisions

If you use predicted probabilities (not just binary predictions), calibration is critical:

**Use cases requiring calibration**:

- Risk scoring (credit, insurance, healthcare)
- Resource allocation (based on probability thresholds)
- Cost-sensitive decisions (expected value calculations)

### 2. Calibration ≠ Accuracy

**Example**:

- Model A: 90% accuracy, ECE = 0.15 (poor calibration)
- Model B: 85% accuracy, ECE = 0.03 (good calibration)

**Which to use?**

- For **binary decisions**: Model A (higher accuracy)
- For **probability-based decisions**: Model B (reliable probabilities)

### 3. Validate on Holdout Set

Calibration can overfit to validation set during tuning:

```
Train → Validation (tune hyperparameters) → Test (final calibration check)
```

**Best practice**: Reserve separate holdout set for final calibration assessment.

### 4. Document Calibration Method

If applying calibration, record in audit manifest:

```yaml
model:
  type: xgboost
  calibration:
    method: platt_scaling # or isotonic, temperature
    fitted_on: validation_set
    ece_before: 0.18
    ece_after: 0.04
```

### 5. Monitor Calibration Drift

Calibration can degrade over time as data distribution shifts:

**Recommendation**: Re-run calibration analysis quarterly or after major data changes.

## Related Features

- **[Fairness Metrics](fairness-metrics.md)**: Group-level performance with CIs
- **[Robustness Testing](robustness.md)**: Stability under perturbations
- **[Shift Testing](../guides/shift-testing.md)**: Robustness to demographic changes
- **[SR 11-7 Mapping](../compliance/sr-11-7-mapping.md)**: Section III.B.2 validation testing

## Implementation Details

**Modules**:

- `glassalpha.metrics.calibration.quality`: ECE and Brier computation
- `glassalpha.metrics.calibration.confidence`: Bootstrap CI computation
- `glassalpha.metrics.calibration.binning`: Binning strategies

**API**:

- `assess_calibration_quality()`: Main entry point
- `compute_calibration_with_ci()`: Calibration with CIs
- `compute_bin_wise_ci()`: Per-bin error bars

**Test Coverage**: 25+ contract tests + German Credit integration tests validating determinism, accuracy, and edge cases.

## Summary

Calibration analysis with statistical rigor:

- ✅ **ECE with 95% CIs**: Point estimate + uncertainty quantification
- ✅ **Brier Score with CIs**: Combined calibration + discrimination metric
- ✅ **Bin-wise error bars**: Visualize uncertainty in calibration curve
- ✅ **Deterministic bootstrap**: Reproducible with random seeds
- ✅ **Flexible binning**: Fixed or adaptive strategies

**Critical for**: Risk scoring, probability-based decisions, regulatory compliance (SR 11-7 Section III.B.2).

**Remember**: A model can be accurate but poorly calibrated. Always check both.
