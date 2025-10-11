# Dataset Bias Detection Guide

!!! tip "Quick Navigation"
**Prerequisites**: [Configuration Guide](../getting-started/configuration.md)
**Related**: [Fairness Metrics](../reference/fairness-metrics.md) | [SR 11-7 Mapping](../compliance/sr-11-7-mapping.md)
**Next**: [Testing Demographic Shifts](shift-testing.md)

# Dataset Bias Detection Guide

Catch bias at the source. Most unfairness originates in data, not models. Dataset-level bias detection identifies problems before model training.

## Overview

GlassAlpha's dataset bias audit (E12) performs five independent checks:

1. **Proxy Correlation Detection**: Non-protected features correlating with protected attributes
2. **Distribution Drift Analysis**: Feature distribution shifts between train and test
3. **Statistical Power for Sampling Bias**: Detects insufficient sample sizes
4. **Train/Test Split Imbalance**: Protected group representation differences
5. **Continuous Attribute Binning**: Configurable age and other continuous protected attributes

All checks run automatically before model evaluation in the audit pipeline.

## Why Dataset Bias Matters

**Model fairness starts with data fairness.** Even the best debiasing algorithms cannot fix:

- Proxy features encoding protected attributes (e.g., zip code → race)
- Underrepresented groups in training data
- Biased labeling or historical discrimination in outcomes
- Distribution mismatch between train and test sets

**Regulatory requirement**: SR 11-7 Section III.C.2 requires data quality assessment for model risk management.

## Configuration

Enable dataset bias analysis in your audit config:

```yaml
data:
  dataset: german_credit
  target_column: credit_risk
  protected_attributes:
    - gender
    - age_group
    - foreign_worker

  # Optional: Configure continuous attribute binning
  binning:
    age:
      strategy: domain_specific # domain_specific, custom, equal_width, equal_frequency
      bins: [18, 25, 35, 50, 65, 100] # Age range boundaries
```

**Dataset bias checks run automatically** when protected attributes are specified. No additional flags needed.

## Check 1: Proxy Correlation Detection

### What It Detects

Non-protected features that correlate strongly with protected attributes. These "proxy features" can encode indirect discrimination.

**Example**: Zip code correlating with race, enabling redlining without explicitly using race as a feature.

### Severity Levels

| Severity   | Correlation Threshold | Interpretation       | Action                                       |
| ---------- | --------------------- | -------------------- | -------------------------------------------- |
| 🔴 ERROR   | \|r\| > 0.5           | Strong correlation   | Remove feature or apply fairness constraints |
| ⚠️ WARNING | 0.3 < \|r\| ≤ 0.5     | Moderate correlation | Monitor feature impact                       |
| ℹ️ INFO    | \|r\| ≤ 0.3           | Weak correlation     | Document and proceed                         |

### Correlation Methods

GlassAlpha uses appropriate correlation metrics based on feature types:

- **Continuous-Continuous**: Pearson correlation
- **Continuous-Categorical**: Point-biserial correlation
- **Categorical-Categorical**: Cramér's V

### PDF Output

Proxy correlations appear in the audit PDF as a table:

```
Protected Attribute | Feature         | Correlation | Method          | Severity
--------------------|-----------------|-------------|-----------------|----------
gender              | occupation_code | 0.62        | Cramér's V      | ERROR
race                | zip_code        | 0.48        | Cramér's V      | WARNING
age_group           | income_level    | 0.28        | Cramér's V      | INFO
```

### Interpretation

**ERROR-level proxy**:

- Feature is effectively a proxy for protected attribute
- Model using this feature may violate anti-discrimination laws
- **Action**: Remove feature or apply fairness constraints

**WARNING-level proxy**:

- Feature has moderate correlation
- May contribute to disparate impact
- **Action**: Monitor feature importance and group metrics

**INFO-level proxy**:

- Weak correlation, likely acceptable
- **Action**: Document correlation, proceed with training

### Example: Occupation as Gender Proxy

```yaml
# Proxy correlation output
protected_attr: gender
feature: occupation_code
correlation: 0.62
method: cramers_v
severity: ERROR
```

**Interpretation**: Occupation is strongly correlated with gender (Cramér's V = 0.62). Using occupation as a feature may encode gender discrimination (e.g., historically male-dominated fields).

**Actions**:

1. Remove occupation_code from features
2. Or: Apply fairness constraints to ensure equal opportunity across genders
3. Or: Use occupation with explicit disparate impact testing

## Check 2: Distribution Drift Analysis

### What It Detects

Feature distribution differences between train and test sets. Large drift indicates:

- Non-representative sampling
- Temporal changes between data collection periods
- Different data sources for train vs test

### Statistical Tests

- **Continuous features**: Kolmogorov-Smirnov (KS) test
- **Categorical features**: Chi-square test

Both tests return p-values. Low p-value (< 0.05) indicates significant drift.

### PDF Output

```
Feature         | Test Statistic | P-Value  | Drift Detected
----------------|----------------|----------|---------------
income          | 0.18           | 0.002    | Yes
education       | 12.3           | 0.015    | Yes
age             | 0.05           | 0.632    | No
```

### Interpretation

**Significant drift (p < 0.05)**:

- Train and test distributions differ
- Model may not generalize well
- **Action**: Investigate data collection process, consider resampling or reweighting

**No drift (p ≥ 0.05)**:

- Train and test distributions are similar
- **Action**: Proceed with training

### Example: Income Drift

```yaml
feature: income
statistic: 0.18
p_value: 0.002
test: kolmogorov_smirnov
drift_detected: true
```

**Interpretation**: Income distribution differs significantly between train (p=0.002 < 0.05) and test. Model trained on one income distribution may not work on another.

**Actions**:

1. Check if train and test come from different time periods (temporal drift)
2. Verify sampling strategy is representative
3. Consider reweighting test set to match train distribution

## Check 3: Statistical Power for Sampling Bias

### What It Detects

Whether sample sizes are sufficient to detect underrepresentation of protected groups.

**Low statistical power** means even if a group is undersampled, you might not detect it reliably.

### Power Calculation

For each protected attribute group:

```
Power = probability of detecting 10% undersampling given current sample size
```

### Severity Levels

| Severity   | Power Threshold   | Interpretation       | Action                       |
| ---------- | ----------------- | -------------------- | ---------------------------- |
| 🔴 ERROR   | power < 0.5       | Insufficient samples | Collect more data            |
| ⚠️ WARNING | 0.5 ≤ power < 0.7 | Low power            | Monitor or collect more data |
| ✅ OK      | power ≥ 0.7       | Sufficient samples   | Proceed                      |

### PDF Output

```
Attribute | Group    | Sample Size | Power  | Severity
----------|----------|-------------|--------|----------
gender    | female   | 35          | 0.42   | ERROR
gender    | male     | 165         | 0.95   | OK
race      | minority | 48          | 0.61   | WARNING
```

### Interpretation

**ERROR (power < 0.5)**:

- Sample size too small to reliably detect bias
- High risk of false confidence in fairness metrics
- **Action**: Collect more data for underrepresented group

**WARNING (0.5 ≤ power < 0.7)**:

- Marginal sample size
- Some risk of missing true bias
- **Action**: Collect more data if possible, or flag limited confidence in results

**OK (power ≥ 0.7)**:

- Sufficient sample size for reliable bias detection
- **Action**: Proceed with fairness analysis

### Example: Small Female Sample

```yaml
protected_attr: gender
group: female
sample_size: 35
power: 0.42
severity: ERROR
```

**Interpretation**: Only 35 female samples. Statistical power is 0.42, meaning there's <50% chance of detecting even a 10% undersampling. Fairness metrics for females will be unreliable.

**Actions**:

1. Collect more female samples (target: n ≥ 100 for power > 0.8)
2. Flag fairness results as having low confidence
3. Do not deploy model without more data

## Check 4: Train/Test Split Imbalance

### What It Detects

Whether protected group proportions differ between train and test sets. Imbalanced splits can cause:

- Biased performance estimates
- Train/test distribution mismatch
- Unfair evaluation of group-specific performance

### Statistical Test

Chi-square test comparing protected attribute distributions between train and test.

### PDF Output

```
Attribute     | Train %  | Test %   | Chi-Square | P-Value | Imbalanced
--------------|----------|----------|------------|---------|------------
gender:female | 32.5%    | 28.0%    | 1.2        | 0.273   | No
race:minority | 18.0%    | 25.0%    | 4.8        | 0.028   | Yes
```

### Interpretation

**Imbalanced (p < 0.05)**:

- Protected group proportions differ significantly
- Evaluation metrics may not reflect true performance
- **Action**: Use stratified splitting or reweight test set

**Balanced (p ≥ 0.05)**:

- Train and test have similar group proportions
- **Action**: Proceed with evaluation

### Example: Race Imbalance

```yaml
protected_attr: race
train_proportion:
  majority: 0.82
  minority: 0.18
test_proportion:
  majority: 0.75
  minority: 0.25
chi_square: 4.8
p_value: 0.028
imbalanced: true
```

**Interpretation**: Minority representation increased from 18% (train) to 25% (test). This imbalance (p=0.028 < 0.05) means model is trained on one distribution but evaluated on another.

**Actions**:

1. Use stratified train/test split: `train_test_split(stratify=df['race'])`
2. Reweight test metrics to match train proportions
3. Report group-specific performance separately

## Check 5: Continuous Attribute Binning

### What It Is

Many protected attributes (age, income) are continuous but fairness analysis requires discrete groups. Binning converts continuous values to categorical groups.

### Binning Strategies

#### 1. Domain-Specific (Recommended)

Use meaningful ranges based on domain knowledge:

```yaml
data:
  binning:
    age:
      strategy: domain_specific
      bins: [18, 25, 35, 50, 65, 100]
      # Creates groups: [18-25), [25-35), [35-50), [50-65), [65-100)
```

**Age groups interpretation**:

- 18-25: Young adults (entry-level workforce)
- 25-35: Early career
- 35-50: Mid-career
- 50-65: Late career
- 65+: Retirement age

#### 2. Custom Bins

Define your own thresholds:

```yaml
data:
  binning:
    income:
      strategy: custom
      bins: [0, 30000, 60000, 100000, 1000000]
      # Groups: low, middle, upper-middle, high income
```

#### 3. Equal-Width

Divide range into N equal-width bins:

```yaml
data:
  binning:
    credit_score:
      strategy: equal_width
      n_bins: 5 # [300-420), [420-540), [540-660), [660-780), [780-900)
```

#### 4. Equal-Frequency

Divide samples into N equal-sized groups:

```yaml
data:
  binning:
    age:
      strategy: equal_frequency
      n_bins: 4 # Each bin has ~25% of samples
```

### Binning Recommendations

| Attribute        | Recommended Strategy | Example Bins              | Rationale              |
| ---------------- | -------------------- | ------------------------- | ---------------------- |
| Age              | Domain-specific      | [18, 25, 35, 50, 65, 100] | Meaningful life stages |
| Income           | Custom               | [0, 30k, 60k, 100k, ∞]    | Economic brackets      |
| Credit Score     | Domain-specific      | [300, 580, 670, 740, 850] | Standard FICO ranges   |
| Years Experience | Equal-width          | 5 bins                    | Continuous progression |

### Reproducibility

**Critical**: Binning strategy and bin edges are recorded in the audit manifest for reproducibility:

```json
{
  "data_preprocessing": {
    "binning": {
      "age": {
        "strategy": "domain_specific",
        "bins": [18, 25, 35, 50, 65, 100],
        "n_groups": 5
      }
    }
  }
}
```

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

  protected_attributes:
    - gender
    - age_group
    - foreign_worker

  # Configure age binning
  binning:
    age:
      strategy: domain_specific
      bins: [18, 25, 35, 50, 65, 100]

model:
  type: xgboost
  params:
    n_estimators: 100
    random_state: 42
```

### Run Audit

```bash
glassalpha audit --config audit_config.yaml --output audit.pdf
```

### Expected Output

```
Running dataset bias analysis...
  ✓ Proxy correlations: 3 features checked
     WARNING: occupation_code ~ gender (Cramér's V = 0.48)
  ✓ Distribution drift: 21 features checked
     INFO: No significant drift detected
  ✓ Sampling bias power: 3 groups checked
     WARNING: gender:female (n=35, power=0.42)
  ✓ Split imbalance: 3 attributes checked
     OK: All splits balanced
  ✓ Binning: age → 5 groups

Dataset Bias Summary:
  Warnings: 2
  Errors: 0
```

## PDF Output Location

Dataset bias analysis appears as **Section 3** in the audit PDF (after Data Overview, before Preprocessing).

Content includes:

- Proxy correlation table with severity badges
- Distribution drift results
- Statistical power by group
- Split imbalance tests
- Binning summary

## JSON Export

All dataset bias results are in the audit manifest:

```json
{
  "dataset_bias": {
    "proxy_correlations": {
      "correlations": {
        "gender": {
          "occupation_code": {
            "value": 0.48,
            "method": "cramers_v",
            "severity": "WARNING"
          }
        }
      }
    },
    "distribution_drift": {
      "drift_tests": {
        "income": {
          "statistic": 0.18,
          "p_value": 0.002,
          "test": "kolmogorov_smirnov",
          "drift_detected": true
        }
      }
    },
    "sampling_bias_power": {
      "power_by_group": {
        "gender": {
          "female": {
            "n": 35,
            "power": 0.42,
            "severity": "ERROR"
          }
        }
      }
    },
    "split_imbalance": { ... }
  }
}
```

## Troubleshooting

### "Dataset bias section missing from PDF"

**Cause**: No protected attributes specified.

**Fix**: Add `data.protected_attributes` to config.

### "All correlations show INFO severity"

**Cause**: No proxy features detected (good news!).

**Interpretation**: Dataset does not have obvious proxy features for protected attributes.

### "ERROR: Insufficient power for all groups"

**Cause**: Small dataset overall.

**Fixes**:

1. Collect more data (preferred)
2. Reduce number of protected groups (combine categories)
3. Flag results as exploratory, not definitive

### "Distribution drift detected for many features"

**Causes**:

1. Train and test from different time periods (temporal drift)
2. Different data sources
3. Non-random train/test split

**Fixes**:

1. Use recent data for both train and test
2. Ensure consistent data collection process
3. Use random stratified splitting

### "Custom bins not working"

**Check**:

1. Bins must be sorted: `[0, 30000, 60000]` not `[60000, 0, 30000]`
2. Bins must cover full data range
3. First bin should be ≤ min value, last bin ≥ max value

## Best Practices

### 1. Run Dataset Bias BEFORE Model Training

Dataset issues are easier to fix than model issues:

```bash
# Check data quality first
glassalpha audit --config config.yaml --dry-run  # Validates data only
```

### 2. Set Explicit Binning for Age

Don't rely on defaults:

```yaml
data:
  binning:
    age:
      strategy: domain_specific
      bins: [18, 25, 35, 50, 65, 100] # Explicit is better
```

### 3. Monitor Proxy Correlations

Track proxy correlations over time. Increasing correlations may indicate:

- Feature engineering introducing proxies
- Data collection changes
- Emerging bias patterns

### 4. Use Stratified Splitting

Always stratify by protected attributes:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=df['protected_attr'],  # Ensures balanced splits
    random_state=42
)
```

### 5. Document Binning Decisions

Record why you chose specific bin edges:

```yaml
# config.yaml
data:
  binning:
    age:
      strategy: domain_specific
      bins: [18, 25, 35, 50, 65, 100]
      # Rationale: Aligns with EEOC age discrimination thresholds
      # References: 29 U.S.C. § 631 (Age Discrimination in Employment Act)
```

## Related Features

- **[Fairness Metrics](../reference/fairness-metrics.md)**: Model-level fairness analysis
- **[Preprocessing Verification](preprocessing.md)**: Production artifact validation
- **[SR 11-7 Mapping](../compliance/sr-11-7-mapping.md)**: Section III.C.2 data quality

## Implementation Details

**Module**: `glassalpha.metrics.fairness.dataset`

**API**:

- `compute_dataset_bias_metrics()`: Main entry point
- `compute_proxy_correlations()`: Proxy correlation detection
- `compute_distribution_drift()`: Train/test drift analysis
- `compute_sampling_bias_power()`: Statistical power calculation
- `detect_split_imbalance()`: Chi-square tests for splits
- `bin_continuous_attribute()`: Binning utility

**Test Coverage**: 27 contract tests + 6 integration tests with German Credit dataset.

## Summary

Dataset bias detection catches problems at the source:

- ✅ **Proxy correlations**: Prevents indirect discrimination
- ✅ **Distribution drift**: Ensures train/test consistency
- ✅ **Statistical power**: Validates sample sizes
- ✅ **Split imbalance**: Prevents biased evaluation
- ✅ **Continuous binning**: Handles age and income attributes

Run automatically in every audit. No model can overcome biased data.
