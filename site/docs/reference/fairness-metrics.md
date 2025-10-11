# Fairness Metrics Reference

!!! info "Part of Understanding Section"
This page explains fairness concepts in depth. For practical usage, see:

    - **[Detecting Dataset Bias](../guides/dataset-bias.md)** - Pre-model fairness checks
    - **[Testing Demographic Shifts](../guides/shift-testing.md)** - Robustness testing
    - **[Quick Start](../getting-started/quickstart.md)** - Get started in 5 minutes

# Fairness Metrics Reference

Comprehensive fairness analysis with statistical confidence intervals, individual consistency testing, and intersectional bias detection.

## Overview

GlassAlpha provides three levels of fairness analysis:

1. **Group Fairness (E10)**: Demographic parity, equal opportunity, with 95% confidence intervals
2. **Intersectional Fairness (E5.1)**: Hidden bias at demographic intersections (e.g., race×gender)
3. **Individual Fairness (E11)**: Consistency score, matched pairs, counterfactual testing

All metrics include:

- Statistical confidence intervals (bootstrap)
- Sample size adequacy checks
- Statistical power analysis
- Deterministic computation (reproducible with seeds)

## Which Fairness Metric Should I Use?

```mermaid
graph TB
    Start[Choose Metric]
    Start --> Context{Context?}

    Context -->|Lending/Credit| Credit{What matters?}
    Credit -->|Equal approval rates| DP[Demographic Parity]
    Credit -->|Equal opportunity| EO[Equal Opportunity]

    Context -->|Hiring| Hire[Demographic Parity + Predictive Parity]
    Context -->|Healthcare| Health[Equal Opportunity + Calibration]
    Context -->|Criminal Justice| CJ[Predictive Equality + Calibration]

    Context -->|Not sure| General{What's the harm?}
    General -->|Denying benefits| EO2[Equal Opportunity]
    General -->|False accusations| PE[Predictive Equality]
    General -->|Unequal treatment| DP2[Demographic Parity]

    style DP fill:#d4edda
    style EO fill:#d4edda
    style Hire fill:#fff3cd
    style Health fill:#fff3cd
```

**Quick Guide**:

- **Lending/Credit**: Equal Opportunity (ensure qualified applicants have equal chance)
- **Hiring**: Demographic Parity (equal consideration across groups)
- **Healthcare**: Equal Opportunity + Calibration (accurate risk across groups)
- **Criminal Justice**: Predictive Equality (equal false positive rates)

**When in doubt**: Use all three (Demographic Parity, Equal Opportunity, Predictive Parity) and compare results.

## Group Fairness with Confidence Intervals (E10)

### Metrics Computed

For each protected attribute group:

| Metric                        | Definition                  | Fairness Criterion  |
| ----------------------------- | --------------------------- | ------------------- |
| **TPR** (True Positive Rate)  | Recall for positive class   | Equal opportunity   |
| **FPR** (False Positive Rate) | Type I error rate           | Predictive equality |
| **Precision**                 | Positive predictive value   | Predictive parity   |
| **Recall**                    | Sensitivity                 | Equal opportunity   |
| **Selection Rate**            | Fraction predicted positive | Demographic parity  |

### Confidence Intervals

**Method**: Bootstrap resampling (default: 1000 samples)
**Interval**: Percentile method (default: 95%)
**Determinism**: Seeded random sampling for reproducibility

### Configuration

```yaml
metrics:
  fairness:
    metrics:
      - demographic_parity
      - equal_opportunity
      - predictive_parity

    # Confidence interval settings
    compute_confidence_intervals: true # Default: true
    n_bootstrap: 1000 # Bootstrap samples (default: 1000)
    confidence_level: 0.95 # CI level (default: 0.95)
```

### Sample Size Adequacy

Automatic warnings for small sample sizes:

| Sample Size (n) | Severity   | Interpretation | Action            |
| --------------- | ---------- | -------------- | ----------------- |
| n < 10          | 🔴 ERROR   | Unreliable     | Collect more data |
| 10 ≤ n < 30     | ⚠️ WARNING | Low confidence | Flag uncertainty  |
| n ≥ 30          | ✅ OK      | Adequate       | Proceed           |

### Sample Size Calculator

**Rule of Thumb**: Minimum samples needed per group:

| Desired Precision | Min Disparity to Detect | Samples Needed per Group |
| ----------------- | ----------------------- | ------------------------ |
| High (CI ±0.02)   | 0.05                    | 200+                     |
| Medium (CI ±0.05) | 0.10                    | 100+                     |
| Low (CI ±0.10)    | 0.15                    | 30+                      |

**Formula**: For 80% power to detect 10% disparity at α=0.05:

```
n ≈ 16 * (p * (1-p)) / (effect_size²)
where p ≈ 0.5, effect_size = 0.10
n ≈ 100 per group
```

**Practical Guidance**:

- **n < 30**: Report only descriptive statistics, flag as inconclusive
- **30 ≤ n < 100**: Report with wide confidence intervals, note limited power
- **n ≥ 100**: Standard reporting with confidence intervals

### Statistical Power

For each group, power calculation estimates:

**Power** = Probability of detecting 10% disparity given current sample size

| Power   | Interpretation                               |
| ------- | -------------------------------------------- |
| < 0.5   | Insufficient power (high Type II error risk) |
| 0.5-0.7 | Marginal power                               |
| ≥ 0.7   | Adequate power                               |

### PDF Output Example

```
Group Fairness Analysis

Protected Attribute: gender

Metric       | Male (n=165) | Female (n=35) | Max Disparity | 95% CI          | Status
-------------|--------------|---------------|---------------|-----------------|--------
TPR          | 0.68         | 0.52          | 0.16          | [0.08, 0.24]    | WARNING
FPR          | 0.12         | 0.15          | 0.03          | [-0.08, 0.14]   | PASS
Precision    | 0.75         | 0.71          | 0.04          | [-0.12, 0.19]   | PASS
Selection    | 0.45         | 0.38          | 0.07          | [-0.06, 0.20]   | PASS

Sample Size Warnings:
  ⚠️ Female: n=35 (WARNING - low statistical power: 0.42)
```

### Interpreting Confidence Intervals

**Narrow CI** ([0.08, 0.12]):

- Precise estimate
- Sufficient sample size
- High confidence in disparity magnitude

**Wide CI** ([-0.10, 0.30]):

- Imprecise estimate
- Small sample size or high variance
- Low confidence in exact disparity

**CI includes zero** ([-0.05, 0.12]):

- Disparity not statistically significant
- Could be no true disparity
- Or insufficient power to detect it

**CI excludes zero** ([0.08, 0.24]):

- Statistically significant disparity
- True difference likely exists
- Action recommended

### JSON Export

```json
{
  "fairness_analysis": {
    "gender": {
      "metrics": {
        "tpr": {
          "male": 0.68,
          "female": 0.52,
          "disparity": 0.16,
          "ci": {
            "ci_lower": 0.08,
            "ci_upper": 0.24,
            "confidence_level": 0.95,
            "n_bootstrap": 1000
          }
        }
      },
      "sample_size_warnings": {
        "female": {
          "n": 35,
          "severity": "WARNING"
        }
      },
      "statistical_power": {
        "female": 0.42
      }
    }
  }
}
```

### Visual Examples

**What Does Disparity Look Like in Practice?**

> **Note**: Visual examples with screenshots coming soon. For now, see [German Credit Audit walkthrough](../examples/german-credit-audit.md) for real report examples showing these scenarios.

#### Example 1: No Bias Detected (PASS)

**Scenario**: Credit model demographic parity analysis

- **Max disparity**: 0.03 (well within 0.10 threshold)
- **Confidence interval**: [-0.02, 0.08] (narrow, precise)
- **Sample sizes**: All groups n>100
- **Interpretation**: No evidence of bias. CI is narrow and doesn't exclude zero by much, but disparity is minimal.
- **Action**: Model passes fairness check

#### Example 2: Potential Bias (WARNING)

**Scenario**: Hiring model equal opportunity analysis

- **Max disparity**: 0.12 (exceeds 0.10 threshold)
- **Confidence interval**: [0.05, 0.19] (excludes zero)
- **Sample sizes**: All groups n>80
- **Interpretation**: Statistically significant disparity. CI excludes zero, indicating true difference likely exists.
- **Action**: Investigate model for bias, review training data, consider fairness constraints

#### Example 3: Insufficient Data (CAUTION)

**Scenario**: Insurance risk model with small minority group

- **Small group**: n=18 samples
- **Disparity**: 0.15
- **Confidence interval**: [-0.15, 0.30] (very wide, includes zero)
- **Interpretation**: Inconclusive - wide CI indicates high uncertainty. Can't determine if disparity is real or noise.
- **Action**: (1) Collect more data for small group, (2) Document limitation in report, (3) Consider aggregating with similar groups

## Intersectional Fairness (E5.1) {#intersectional-fairness-e51}

### What It Is

Bias at the intersection of multiple protected attributes:

- **Example**: Black women may face unique discrimination not captured by race or gender alone
- **Kimberlé Crenshaw (1989)**: Coined "intersectionality" to describe compounded discrimination

### How It Works

Creates all combinations (Cartesian product) of protected attributes:

**Example**: gender × race

```
Groups:
  - male_white
  - male_black
  - female_white
  - female_black
```

Computes full fairness metrics (TPR, FPR, precision, recall, selection rate) for each intersectional group.

### Configuration

```yaml
data:
  protected_attributes:
    - gender
    - race
    - age_group

  # Specify intersections to analyze
  intersections:
    - "gender*race" # 2-way: gender × race
    - "age_group*race" # 2-way: age × race
```

**Syntax**: Use `*` to combine attributes (e.g., `"attr1*attr2"`)

**Limit**: Currently supports 2-way intersections (3+ deferred to enterprise)

### PDF Output Example

```
Intersectional Fairness Analysis

Intersection: gender × race

Group          | n   | TPR  | 95% CI       | FPR  | 95% CI       | Selection Rate
---------------|-----|------|--------------|------|--------------|---------------
male_white     | 82  | 0.72 | [0.61, 0.83] | 0.10 | [0.04, 0.16] | 0.48
male_black     | 35  | 0.64 | [0.47, 0.81] | 0.15 | [0.03, 0.27] | 0.41
female_white   | 18  | 0.55 | [0.28, 0.82] | 0.12 | [0.00, 0.28] | 0.39
female_black   | 8   | 0.38 | [0.00, 0.75] | 0.20 | [0.00, 0.50] | 0.25

Sample Size Warnings:
  ⚠️ female_white: n=18 (WARNING)
  🔴 female_black: n=8 (ERROR - unreliable)

Disparity Metrics:
  Max TPR difference: 0.34 (male_white vs female_black)
  Max FPR difference: 0.10 (female_black vs male_white)
```

### Disparity Metrics

For each metric, GlassAlpha computes:

- **Max-min difference**: Largest absolute difference between any two groups
- **Max-min ratio**: Largest ratio between any two groups

**Example**:

- TPR range: 0.38 (female_black) to 0.72 (male_white)
- Max difference: 0.34
- Max ratio: 1.89 (male_white / female_black)

### When to Use

**Use intersectional analysis when:**

- Protected attributes may interact (gender + race, age + disability)
- Historical discrimination affects specific combinations
- Legal requirements (e.g., Title VII intersectional claims)

**Skip intersectional analysis when:**

- Total sample size < 200 (most intersections will have insufficient power)
- No hypothesis of interaction effects
- Exploratory phase (start with group fairness first)

### Sample Size Challenge

Intersections multiply sample requirements:

| Groups                  | Samples per Group | Total Required |
| ----------------------- | ----------------- | -------------- |
| 2 (gender)              | 30                | 60             |
| 4 (gender × race)       | 30                | 120            |
| 8 (gender × race × age) | 30                | 240            |

**Recommendation**: Need ≥30 samples per intersection for reliable metrics.

### JSON Export

```json
{
  "intersectional_fairness": {
    "gender*race": {
      "groups": {
        "male_white": {
          "n": 82,
          "metrics": {
            "tpr": 0.72,
            "tpr_ci": { "ci_lower": 0.61, "ci_upper": 0.83 }
          }
        },
        "female_black": {
          "n": 8,
          "metrics": {
            "tpr": 0.38,
            "tpr_ci": { "ci_lower": 0.0, "ci_upper": 0.75 }
          }
        }
      },
      "disparity": {
        "tpr_max_diff": 0.34,
        "tpr_max_ratio": 1.89
      },
      "sample_size_warnings": {
        "female_black": { "n": 8, "severity": "ERROR" }
      }
    }
  }
}
```

## Individual Fairness (E11) {#individual-fairness-e11}

### What It Is

**Principle**: Similar individuals should receive similar predictions.

**Legal basis**:

- Equal Protection Clause (14th Amendment)
- Civil Rights Act Title VI/VII (disparate treatment)
- ECOA (Equal Credit Opportunity Act)

### Three Tests

#### 1. Consistency Score

**Definition**: Lipschitz-like metric measuring prediction stability for similar individuals.

**Method**:

1. Compute pairwise distances between all individuals (feature space)
2. Identify "similar pairs" (distance below threshold)
3. Measure prediction differences for similar pairs
4. Consistency score = 1 - (mean prediction difference)

**Higher score = more consistent = more fair**

#### 2. Matched Pairs Report

**Definition**: Identifies specific individuals with similar features but different predictions.

**Purpose**: Flag potential disparate treatment cases for manual review.

**Output**: List of (individual_A, individual_B) pairs where:

- Feature distance < threshold
- Prediction difference > threshold
- Protected attributes differ

#### 3. Counterfactual Flip Test

**Definition**: Tests if changing only protected attribute changes prediction.

**Method**:

1. For each individual, create counterfactual by flipping protected attribute
2. Re-predict with counterfactual
3. Measure prediction change
4. Disparate treatment rate = fraction with significant change

**High rate = model relies on protected attribute = discriminatory**

### Configuration

```yaml
metrics:
  fairness:
    individual_fairness:
      enabled: true

      # Distance metric for similarity
      distance_metric: euclidean # euclidean or mahalanobis

      # Similarity threshold (percentile of pairwise distances)
      similarity_percentile: 90 # Top 10% most similar pairs

      # Prediction difference threshold
      prediction_threshold: 0.10 # 10% difference
```

### PDF Output Example

```
Individual Fairness Analysis

Consistency Score: 0.82 (Good)
  • Distance metric: Euclidean
  • Similar pairs: 1,245 (top 10% by distance)
  • Mean prediction difference: 0.18
  • Max prediction difference: 0.45

Matched Pairs Report (5 flagged):
  Pair 1: Individual 42 vs Individual 89
    Feature distance: 0.08
    Prediction difference: 0.32
    Protected attribute differs: gender (male vs female)

  Pair 2: Individual 103 vs Individual 157
    Feature distance: 0.12
    Prediction difference: 0.28
    Protected attribute differs: race (white vs black)

Counterfactual Flip Test:
  Protected attribute: gender
  Disparate treatment rate: 8.5% (17 of 200 cases)
  Mean prediction change: 0.04
  Max prediction change: 0.22
```

### Consistency Score Interpretation

| Score     | Interpretation        | Status     |
| --------- | --------------------- | ---------- |
| ≥ 0.90    | Excellent consistency | ✅ PASS    |
| 0.80-0.90 | Good consistency      | ✅ PASS    |
| 0.70-0.80 | Fair consistency      | ⚠️ WARNING |
| < 0.70    | Poor consistency      | 🔴 FAIL    |

### Matched Pairs - Legal Risk

**High risk pairs**:

- Feature distance < 0.10 (very similar)
- Prediction difference > 0.20 (substantially different)
- Protected attributes differ

**Example disparate treatment case**: Two applicants with identical credit history, income, and employment, but different race, receive substantially different loan approval probabilities.

**Action**: Manual review of flagged pairs for legitimate reasons for difference.

### Counterfactual Flip Rate Interpretation

| Rate  | Interpretation                       | Action                      |
| ----- | ------------------------------------ | --------------------------- |
| < 5%  | Minimal protected attribute reliance | ✅ PASS                     |
| 5-10% | Moderate reliance                    | ⚠️ Investigate              |
| > 10% | Strong reliance                      | 🔴 Audit for discrimination |

### Distance Metrics

**Euclidean** (default):

- Simple distance: sqrt(Σ(xᵢ - yᵢ)²)
- Treats all features equally
- Good for normalized features

**Mahalanobis**:

- Accounts for feature correlations: sqrt((x-y)ᵀ Σ⁻¹ (x-y))
- Better for correlated features
- More computationally expensive

```yaml
metrics:
  fairness:
    individual_fairness:
      distance_metric: mahalanobis # For correlated features
```

### Performance

Individual fairness requires pairwise distance computation:

- **Complexity**: O(n²) for n samples
- **Runtime**: ~2-5 seconds for n=200, ~30-60 seconds for n=1000
- **Optimization**: Vectorized with NumPy for speed

**Recommendation**: For large datasets (n > 5000), consider sampling:

```yaml
metrics:
  fairness:
    individual_fairness:
      max_samples: 1000 # Random sample for pairwise computation
```

### JSON Export

```json
{
  "individual_fairness": {
    "consistency_score": {
      "score": 0.82,
      "distance_metric": "euclidean",
      "n_similar_pairs": 1245,
      "mean_prediction_diff": 0.18,
      "max_prediction_diff": 0.45
    },
    "matched_pairs": [
      {
        "individual_a": 42,
        "individual_b": 89,
        "distance": 0.08,
        "prediction_diff": 0.32,
        "protected_attr_differs": "gender"
      }
    ],
    "counterfactual_flip": {
      "protected_attr": "gender",
      "disparate_treatment_rate": 0.085,
      "mean_change": 0.04,
      "max_change": 0.22
    }
  }
}
```

## Deterministic Execution

All fairness metrics are fully deterministic with explicit seeds:

```yaml
reproducibility:
  random_seed: 42 # Required for reproducible bootstrap CIs
```

**What's deterministic:**

- Bootstrap sample selection
- Pairwise distance computation order
- Matched pairs ordering
- Counterfactual flip order

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

  protected_attributes:
    - gender
    - age_group
    - foreign_worker

  # Intersectional analysis
  intersections:
    - "gender*age_group"
    - "gender*foreign_worker"

metrics:
  fairness:
    # Group fairness
    metrics:
      - demographic_parity
      - equal_opportunity
      - predictive_parity

    # Statistical confidence
    compute_confidence_intervals: true
    n_bootstrap: 1000
    confidence_level: 0.95

    # Individual fairness
    individual_fairness:
      enabled: true
      distance_metric: euclidean
      similarity_percentile: 90
      prediction_threshold: 0.10
```

### Run Audit

```bash
glassalpha audit --config audit_config.yaml --output audit.pdf
```

### Expected Output

```
Running fairness analysis...
  ✓ Group fairness: 3 protected attributes
     gender: 2 groups (n=165, n=35)
     age_group: 5 groups (n=45, n=72, n=58, n=18, n=7)
     foreign_worker: 2 groups (n=167, n=33)

  ⚠️ Sample size warnings:
     age_group[65+]: n=7 (ERROR - unreliable)
     foreign_worker[yes]: n=33 (WARNING - low power)

  ✓ Intersectional fairness: 2 intersections
     gender*age_group: 10 groups
     gender*foreign_worker: 4 groups

  ✓ Individual fairness:
     Consistency score: 0.84 (Good)
     Matched pairs: 3 flagged
     Disparate treatment rate: 6.5%

Fairness Summary:
  Warnings: 2
  Errors: 1
  Action required: Collect more data for age_group[65+]
```

## Troubleshooting

### "Fairness section missing from PDF"

**Cause**: No protected attributes specified.

**Fix**: Add `data.protected_attributes` to config.

### "Wide confidence intervals"

**Cause**: Small sample size or high variance.

**Fixes**:

1. Collect more data (preferred)
2. Increase bootstrap samples: `n_bootstrap: 5000`
3. Flag uncertainty in report

### "ERROR: All intersectional groups have low n"

**Cause**: Total dataset too small for intersections.

**Fix**: Need total n > (# groups × 30). For 8 intersectional groups, need n > 240.

### "Individual fairness too slow"

**Cause**: Large dataset (n > 2000).

**Fixes**:

1. Sample: `individual_fairness.max_samples: 1000`
2. Disable if not needed: `individual_fairness.enabled: false`

### "Disparate treatment rate is 0%"

**Possible causes**:

1. Model truly doesn't use protected attribute (good!)
2. Protected attribute encoded in proxies (bad - check proxy correlations)
3. Threshold too high

**Check**: Review proxy correlations in dataset bias analysis.

## Best Practices

### 1. Start with Group Fairness

Don't jump to intersectional/individual before checking groups:

```bash
# First audit: Group fairness only
glassalpha audit --config base_config.yaml
```

### 2. Set Explicit Thresholds

Document fairness thresholds based on legal/policy requirements:

```yaml
metrics:
  fairness:
    thresholds:
      demographic_parity: 0.10 # Max 10% selection rate difference
      equal_opportunity: 0.05 # Max 5% TPR difference
```

### 3. Use Confidence Intervals for Decision-Making

**Don't**: Rely on point estimates alone
**Do**: Check if disparity CI excludes zero

**Example**:

- TPR disparity = 0.08, CI = [-0.02, 0.18]
- **Interpretation**: Not statistically significant (CI includes 0)
- **Action**: Collect more data before concluding disparity exists

### 4. Document Small Sample Sizes

Always flag low-power groups in reports:

```
Note: Fairness metrics for group X (n=12) have low statistical power.
Results should be interpreted with caution. Recommend collecting
additional data (target: n≥30) for reliable disparity detection.
```

### 5. Combine with Dataset Bias Analysis

Individual fairness complements dataset bias:

- **Dataset bias**: Catches proxies and sampling issues
- **Individual fairness**: Catches model reliance on protected attributes

Run both for comprehensive fairness assessment.

## Related Features

- **[Dataset Bias Detection](../guides/dataset-bias.md)**: Pre-model fairness checks
- **[Shift Testing](../guides/shift-testing.md)**: Robustness under demographic changes
- **[Calibration Analysis](calibration.md)**: Probability accuracy by group
- **[SR 11-7 Mapping](../compliance/sr-11-7-mapping.md)**: Section V fairness requirements

## Implementation Details

**Modules**:

- `glassalpha.metrics.fairness.runner`: Main fairness pipeline
- `glassalpha.metrics.fairness.individual`: Individual fairness (E11)
- `glassalpha.metrics.fairness.intersectional`: Intersectional fairness (E5.1)
- `glassalpha.metrics.fairness.bootstrap`: Confidence intervals (E10)

**Test Coverage**: 64 contract tests covering determinism, edge cases, and integration.

## Summary

Three-level fairness analysis:

1. **Group Fairness (E10)**: Standard metrics with statistical confidence

   - TPR, FPR, precision, recall, selection rate
   - Bootstrap 95% CIs
   - Sample size warnings

2. **Intersectional Fairness (E5.1)**: Hidden bias detection

   - 2-way interactions (gender×race, age×income, etc.)
   - Disparity metrics (max-min difference/ratio)
   - Per-intersection sample warnings

3. **Individual Fairness (E11)**: Consistency and disparate treatment
   - Consistency score (similar individuals → similar predictions)
   - Matched pairs report (flag disparate treatment cases)
   - Counterfactual flip test (protected attribute reliance)

All deterministic, reproducible, and export-ready for regulatory submission.
