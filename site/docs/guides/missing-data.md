# Handling Missing Data in Protected Attributes

Guide to handling missing values (NaN) in protected attributes for fairness analysis.

---

## Quick Summary

- **Default behavior**: NaN → "Unknown" (third category)
- **No imputation**: GlassAlpha never imputes protected attributes
- **Fairness impact**: "Unknown" treated as separate demographic group
- **Best practice**: Address upstream; use "Unknown" category for audit transparency

---

## Default Behavior

GlassAlpha automatically maps NaN to "Unknown" category:

```python
import numpy as np
import pandas as pd
import glassalpha as ga

# Protected attribute with missing values
gender = pd.Series([0, 1, np.nan, 1, 0, np.nan])

result = ga.audit.from_model(
    model=model,
    X=X_test,
    y=y_test,
    protected_attributes={"gender": gender},
    random_seed=42
)

# "Unknown" treated as third category
print(result.fairness.group_gender_Unknown.tpr)
```

### Why "Unknown" Instead of Imputation

1. **Transparency**: Auditors see missing data explicitly
2. **No assumptions**: Don't assume missing values match distribution
3. **Fairness risk**: "Unknown" may be proxy for disadvantaged group
4. **Regulatory**: CFPB/NAIC guidance requires transparency about missing protected attributes

---

## Common Scenarios

### 1. Self-Reported Demographics (Voluntary)

**Scenario**: Users optionally provide gender/race (e.g., job applications).

**Challenge**: Missing may indicate discomfort with disclosure → potential proxy for protected status.

```python
# Example: 15% missing gender
gender = pd.Series(["F", "M", None, "M", "F", None, ...])

result = ga.audit.from_model(...)

# Check fairness for "Unknown" group
unknown_metrics = result.fairness.group_gender_Unknown
if unknown_metrics["tpr"] < 0.8:  # Substantially worse outcomes
    print("⚠️ 'Unknown' group has lower TPR - investigate upstream")
```

**Best practice**:

1. Measure: % missing, outcomes by "Unknown" group
2. Investigate: Why missing? Correlated with outcomes?
3. Document: Include "Unknown" group in fairness report

### 2. Historical Data (Incompletely Recorded)

**Scenario**: Legacy systems didn't collect protected attributes consistently.

**Challenge**: Missing may correlate with time period, geography, outcomes.

```python
# Example: 40% missing race in older records
race = pd.Series([np.nan] * 400 + ["White", "Black", "Hispanic", ...] * 600)

result = ga.audit.from_model(...)

# Check if "Unknown" correlates with other features
if result.fairness.group_race_Unknown.n_samples > 0.3 * len(X_test):
    print("⚠️ High missingness (>30%) - check data quality")
```

**Best practice**:

1. Stratify: Analyze by time period, geography
2. Document: % missing by cohort
3. Consider: Separate audit for complete vs incomplete records

### 3. Legal Restrictions (Cannot Collect)

**Scenario**: Some jurisdictions prohibit collecting race/ethnicity.

**Challenge**: Cannot audit fairness without protected attributes.

```python
# Example: EU prohibits collecting race (GDPR)
# Use proxies cautiously: geography, language, surname analysis
# But document limitations explicitly

result = ga.audit.from_model(
    model=model,
    X=X_test,
    y=y_test,
    # No protected_attributes
    random_seed=42
)

# Note: Cannot audit race fairness (legal restriction)
```

**Best practice**:

1. Document: Legal constraints on protected attribute collection
2. Proxies: Use only with expert guidance (risk of incorrect assignment)
3. Alternative: Audit on other axes (gender, age, disability)

---

## Validation Rules

### 1. Too Much Missing Data

```python
import glassalpha as ga

gender = pd.Series([0, 1, np.nan, np.nan, np.nan, ...])  # 60% missing

if gender.isna().mean() > 0.50:
    print("⚠️ Warning: >50% missing - fairness analysis unreliable")

# Still runs, but results questionable
result = ga.audit.from_model(...)
```

**Threshold guidance**:

- <10% missing: Acceptable
- 10-30% missing: Document in report
- 30-50% missing: Investigate data quality
- > 50% missing: Fairness analysis unreliable

### 2. All Missing Data

```python
# Example: All NaN
gender = pd.Series([np.nan] * len(X_test))

# Runs but fairness metrics meaningless (single group)
result = ga.audit.from_model(...)
print(result.fairness)  # Only "Unknown" group
```

**Best practice**: Skip fairness analysis if all values missing.

---

## Auditing the "Unknown" Group

### Check "Unknown" Group Metrics

```python
result = ga.audit.from_model(...)

# Get "Unknown" group metrics
unknown = result.fairness.group_gender_Unknown

print(f"Sample size: {unknown['n_samples']}")
print(f"TPR: {unknown['tpr']:.3f}")
print(f"FPR: {unknown['fpr']:.3f}")
print(f"Precision: {unknown['precision']:.3f}")

# Compare to known groups
male = result.fairness.group_gender_M
female = result.fairness.group_gender_F

tpr_gap = max(male["tpr"], female["tpr"]) - unknown["tpr"]
if tpr_gap > 0.10:
    print("⚠️ 'Unknown' group has substantially lower TPR")
```

### Statistical Significance

```python
# Small sample sizes unreliable
if unknown["n_samples"] < 30:
    print("⚠️ 'Unknown' group too small for reliable metrics (n<30)")
```

**Guideline**: Require n≥30 per group for reliable fairness metrics.

---

## Upstream Solutions

### 1. Improve Collection Process

**Before GlassAlpha audit:**

1. Review: Why missing? Confusing questions? Legal concerns?
2. Simplify: Clear, optional questions
3. Incentivize: Explain how data used (fairness monitoring)
4. Monitor: Track response rates by demographic

### 2. Proxy Variables (Use Cautiously)

**Example**: Use surname + geography to infer race/ethnicity.

**Risks**:

- Incorrect assignment (privacy violation)
- Stereotyping
- Legal issues (ECOA, FCRA)

**When acceptable**:

- Expert-validated methodology
- Statistical correction for errors
- Documented limitations
- Legal review

**GlassAlpha support**:

```python
# If using proxies, document explicitly in manifest
result = ga.audit.from_model(...)

# Add note to manifest
custom_manifest = result.manifest.copy()
custom_manifest["protected_attributes_note"] = (
    "Race inferred via BISG (Bayesian Improved Surname Geocoding). "
    "Estimated accuracy: 85%. See methodology doc."
)
```

### 3. Separate Audits for Complete Data

```python
# Option 1: Drop rows with missing protected attributes
complete_mask = ~gender.isna()
X_complete = X_test[complete_mask]
y_complete = y_test[complete_mask]
gender_complete = gender[complete_mask]

result_complete = ga.audit.from_model(
    model=model,
    X=X_complete,
    y=y_complete,
    protected_attributes={"gender": gender_complete},
    random_seed=42
)

# Option 2: Audit both and compare
result_all = ga.audit.from_model(...)  # Includes "Unknown"
result_complete = ga.audit.from_model(...)  # Complete only

# Check if conclusions differ
if abs(result_all.fairness.demographic_parity_max_diff -
       result_complete.fairness.demographic_parity_max_diff) > 0.05:
    print("⚠️ Missing data affects fairness conclusions")
```

---

## Reporting Missing Data

### In Audit Report

Include in report narrative:

```
## Protected Attributes

- Gender:
  - Female: 412 (41.2%)
  - Male: 503 (50.3%)
  - Unknown: 85 (8.5%)

- Race:
  - White: 645 (64.5%)
  - Black: 198 (19.8%)
  - Hispanic: 102 (10.2%)
  - Unknown: 55 (5.5%)

## Missing Data Impact

The "Unknown" groups represent self-reported demographics with opt-out.
Analysis shows:

- Gender Unknown: n=85, TPR=0.72 (vs 0.81 for known groups)
- Race Unknown: n=55, TPR=0.68 (vs 0.79 for known groups)

Recommendation: Investigate why "Unknown" groups have lower outcomes.
Potential causes: selection bias, unmeasured confounders, or model bias.
```

### In Manifest

```python
result.manifest["protected_attributes_completeness"] = {
    "gender": {"complete": 0.915, "unknown": 0.085},
    "race": {"complete": 0.945, "unknown": 0.055}
}
```

---

## Advanced: Custom Missing Handling

### Multiple Missing Categories

If you have multiple missing types (e.g., "Declined", "Unknown", "Not Collected"):

```python
# Recode before audit
gender_recoded = gender.copy()
gender_recoded = gender_recoded.replace({
    "Declined": "Unknown",
    "Not Collected": "Unknown",
    np.nan: "Unknown"
})

result = ga.audit.from_model(
    protected_attributes={"gender": gender_recoded},
    ...
)
```

### Document Custom Mapping

```python
result.manifest["protected_attributes_missing_policy"] = {
    "gender": "All missing types mapped to 'Unknown' category",
    "categories": ["F", "M", "Unknown"],
    "missing_types_merged": ["NaN", "Declined", "Not Collected"]
}
```

---

## Checklist

**Before audit:**

- [ ] Check missingness rate per protected attribute
- [ ] Document why missing (voluntary, data quality, legal)
- [ ] Consider if "Unknown" is meaningful category

**During audit:**

- [ ] Verify "Unknown" group has n≥30 for reliable metrics
- [ ] Check "Unknown" group outcomes vs known groups
- [ ] Document findings in report narrative

**After audit:**

- [ ] Investigate if "Unknown" has worse outcomes
- [ ] Consider upstream improvements to collection
- [ ] Document limitations in final report

---

## Related

- **[Audit Entry Points](../reference/api/audit-entry-points.md)** - from_model() API reference
- **[Fairness 101](fairness-101.md)** - Fairness concepts
- **[Probability Requirements](probability-requirements.md)** - When probabilities needed
- **[Data Requirements](../getting-started/custom-data.md)** - Data format requirements

---

## Support

- **GitHub Issues**: [Report bugs](https://github.com/GlassAlpha/glassalpha/issues)
- **Discussions**: [Ask questions](https://github.com/GlassAlpha/glassalpha/discussions)
