# Healthcare Compliance Guide

Guide to using GlassAlpha for healthcare ML model audits. Covers HIPAA, health equity mandates, clinical validation, and bias detection in medical decision support systems.

## Regulatory Context

Healthcare organizations using ML models for clinical decision support, risk stratification, or resource allocation face unique compliance requirements balancing privacy, equity, and clinical validity.

### HIPAA: Health Insurance Portability and Accountability Act

**Issuer**: Department of Health and Human Services (HHS)

**Scope**: Privacy and security of Protected Health Information (PHI)

**Key Requirements for ML Models**:

- De-identification of training data (Safe Harbor or Expert Determination)
- Business Associate Agreements for third-party model vendors
- Audit trails for PHI access
- Security safeguards for model storage and deployment

**GlassAlpha features**:

- Works with de-identified data (no PHI storage)
- Local-only processing (no network calls, no telemetry by default)
- Audit trail via provenance manifest (tracks data access)
- Evidence pack for HIPAA compliance audits

**Note**: GlassAlpha does not handle PHI directly. Use de-identified datasets or aggregate statistics.

### Health Equity Mandates

**CMS Quality Measures**:

- Health equity focus areas (maternal health, chronic disease management)
- Stratified reporting by race, ethnicity, language, disability
- Disparity reduction goals

**State Requirements**:

- California AB 2119: Health equity reporting for hospitals
- Massachusetts health equity law: Disparity reduction plans
- New York health equity assessment requirements

**GlassAlpha features**:

- Fairness metrics stratified by race, ethnicity, gender, age
- Statistical confidence intervals for group comparisons
- Dataset bias detection (disparities in training data)
- Intersectional analysis (2-way interactions)

### Clinical Validation Requirements

**FDA Guidance (Software as a Medical Device)**:

- Clinical validation: Does model improve patient outcomes?
- Analytical validation: Does model accurately predict target condition?
- Performance across subgroups (equity in diagnostic accuracy)

**IRB Requirements**:

- Informed consent for algorithm-assisted decisions
- Ongoing monitoring of model performance
- Adverse event reporting

**GlassAlpha features**:

- Performance metrics with confidence intervals (analytical validation)
- Subgroup analysis (equity in sensitivity, specificity, PPV, NPV)
- Calibration testing (predicted vs actual risk)
- Robustness testing (performance under distribution shift)

## Common Healthcare Use Cases

### Risk Stratification Models

Models that predict patient risk for hospitalization, readmission, or disease progression.

**Compliance focus**: Equity, calibration, clinical validity

**Key metrics**:

- Sensitivity/specificity parity across race/ethnicity groups
- Calibration: Do predicted risks match actual event rates?
- Positive predictive value (PPV) parity (avoid over-referral of one group)

**CLI workflow**:

```bash
glassalpha audit \
  --config risk_stratification.yaml \
  --output risk_model_audit.pdf \
  --strict \
  --fairness-focus equalized_odds \
  --check-shift race:+0.1
```

### Diagnostic Support Models

Models that assist with disease diagnosis (e.g., diabetic retinopathy detection, skin cancer classification).

**Compliance focus**: Equity in diagnostic accuracy, false negative rate

**Key metrics**:

- Sensitivity (true positive rate) parity across groups
- False negative rate parity (equity in missed diagnoses)
- PPV/NPV parity

**CLI workflow**:

```bash
glassalpha audit \
  --config diagnostic_support.yaml \
  --output diagnostic_audit.pdf \
  --fairness-focus fnr \
  --policy-gates policy/clinical_equity.yaml
```

### Resource Allocation Models

Models that prioritize patients for interventions (e.g., care management programs, transplant waitlists).

**Compliance focus**: Equalized odds, individual fairness, contestability

**Key metrics**:

- Selection rate parity (who gets selected for intervention)
- Equalized odds (fairness in both benefited and non-benefited groups)
- Individual fairness (similar patients, similar priority)

**CLI workflow**:

```bash
glassalpha audit \
  --config resource_allocation.yaml \
  --output allocation_audit.pdf \
  --fairness-focus equalized_odds \
  --policy-gates policy/allocation_fairness.yaml
```

## Typical Audit Workflow

### Step 1: Configure Audit with De-identified Data

Ensure data is HIPAA-compliant:

```yaml
# risk_stratification.yaml
model:
  path: "models/readmission_risk.pkl"
  type: "xgboost"

data:
  train: "data/train_deidentified.csv"
  test: "data/test_deidentified.csv"
  target_column: "readmitted_30d"
  protected_attributes:
    - "race"
    - "ethnicity"
    - "language"
    - "gender"
    - "age_group"

# Note: Data must be de-identified per HIPAA Safe Harbor
# No direct identifiers, dates generalized to year, zip codes >3 digits only

audit_profile: "healthcare_equity"
random_seed: 42
strict_mode: true

explainer:
  type: "treeshap"
  background_samples: 500

fairness:
  metrics: ["equalized_odds", "calibration_parity", "individual_fairness"]
  threshold: 0.5 # High-risk threshold

calibration:
  enabled: true
  bins: 10
```

### Step 2: Generate Audit Report

```bash
glassalpha audit \
  --config risk_stratification.yaml \
  --output reports/readmission_model_2025Q4.pdf \
  --policy-gates policy/clinical_equity.yaml \
  --strict
```

**Output artifacts**:

- `readmission_model_2025Q4.pdf` - Clinical validation report
- `readmission_model_2025Q4.manifest.json` - Provenance manifest
- `policy_decision.json` - Equity gate pass/fail results

### Step 3: Review for Health Equity

**Checklist**:

- [ ] Sensitivity parity: All groups have similar true positive rates (within 5%)
- [ ] Specificity parity: All groups have similar true negative rates
- [ ] PPV/NPV parity: Predictive value similar across groups
- [ ] Calibration by group: Predicted risk matches actual risk for all groups
- [ ] Sample sizes: Each group has n ≥ 30 for statistical power
- [ ] Confidence intervals: Documented for all group comparisons

**Red flags**:

- Large sensitivity differences (one group has many missed diagnoses)
- Poor calibration for minority groups (over/under-estimated risk)
- Small sample sizes (underpowered comparisons)
- Individual fairness violations (similar patients, different predictions)

### Step 4: Document Clinical Justification

Include in IRB or quality committee submission:

```yaml
# Clinical Validation Summary
performance_validation:
  overall_auc: 0.82
  sensitivity: 0.75
  specificity: 0.80
  calibration_ece: 0.032

equity_validation:
  race_groups:
    - group: "White"
      n: 5000
      sensitivity: 0.76
      specificity: 0.81
    - group: "Black/African American"
      n: 1200
      sensitivity: 0.74
      specificity: 0.79
      difference_from_white: "Not statistically significant (p=0.32)"
    - group: "Hispanic/Latino"
      n: 800
      sensitivity: 0.73
      specificity: 0.80
      difference_from_white: "Not statistically significant (p=0.45)"

clinical_interpretation:
  "Model demonstrates equitable performance across racial/ethnic groups.
  Sensitivity differences are within 3 percentage points and not statistically
  significant. Calibration testing shows accurate risk predictions for all groups."
```

## Policy Gates for Healthcare

Example policy configuration for health equity:

```yaml
# policy/clinical_equity.yaml
policy_name: "Clinical Health Equity Baseline"
version: "1.0"
citation: "CMS quality measures, institutional equity policy"

gates:
  - name: "Sensitivity Parity"
    clause: "Equitable diagnostic accuracy"
    metric: "equalized_odds_difference"
    threshold: 0.05
    comparison: "less_than"
    severity: "error"

  - name: "Calibration by Group"
    clause: "Accurate risk predictions for all groups"
    metric: "calibration_parity_difference"
    threshold: 0.05
    comparison: "less_than"
    severity: "warning"

  - name: "Minimum Group Sample Size"
    clause: "Statistical power for group comparisons"
    metric: "min_group_size"
    threshold: 30
    comparison: "greater_than"
    severity: "error"

  - name: "Individual Fairness"
    clause: "Consistent treatment"
    metric: "individual_fairness_violation_rate"
    threshold: 0.05
    comparison: "less_than"
    severity: "warning"

  - name: "No Dataset Bias Amplification"
    clause: "Model does not amplify existing disparities"
    metric: "disparity_amplification_ratio"
    threshold: 1.1
    comparison: "less_than"
    severity: "warning"
```

## Addressing Dataset Bias

Healthcare datasets often reflect existing disparities:

- Underrepresentation of minority groups
- Differential missingness (e.g., lab tests ordered less frequently for some groups)
- Historical bias (past discriminatory practices encoded in data)

**GlassAlpha dataset bias detection**:

```bash
# Analyze training data for bias before modeling
glassalpha dataset-bias \
  --data data/train_deidentified.csv \
  --protected-attrs race ethnicity gender \
  --output dataset_bias_report.pdf
```

**Checks**:

- Representation: Are groups balanced or severely imbalanced?
- Proxy correlation: Are features correlated with protected attributes?
- Outcome disparity: Do baseline outcome rates differ by group?
- Missingness patterns: Is missingness associated with protected attributes?

## Stress Testing for Health Equity

Test model robustness to demographic shifts:

```bash
# Test sensitivity to racial composition changes
glassalpha audit \
  --config risk_model.yaml \
  --check-shift race:+0.1 \
  --fail-on-degradation 0.03

# Test sensitivity to age distribution changes
glassalpha audit \
  --config risk_model.yaml \
  --check-shift age_group:+0.05 \
  --fail-on-degradation 0.03
```

**Use case**: Ensure model maintains equity if patient population demographics change (e.g., hospital serves more diverse community).

## Documentation Requirements

IRBs and quality committees typically require:

1. **Clinical validation** - Sensitivity, specificity, PPV, NPV with confidence intervals
2. **Equity analysis** - Performance stratified by race, ethnicity, gender, age
3. **Calibration testing** - Predicted vs actual risk by subgroup
4. **Dataset quality** - Representativeness, missingness patterns, bias detection
5. **Explainability** - How model makes predictions (feature importance, SHAP)
6. **Ongoing monitoring** - Performance tracking over time, disparity monitoring
7. **Adverse event plan** - How to identify and respond to model failures

**GlassAlpha audit PDF includes sections 1-5. Sections 6-7 require operational monitoring.**

## Common Audit Failures

### Failure 1: Sensitivity Disparity

**Symptom**: One racial/ethnic group has significantly lower sensitivity (more missed diagnoses)

**Health equity issue**: Inequitable diagnostic accuracy

**Fix**:

- Check sample size (underpowered comparison?)
- Check dataset bias (group underrepresented in training?)
- Retrain with stratified sampling or fairness constraints
- Document mitigation in IRB submission

### Failure 2: Calibration by Group

**Symptom**: Model overestimates or underestimates risk for specific groups

**Clinical issue**: Inaccurate risk predictions lead to under/over-treatment

**Fix**:

- Apply group-specific calibration (isotonic regression per group)
- Retest calibration after adjustment
- Document calibration approach in model card

### Failure 3: Dataset Bias Amplification

**Symptom**: Model amplifies existing disparities (disparity ratio > 1.1)

**Example**: Training data has 10% outcome disparity, model predictions have 15% disparity

**Fix**:

- Investigate features driving amplification
- Check for proxy features (correlated with protected attributes)
- Retrain with fairness constraints or adversarial debiasing
- Document bias mitigation strategy

### Failure 4: Small Sample Sizes

**Symptom**: Subgroup has n < 30, wide confidence intervals, underpowered comparisons

**Statistical issue**: Cannot reliably assess equity

**Options**:

- Collect more data from underrepresented groups
- Aggregate groups if clinically appropriate (e.g., combine age brackets)
- Document limitation and plan for expanded validation
- Defer deployment until adequate validation possible

## HIPAA Compliance Considerations

### Data Handling

GlassAlpha operates on de-identified data only:

- Safe Harbor method: Remove 18 identifiers per HIPAA rules
- Expert Determination: Statistical certification by qualified expert
- Aggregate statistics: Summary metrics without individual records

### Audit Trails

Provenance manifest tracks:

- Dataset hash (SHA256 of de-identified data)
- Model lineage (training code, package versions)
- Audit timestamp and operator
- Configuration parameters

**For HIPAA audit**: Evidence pack provides tamper-evident audit trail.

### Third-Party Risk

If using GlassAlpha for covered entity:

- Business Associate Agreement (BAA) not required (no PHI access)
- Local-only processing (no network calls, no cloud storage)
- Open-source (code is auditable)

**Best practice**: Run GlassAlpha on-premises or in organization's HIPAA-compliant environment.

## Related Resources

- [Compliance Overview](index.md) - Role/industry navigation
- [Compliance Officer Workflow](../guides/compliance-workflow.md) - Evidence packs, policy gates
- [ML Engineer Workflow](../guides/ml-engineer-workflow.md) - Implementation details
- [Dataset Bias Detection](../guides/dataset-bias.md) - Bias analysis before modeling
- [Fairness Metrics Reference](../reference/fairness-metrics.md) - Technical definitions
- [Healthcare Bias Detection Example](../examples/healthcare-bias-detection.md) - Complete walkthrough

## Support

For healthcare-specific questions:

- GitHub Discussions: [GlassAlpha/glassalpha/discussions](https://github.com/GlassAlpha/glassalpha/discussions)
- Email: [contact@glassalpha.com](mailto:contact@glassalpha.com)
- Documentation: [glassalpha.com](https://glassalpha.com)
