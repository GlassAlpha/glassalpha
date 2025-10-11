# Insurance Compliance Guide

Guide to using GlassAlpha for insurance underwriting and pricing model audits. Covers NAIC Model Act #670, state regulations, and anti-discrimination requirements.

## Regulatory Context

Insurance companies using ML models for underwriting, pricing, and claims decisions face regulatory requirements focused on preventing unfair discrimination while allowing actuarially justified risk differentiation.

### NAIC Model Act #670: Unfair Discrimination

**Issuer**: National Association of Insurance Commissioners (NAIC)

**Scope**: Prohibits unfair discrimination in insurance rates, underwriting, and claims practices.

**Key Principles**:

- Rates must not be excessive, inadequate, or unfairly discriminatory
- Discrimination based on race, color, creed, national origin is prohibited
- Risk classification must be actuarially justified
- Similar risks should be treated similarly

**GlassAlpha features**:

- Calibration testing: Do predicted risk levels match actual claim rates?
- Rate parity analysis: Are premiums equitable across protected groups after risk adjustment?
- Individual consistency: Similar applicants receive similar quotes

### State Regulations

Insurance is regulated at the state level with varying requirements:

**California (Proposition 103)**:

- Rate factors must be cost-based
- Prior approval required for rate changes
- Explicit fairness requirements

**New York (Regulation 187)**:

- Best interest standard for life insurance
- Suitability requirements
- Enhanced disclosure

**Colorado (SB 21-169)**:

- External model risk management for life insurance
- Algorithm transparency requirements
- Consumer notice of AI use

**GlassAlpha features**:

- Model documentation (transparent methodology)
- Explainability (SHAP values, feature contributions)
- Reason codes (why this rate was assigned)

### Anti-Discrimination Laws

Federal and state laws prohibit discrimination based on protected characteristics:

- Title VII (employment-related insurance)
- Fair Housing Act (homeowners/renters insurance)
- Genetic Information Nondiscrimination Act (GINA)
- State-specific protected classes (marital status, sexual orientation, etc.)

**GlassAlpha features**:

- Protected attribute isolation (detect proxy features)
- Fairness metrics (demographic parity, equalized odds)
- Dataset bias detection (proxy correlation analysis)

## Common Insurance Use Cases

### Pricing Models

Models that determine premium rates based on risk assessment.

**Compliance focus**: Rate parity, calibration, actuarial justification

**Key metrics**:

- Calibration: Do predicted loss rates match actual loss experience?
- Rate parity: Are premiums equitable across protected groups (risk-adjusted)?
- Individual consistency: Similar risk profiles receive similar rates

**CLI workflow**:

```bash
glassalpha audit \
  --config insurance_pricing.yaml \
  --output pricing_audit.pdf \
  --strict \
  --fairness-focus rate_parity \
  --check-shift race:+0.1
```

### Underwriting Models

Models that determine eligibility and coverage limits.

**Compliance focus**: Equalized odds, disparate impact, explainability

**Key metrics**:

- Approval rate parity across protected groups
- False positive rate parity (incorrectly denied coverage)
- Precision at threshold (accuracy of denials)

**CLI workflow**:

```bash
glassalpha audit \
  --config underwriting.yaml \
  --output underwriting_audit.pdf \
  --policy-gates policy/naic_underwriting.yaml \
  --fairness-focus approval_rate
```

### Claims Models

Models that predict claim severity, detect fraud, or automate claims decisions.

**Compliance focus**: Treatment equity, false positive rate, contestability

**Key metrics**:

- False positive rate parity (incorrect fraud flags)
- Claim payout equity across groups
- Recourse (why flagged, how to contest)

**CLI workflow**:

```bash
glassalpha audit \
  --config claims_fraud.yaml \
  --output claims_audit.pdf \
  --fairness-focus fpr \
  --policy-gates policy/claims_fairness.yaml
```

## Typical Audit Workflow

### Step 1: Configure Audit

Create a config file with your model, data, and protected attributes:

```yaml
# insurance_pricing.yaml
model:
  path: "models/pricing_model.pkl"
  type: "xgboost"

data:
  train: "data/train.csv"
  test: "data/test.csv"
  target_column: "loss_ratio"
  protected_attributes:
    - "gender"
    - "age_group"
    - "zip_code" # Potential proxy for race/ethnicity

# Direct configuration: "insurance_pricing"
random_seed: 42
strict_mode: true

explainer:
  type: "treeshap"
  background_samples: 1000

fairness:
  metrics: ["rate_parity", "calibration_parity", "individual_fairness"]
  threshold: null # Regression model, no threshold

calibration:
  enabled: true
  bins: 10

recourse:
  enabled: true
  max_features: 3
  immutable: ["gender", "race", "age"]
```

### Step 2: Generate Audit Report

```bash
glassalpha audit \
  --config insurance_pricing.yaml \
  --output reports/pricing_model_2025Q4.pdf \
  --policy-gates policy/naic_pricing.yaml \
  --strict
```

**Output artifacts**:

- `pricing_model_2025Q4.pdf` - Complete audit report
- `pricing_model_2025Q4.manifest.json` - Provenance manifest
- `policy_decision.json` - Pass/fail results for each gate

### Step 3: Review for Actuarial Justification

**Checklist**:

- [ ] Risk factors are statistically significant predictors of loss
- [ ] Calibration: Predicted loss ratios match actual loss ratios (ECE < 0.05)
- [ ] Rate parity: Similar risks pay similar premiums (after risk adjustment)
- [ ] No proxy discrimination: Features like zip code don't encode race/ethnicity
- [ ] Individual fairness: Similar policyholders receive similar rates

**Red flags**:

- Poor calibration (ECE > 0.05): Rates not actuarially justified
- Rate disparities without risk justification
- High correlation between zip code and race (proxy discrimination)
- Large individual fairness violations (similar risks, different rates)

### Step 4: Document Actuarial Justification

Include in regulatory submission:

```yaml
# Actuarial Justification Statement
risk_factors:
  credit_score:
    correlation_with_loss: 0.45
    statistical_significance: p < 0.001
    actuarial_basis: "Strong predictor of claim frequency and severity"

  vehicle_age:
    correlation_with_loss: 0.32
    statistical_significance: p < 0.001
    actuarial_basis: "Older vehicles have higher repair costs"

  zip_code:
    correlation_with_loss: 0.18
    statistical_significance: p < 0.05
    actuarial_basis: "Reflects regional accident rates and repair costs"
    proxy_analysis: "Correlation with race: 0.08 (below 0.1 threshold)"
```

## Policy Gates for Insurance

Example policy configuration for NAIC compliance:

```yaml
# policy/naic_pricing.yaml
policy_name: "NAIC Model Act #670 Pricing Baseline"
version: "1.0"
citation: "NAIC Model Act #670, state anti-discrimination laws"

gates:
  - name: "Calibration Quality"
    clause: "Rates must be actuarially justified"
    metric: "expected_calibration_error"
    threshold: 0.05
    comparison: "less_than"
    severity: "error"

  - name: "Rate Parity (Risk-Adjusted)"
    clause: "Similar risks, similar rates"
    metric: "demographic_parity_difference"
    threshold: 0.10
    comparison: "less_than"
    severity: "warning"

  - name: "Individual Fairness"
    clause: "Consistent treatment"
    metric: "individual_fairness_violation_rate"
    threshold: 0.05
    comparison: "less_than"
    severity: "warning"

  - name: "Proxy Feature Detection"
    clause: "No unfair discrimination"
    metric: "max_proxy_correlation"
    threshold: 0.15
    comparison: "less_than"
    severity: "error"

  - name: "Statistical Significance"
    clause: "Risk factors must predict loss"
    metric: "min_feature_importance"
    threshold: 0.01
    comparison: "greater_than"
    severity: "warning"
```

## Stress Testing for Rate Stability

Test model robustness to demographic shifts:

```bash
# Test sensitivity to gender composition changes
glassalpha audit \
  --config pricing.yaml \
  --check-shift gender:+0.1 \
  --fail-on-degradation 0.03

# Test sensitivity to age distribution changes
glassalpha audit \
  --config pricing.yaml \
  --check-shift age_group:+0.05 \
  --fail-on-degradation 0.03
```

**Use case**: Ensure rates remain stable if demographic mix of applicant pool changes over time.

## Documentation Requirements

State regulators typically require:

1. **Model documentation** - Algorithm description, training process, feature definitions
2. **Actuarial justification** - Statistical evidence that risk factors predict loss
3. **Calibration testing** - Predicted vs actual loss ratios by risk segment
4. **Fairness analysis** - Rate parity and disparate impact testing
5. **Proxy detection** - Analysis of correlations between features and protected attributes
6. **Individual consistency** - Similar risks receive similar treatment
7. **Ongoing monitoring** - Rate adequacy and fairness over time

**GlassAlpha audit PDF includes all sections above.**

## Common Audit Failures

### Failure 1: Poor Calibration

**Symptom**: Predicted loss ratios don't match actual loss experience (ECE > 0.05)

**Actuarial issue**: Rates not actuarially justified, violates NAIC requirement

**Fix**:

- Recalibrate model using isotonic regression
- Retest with updated model
- Document calibration method in actuarial memo

### Failure 2: Proxy Discrimination

**Symptom**: Zip code or other feature highly correlated with race (correlation > 0.15)

**Regulatory issue**: Potential unfair discrimination

**Fix**:

- Check if correlation reflects actuarial risk (regional accident rates) or proxy discrimination
- If proxy: Remove feature or use fairness constraints during training
- Document analysis in actuarial justification

### Failure 3: Rate Disparities Without Risk Justification

**Symptom**: Protected groups pay different average premiums despite similar risk profiles

**Regulatory issue**: Unfair discrimination

**Fix**:

- Investigate: Is disparity due to risk factors or model bias?
- If bias: Retrain with fairness constraints
- If risk factors: Document actuarial justification explicitly

### Failure 4: Individual Fairness Violations

**Symptom**: Two policyholders with nearly identical risk profiles receive significantly different rates

**Regulatory issue**: Inconsistent treatment, potential discrimination

**Fix**:

- Identify causes (noise, edge cases, bias)
- Apply smoothing or monotonicity constraints
- Test consistency metrics after fix

## State-Specific Considerations

### California (Prop 103)

**Additional requirements**:

- Prior approval for rate changes
- Rate factors must be cost-based
- Public rate hearings for large insurers

**GlassAlpha artifacts**: Calibration testing (cost basis), actuarial justification (rate factor analysis)

### New York (Reg 187)

**Additional requirements**:

- Best interest standard (life insurance)
- Enhanced disclosure of AI use

**GlassAlpha artifacts**: Explainability section (SHAP values), reason codes, model card

### Colorado (SB 21-169)

**Additional requirements**:

- External model risk management
- Algorithm transparency
- Consumer notice of AI use

**GlassAlpha artifacts**: Evidence pack (for external validation), model card, audit PDF

## Related Resources

- [Compliance Overview](index.md) - Role/industry navigation
- [Compliance Officer Workflow](../guides/compliance-workflow.md) - Evidence packs, policy gates
- [ML Engineer Workflow](../guides/ml-engineer-workflow.md) - Implementation details
- [Dataset Bias Detection](../guides/dataset-bias.md) - Proxy correlation analysis
- [Shift Testing Guide](../guides/shift-testing.md) - Demographic robustness

## Support

For insurance-specific questions:

- GitHub Discussions: [GlassAlpha/glassalpha/discussions](https://github.com/GlassAlpha/glassalpha/discussions)
- Email: [contact@glassalpha.com](mailto:contact@glassalpha.com)
- Documentation: [glassalpha.com](https://glassalpha.com)
