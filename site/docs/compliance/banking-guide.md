# Banking & Credit Compliance Guide

Guide to using GlassAlpha for banking and credit risk model audits. Covers SR 11-7, ECOA, FCRA, and fair lending requirements.

## Regulatory Context

Banking institutions using ML models for credit decisions face strict compliance requirements:

### SR 11-7: Model Risk Management

**Issuer**: Federal Reserve (April 2011)

**Scope**: All models used in bank operations, including credit scoring, loan pricing, fraud detection, and risk assessment.

**Key Requirements**:

- Comprehensive model documentation with clear methodology
- Validation testing by independent parties
- Ongoing performance monitoring and outcomes analysis
- Documentation of model limitations and assumptions
- Reproducibility and auditability

**GlassAlpha mapping**: See [SR 11-7 Technical Mapping](sr-11-7-mapping.md) for clause-by-clause artifact coverage.

### ECOA: Equal Credit Opportunity Act

**Issuer**: Consumer Financial Protection Bureau (CFPB)

**Scope**: Prohibits discrimination in any aspect of credit transactions based on protected characteristics (race, color, religion, national origin, sex, marital status, age, receipt of public assistance).

**Key Requirements**:

- Adverse action notices must include specific reasons for denial
- Creditors must provide statements of specific reasons (reason codes)
- Models must not use protected attributes as direct input features
- Disparate impact must be monitored and addressed

**GlassAlpha features**:

- Reason code generation: `glassalpha reasons --model <path> --instance <id>`
- Fairness analysis across protected groups
- Protected attribute isolation testing

### FCRA: Fair Credit Reporting Act

**Issuer**: Federal Trade Commission (FTC)

**Scope**: Regulates collection, dissemination, and use of consumer credit information.

**Key Requirements**:

- Accuracy and fairness in credit reporting
- Adverse action notices for credit denials
- Consumer right to dispute inaccurate information
- Model predictions must be explainable and contestable

**GlassAlpha features**:

- Explanation generation (SHAP values, feature contributions)
- Recourse analysis (what changes would improve credit decision)
- Audit trail for reproducibility

## Common Banking Use Cases

### Credit Scoring Models

Models that predict probability of default or creditworthiness.

**Compliance focus**: Fairness, calibration accuracy, reason codes

**Key metrics**:

- Calibration: Do predicted default rates match actual default rates?
- Fairness at threshold: Are approval rates equitable across protected groups?
- Individual consistency: Similar applicants receive similar decisions

**CLI workflow**:

```bash
glassalpha audit \
  --config credit_scoring.yaml \
  --output credit_audit.pdf \
  --strict \
  --fairness-focus approval_rate \
  --check-shift gender:+0.1
```

### Loan Pricing Models

Models that set interest rates based on risk assessment.

**Compliance focus**: Disparate impact, calibration, explainability

**Key metrics**:

- Rate parity: Average rates across protected groups (risk-adjusted)
- Calibration: Predicted risk aligns with actual loss rates
- Recourse: Clear path to better rates (e.g., improve credit score)

**CLI workflow**:

```bash
glassalpha audit \
  --config loan_pricing.yaml \
  --output pricing_audit.pdf \
  --policy-gates policy/sr_11_7_banking.yaml \
  --strict
```

### Fraud Detection Models

Models that flag suspicious transactions or applications.

**Compliance focus**: False positive equity, disparate impact, contestability

**Key metrics**:

- False positive rate parity across groups
- Precision at operating threshold
- Recourse (why flagged, how to contest)

**CLI workflow**:

```bash
glassalpha audit \
  --config fraud_detection.yaml \
  --output fraud_audit.pdf \
  --fairness-focus fpr \
  --policy-gates policy/fraud_fairness.yaml
```

## Typical Audit Workflow

### Step 1: Configure Audit

Create a config file with your model, data, and protected attributes:

```yaml
# credit_audit.yaml
model:
  path: "models/credit_model.pkl"
  type: "xgboost"

data:
  train: "data/train.csv"
  test: "data/test.csv"
  target_column: "default"
  protected_attributes:
    - "gender"
    - "age_group"
    - "race"

# Direct configuration for SR 11-7 compliance
random_seed: 42
strict_mode: true

explainer:
  type: "treeshap"
  background_samples: 1000

fairness:
  metrics: ["demographic_parity", "equalized_odds", "calibration"]
  threshold: 0.5

recourse:
  enabled: true
  max_features: 3
  immutable: ["gender", "race", "age"]
```

### Step 2: Generate Audit Report

```bash
glassalpha audit \
  --config credit_audit.yaml \
  --output reports/credit_model_audit_2025Q4.pdf \
  --policy-gates policy/sr_11_7_banking.yaml \
  --strict
```

**Output artifacts**:

- `credit_model_audit_2025Q4.pdf` - Complete audit report
- `credit_model_audit_2025Q4.manifest.json` - Provenance manifest (hashes, versions, seeds)
- `policy_decision.json` - Pass/fail results for each gate

### Step 3: Export Evidence Pack

For regulator submission or internal audit:

```bash
glassalpha export-evidence-pack \
  --audit reports/credit_model_audit_2025Q4.pdf \
  --output evidence_packs/credit_model_2025Q4.zip
```

**Evidence pack contents**:

- Audit PDF
- Provenance manifest (with SHA256 checksums)
- Policy decision log
- Configuration files
- Dataset schema

### Step 4: Verify Evidence Pack

Independent validators can verify integrity:

```bash
glassalpha verify-evidence-pack \
  --input evidence_packs/credit_model_2025Q4.zip
```

Checks:

- All checksums match (SHA256)
- Manifest is complete and valid
- Policy gates are documented
- Reproducibility information present

## Policy Gates for Banking

Example policy configuration for SR 11-7 compliance:

```yaml
# policy/sr_11_7_banking.yaml
policy_name: "SR 11-7 Banking Baseline"
version: "1.0"
citation: "Federal Reserve SR 11-7 (April 2011)"

gates:
  - name: "Minimum Calibration Quality"
    clause: "III.B.1 - Model development testing"
    metric: "expected_calibration_error"
    threshold: 0.05
    comparison: "less_than"
    severity: "error"

  - name: "Fairness: Demographic Parity"
    clause: "V - Fair lending compliance"
    metric: "demographic_parity_difference"
    threshold: 0.1
    comparison: "less_than"
    severity: "error"

  - name: "Fairness: Equalized Odds"
    clause: "V - Fair lending compliance"
    metric: "equalized_odds_difference"
    threshold: 0.15
    comparison: "less_than"
    severity: "warning"

  - name: "Minimum Group Sample Size"
    clause: "III.B.2 - Statistical power"
    metric: "min_group_size"
    threshold: 30
    comparison: "greater_than"
    severity: "error"

  - name: "Robustness to Shift"
    clause: "III.A.3 - Ongoing monitoring"
    metric: "max_metric_degradation_under_shift"
    threshold: 0.05
    comparison: "less_than"
    severity: "warning"
```

## Documentation Requirements

SR 11-7 auditors typically require:

1. **Model documentation** - Algorithm description, training process, feature engineering
2. **Validation testing** - Performance metrics, calibration analysis, stability tests
3. **Outcomes analysis** - Actual vs predicted performance, fairness metrics
4. **Ongoing monitoring** - Drift detection, performance tracking, recalibration triggers
5. **Limitations and assumptions** - Data quality requirements, edge cases, known biases
6. **Reproducibility** - Seeds, versions, data hashes, environment details

**GlassAlpha audit PDF includes all sections above.**

## Common Audit Failures

### Failure 1: Poor Calibration

**Symptom**: Predicted default rates don't match actual default rates (ECE > 0.05)

**SR 11-7 clause**: III.B.1 (Model development testing)

**Fix**:

- Recalibrate using isotonic regression or Platt scaling
- Retest with `glassalpha audit --config <updated_config>`
- Document calibration method in model card

### Failure 2: Fairness Violations

**Symptom**: Demographic parity difference > 0.1 or equalized odds difference > 0.15

**ECOA requirement**: No disparate impact on protected groups

**Fix**:

- Check for proxy features (zip code → race correlation)
- Adjust decision threshold per group (if legally permissible)
- Use fairness-aware training methods
- Document mitigation strategy in audit report

### Failure 3: Inadequate Reason Codes

**Symptom**: Reason codes are generic ("credit score too low") rather than specific

**FCRA requirement**: Specific reasons for adverse action

**Fix**:

- Generate detailed reason codes: `glassalpha reasons --model <path> --instance <id>`
- Include top 3-5 feature contributions
- Exclude protected attributes from reason codes
- Test reason code quality with compliance team

### Failure 4: Non-Reproducible Results

**Symptom**: Rerunning audit produces different metrics or PDF output

**SR 11-7 clause**: IV.A (Reproducibility requirements)

**Fix**:

- Set explicit random seed in config: `random_seed: 42`
- Run in strict mode: `--strict` flag enforces determinism
- Lock environment: use constraints file or Docker image
- Verify with: Run audit twice, compare manifest SHA256 hashes

## Related Resources

- [SR 11-7 Technical Mapping](sr-11-7-mapping.md) - Detailed clause-by-clause coverage
- [Compliance Officer Workflow](../guides/compliance-workflow.md) - Evidence pack generation and verification
- [Reason Codes Guide](../guides/reason-codes.md) - ECOA-compliant adverse action notices
- [Recourse Guide](../guides/recourse.md) - Counterfactual recommendations for applicants
- [German Credit Audit Example](../examples/german-credit-audit.md) - Complete credit scoring audit walkthrough

## Support

For banking-specific questions:

- GitHub Discussions: [GlassAlpha/glassalpha/discussions](https://github.com/GlassAlpha/glassalpha/discussions)
- Email: [contact@glassalpha.com](mailto:contact@glassalpha.com)
- Documentation: [glassalpha.com](https://glassalpha.com)
