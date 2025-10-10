# Fraud Detection Compliance Guide

Guide to using GlassAlpha for fraud detection model audits. Covers FCRA adverse action requirements, FTC fairness guidelines, and false positive equity analysis.

## Regulatory Context

Organizations using ML models for fraud detection face requirements balancing fraud prevention effectiveness with consumer protection and fairness.

### FCRA: Fair Credit Reporting Act

**Issuer**: Federal Trade Commission (FTC)

**Scope**: Use of consumer reports for fraud detection, account reviews, and adverse actions.

**Key Requirements**:

- **Adverse action notices**: If a consumer is flagged for fraud, they must receive notice
- **Specific reasons**: Notice must include specific reasons for the action
- **Right to dispute**: Consumer has right to contest fraud flags
- **Accuracy requirements**: Fraud models must be reasonably accurate

**GlassAlpha features**:

- Reason code generation for fraud flags
- Explainability (why flagged as fraudulent)
- Recourse analysis (how to contest/resolve)
- Precision metrics (accuracy of fraud predictions)

### FTC Fairness Guidelines

**Issuer**: Federal Trade Commission

**Guidance**: "Using Artificial Intelligence and Algorithms" (April 2020)

**Key Principles**:

- **Fairness**: Models must not discriminate based on protected characteristics
- **Transparency**: Companies must understand how models work
- **Robustness**: Models must be tested for accuracy and bias
- **Accountability**: Clear processes for consumer recourse

**GlassAlpha features**:

- Fairness metrics (false positive rate parity)
- Model documentation and explainability
- Robustness testing (shift scenarios, adversarial tests)
- Evidence pack for accountability

### Consumer Protection Laws

**Scope**: Federal and state laws protecting consumers from unfair practices.

**Key Concerns**:

- Disparate impact (fraud models disproportionately flag certain groups)
- False positive harm (legitimate transactions blocked, accounts frozen)
- Lack of recourse (difficult to contest fraud flags)
- Reputational damage (fraud flag affects credit or future transactions)

**GlassAlpha features**:

- False positive rate analysis by protected group
- Individual consistency testing
- Recourse recommendations
- Contestability (clear explanations)

## Common Fraud Detection Use Cases

### Transaction Fraud Models

Models that flag suspicious credit card or payment transactions in real-time.

**Compliance focus**: False positive equity, precision, explainability

**Key metrics**:

- False positive rate (FPR) parity across protected groups
- Precision at threshold (accuracy of fraud flags)
- Individual fairness (similar transactions, similar predictions)

**CLI workflow**:

```bash
glassalpha audit \
  --config transaction_fraud.yaml \
  --output transaction_fraud_audit.pdf \
  --fairness-focus fpr \
  --policy-gates policy/fraud_fairness.yaml \
  --strict
```

### Account Takeover Detection

Models that detect suspicious login attempts or account activity.

**Compliance focus**: False positive rate, user experience, recourse

**Key metrics**:

- FPR parity (who gets locked out of accounts)
- Precision (accuracy of account takeover flags)
- Recourse (how to regain access)

**CLI workflow**:

```bash
glassalpha audit \
  --config account_takeover.yaml \
  --output takeover_audit.pdf \
  --fairness-focus fpr \
  --check-shift geography:+0.1
```

### Application Fraud Models

Models that flag fraudulent applications (new accounts, loans, insurance claims).

**Compliance focus**: FCRA adverse action, equalized odds, reason codes

**Key metrics**:

- False positive rate (legitimate applicants flagged)
- Equalized odds (fairness for both fraudulent and legitimate)
- Reason codes (why application flagged)

**CLI workflow**:

```bash
glassalpha audit \
  --config application_fraud.yaml \
  --output application_fraud_audit.pdf \
  --fairness-focus equalized_odds \
  --policy-gates policy/adverse_action.yaml
```

## Typical Audit Workflow

### Step 1: Configure Audit

Create a config file with your model, data, and protected attributes:

```yaml
# transaction_fraud.yaml
model:
  path: "models/fraud_model.pkl"
  type: "xgboost"

data:
  train: "data/train.csv"
  test: "data/test.csv"
  target_column: "is_fraud"
  protected_attributes:
    - "geography" # Proxy for race/ethnicity
    - "merchant_category" # Could correlate with demographics
    - "device_type" # Could correlate with income

audit_profile: "fraud_detection"
random_seed: 42
strict_mode: true

explainer:
  type: "treeshap"
  background_samples: 1000

fairness:
  metrics: ["fpr_parity", "precision_parity", "equalized_odds"]
  threshold: 0.7 # Fraud flag threshold

recourse:
  enabled: true
  max_features: 3
  immutable: ["transaction_history", "geography"]

reason_codes:
  enabled: true
  top_n: 3
  exclude: ["geography", "device_type"] # Exclude proxies
```

### Step 2: Generate Audit Report

```bash
glassalpha audit \
  --config transaction_fraud.yaml \
  --output reports/fraud_model_2025Q4.pdf \
  --policy-gates policy/fraud_fairness.yaml \
  --strict
```

**Output artifacts**:

- `fraud_model_2025Q4.pdf` - Complete audit report
- `fraud_model_2025Q4.manifest.json` - Provenance manifest
- `policy_decision.json` - Pass/fail results for each gate

### Step 3: Review for Consumer Protection

**Checklist**:

- [ ] FPR parity: Similar false positive rates across protected groups (within 5%)
- [ ] Precision: Fraud flags are accurate (precision ≥ 0.70)
- [ ] Reason codes: Clear, specific, actionable
- [ ] Recourse: Clear path to contest fraud flags
- [ ] Individual fairness: Similar transactions receive similar treatment

**Red flags**:

- High FPR for specific groups (disproportionate harm)
- Low precision (many false positives, poor user experience)
- Vague reason codes ("unusual activity" without specifics)
- No clear recourse mechanism
- Individual fairness violations (similar transactions, different outcomes)

### Step 4: Generate Reason Codes

For adverse action notices (FCRA requirement):

```bash
glassalpha reasons \
  --model models/fraud_model.pkl \
  --instance data/flagged_transaction_12345.json \
  --output reason_codes/transaction_12345.json \
  --top-n 3 \
  --exclude geography device_type
```

**Example output**:

```json
{
  "transaction_id": "12345",
  "fraud_score": 0.85,
  "decision": "FLAGGED",
  "reason_codes": [
    {
      "rank": 1,
      "feature": "transaction_amount",
      "value": "$1,450.00",
      "contribution": 0.32,
      "explanation": "Transaction amount significantly higher than historical average ($75.00)"
    },
    {
      "rank": 2,
      "feature": "merchant_country",
      "value": "Romania",
      "contribution": 0.24,
      "explanation": "Merchant location inconsistent with cardholder's travel history"
    },
    {
      "rank": 3,
      "feature": "time_since_last_transaction",
      "value": "5 minutes",
      "contribution": 0.18,
      "explanation": "Multiple transactions in short time window (velocity pattern)"
    }
  ],
  "recourse": {
    "actions": [
      "Verify transaction with cardholder",
      "Confirm merchant legitimacy",
      "Check for authorized travel to merchant country"
    ]
  }
}
```

## Policy Gates for Fraud Detection

Example policy configuration for fairness and consumer protection:

```yaml
# policy/fraud_fairness.yaml
policy_name: "Fraud Detection Fairness Baseline"
version: "1.0"
citation: "FTC fairness guidelines, FCRA adverse action requirements"

gates:
  - name: "False Positive Rate Parity"
    clause: "Equitable false positive burden"
    metric: "fpr_difference"
    threshold: 0.05
    comparison: "less_than"
    severity: "error"

  - name: "Minimum Precision"
    clause: "Accuracy of fraud flags"
    metric: "precision"
    threshold: 0.70
    comparison: "greater_than"
    severity: "error"

  - name: "Reason Code Specificity"
    clause: "FCRA adverse action requirements"
    metric: "reason_code_quality"
    threshold: 0.8
    comparison: "greater_than"
    severity: "warning"

  - name: "Individual Fairness"
    clause: "Consistent treatment"
    metric: "individual_fairness_violation_rate"
    threshold: 0.05
    comparison: "less_than"
    severity: "warning"

  - name: "No Proxy Feature Over-reliance"
    clause: "Avoid disparate impact"
    metric: "max_proxy_importance"
    threshold: 0.10
    comparison: "less_than"
    severity: "warning"
```

## Stress Testing for Robustness

Test model performance under distribution shifts:

```bash
# Test sensitivity to geographic distribution changes
glassalpha audit \
  --config fraud.yaml \
  --check-shift geography:+0.1 \
  --fail-on-degradation 0.05

# Test sensitivity to device type distribution changes
glassalpha audit \
  --config fraud.yaml \
  --check-shift device_type:+0.05 \
  --fail-on-degradation 0.05
```

**Use case**: Ensure model remains fair if transaction patterns change (e.g., more mobile transactions, new geographies).

## Documentation Requirements

Internal compliance and external audits typically require:

1. **Model documentation** - Algorithm, features, training process
2. **Performance metrics** - Precision, recall, FPR, with confidence intervals
3. **Fairness analysis** - FPR parity, precision parity across groups
4. **Reason code quality** - Specificity, actionability, compliance with FCRA
5. **Recourse mechanisms** - How consumers contest fraud flags
6. **Ongoing monitoring** - Performance and fairness over time
7. **Adverse event tracking** - False positive complaints, disputes

**GlassAlpha audit PDF includes sections 1-5. Sections 6-7 require operational monitoring.**

## Common Audit Failures

### Failure 1: False Positive Rate Disparity

**Symptom**: One geographic region or demographic group has 2x the false positive rate

**Fairness issue**: Disproportionate burden on specific groups

**Fix**:

- Check for proxy features (geography, merchant category correlating with demographics)
- Adjust threshold per group (if legally permissible)
- Retrain with fairness constraints
- Document mitigation strategy

### Failure 2: Low Precision

**Symptom**: Precision < 0.70 (30%+ of fraud flags are false positives)

**Consumer protection issue**: Poor user experience, legitimate transactions blocked

**Fix**:

- Increase threshold (trade-off: fewer fraud catches)
- Improve features (reduce noise)
- Implement tiered response (low-score flags → soft decline, high-score → hard decline)

### Failure 3: Vague Reason Codes

**Symptom**: Reason codes like "unusual activity" or "risk score too high"

**FCRA issue**: Not specific enough for adverse action notices

**Fix**:

- Generate feature-specific reason codes
- Provide actionable recommendations
- Test reason code quality with compliance team

### Failure 4: Individual Fairness Violations

**Symptom**: Two nearly identical transactions receive very different fraud scores

**Consistency issue**: Unpredictable model behavior, fairness concern

**Fix**:

- Identify causes (noise, outliers, model instability)
- Apply smoothing or regularization
- Test individual fairness metrics after fix

## FCRA Adverse Action Requirements

When a consumer is flagged for fraud and experiences adverse action (blocked transaction, frozen account):

**Required notice elements**:

1. Statement that adverse action was taken
2. Name and contact info of organization
3. Statement that consumer report was used
4. **Specific reasons** for the action
5. Right to obtain copy of report
6. Right to dispute inaccurate information

**GlassAlpha reason codes satisfy requirement #4** with specific feature contributions.

### Example Adverse Action Notice Template

```
ADVERSE ACTION NOTICE

Date: [DATE]
Transaction ID: [ID]

We have flagged your transaction for potential fraud based on the following reasons:

1. Transaction amount ($1,450.00) significantly higher than your typical transactions ($75 average)
2. Merchant location (Romania) inconsistent with your travel history
3. Multiple transactions within short time window (5 minutes)

This decision was made in part using an automated fraud detection system.

You have the right to:
- Contest this decision by calling [PHONE] or visiting [WEBSITE]
- Request a manual review of your transaction
- Provide additional documentation to verify legitimacy

For more information about our fraud detection practices, see [PRIVACY POLICY LINK].

[ORGANIZATION NAME]
[ADDRESS]
[CONTACT INFO]
```

## Related Resources

- [Compliance Overview](index.md) - Role/industry navigation
- [Compliance Officer Workflow](../guides/compliance-workflow.md) - Evidence packs, policy gates
- [Reason Codes Guide](../guides/reason-codes.md) - FCRA-compliant adverse action notices
- [Recourse Guide](../guides/recourse.md) - Contestability and consumer recourse
- [Fraud Detection Example](../examples/fraud-detection-audit.md) - Complete fraud model audit

## Support

For fraud detection-specific questions:

- GitHub Discussions: [GlassAlpha/glassalpha/discussions](https://github.com/GlassAlpha/glassalpha/discussions)
- Email: [contact@glassalpha.com](mailto:contact@glassalpha.com)
- Documentation: [glassalpha.com](https://glassalpha.com)
