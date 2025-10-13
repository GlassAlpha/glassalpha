# Model Validator Workflow

Guide for model validators, internal auditors, and third-party consultants performing independent verification of ML model audits.

## Overview

This guide is for validators who need to:

- Verify audit completeness and accuracy
- Challenge model assumptions and findings
- Identify red flags in documentation
- Reproduce audit results independently
- Provide validation opinions for regulatory submissions

**Not a validator?** For implementation, see [ML Engineer Workflow](ml-engineer-workflow.md). For compliance workflows, see [Compliance Officer Workflow](compliance-workflow.md).

## Key Capabilities

### Evidence Pack Verification

Validate integrity of audit artifacts:

- SHA256 checksum verification
- Manifest completeness checks
- Reproducibility validation

### Independent Reproduction

Reproduce audit results from scratch:

- Deterministic outputs under same conditions
- Platform-independent reproducibility
- Full audit trail from data to report

### Challenge Testing

Stress test model claims:

- Robustness to distribution shifts
- Sensitivity to threshold changes
- Edge case exploration

## Typical Workflows

### Workflow 1: Initial Audit Review

**Scenario**: You receive an audit report and documentation bundle for validation.

#### Step 1: Verify evidence bundle integrity

```bash
# Extract and verify bundle
unzip evidence_packs/credit_model_2025Q4.zip -d validation_check/
cd validation_check/

# Verify all checksums
sha256sum -c SHA256SUMS.txt

# Expected output:
# credit_model_audit_2025Q4.pdf: OK
# credit_model_audit_2025Q4.manifest.json: OK
# audit_config.yaml: OK
```

**What to verify**:

- All checksums pass (SHA256)
- Manifest is complete and valid JSON
- Configuration file is present and valid YAML
- Random seed is documented in manifest
- Package versions are documented in manifest
- Git SHA is present (for reproducibility)

**Note**: Automated `verify-evidence-pack` command is planned for v0.3.0 (Enhancement E3).

**Red flags**:

- Checksum mismatches (evidence tampering or corruption)
- Missing manifest fields (incomplete documentation)
- No random seed in manifest (non-reproducible)
- No package versions (environment not documented)
- Missing configuration file (can't reproduce)

#### Step 2: Review audit PDF systematically

Use this checklist for comprehensive review:

**Section 1: Executive Summary**

- [ ] Model purpose clearly stated
- [ ] Key findings summarized with numbers
- [ ] Limitations explicitly called out
- [ ] Date, version, and operator documented

**Section 2: Data & Preprocessing**

- [ ] Data sources documented
- [ ] Sample sizes reported (train, test, per group)
- [ ] Protected attributes identified
- [ ] Preprocessing steps documented
- [ ] Dataset bias analysis included (if applicable)

**Section 3: Model Documentation**

- [ ] Algorithm type clearly stated
- [ ] Feature list complete
- [ ] Training process documented
- [ ] Hyperparameters recorded

**Section 4: Performance Metrics**

- [ ] Key metrics reported (accuracy, precision, recall, AUC)
- [ ] Confidence intervals provided (95% CI)
- [ ] Sample sizes adequate (n ≥ 30 per group)
- [ ] Test set used (not train set)

**Section 5: Calibration**

- [ ] Calibration curve included
- [ ] ECE (Expected Calibration Error) reported
- [ ] Confidence intervals for ECE
- [ ] Calibration tested by group (if fairness concern)

**Section 6: Fairness Analysis**

- [ ] All protected attributes analyzed
- [ ] Multiple fairness metrics (demographic parity, equalized odds)
- [ ] Statistical significance tested
- [ ] Confidence intervals for group differences
- [ ] Sample sizes per group adequate

**Section 7: Explainability**

- [ ] Feature importance reported
- [ ] SHAP values or similar method used
- [ ] Example explanations provided
- [ ] Protected attributes not dominant features

**Section 8: Reason Codes (if applicable)**

- [ ] Specific, not vague ("low credit score" not "risk too high")
- [ ] Top 3-5 factors provided
- [ ] Protected attributes excluded from reason codes
- [ ] Actionable (customer can understand and respond)

**Section 9: Recourse (if applicable)**

- [ ] Counterfactual recommendations provided
- [ ] Immutable features respected
- [ ] Realistic changes (not "increase age")
- [ ] Cost/effort documented

**Section 10: Robustness (if applicable)**

- [ ] Shift testing results
- [ ] Adversarial perturbation results
- [ ] Stability metrics

**Section 11: Limitations**

- [ ] Assumptions explicitly stated
- [ ] Edge cases documented
- [ ] Known biases acknowledged
- [ ] Mitigation strategies proposed

**Section 12: Audit Trail**

- [ ] Random seeds documented
- [ ] Package versions listed
- [ ] Git commit SHA included
- [ ] Dataset hashes provided

#### Step 3: Flag issues for further investigation

**Common red flags**:

| Red Flag                         | Implication                           | Next Step                                     |
| -------------------------------- | ------------------------------------- | --------------------------------------------- |
| No confidence intervals          | Can't assess statistical significance | Request rerun with CI estimation              |
| Small group sizes (n<30)         | Underpowered comparisons              | Request more data or aggregate groups         |
| Wide confidence intervals        | High uncertainty, unreliable metrics  | Request more data or acknowledge limitation   |
| Missing limitations section      | Overconfident, incomplete             | Request explicit limitation statement         |
| Vague reason codes               | FCRA/ECOA violation                   | Request feature-specific reasons              |
| Poor calibration (ECE>0.10)      | Predictions inaccurate                | Request recalibration or threshold adjustment |
| Large fairness violations (>15%) | Disparate impact concern              | Request mitigation strategy                   |
| No reproducibility info          | Can't validate                        | Request manifest with seeds/versions          |

#### Step 4: Document findings

Create validation memo:

```markdown
# Validation Memo: Credit Model Audit (2025 Q4)

**Validator**: [Your Name]
**Date**: [Date]
**Model**: Credit Scoring Model v2.5
**Audit Date**: 2025-10-15

## Overall Assessment

[PASS / PASS WITH CONDITIONS / FAIL]

## Key Findings

### Strengths

- Comprehensive fairness analysis across 3 protected attributes
- Statistical confidence intervals provided for all key metrics
- Explicit limitations section

### Concerns

- Small sample size for Hispanic group (n=45, recommend n≥100)
- Wide confidence interval for calibration (ECE: 0.042 ± 0.028)
- No robustness testing to demographic shifts

### Recommendations

1. Collect additional data for Hispanic group to improve statistical power
2. Implement ongoing calibration monitoring
3. Conduct shift testing before production deployment

## Detailed Review

[Section-by-section findings...]

## Reproducibility Check

[See Workflow 2 below]
```

### Workflow 2: Independent Reproduction

**Scenario**: Verify audit results are reproducible and deterministic.

#### Step 1: Set up environment

```bash
# Create clean environment
python -m venv validator_env
source validator_env/bin/activate

# Install exact versions from manifest
pip install glassalpha==0.2.0  # Use version from manifest
pip install -r constraints.txt  # Use constraints from evidence pack
```

**Best practice**: Use Docker to match exact environment:

```bash
# Extract Dockerfile from evidence pack (if provided)
docker build -t audit-validator .
docker run -v $(pwd)/output:/output audit-validator
```

#### Step 2: Reproduce audit

```bash
# Extract config and data from evidence pack
unzip evidence_packs/credit_model_2025Q4.zip

# Run audit with same config
glassalpha audit \
  --config config/audit_config.yaml \
  --output reproduced_audit.pdf \
  --strict

# Generate manifest
mv reproduced_audit.manifest.json reproduced.manifest.json
```

#### Step 3: Compare results

```bash
# Compare PDF checksums (should match if deterministic)
sha256sum original_audit.pdf reproduced_audit.pdf

# Compare manifests
diff original.manifest.json reproduced.manifest.json
```

**Expected**: Byte-identical PDFs and manifests (if environment and config match).

**If different**:

- Compare environment: Check package versions, platform (macOS vs Linux)
- Compare data: Check dataset hashes in manifests
- Compare config: Ensure exact same random seed
- Compare timestamps: May differ (not a concern if metrics match)

#### Step 4: Validate key metrics

Even if PDFs differ slightly (due to platform differences), key metrics should match:

```bash
# Extract metrics from both audits
glassalpha inspect --audit original_audit.pdf --output original_metrics.json
glassalpha inspect --audit reproduced_audit.pdf --output reproduced_metrics.json

# Compare metrics
python compare_metrics.py original_metrics.json reproduced_metrics.json
```

**Tolerance**: Metrics should match within floating-point precision (< 0.001 difference).

### Workflow 3: Challenge Testing

**Scenario**: Stress test model to identify weaknesses not covered in original audit.

#### Challenge 1: Threshold Sensitivity

Test how fairness changes at different decision thresholds:

```bash
# Test multiple thresholds
for threshold in 0.3 0.4 0.5 0.6 0.7; do
  glassalpha audit \
    --config audit.yaml \
    --threshold $threshold \
    --output threshold_$threshold\_audit.pdf
done

# Compare fairness metrics across thresholds
glassalpha compare \
  --audits threshold_*_audit.pdf \
  --metric demographic_parity_difference \
  --output threshold_sensitivity.png
```

**Red flags**:

- Large fairness swings across thresholds (unstable)
- No threshold achieves acceptable fairness (systemic bias)

#### Challenge 2: Distribution Shift Stress Test

Test robustness to demographic changes:

```bash
# Test extreme shifts (±20%)
glassalpha audit \
  --config audit.yaml \
  --check-shift gender:+0.2 \
  --check-shift race:-0.2 \
  --output stress_test_audit.pdf

# Check metric degradation
# Original: accuracy 0.85, fairness 0.08
# Shifted: accuracy 0.78, fairness 0.15
# Degradation: 8% accuracy drop, 88% fairness degradation
```

**Red flags**:

- Large metric degradation (>10% accuracy, >50% fairness)
- Model only works on training distribution

#### Challenge 3: Edge Case Exploration

Test model on edge cases:

```bash
# Test on specific subgroups
glassalpha audit \
  --config audit.yaml \
  --filter "age > 70 AND gender == 'Female'" \
  --output elderly_women_audit.pdf

# Test on near-threshold cases
glassalpha audit \
  --config audit.yaml \
  --filter "0.45 < score < 0.55" \
  --output borderline_audit.pdf
```

**Red flags**:

- Performance drops on edge cases
- Large uncertainty on borderline predictions

#### Challenge 4: Feature Ablation

Test feature importance claims:

```bash
# Remove top feature and reaudit
python ablate_feature.py --remove credit_score

glassalpha audit \
  --config audit_without_credit_score.yaml \
  --output ablation_audit.pdf

# Compare performance
# If top feature removed but performance unchanged → feature not important
```

**Red flags**:

- Claimed important features are not actually important
- Model relies heavily on protected attribute proxies

### Workflow 4: Validation Opinion

**Scenario**: Provide formal validation opinion for regulatory submission.

#### Template: Validation Opinion Letter

```
INDEPENDENT MODEL VALIDATION OPINION

Model: Credit Scoring Model v2.5
Audit Date: 2025-10-15
Validation Date: 2025-10-22
Validator: [Name, Credentials]

I. SCOPE OF VALIDATION

I was engaged to perform an independent validation of the audit report prepared
for [Model Name] by [Organization]. My validation included:

1. Verification of evidence pack integrity (SHA256 checksums)
2. Independent reproduction of audit results
3. Challenge testing (threshold sensitivity, distribution shift, edge cases)
4. Review of documentation completeness and quality

II. METHODOLOGY

- Reproduced audit in controlled environment using provided evidence pack
- Validated key metrics within 0.1% tolerance
- Conducted threshold sensitivity analysis (0.3 to 0.7 range)
- Tested robustness to ±10% demographic shifts
- Reviewed against SR 11-7 / [relevant regulation] requirements

III. FINDINGS

Strengths:
- Audit results are reproducible (byte-identical under same environment)
- Comprehensive fairness analysis with statistical rigor
- Explicit documentation of limitations

Concerns:
- [List specific concerns from review]

Red Flags Identified:
- [None / List critical issues]

IV. OPINION

Based on my independent validation, I conclude that:

[PASS]: The audit report provides a reliable and comprehensive assessment of
the model's performance, fairness, and limitations. The model is suitable for
deployment subject to [conditions].

[PASS WITH CONDITIONS]: The audit report is generally reliable, but the
following conditions must be met before deployment: [conditions].

[FAIL]: The audit report contains [material deficiencies / insufficient
documentation / unreproducible results] and cannot be relied upon for
regulatory compliance without [remediation].

V. LIMITATIONS OF VALIDATION

This validation was based on [data / model / documentation] provided by
[Organization]. I did not independently verify [data quality / model training
process / production deployment]. This opinion is subject to [limitations].

Validator Signature: ________________
Date: ________________
```

## Red Flag Taxonomy

### Critical Red Flags (Block Deployment)

- **Checksum mismatch**: Evidence tampering or corruption
- **Non-reproducible**: Results change without explanation
- **No confidence intervals**: Can't assess statistical significance
- **Large fairness violations (>20%)**: Severe disparate impact
- **Poor calibration (ECE>0.15)**: Predictions highly inaccurate
- **Missing limitations section**: Overconfident, incomplete

### Warning Red Flags (Require Mitigation)

- **Small sample sizes (n<30)**: Underpowered comparisons
- **Wide confidence intervals**: High uncertainty
- **Moderate fairness violations (10-20%)**: Disparate impact concern
- **Moderate calibration issues (ECE 0.05-0.15)**: Predictions somewhat inaccurate
- **Vague reason codes**: FCRA/ECOA compliance issue
- **No robustness testing**: Unknown sensitivity to distribution changes

### Advisory Red Flags (Document but Don't Block)

- **Edge case performance unknown**: Needs more testing
- **Limited diversity in training data**: Generalization concern
- **Proxy features present**: Monitor for indirect discrimination
- **Threshold not optimized**: May not be best operating point

## Best Practices

### Independence

- Use separate environment from original audit team
- Don't accept pre-processed data (verify from source)
- Challenge assumptions, don't rubber-stamp

### Documentation

- Document all steps taken during validation
- Record exact versions, commands, data used
- Preserve artifacts for future reference

### Communication

- Be specific about concerns (not "model seems biased")
- Quantify findings (not "poor performance" but "accuracy 0.65, below 0.75 threshold")
- Distinguish critical from advisory issues

### Reproducibility

- Always reproduce in clean environment
- Use Docker when possible for platform consistency
- Document any differences from original audit

## Troubleshooting

### Issue: Can't reproduce audit results

**Possible causes**:

- Different package versions
- Different platform (macOS vs Linux)
- Different random seed
- Different data version

**Debugging**:

- Compare manifests field-by-field
- Use exact constraints file from evidence pack
- Run in Docker to match platform
- Verify data hashes match

### Issue: Metrics differ slightly (<1%)

**Explanation**: Floating-point differences across platforms are expected and acceptable.

**Validation**: If differences are <0.1% for all metrics, audit is reproducible.

### Issue: Evidence pack verification fails

**Possible causes**:

- File corruption during transfer
- Evidence pack created incorrectly
- Intentional tampering

**Action**: Request new evidence pack from source.

## Related Resources

- [Compliance Officer Workflow](compliance-workflow.md) - Evidence pack generation
- [ML Engineer Workflow](ml-engineer-workflow.md) - Implementation details
- [Banking Compliance Guide](../compliance/banking-guide.md) - SR 11-7 requirements
- [Troubleshooting](../reference/troubleshooting.md) - Common issues
- [Trust & Deployment](../reference/trust-deployment.md) - Reproducibility details

## Support

For validation-specific questions:

- GitHub Discussions: [GlassAlpha/glassalpha/discussions](https://github.com/GlassAlpha/glassalpha/discussions)
- Email: [contact@glassalpha.com](mailto:contact@glassalpha.com)
- Documentation: [glassalpha.com](https://glassalpha.com)
