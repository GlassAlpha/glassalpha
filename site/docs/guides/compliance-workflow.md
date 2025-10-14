# Compliance Officer Workflow

Guide for compliance officers, risk managers, and regulatory liaisons using GlassAlpha to generate audit evidence and communicate with regulators.

## Overview

This guide is for compliance professionals who need to:

- Generate audit evidence for regulator submissions
- Verify model documentation completeness
- Establish policy gates for automated compliance checks
- Communicate audit results to examiners and auditors
- Maintain audit trails and reproducibility

**Not an ML engineer?** This guide assumes minimal technical background. For implementation details, see [ML Engineer Workflow](ml-engineer-workflow.md).

## Key Capabilities

### Evidence Pack Generation

Create tamper-evident zip files containing:

- Audit PDF report
- Provenance manifest (hashes, versions, seeds)
- Policy decision log (pass/fail for each compliance gate)
- Configuration files
- Dataset schema

**Use case**: Regulator requests model documentation for SR 11-7 audit or fair lending review.

### Policy-as-Code Gates

Define compliance thresholds in YAML files:

- Minimum calibration accuracy
- Maximum fairness metric values
- Required sample sizes
- Robustness requirements

**Use case**: Establish firm-wide model acceptance criteria that automatically fail non-compliant models.

### Verification Workflow

Independent validators can verify audit integrity:

- Checksum validation (SHA256)
- Manifest completeness
- Reproducibility checks

**Use case**: Internal audit or third-party validation of model documentation.

## Typical Workflows

### Workflow 1: Generating Evidence for Regulator Submission

**Scenario**: Federal Reserve examiner requests SR 11-7 documentation for your credit scoring model.

#### Step 1: Request audit from ML team

Provide ML team with compliance requirements:

```yaml
# Requirements for compliance review
# Direct configuration for SR 11-7 compliance
strict_mode: true # Enforce determinism
policy_gates: "configs/policy/sr_11_7_banking.yaml"

# Required sections
fairness_analysis: true
calibration_testing: true
reason_codes: true
recourse_analysis: true
```

**What to communicate to ML team**:

- "We need SR 11-7 documentation for the Q4 credit model"
- "Use strict mode to ensure reproducibility"
- "Apply sr_11_7_banking policy gates"
- "Audit must pass all error-level gates"

#### Step 2: Review audit report

ML team provides `credit_model_audit_2025Q4.pdf` and `credit_model_audit_2025Q4.manifest.json`.

**Review checklist**:

- [ ] All sections present (model card, performance, fairness, calibration, explainability)
- [ ] Protected attributes analyzed (race, gender, age)
- [ ] Statistical significance documented (confidence intervals, sample sizes)
- [ ] Limitations section lists assumptions and edge cases
- [ ] Provenance manifest includes git SHA, package versions, data hashes
- [ ] Policy gates documented with pass/fail status

**Red flags**:

- Missing sections or "N/A" without explanation
- Small group sample sizes (n < 30)
- Wide confidence intervals (statistical power issues)
- Failed policy gates without documented mitigation
- Missing reproducibility information (no seed, no versions)

#### Step 3: Package evidence for submission

Once audit passes review, create evidence bundle:

```bash
# Create evidence directory
mkdir -p evidence_packs/credit_model_2025Q4

# Copy artifacts
cp reports/credit_model_audit_2025Q4.pdf evidence_packs/credit_model_2025Q4/
cp reports/credit_model_audit_2025Q4.manifest.json evidence_packs/credit_model_2025Q4/
cp configs/audit_config.yaml evidence_packs/credit_model_2025Q4/

# Generate checksums
cd evidence_packs/credit_model_2025Q4
sha256sum *.pdf *.json *.yaml > SHA256SUMS.txt

# Add verification instructions
cat > README.txt << 'EOF'
# Verification Instructions

1. Verify checksums:
   sha256sum -c SHA256SUMS.txt

2. Reproduce audit:
   glassalpha audit --config audit_config.yaml --output validation_rerun.pdf

3. Compare outputs:
   sha256sum validation_rerun.pdf credit_model_audit_2025Q4.pdf

All artifacts should be byte-identical when regenerated with same configuration.
EOF

# Create ZIP for submission
zip -r ../credit_model_2025Q4.zip .
```

**Evidence bundle contents**:

```
credit_model_2025Q4.zip
├── credit_model_audit_2025Q4.pdf          # Complete audit
├── credit_model_audit_2025Q4.manifest.json # Provenance (seeds, versions, hashes)
├── audit_config.yaml                       # Exact configuration used
├── SHA256SUMS.txt                          # Checksums for all files
└── README.txt                              # Verification instructions
```

**Note**: Evidence pack commands (`export-evidence-pack`, `verify-evidence-pack`) are available starting in v0.2.1.

[Complete evidence pack guide →](evidence-packs.md)

#### Step 4: Submit to regulator

Include evidence bundle in submission with cover letter:

**Example cover letter language**:

> "Model documentation was generated using GlassAlpha v0.2.0, an open-source audit framework. The attached evidence bundle contains a comprehensive audit report addressing SR 11-7 requirements (Sections III.A through V), including model validation testing, fairness analysis, and outcomes monitoring.
>
> All artifacts in the evidence bundle are checksummed for integrity verification. Independent validation can be performed using the included verification instructions. The provenance manifest documents all random seeds, package versions, and data hashes for full reproducibility."

### Workflow 2: Establishing Policy Gates

**Scenario**: Your firm needs standardized model acceptance criteria for all credit models.

#### Step 1: Define compliance requirements

Work with legal, risk, and compliance teams to establish thresholds:

**Example requirements**:

- Calibration error < 5% (SR 11-7 III.B.1)
- Demographic parity difference < 10% (ECOA, fair lending)
- Equalized odds difference < 15% (ECOA, fair lending)
- Minimum group size ≥ 30 (statistical power)
- Robustness to demographic shift < 5% degradation (SR 11-7 III.A.3)

#### Step 2: Create policy configuration

Document requirements in YAML:

```yaml
# policy/firm_credit_baseline.yaml
policy_name: "Firm Credit Model Baseline"
version: "1.0"
effective_date: "2025-10-01"
citation: "SR 11-7, ECOA, internal risk policy"

gates:
  - name: "Calibration Quality"
    clause: "SR 11-7 III.B.1"
    metric: "expected_calibration_error"
    threshold: 0.05
    comparison: "less_than"
    severity: "error" # Blocks deployment

  - name: "Demographic Parity"
    clause: "ECOA fair lending"
    metric: "demographic_parity_difference"
    threshold: 0.10
    comparison: "less_than"
    severity: "error"

  - name: "Equalized Odds"
    clause: "ECOA fair lending"
    metric: "equalized_odds_difference"
    threshold: 0.15
    comparison: "less_than"
    severity: "warning" # Requires documentation

  - name: "Minimum Statistical Power"
    clause: "SR 11-7 III.B.2"
    metric: "min_group_size"
    threshold: 30
    comparison: "greater_than"
    severity: "error"
```

**Severity levels**:

- `error`: Model cannot be deployed (CI fails, exit code 1)
- `warning`: Requires documentation but doesn't block deployment
- `info`: Tracked for monitoring, no action required

#### Step 3: Apply to all audits

ML engineers reference policy in audit configs:

```yaml
# Standard audit config
# Direct configuration for SR 11-7 compliance
policy_gates: "configs/policy/firm_credit_baseline.yaml"
strict_mode: true
```

When audit runs, policy gates are checked:

```bash
glassalpha audit --config audit.yaml --output report.pdf

# Exit codes:
# 0 = All gates passed
# 1 = One or more error-level gates failed
# 2 = Runtime error
```

#### Step 4: Monitor compliance

Track gate failures across model portfolio by maintaining a log of audit results:

```python
# Example: Track audit results in a simple log
import json
from pathlib import Path

audit_log_path = Path("audit_results_log.jsonl")

# After each audit, append results
with open(audit_log_path, 'a') as f:
    result_entry = {
        "model_id": "credit_v2",
        "timestamp": "2025-10-14",
        "gates": result.gates,  # Gate results from audit
        "passed": all(g["status"] == "PASS" for g in result.gates)
    }
    f.write(json.dumps(result_entry) + "\n")

# Query failed gates
with open(audit_log_path) as f:
    failed = [json.loads(line) for line in f if not json.loads(line)["passed"]]
    for entry in failed:
        print(f"{entry['model_id']}: {entry['timestamp']}")
```

**Action items for failures**:

- FAILED gates: Model must be retrained or threshold adjusted
- WARNING gates: Document mitigation strategy in model card
- INFO gates: Monitor for trends, no immediate action

### Workflow 3: Responding to Audit Findings

**Scenario**: Internal audit or regulator identifies concern about fairness metrics.

#### Step 1: Reproduce audit results

Verify audit is reproducible:

```bash
# Verify evidence pack integrity
glassalpha verify-evidence-pack \
  --input evidence_packs/credit_model_2025Q4.zip

# Output:
# ✓ All checksums valid
# ✓ Manifest complete
# ✓ Policy gates documented
# ✓ Reproducibility info present
```

If checksums match, audit is tamper-evident.

#### Step 2: Investigate specific finding

Example finding: "Demographic parity difference of 0.12 exceeds 0.10 threshold for gender."

**Investigation steps**:

1. Review fairness section of audit PDF (Section 7)
2. Check sample sizes per group (are groups balanced?)
3. Review confidence intervals (is difference statistically significant?)
4. Check for proxy features (features correlated with gender)

**Questions to ask ML team**:

- "What is the confidence interval for this metric?" (0.12 ± 0.03 vs 0.12 ± 0.10)
- "What features are most correlated with the protected attribute?"
- "What is the baseline disparity in the training data?" (model vs data bias)

#### Step 3: Document mitigation or remediation

**If finding is valid**:

Option A: Retrain model with fairness constraints
Option B: Adjust decision threshold (if permissible)
Option C: Remove proxy features and retest

**If finding is statistical artifact**:

Document in response:

> "Demographic parity difference of 0.12 has a 95% confidence interval of [0.08, 0.16], overlapping with the 0.10 threshold. Sample size of n=150 for the minority group results in wide confidence intervals. With increased sample size, we expect the metric to stabilize within policy bounds. We will monitor this metric in ongoing validation."

#### Step 4: Update audit and resubmit

After mitigation, generate new audit:

```bash
glassalpha audit \
  --config audit.yaml \
  --output credit_model_audit_2025Q4_v2.pdf
```

Compare old vs new:

```bash
# Extract key metrics
glassalpha inspect --audit report_v1.pdf --output metrics_v1.json
glassalpha inspect --audit report_v2.pdf --output metrics_v2.json

# Compare
diff metrics_v1.json metrics_v2.json
```

Document changes in cover letter to regulator.

## Communication Templates

### Template 1: Evidence Pack Cover Letter

> **Subject**: Model Documentation Submission - [Model Name]
>
> Attached is comprehensive documentation for [Model Name] addressing [SR 11-7 / ECOA / other regulation] requirements.
>
> **Evidence Pack Contents**:
>
> - Audit report (PDF): 45 pages covering model validation, fairness analysis, calibration testing, and outcomes monitoring
> - Provenance manifest (JSON): Complete lineage with data hashes, package versions, and random seeds
> - Policy decision log (JSON): Pass/fail results for all compliance gates
> - Configuration files: Audit settings, policy thresholds, data schema
>
> **Verification**: The evidence pack includes SHA256 checksums for all artifacts. Independent verification can be performed using the included instructions.
>
> **Tool**: Documentation generated using GlassAlpha v[X.Y.Z], an open-source audit framework ([glassalpha.com](https://glassalpha.com)).
>
> **Contact**: [Your Name], [Title], [Email], [Phone]

### Template 2: Policy Gate Failure Notification

> **Subject**: Model Deployment Blocked - Policy Gate Failure
>
> Model [Model Name] failed mandatory compliance gate(s) and cannot be deployed.
>
> **Failed Gate**: [Gate Name] > **Metric**: [Metric Name] = [Value] > **Threshold**: [Comparison] [Threshold] > **Severity**: ERROR
> **Regulatory Citation**: [Clause]
>
> **Required Action**: Model must be retrained or threshold adjusted. Rerun audit after changes:
>
> ```
> glassalpha audit --config audit.yaml --output report_v2.pdf
> ```
>
> **Deployment Status**: BLOCKED until all error-level gates pass.
>
> **Contact**: Compliance team at [email] for questions.

### Template 3: Response to Audit Finding

> **Subject**: Response to Audit Finding - [Model Name]
>
> **Finding**: [Summary of finding, e.g., "Demographic parity difference exceeds threshold"]
>
> **Investigation Results**:
>
> - Metric value: [Value with confidence interval]
> - Sample sizes: [Per-group sample sizes]
> - Statistical significance: [P-value or confidence interval interpretation]
> - Root cause: [Data imbalance / proxy feature / model artifact]
>
> **Mitigation Plan**:
>
> [Option A: Retrain with fairness constraints] > [Option B: Adjust threshold] > [Option C: Monitor with increased sample size]
>
> **Timeline**: [Expected completion date]
>
> **Updated Audit**: [Attached if remediation complete]

## Best Practices

### Documentation Hygiene

- Store evidence packs in tamper-evident storage (write-once, append-only)
- Version all policy gate configurations (track changes over time)
- Archive audits with date and model version in filename
- Maintain audit logs for portfolio-level tracking (e.g., JSONL files with audit results)

### Reproducibility

- Always run audits in strict mode for regulator submissions
- Verify evidence pack integrity before submission
- Include verification instructions for independent validators
- Document any deviations from standard policy gates

### Regulator Communication

- Use neutral, factual language (not marketing tone)
- Cite specific regulatory clauses for all requirements
- Provide confidence intervals and sample sizes for metrics
- Document limitations and assumptions clearly
- Include contact information for technical questions

### Policy Gate Governance

- Review and update policy thresholds annually
- Document rationale for all threshold values
- Track gate failures across model portfolio
- Require executive approval for policy exceptions

## Troubleshooting

### Issue: Evidence pack verification fails

**Symptom**: `glassalpha verify-evidence-pack` reports checksum mismatch

**Causes**:

- File was modified after evidence pack creation
- Incorrect file uploaded (wrong version)
- Transmission corruption

**Resolution**:

- Regenerate evidence pack from source audit
- Verify file integrity with `sha256sum <file>`
- Use secure transfer method (SFTP, not email)

### Issue: Policy gate failure near threshold

**Symptom**: Metric is 0.102, threshold is 0.10 (just over)

**Investigation**:

- Check confidence interval (does it overlap threshold?)
- Check sample size (is difference statistically significant?)
- Review trend over time (improving or degrading?)

**Options**:

- If CI overlaps threshold: Document uncertainty, monitor
- If significant: Remediate
- If improving trend: Request waiver with monitoring plan

### Issue: Regulator questions reproducibility

**Symptom**: Regulator asks "How do I know these results are accurate?"

**Response**:

- Provide evidence pack with checksums
- Offer to reproduce audit in front of regulator (deterministic)
- Share provenance manifest showing all seeds, versions, hashes
- Reference open-source tool (code is auditable)
- Provide verification instructions

## Related Resources

- [Banking Compliance Guide](../compliance/banking-guide.md) - SR 11-7, ECOA, FCRA requirements
- [SR 11-7 Technical Mapping](../compliance/sr-11-7-mapping.md) - Clause-by-clause coverage
- [Model Validator Workflow](validator-workflow.md) - Independent verification procedures
- [Trust & Deployment](../reference/trust-deployment.md) - Reproducibility and audit trail details

## Support

For compliance-specific questions:

- GitHub Discussions: [GlassAlpha/glassalpha/discussions](https://github.com/GlassAlpha/glassalpha/discussions)
- Email: [contact@glassalpha.com](mailto:contact@glassalpha.com)
- Documentation: [glassalpha.com](https://glassalpha.com)
