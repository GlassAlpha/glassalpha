# Compliance Readiness Checklist

**Purpose**: Pre-flight checklist for regulatory submission or audit readiness. Print this page and verify all items before submitting to regulators or internal audit.

**Version**: 1.0 | **Last Updated**: January 2025

---

## Model Documentation

- [ ] **Model Card Complete**

  - Model purpose and use case documented
  - Algorithm type and training process described
  - Hyperparameters and configuration recorded
  - Limitations explicitly stated

- [ ] **Audit Report Generated**

  - Professional PDF with all required sections
  - Includes executive summary with key findings
  - Performance metrics with confidence intervals
  - Fairness analysis across all protected attributes

- [ ] **Reproducibility Manifest**

  - Random seeds documented
  - Package versions captured
  - Git commit SHA included
  - Dataset hash recorded

- [ ] **Provenance Trail**
  - Data sources documented
  - Preprocessing steps recorded
  - Model training process documented
  - All transformations logged

---

## Fairness & Bias Testing

- [ ] **Protected Attributes Identified**

  - All legally protected attributes documented (race, gender, age, etc.)
  - Protected attributes NOT used as model features (unless legally permitted)
  - Proxy features identified and documented

- [ ] **Group Fairness Analyzed**

  - Demographic parity computed and documented
  - Equal opportunity (TPR parity) tested
  - Equalized odds (TPR + FPR parity) verified
  - Statistical significance tested (p-values reported)

- [ ] **Statistical Confidence**

  - Confidence intervals provided for all fairness metrics (95% CI)
  - Sample sizes documented per group (minimum n≥30)
  - Statistical power calculated (minimum 0.80 for bias detection)
  - Wide CIs acknowledged as limitation

- [ ] **Individual Fairness** (if applicable)

  - Consistency score computed
  - Similar cases treated similarly
  - Flip tests conducted (protected attribute changes)

- [ ] **Intersectional Analysis** (if applicable)
  - Multi-way fairness analyzed (e.g., race × gender)
  - Hidden bias in subgroups identified
  - Sample size warnings for small subgroups

---

## Model Performance

- [ ] **Out-of-Sample Validation**

  - Test set distinct from training data
  - Performance metrics on test set only
  - No data leakage verified

- [ ] **Core Metrics Reported**

  - Accuracy, precision, recall, F1, AUC-ROC documented
  - Confusion matrix included
  - Confidence intervals for all metrics

- [ ] **Calibration Testing**

  - Expected Calibration Error (ECE) computed
  - Calibration curve included in report
  - Confidence intervals for ECE reported
  - Calibration by protected group (if fairness concern)

- [ ] **Performance by Group**
  - Metrics broken down by protected attributes
  - No single group has significantly worse performance
  - Disparities documented with explanations

---

## Explainability

- [ ] **Global Explanations**

  - Feature importance rankings provided
  - Top-N most important features documented
  - Protected attributes not dominant features (red flag if they are)

- [ ] **Individual Explanations**

  - Example predictions with explanations included
  - SHAP values or coefficient-based explanations
  - Clear visualization of feature contributions

- [ ] **Reason Codes** (if applicable for credit decisions)
  - Top 3-5 specific reasons for adverse actions
  - Protected attributes excluded from reason codes
  - ECOA-compliant adverse action notice template
  - Actionable (customer can understand and respond)

---

## Robustness & Stability

- [ ] **Stability Testing**

  - Monotonicity checks conducted (where expected)
  - Feature importance stability verified
  - No unexplained violations

- [ ] **Demographic Shift Testing**

  - Model tested under ±10% demographic shifts
  - Maximum degradation documented
  - Fails if degradation >10% (adjust threshold per policy)

- [ ] **Adversarial Testing** (if applicable)
  - Perturbation sweeps conducted (ε = 1%, 5%, 10%)
  - Maximum prediction delta documented
  - Robustness score computed

---

## Data Quality

- [ ] **Dataset Bias Audit** (if applicable)

  - Proxy features identified (correlation with protected attributes)
  - Distribution drift tested (KS/χ² tests)
  - Sampling bias assessed (statistical power per group)
  - Train/test balance verified

- [ ] **Missing Data Handled**

  - Missing value strategy documented
  - Imputation method justified
  - No systematic bias from missing data

- [ ] **Data Schema Locked**
  - Column names and types documented
  - Feature definitions clear
  - Changes to schema version-controlled

---

## Preprocessing Artifacts

- [ ] **Production Artifacts Verified** (if using production preprocessing)

  - Preprocessing pipeline artifact hash documented
  - Dual-hash system (file + params)
  - Class allowlisting applied (security)
  - Artifact inspection passed

- [ ] **Transformations Documented**
  - All preprocessing steps recorded
  - Transformation logic explainable
  - Reversibility confirmed (where needed)

---

## Policy Compliance

- [ ] **Policy Gates Defined**

  - Organization policy applied (if exists)
  - All gates documented with thresholds
  - Severity levels assigned (error/warning/info)
  - Regulatory citations included

- [ ] **Gate Results Documented**

  - Pass/fail status for each gate
  - Failed gates explained with remediation plan
  - Warning-level gates documented with mitigation
  - Policy decision JSON exported

- [ ] **Thresholds Justified**
  - Rationale documented for all thresholds
  - Aligned with regulatory guidance
  - Reviewed by legal/compliance team

---

## Regulatory Alignment

### For Banking/Credit (SR 11-7, ECOA, FCRA)

- [ ] **SR 11-7 Requirements**

  - Model documentation comprehensive (§III.A.1)
  - Validation testing complete (§III.B.1-3)
  - Ongoing monitoring plan defined (§III.A.3)
  - Limitations documented (§III.C.3)
  - See [SR 11-7 Mapping](sr-11-7-mapping.md) for clause-by-clause coverage

- [ ] **ECOA Compliance**

  - Adverse action reasons specific and actionable (Reg B §1002.9)
  - Protected attributes not used in decision (unless permitted exception)
  - Disparate impact monitored and mitigated
  - Reason codes exclude protected attributes

- [ ] **FCRA Compliance**
  - Model predictions are accurate and fair
  - Adverse action procedures documented
  - Consumer dispute process defined

### For EU Operations (GDPR, EU AI Act)

- [ ] **GDPR Article 22**

  - Right to explanation satisfied
  - Automated decision-making documented
  - Human review process defined (where required)
  - Data subject rights procedures established

- [ ] **EU AI Act** (if applicable)
  - High-risk AI system assessment complete
  - Risk management system documented
  - Data governance procedures established
  - Technical documentation complete
  - See [EU AI Act Mapping](eu-ai-act-mapping.md) for detailed requirements

---

## Evidence Pack

- [ ] **Complete Artifacts**

  - Audit PDF report
  - Provenance manifest (JSON)
  - Policy decision log (JSON)
  - Configuration files (YAML)
  - Dataset schema

- [ ] **Integrity Verification**

  - SHA256 checksums computed for all artifacts (use `sha256sum *.pdf *.json *.yaml > SHA256SUMS.txt`)
  - Checksums file included in evidence bundle
  - Verification instructions included (README.txt)

- [ ] **Evidence Bundle Packaged**
  - All artifacts collected (audit PDF, manifest JSON, config YAML)
  - Checksums generated and verified
  - Verification instructions provided
  - Bundle stored in tamper-evident location (ZIP with checksums)
  - Note: Automated `export-evidence-pack` command planned for v0.3.0 (Enhancement E3)

---

## Operational Readiness

- [ ] **Strict Mode Used**

  - Audit run with `--strict` flag
  - All determinism checks passed
  - No warnings as errors (strict mode enforcement)

- [ ] **Reproducibility Verified**

  - Audit reproduced in clean environment
  - Byte-identical PDF achieved
  - Manifests match on rerun

- [ ] **Deployment Gate Configured** (if using CI/CD)

  - Policy gates enforced in CI pipeline
  - Failed audits block deployment
  - Audit artifacts uploaded to registry

- [ ] **Monitoring Plan** (for production models)
  - Metrics to track defined (calibration, fairness drift)
  - Monitoring frequency specified (monthly, quarterly)
  - Alert thresholds configured
  - Remediation procedures documented

---

## Final Review

- [ ] **Compliance Officer Review**

  - Audit reviewed by compliance team
  - Findings documented
  - Sign-off obtained

- [ ] **Legal Review** (if high-risk model)

  - Legal team consulted
  - Regulatory risk assessed
  - Approval granted

- [ ] **Executive Approval** (if required)

  - Model Review Board presentation complete
  - Executive sign-off obtained
  - Deployment authorization granted

- [ ] **Documentation Archive**
  - All artifacts archived (7-year retention typical)
  - Version control committed
  - Registry entry created
  - Audit trail complete

---

## Quick Reference: Minimum Requirements

**For ANY model audit submission:**
✅ Audit PDF with all sections
✅ Reproducibility manifest
✅ Fairness analysis (if protected attributes exist)
✅ Confidence intervals for key metrics
✅ Limitations section

**For BANKING/CREDIT models:**
✅ All of the above
✅ SR 11-7 compliance mapping
✅ Reason codes (if adverse actions)
✅ Calibration testing
✅ Evidence pack with checksums

**For EU HIGH-RISK AI systems:**
✅ All of the above
✅ EU AI Act Article 9-15 compliance
✅ Risk management documentation
✅ Human oversight procedures
✅ Data governance documentation

---

## Troubleshooting Common Issues

| Issue                        | Fix                                                                                            |
| ---------------------------- | ---------------------------------------------------------------------------------------------- |
| Missing confidence intervals | Re-run with bootstrap enabled: `calibration.n_bootstrap: 1000`                                 |
| Small sample sizes (n<30)    | Collect more data or aggregate groups                                                          |
| Wide confidence intervals    | Acknowledge as limitation in report                                                            |
| Failed policy gates          | See gate-specific remediation in [Policy Guide](../guides/ml-manager-workflow.md#policy-gates) |
| Non-reproducible results     | Ensure seed set: `random_seed: 42` and `--strict` flag used                                    |
| Missing limitations section  | Add explicit limitations in model card                                                         |

---

## Resources

- **Workflow Guides**: [Compliance Officer](../guides/compliance-workflow.md) | [ML Engineer](../guides/ml-engineer-workflow.md) | [Validator](../guides/validator-workflow.md)
- **Regulatory Mappings**: [SR 11-7](sr-11-7-mapping.md) | [EU AI Act](eu-ai-act-mapping.md)
- **Industry Guides**: [Banking](banking-guide.md) | [Insurance](insurance-guide.md) | [Healthcare](healthcare-guide.md)
- **Technical Docs**: [Trust & Deployment](../reference/trust-deployment.md) | [Troubleshooting](../reference/troubleshooting.md)

---

**Completed this checklist?** You're ready for regulatory submission or internal audit.

**Print this page** and use it as a physical checklist during audit review sessions.
