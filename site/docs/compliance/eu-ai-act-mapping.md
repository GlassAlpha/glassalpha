# EU AI Act Compliance Mapping

## Overview

This document maps **GlassAlpha features** to **EU AI Act requirements** for high-risk AI systems. Use this as a reference when demonstrating compliance with the EU Artificial Intelligence Act.

**Regulation**: [EU Artificial Intelligence Act](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52021PC0206) (Regulation 2024/1689)

**Scope**: High-risk AI systems as defined in Annex III, including credit scoring, employment decisions, and other uses affecting fundamental rights.

**Status**: The EU AI Act entered into force in August 2024. Full compliance required by August 2026 for most high-risk systems.

## Related Guides

This is a **technical reference** document mapping GlassAlpha features to specific EU AI Act articles. For practical guidance, see:

- **[Compliance Readiness Checklist](compliance-readiness-checklist.md)** - Pre-submission verification
- **[Compliance Officer Workflow](../guides/compliance-workflow.md)** - Evidence pack generation and submission
- **[Banking Compliance Guide](banking-guide.md)** - Additional requirements for financial services

**New to EU AI Act?** Start with the compliance checklist, then return here for article-by-article details.

## Quick Reference Table

| Article  | Requirement Summary     | GlassAlpha Artifact               | Location in Audit       |
| -------- | ----------------------- | --------------------------------- | ----------------------- |
| Art. 9   | Risk management system  | Risk assessment + mitigation docs | Model Card (Section 11) |
| Art. 10  | Data governance         | Dataset bias audit + schema       | Section 3, Manifest     |
| Art. 11  | Technical documentation | Audit PDF + manifest              | Full report             |
| Art. 12  | Record keeping          | Audit trail + logs                | Manifest JSON           |
| Art. 13  | Transparency            | Explanations + reason codes       | Sections 6-7, CLI       |
| Art. 14  | Human oversight         | Decision review procedures        | Documentation           |
| Art. 15  | Accuracy & robustness   | Performance + stability testing   | Sections 4-5, 10        |
| Art. 17  | Quality management      | Audit pipeline + versioning       | CI/CD integration       |
| Annex IV | Technical documentation | Complete audit package            | Evidence pack           |

## Detailed Mapping

### Title III, Chapter 2: Requirements for High-Risk AI Systems

#### Article 9: Risk Management System

**Requirement**: "A risk management system shall be established, implemented, documented and maintained in relation to high-risk AI systems."

**GlassAlpha Artifacts:**

| Component               | Feature                                          | How to Document                 |
| ----------------------- | ------------------------------------------------ | ------------------------------- |
| **Risk identification** | Fairness analysis, bias detection                | Sections 7-9 of audit PDF       |
| **Risk estimation**     | Statistical confidence intervals, power analysis | Throughout audit (95% CIs)      |
| **Risk evaluation**     | Policy gates with severity levels                | `policy_decision.json`          |
| **Risk mitigation**     | Recourse analysis, threshold optimization        | Recourse guide, threshold sweep |

**Implementation:**

```yaml
# Risk management config
risk_management:
  identified_risks:
    - name: "Gender bias in credit decisions"
      severity: "high"
      likelihood: "medium"
      mitigation: "Fairness constraints applied, monitoring enabled"

    - name: "Calibration drift over time"
      severity: "medium"
      likelihood: "high"
      mitigation: "Monthly recalibration, automated alerts"

  residual_risks:
    - name: "Small sample size for minority groups"
      severity: "low"
      acceptance_rationale: "Sample size adequate for power=0.80, limitations documented"
```

**Evidence:**

- Risk assessment matrix in model card
- Mitigation strategies documented
- Residual risk acceptance with rationale

---

#### Article 10: Data and Data Governance

**Requirement**: "High-risk AI systems which make use of techniques involving the training of models with data shall be developed on the basis of training, validation and testing data sets that meet the quality criteria..."

**GlassAlpha Artifacts:**

| Requirement             | Feature                                             | CLI/Config               |
| ----------------------- | --------------------------------------------------- | ------------------------ |
| **Relevant data**       | Dataset bias audit (E12)                            | `glassalpha detect-bias` |
| **Representative data** | Statistical power analysis, sampling bias detection | E12: distribution tests  |
| **Error-free data**     | Missing value handling, outlier detection           | Preprocessing manifest   |
| **Data governance**     | Dataset schema, provenance tracking                 | `manifest.data_schema`   |

**Configuration:**

```yaml
data_governance:
  training_data:
    source: "CRM system exports 2023-2024"
    schema: "data/schemas/credit_applications_v2.yaml"
    quality_checks:
      - name: "Missing value rate"
        threshold: 0.30
        result: "PASS (max 12% missing)"
      - name: "Outlier detection"
        method: "IQR"
        result: "3 outliers removed, documented"
      - name: "Proxy correlation"
        threshold: 0.50
        result: "WARNING: ZIP code correlation with race=0.67"

  validation_data:
    split_method: "stratified"
    test_size: 0.30
    stratify_by: ["credit_risk", "gender", "age_group"]

  bias_monitoring:
    frequency: "monthly"
    protected_attributes: ["gender", "age", "nationality"]
    statistical_tests: ["KS test", "chi-square", "power analysis"]
```

**Evidence:**

- Dataset bias audit report (E12)
- Data quality checks documented
- Sampling strategy justified
- Protected attribute distributions analyzed

---

#### Article 11: Technical Documentation

**Requirement**: "The technical documentation of a high-risk AI system shall be drawn up before that system is placed on the market..."

**GlassAlpha Artifacts:**

**Annex IV requirements mapped to GlassAlpha:**

| Annex IV Requirement       | GlassAlpha Artifact                 | Location                         |
| -------------------------- | ----------------------------------- | -------------------------------- |
| 1. General description     | Model card                          | Section 11                       |
| 2. Detailed description    | Model architecture, hyperparameters | Manifest JSON                    |
| 3. Development process     | Training logs, validation process   | Provenance manifest              |
| 4. Data requirements       | Data schema, preprocessing          | Section 2, manifest              |
| 5. Computational resources | Environment specs                   | Manifest: `platform`, `versions` |
| 6. Testing procedures      | Validation testing, audit results   | Sections 4-10                    |
| 7. Performance metrics     | Accuracy, fairness, calibration     | Sections 5, 7, 6                 |
| 8. Safety & security       | Adversarial testing, robustness     | Section 10 (E6+)                 |
| 9. Human oversight         | Decision review procedures          | Documentation                    |

**Evidence Pack Structure:**

```
technical_documentation_eu_ai_act/
├── 01_general_description/
│   └── model_card.pdf              # Audit PDF Section 11
├── 02_detailed_description/
│   ├── architecture.yaml           # Model config
│   └── hyperparameters.json        # From manifest
├── 03_development_process/
│   ├── training_log.txt
│   └── validation_results.pdf      # Audit PDF Sections 4-5
├── 04_data_requirements/
│   ├── data_schema.yaml
│   ├── preprocessing_manifest.json
│   └── dataset_bias_audit.pdf      # E12 output
├── 05_computational_resources/
│   ├── environment.yaml            # Manifest: versions
│   └── platform_specs.txt          # Manifest: platform
├── 06_testing_procedures/
│   ├── audit_report.pdf            # Full audit
│   └── validation_manifest.json
├── 07_performance_metrics/
│   ├── performance_summary.json
│   └── metrics_with_CIs.csv
├── 08_safety_security/
│   ├── robustness_testing.pdf      # E6+ output
│   └── adversarial_results.json
├── 09_human_oversight/
│   └── review_procedures.pdf
└── checksums.sha256                # Integrity verification
```

**Export command:**

```bash
glassalpha export-eu-documentation \
  --audit audit.pdf \
  --manifest audit.manifest.json \
  --output eu_technical_docs.zip
```

---

#### Article 12: Record Keeping

**Requirement**: "High-risk AI systems shall be designed and developed with capabilities enabling the automatic recording of events ('logs') over the lifetime of the system."

**GlassAlpha Artifacts:**

| Log Type            | Feature                                  | Format                          |
| ------------------- | ---------------------------------------- | ------------------------------- |
| **Decision logs**   | Individual predictions with explanations | JSON export                     |
| **Monitoring logs** | Ongoing fairness/calibration tracking    | Registry database               |
| **Audit trail**     | Provenance manifest per audit run        | Manifest JSON                   |
| **Version logs**    | Git commit, package versions             | Manifest: `git_sha`, `versions` |

**Configuration:**

```yaml
logging:
  decision_logging:
    enabled: true
    include_explanations: true
    include_confidence: true
    retention_days: 2555 # 7 years
    storage: "s3://audit-logs/decisions/"

  audit_logging:
    enabled: true
    include_manifest: true
    include_policy_decisions: true
    storage: "s3://audit-logs/audits/"

  monitoring_logging:
    frequency: "monthly"
    metrics: ["fairness", "calibration", "performance"]
    alerts:
      - metric: "demographic_parity_difference"
        threshold: 0.10
        action: "email_compliance_team"
```

**Evidence:**

- Audit manifests for all production deployments
- Decision logs with unique IDs
- Monitoring results with timestamps
- Version history (Git log)

---

#### Article 13: Transparency and Provision of Information to Users

**Requirement**: "High-risk AI systems shall be designed and developed in such a way to ensure that their operation is sufficiently transparent..."

**GlassAlpha Artifacts:**

| Transparency Requirement     | Feature                               | Output                      |
| ---------------------------- | ------------------------------------- | --------------------------- |
| **Explanation of decisions** | SHAP values, reason codes             | Sections 6-7, `reasons` CLI |
| **System capabilities**      | Model card with limitations           | Section 11                  |
| **System limitations**       | Explicit limitations section          | Section 11                  |
| **Accuracy levels**          | Performance with confidence intervals | Section 5                   |
| **Context of use**           | Use case documentation                | Model card                  |

**User-facing transparency:**

```bash
# Generate user-facing explanation
glassalpha reasons \
  --model model.pkl \
  --instance 12345 \
  --output explanations/user_12345.txt \
  --language en \
  --format plain-text
```

**Output example:**

```
AUTOMATED DECISION EXPLANATION

Decision: Application DENIED
Confidence: 72%

Primary factors contributing to this decision:
1. Credit History (negative): Previous default on credit account
2. Debt-to-Income Ratio (negative): 45% (high relative to income)
3. Employment Duration (negative): 6 months (short tenure)

This decision was made by an automated system. You have the right to:
- Request human review of this decision
- Provide additional information for reconsideration
- Understand how the decision was reached

For more information or to request review, contact: [compliance contact]
```

**Evidence:**

- Explanation methodology documented
- User-facing explanation templates
- Limitations clearly communicated
- Human review process defined

---

#### Article 14: Human Oversight

**Requirement**: "High-risk AI systems shall be designed and developed in such a way...that they can be effectively overseen by natural persons..."

**GlassAlpha Support:**

| Oversight Requirement       | GlassAlpha Feature                | Documentation                    |
| --------------------------- | --------------------------------- | -------------------------------- |
| **Understand capabilities** | Audit report with limitations     | Model card (Section 11)          |
| **Understand limitations**  | Explicit limitations section      | Section 11, confidence intervals |
| **Monitor operation**       | Ongoing audits, registry tracking | CI/CD integration, registry      |
| **Intervene on decisions**  | Flag uncertain predictions        | Confidence thresholds            |
| **Override decisions**      | Manual review workflows           | Process documentation            |

**Human oversight configuration:**

```yaml
human_oversight:
  mandatory_review:
    - condition: "prediction_confidence < 0.60"
      action: "Flag for human review"
    - condition: "protected_attribute_flagged = true"
      action: "Require supervisory approval"
    - condition: "adverse_action = true"
      action: "Generate explanation + review option"

  review_process:
    reviewer_role: "Credit Officer"
    review_sla: "24 hours"
    override_authority: "Senior Credit Manager"
    documentation_required: true

  monitoring:
    dashboard: "https://monitoring.company.com/ml-models"
    alerts:
      - condition: "fairness_drift > 0.05"
        notification: "Compliance team"
      - condition: "calibration_error > 0.10"
        notification: "Model owner + Risk team"
```

**Evidence:**

- Human oversight procedures documented
- Override process defined
- Training for human reviewers
- Audit logs of human interventions

---

#### Article 15: Accuracy, Robustness and Cybersecurity

**Requirement**: "High-risk AI systems shall be designed and developed in such a way that they achieve...an appropriate level of accuracy, robustness and cybersecurity..."

**GlassAlpha Artifacts:**

| Requirement                  | Feature                                | CLI/Output                            |
| ---------------------------- | -------------------------------------- | ------------------------------------- |
| **Accuracy**                 | Performance metrics with CIs           | Section 5, bootstrap CIs              |
| **Robustness (stability)**   | Monotonicity, feature importance tests | E6: stability tests                   |
| **Robustness (shifts)**      | Demographic shift testing              | E6.5: `--check-shift`                 |
| **Robustness (adversarial)** | Perturbation sweeps                    | E6+: adversarial testing              |
| **Cybersecurity**            | Preprocessing artifact security        | Class allowlisting, hash verification |

**Configuration:**

```yaml
accuracy_robustness:
  accuracy_requirements:
    min_auc: 0.75
    min_calibration: 0.95 # ECE < 0.05
    statistical_confidence: 0.95 # 95% CIs

  robustness_testing:
    demographic_shifts:
      - attribute: "gender"
        shifts: [-0.10, -0.05, 0.05, 0.10]
        max_degradation: 0.10
      - attribute: "age"
        shifts: [-0.10, 0.10]
        max_degradation: 0.10

    adversarial_perturbations:
      epsilons: [0.01, 0.05, 0.10]
      max_prediction_delta: 0.15

    stability_tests:
      monotonicity_checks: true
      feature_importance_stability: true

  cybersecurity:
    preprocessing_artifacts:
      hash_verification: true
      class_allowlisting: true
      signature_required: false # Optional: KMS signing
```

**Evidence:**

- Performance documented with confidence intervals
- Robustness tested under multiple scenarios
- Degradation bounds quantified
- Security measures documented

---

#### Article 17: Quality Management System

**Requirement**: "Providers of high-risk AI systems shall put a quality management system in place..."

**GlassAlpha Support:**

| QMS Component              | GlassAlpha Feature                | Implementation                |
| -------------------------- | --------------------------------- | ----------------------------- |
| **Design & development**   | Config-driven audit pipeline      | YAML configs, version control |
| **Quality control**        | Policy gates, CI/CD integration   | Automated compliance checks   |
| **Testing & validation**   | Comprehensive audit suite         | Full audit report             |
| **Post-market monitoring** | Registry tracking, ongoing audits | Monthly/quarterly audits      |
| **Documentation**          | Evidence packs, manifests         | Tamper-evident archives       |

**CI/CD Integration:**

```yaml
# .github/workflows/eu-ai-act-compliance.yml
name: EU AI Act Compliance

on:
  pull_request:
    paths: ["models/**", "data/**"]

jobs:
  compliance-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run compliance audit
        run: |
          glassalpha audit \
            --config eu_compliance.yaml \
            --policy-gates policy/eu_ai_act_gates.yaml \
            --output audit_report.pdf \
            --strict

      - name: Check Article 15 robustness
        run: |
          glassalpha audit \
            --config eu_compliance.yaml \
            --check-shift gender:+0.10 \
            --check-shift age:-0.10 \
            --fail-on-degradation 0.10

      - name: Export technical documentation
        run: |
          glassalpha export-eu-documentation \
            --audit audit_report.pdf \
            --output eu_docs.zip

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: eu-compliance-package
          path: |
            audit_report.pdf
            audit_report.manifest.json
            policy_decision.json
            eu_docs.zip
```

**Evidence:**

- QMS procedures documented
- Automated quality gates
- Version-controlled configurations
- Audit trail for all changes

---

## Annex IV: Technical Documentation Requirements

### Complete Documentation Checklist

- [ ] **Section 1: General description of the AI system**

  - [ ] Intended purpose and use case
  - [ ] Provider information
  - [ ] Version and release date
  - [ ] High-level architecture

- [ ] **Section 2: Detailed description**

  - [ ] Model architecture (XGBoost, LightGBM, etc.)
  - [ ] Hyperparameters and training configuration
  - [ ] Preprocessing pipeline
  - [ ] Feature engineering

- [ ] **Section 3: Development process**

  - [ ] Training methodology
  - [ ] Validation approach
  - [ ] Testing procedures
  - [ ] Development tools and frameworks

- [ ] **Section 4: Data requirements**

  - [ ] Training data characteristics
  - [ ] Data quality checks
  - [ ] Sampling strategy
  - [ ] Protected attribute handling

- [ ] **Section 5: Computational resources**

  - [ ] Hardware requirements
  - [ ] Software dependencies (with versions)
  - [ ] Platform specifications
  - [ ] Performance benchmarks

- [ ] **Section 6: Testing procedures**

  - [ ] Validation testing results
  - [ ] Robustness testing results
  - [ ] Fairness testing results
  - [ ] Edge case testing

- [ ] **Section 7: Performance metrics**

  - [ ] Accuracy (with confidence intervals)
  - [ ] Fairness metrics (with statistical significance)
  - [ ] Calibration metrics
  - [ ] Robustness scores

- [ ] **Section 8: Safety and security**

  - [ ] Adversarial testing results
  - [ ] Security measures
  - [ ] Risk mitigation strategies
  - [ ] Residual risks documented

- [ ] **Section 9: Human oversight**
  - [ ] Oversight procedures
  - [ ] Review workflows
  - [ ] Override mechanisms
  - [ ] Training requirements

---

## Example Policy Gates for EU AI Act

```yaml
# policy/eu_ai_act_gates.yaml
policy_name: "EU AI Act High-Risk AI Compliance"
version: "1.0"
effective_date: "2026-08-02"
regulation: "EU AI Act (Regulation 2024/1689)"

gates:
  # Article 15: Accuracy
  - name: "Minimum Accuracy (Art. 15)"
    metric: "roc_auc"
    threshold: 0.75
    comparison: "greater_than"
    severity: "error"
    article: "Article 15"

  # Article 15: Robustness
  - name: "Robustness to Demographic Shifts (Art. 15)"
    metric: "max_shift_degradation"
    threshold: 0.10
    comparison: "less_than"
    severity: "error"
    article: "Article 15"

  # Article 10: Data quality
  - name: "Minimum Statistical Power (Art. 10)"
    metric: "min_group_size"
    threshold: 30
    comparison: "greater_than"
    severity: "error"
    article: "Article 10(3)"

  # Article 13: Transparency
  - name: "Explainability Available (Art. 13)"
    metric: "explainer_available"
    threshold: true
    comparison: "equals"
    severity: "error"
    article: "Article 13"

  # Fundamental rights (implicit)
  - name: "No Unjustified Discrimination"
    metric: "demographic_parity_difference"
    threshold: 0.10
    comparison: "less_than"
    severity: "error"
    article: "Recital 44, Annex III"
```

---

## Compliance Timeline

| Date          | Milestone                         | Action Required              |
| ------------- | --------------------------------- | ---------------------------- |
| August 2024   | Regulation enters into force      | Begin compliance planning    |
| February 2025 | Prohibited practices ban          | Audit for prohibited uses    |
| August 2025   | Code of practice deadline         | Adopt best practices         |
| August 2026   | **High-risk compliance deadline** | **Full compliance required** |
| August 2027   | General-purpose AI requirements   | Additional requirements      |

**Recommended timeline for GlassAlpha users:**

- **Q1 2025**: Assess if your AI system is high-risk (Annex III)
- **Q2 2025**: Implement technical documentation (Article 11, Annex IV)
- **Q3 2025**: Establish quality management system (Article 17)
- **Q4 2025**: Conduct conformity assessment
- **Q1 2026**: Register system in EU database
- **Q2 2026**: Train personnel, finalize human oversight
- **August 2026**: Full compliance deadline

---

## Summary: GlassAlpha Coverage of EU AI Act

| Article                       | Compliance Level                 | Key Artifacts                    |
| ----------------------------- | -------------------------------- | -------------------------------- |
| Art. 9 (Risk management)      | ✅ Full coverage                 | Risk assessment, mitigation docs |
| Art. 10 (Data governance)     | ✅ Full coverage                 | Dataset bias audit, schema       |
| Art. 11 (Documentation)       | ✅ Full coverage                 | Audit PDF + manifest             |
| Art. 12 (Record keeping)      | ✅ Full coverage                 | Manifests, logs, registry        |
| Art. 13 (Transparency)        | ✅ Full coverage                 | Explanations, reason codes       |
| Art. 14 (Human oversight)     | ⚠️ Partial (process docs needed) | Review procedures                |
| Art. 15 (Accuracy/robustness) | ✅ Full coverage                 | Performance + stability testing  |
| Art. 17 (Quality management)  | ✅ Full coverage                 | CI/CD, policy gates              |
| Annex IV (Tech docs)          | ✅ Full coverage                 | Evidence pack                    |

**Overall Assessment**: GlassAlpha provides comprehensive technical compliance for Articles 9-13, 15, 17, and Annex IV. Article 14 (human oversight) requires process documentation beyond technical auditing.

---

## Related Resources

### Compliance Guides

- [Compliance Readiness Checklist](compliance-readiness-checklist.md) - Pre-submission verification
- [Compliance Overview](index.md) - Role/industry navigation
- [Trust & Deployment](../reference/trust-deployment.md) - Reproducibility and audit trails

### Workflow Guides

- [Compliance Officer Workflow](../guides/compliance-workflow.md) - Evidence packs, submissions
- [ML Engineer Workflow](../guides/ml-engineer-workflow.md) - Implementation, CI integration
- [ML Manager Workflow](../guides/ml-manager-workflow.md) - Policy configuration, portfolio tracking

### Feature Guides

- [Dataset Bias Detection](../guides/dataset-bias.md) - Article 10 compliance
- [Shift Testing](../guides/shift-testing.md) - Article 15 robustness
- [Reason Codes](../guides/reason-codes.md) - Article 13 transparency

---

**Questions?** See [Troubleshooting](../reference/troubleshooting.md) or contact [support](mailto:contact@glassalpha.com).
