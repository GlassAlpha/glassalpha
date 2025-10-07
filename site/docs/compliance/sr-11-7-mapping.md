# SR 11-7 Compliance Mapping

## Overview

This document maps **GlassAlpha features** to **SR 11-7 requirements** (Federal Reserve Supervisory Guidance on Model Risk Management). Use this as a reference when demonstrating regulatory compliance to auditors and examiners.

**Document**: [SR 11-7: Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm) (April 4, 2011)

**Scope**: Banking institutions using ML models for credit decisions, risk assessment, and other regulated activities.

## Quick Reference Table

| SR 11-7 Section | Requirement Summary             | GlassAlpha Artifact                    | Location in Audit        |
| --------------- | ------------------------------- | -------------------------------------- | ------------------------ |
| III.A.1         | Model documentation             | Audit PDF + Manifest                   | Full report              |
| III.A.2         | Conceptual soundness            | Model Card                             | Section 11               |
| III.A.3         | Ongoing monitoring              | Shift testing, drift analysis          | CLI flags, E6.5          |
| III.B.1         | Model development testing       | Calibration, performance metrics       | Sections 4-5             |
| III.B.2         | Validation testing              | Statistical confidence intervals       | E10, E10+                |
| III.B.3         | Outcomes analysis               | Recourse, reason codes                 | E2, E2.5                 |
| III.C.1         | Model assumptions documentation | Preprocessing manifest                 | Section 2                |
| III.C.2         | Data quality assessment         | Dataset bias audit                     | Section 3, E12           |
| III.C.3         | Model limitations               | Stability tests, robustness sweeps     | E6, E6+                  |
| IV.A            | Reproducibility requirements    | Provenance manifest, determinism       | Audit Trail (Section 10) |
| IV.B            | Independent validation          | Evidence pack export                   | E3                       |
| V               | Fairness and discrimination     | Fairness analysis (group + individual) | Sections 7-9, E5.1, E11  |

## Detailed Mapping

### Section III: Elements of Sound Model Risk Management

#### III.A.1: Effective Model Documentation

**Requirement**: "Comprehensive and clear documentation is essential... The level of documentation should be commensurate with the potential risk presented by the use of the model."

**GlassAlpha Artifacts:**

| What                         | Where                                                      | How to Cite                                              |
| ---------------------------- | ---------------------------------------------------------- | -------------------------------------------------------- |
| Complete audit report (PDF)  | `glassalpha audit --config audit.yaml --output report.pdf` | "See comprehensive audit report (GlassAlpha v{version})" |
| Provenance manifest (JSON)   | `report.manifest.json` sidecar                             | "Model lineage documented in manifest SHA256:{hash}"     |
| Model card                   | Section 11 of PDF                                          | "Model card per GlassAlpha SR 11-7 template"             |
| Policy-as-code configuration | `configs/policy/*.yaml`                                    | "Policy constraints codified in {policy_name}.yaml"      |

**Example citation**:

> "Model documentation prepared using GlassAlpha audit framework (Section III.A.1 compliance). See `german_credit_audit.pdf` (SHA256:{hash}) and provenance manifest for complete lineage."

---

#### III.A.2: Rigorous Testing of Model Concepts and Outcomes

**Requirement**: "Model development should include rigorous assessment of model concepts and their suitability... testing should evaluate whether the model is performing as intended and in line with design objectives."

**GlassAlpha Artifacts:**

| Test Type                   | GlassAlpha Feature                            | Section  | Command                             |
| --------------------------- | --------------------------------------------- | -------- | ----------------------------------- |
| **Calibration testing**     | Expected Calibration Error (ECE) with 95% CI  | 6        | `metrics.calibration.enabled: true` |
| **Performance validation**  | Accuracy, precision, recall, F1               | 5        | Automatic                           |
| **Fairness assessment**     | Group fairness metrics with statistical power | 7        | `metrics.fairness.*`                |
| **Stability testing**       | Perturbation sweeps, robustness score         | 10 (E6+) | `metrics.stability.enabled: true`   |
| **Distribution robustness** | Demographic shift testing                     | E6.5     | `--check-shift gender:+0.1`         |

**Example citation**:

> "Model concepts validated per SR 11-7 Section III.A.2:
>
> - Calibration: ECE = 0.032 ± 0.006 (95% CI) [Section 6]
> - Robustness: max Δ = 0.084 under ε=0.1 perturbation [Section 10]
> - Fairness: TPR difference = 0.045 (p<0.05, power=0.94) [Section 7]"

---

#### III.A.3: Ongoing Monitoring

**Requirement**: "The institution should establish a program for ongoing monitoring... to confirm the model is operating as expected and to identify the need for recalibration or remediation."

**GlassAlpha Artifacts:**

| Monitoring Type        | Feature                           | Implementation                               | Frequency        |
| ---------------------- | --------------------------------- | -------------------------------------------- | ---------------- |
| **Demographic drift**  | Shift testing in CI/CD            | `--check-shift` with `--fail-on-degradation` | Every deployment |
| **Calibration drift**  | ECE comparison over time          | Track ECE in CI artifacts                    | Monthly          |
| **Fairness drift**     | Group metric tracking             | Compare baseline to production               | Quarterly        |
| **Data quality drift** | Distribution drift tests (KS, χ²) | Dataset bias audit (E12)                     | Monthly          |

**Example CI/CD integration**:

```yaml
# .github/workflows/model-monitoring.yml
- name: Monthly shift test
  run: |
    glassalpha audit --config prod.yaml \
      --check-shift gender:+0.05 \
      --check-shift age:-0.05 \
      --fail-on-degradation 0.03
```

**Example citation**:

> "Ongoing monitoring program per SR 11-7 Section III.A.3:
>
> - Deployment gates: Shift testing blocks deployment if degradation >3pp
> - Monthly audits: ECE, fairness metrics tracked in CI artifacts
> - Quarterly review: Fairness drift analysis with statistical significance testing"

---

#### III.B.1: Model Development Testing - In-Sample and Out-of-Sample

**Requirement**: "The model's development should include testing to demonstrate its accuracy and robustness."

**GlassAlpha Artifacts:**

| Validation Type                | Feature                         | How It Works                 | Citation                             |
| ------------------------------ | ------------------------------- | ---------------------------- | ------------------------------------ |
| **Out-of-sample performance**  | Test set metrics                | Separate test data in config | "Test set accuracy: 87.6% (n=1,000)" |
| **Cross-validated confidence** | Bootstrap CIs (E10)             | Resampling validation        | "95% CI: [0.842, 0.901]"             |
| **Robustness validation**      | Adversarial perturbations (E6+) | ε-perturbation sweeps        | "Stable under 10% noise (Δ=0.084)"   |

**Configuration example**:

```yaml
# audit.yaml
data:
  path: "data/test.csv" # Hold-out test set
  split: "test" # Explicit test designation

metrics:
  calibration:
    enabled: true
    n_bootstrap: 1000 # Statistical validation
```

**Example citation**:

> "Out-of-sample validation per SR 11-7 Section III.B.1:
>
> - Test set: 1,000 samples (30% hold-out, stratified)
> - Accuracy: 87.6% [84.2%, 90.1%] (95% CI, bootstrap n=1,000)
> - Robustness: Δ<0.1 under adversarial perturbations
>   See GlassAlpha audit Section 5 for complete validation results."

---

#### III.B.2: Sensitivity Analysis

**Requirement**: "Sensitivity analysis should be designed to assess the model's response to variations in inputs and assumptions."

**GlassAlpha Artifacts:**

| Analysis Type               | Feature                  | What It Tests                  | Output                       |
| --------------------------- | ------------------------ | ------------------------------ | ---------------------------- |
| **Demographic sensitivity** | Shift simulator (E6.5)   | Response to population changes | Degradation metrics by shift |
| **Feature perturbation**    | Adversarial sweeps (E6+) | Response to input noise        | Max prediction delta         |
| **Threshold sensitivity**   | Fairness@threshold sweep | Impact of decision threshold   | TPR/FPR trade-offs           |

**Example analysis**:

```bash
# Test ±10% demographic shifts
glassalpha audit --config audit.yaml \
  --check-shift gender:+0.1 \
  --check-shift gender:-0.1 \
  --check-shift age:+0.1 \
  --check-shift age:-0.1
```

**Example citation**:

> "Sensitivity analysis per SR 11-7 Section III.B.2:
>
> - Demographic shifts: ±10pp in gender/age → max degradation 4.2pp (within tolerance)
> - Input perturbations: ε ∈ {1%, 5%, 10%} → max Δ = 0.084 (PASS threshold)
> - Threshold sweep: Decision boundary ∈ [0.3, 0.7] → TPR varies [0.72, 0.91]
>   See `audit.shift_analysis.json` for complete sensitivity results."

---

#### III.B.3: Outcomes Analysis

**Requirement**: "Model outcomes should be analyzed to verify that the model is performing as intended... analysis should be at a frequency commensurate with the level of model risk."

**GlassAlpha Artifacts:**

| Outcome Type                  | Feature                           | What It Shows                | Regulatory Value             |
| ----------------------------- | --------------------------------- | ---------------------------- | ---------------------------- |
| **Individual outcomes**       | Reason codes (E2)                 | Why each prediction was made | ECOA/Reg B compliance        |
| **Actionable recourse**       | Recourse generation (E2.5)        | How to change outcome        | Fair lending compliance      |
| **Group outcome disparities** | Fairness analysis (E5, E5.1, E11) | Disparate impact detection   | Fair Housing Act compliance  |
| **Outcome stability**         | Individual fairness (E11)         | Consistency of similar cases | Equal treatment verification |

**Example usage**:

```bash
# Generate reason codes for adverse action notices
glassalpha reasons --model model.pkl --instance 42

# Analyze group outcome disparities
glassalpha fairness --model model.pkl --data test.csv --group gender
```

**Example citation**:

> "Outcomes analysis per SR 11-7 Section III.B.3:
>
> - Individual explanations: Top-N reason codes per ECOA requirements [E2]
> - Recourse: Actionable recommendations for 94% of denials [E2.5]
> - Disparate impact: TPR difference = 4.5pp (p=0.02), within 5pp tolerance [E5]
> - Consistency: 87% of similar applicants receive similar decisions [E11]"

---

#### III.C.1: Model Assumptions and Conceptual Soundness

**Requirement**: "Assumptions should be conceptually sound and well-documented."

**GlassAlpha Artifacts:**

| Assumption Type          | Documentation           | Location          | Validation                        |
| ------------------------ | ----------------------- | ----------------- | --------------------------------- |
| **Data preprocessing**   | Preprocessing manifest  | Section 2         | Artifact hash verification        |
| **Feature engineering**  | Transformation pipeline | Manifest JSON     | Reproducible pipeline             |
| **Model architecture**   | Model card              | Section 11        | Hyperparameter documentation      |
| **Fairness constraints** | Policy-as-code YAML     | `configs/policy/` | Explicit constraint specification |

**Example manifest entry**:

```json
{
  "preprocessing": {
    "mode": "artifact",
    "file_hash": "sha256:abc123...",
    "components": [
      { "name": "imputer", "method": "mean" },
      { "name": "scaler", "method": "standardization" },
      { "name": "onehot", "categories": ["A", "B", "C"] }
    ]
  }
}
```

**Example citation**:

> "Model assumptions documented per SR 11-7 Section III.C.1:
>
> - Preprocessing: StandardScaler + MeanImputer (manifest SHA256:{hash})
> - Feature engineering: 15 features, no synthetic transformations
> - Architecture: XGBoost (n_estimators=100, max_depth=6)
> - Constraints: Debt-to-income monotone decreasing, age immutable
>   See preprocessing manifest and model card for complete assumption documentation."

---

#### III.C.2: Data Quality and Suitability

**Requirement**: "Data should be suitable for the model's purpose... Data quality issues should be identified and addressed."

**GlassAlpha Artifacts:**

| Data Quality Check          | Feature                  | What It Detects                          | Action                         |
| --------------------------- | ------------------------ | ---------------------------------------- | ------------------------------ |
| **Proxy feature detection** | Dataset bias audit (E12) | Features correlated with protected attrs | Flag for removal/investigation |
| **Distribution drift**      | KS/χ² tests (E12)        | Train/test distribution mismatch         | Resample or reweight           |
| **Sampling bias power**     | Statistical power (E12)  | Insufficient samples for bias detection  | Collect more data              |
| **Split imbalance**         | Train/test balance (E12) | Demographic imbalance across splits      | Stratified sampling            |

**Example output**:

```
Dataset-Level Bias Analysis (E12):
- Proxy Correlations:
  * income × race: r=0.42 (WARNING - moderate proxy risk)
  * ZIP × race: r=0.67 (ERROR - high proxy risk)
- Distribution Drift:
  * age: KS=0.12, p=0.03 (SIGNIFICANT drift detected)
- Sampling Power:
  * gender=female: power=0.94 (adequate, min_n=30)
  * race=minority: power=0.61 (marginal, recommend n>100)
```

**Example citation**:

> "Data quality assessment per SR 11-7 Section III.C.2:
>
> - Proxy analysis: 2 features flagged (correlation >0.5 with protected attrs)
> - Distribution validation: KS test p<0.05 for 1/15 features (age drift)
> - Sample adequacy: Statistical power ≥0.80 for all protected groups
> - Action taken: ZIP code removed, age distributions rebalanced
>   See Section 3 (Dataset Bias Analysis) for complete data quality audit."

---

#### III.C.3: Model Limitations and Assumptions

**Requirement**: "Limitations and assumptions should be identified and documented."

**GlassAlpha Artifacts:**

| Limitation Type             | How Documented               | Where         | Example                                    |
| --------------------------- | ---------------------------- | ------------- | ------------------------------------------ |
| **Performance bounds**      | Confidence intervals (E10)   | Section 5     | "Accuracy: 87.6% ±3.2% (95% CI)"           |
| **Calibration limits**      | ECE with CIs (E10+)          | Section 6     | "ECE: 0.032 ±0.006 (marginal calibration)" |
| **Stability bounds**        | Robustness threshold (E6+)   | Section 10    | "Stable for Δ<10% input noise only"        |
| **Sample size constraints** | Statistical power (E10, E12) | Sections 3, 7 | "Bias detection power limited for n<100"   |

**Example Model Card entry**:

```markdown
## Limitations (SR 11-7 Section III.C.3)

1. **Performance bounds**: Model accuracy estimated at 87.6% with 95% CI [84.2%, 90.1%].
   Performance outside this range should trigger investigation.

2. **Calibration**: ECE=0.032 indicates slight miscalibration. Model predictions
   should not be interpreted as precise probabilities without recalibration.

3. **Robustness**: Model stable under ≤10% input perturbations. Adversarial inputs
   or out-of-distribution data may produce unreliable predictions.

4. **Sample coverage**: Fairness analysis limited to binary gender (male/female).
   Other gender identities not represented in training data.

5. **Demographic applicability**: Trained on US population ages 18-75. Predictions
   for international applicants or outside age range are unreliable.
```

**Example citation**:

> "Model limitations documented per SR 11-7 Section III.C.3:
>
> - Known performance bounds with statistical confidence
> - Calibration constraints quantified (ECE with 95% CI)
> - Robustness thresholds validated empirically
> - Sample size limitations disclosed with power analysis
>   See Model Card (Section 11) for complete limitations documentation."

---

### Section IV: Model Validation

#### IV.A: Independence and Reproducibility

**Requirement**: "Validation should include... documentation and review adequate for a third party to replicate the model implementation or validation review."

**GlassAlpha Artifacts:**

| Reproducibility Element      | Feature                | Verification Method     | Regulatory Evidence            |
| ---------------------------- | ---------------------- | ----------------------- | ------------------------------ |
| **Deterministic execution**  | Global seed management | `seed: 42` in config    | Byte-identical PDF on rerun    |
| **Environment capture**      | Version manifest       | Package versions logged | `{"sklearn": "1.3.0", ...}`    |
| **Data provenance**          | Dataset hash           | SHA256 in manifest      | Tamper-proof data integrity    |
| **Model provenance**         | Model hash             | SHA256 in manifest      | Verifiable model artifact      |
| **Pipeline reproducibility** | Preprocessing hash     | Dual-hash system        | Byte-identical transformations |

**Validation workflow**:

```bash
# Original audit
glassalpha audit --config audit.yaml --output report_original.pdf

# Independent validation (reproduces exactly)
glassalpha audit --config audit.yaml --output report_validation.pdf

# Verify byte-identical
sha256sum report_original.pdf report_validation.pdf
# Both should have identical hashes
```

**Example citation**:

> "Model validation independence per SR 11-7 Section IV.A:
>
> - Reproducibility: Byte-identical audit reports achieved with fixed seed (verified SHA256)
> - Environment: Package versions captured in manifest (sklearn 1.3.0, xgboost 1.7.6)
> - Data integrity: Test set hash SHA256:{hash} (unchanged since validation)
> - Pipeline integrity: Preprocessing artifact hash SHA256:{hash} (verified)
>   Third-party validator can reproduce results using audit.yaml + manifest."

---

#### IV.B: Evidence Pack for Independent Review

**Requirement**: "Validation findings should be well documented."

**GlassAlpha Artifacts:**

| Artifact                  | Purpose               | Command                           | Contents                        |
| ------------------------- | --------------------- | --------------------------------- | ------------------------------- |
| **Evidence pack (ZIP)**   | Complete audit bundle | `glassalpha export-evidence-pack` | PDF + manifest + policy + gates |
| **Verification tool**     | Validate integrity    | `glassalpha verify-evidence-pack` | SHA256 checksums                |
| **Policy decisions JSON** | Gates/compliance      | `policy_decision.json` sidecar    | PASS/FAIL per policy rule       |
| **Shift analysis JSON**   | Robustness evidence   | `.shift_analysis.json` sidecar    | Degradation under shifts        |

**Evidence pack structure**:

```
evidence_pack_20241007.zip
├── audit_report.pdf          # Main audit (SHA256: abc123...)
├── audit.manifest.json        # Provenance (SHA256: def456...)
├── policy_decision.json       # Gate results (SHA256: ghi789...)
├── audit.shift_analysis.json  # Shift testing (SHA256: jkl012...)
├── gates.yaml                 # Policy configuration
└── checksums.sha256           # Verification file
```

**Example validation**:

```bash
# Export evidence pack
glassalpha export-evidence-pack \
  --audit report.pdf \
  --manifest report.manifest.json \
  --output evidence_pack.zip

# Independent validator verifies
glassalpha verify-evidence-pack evidence_pack.zip
# ✓ All checksums valid
# ✓ Manifest integrity verified
# ✓ Policy decisions reproducible
```

**Example citation**:

> "Validation evidence per SR 11-7 Section IV.B:
>
> - Evidence pack: `evidence_pack_20241007.zip` (SHA256:{hash})
> - Contents: Audit PDF, manifest, policy decisions, shift analysis
> - Verification: All artifacts SHA256-validated, byte-identical on rerun
> - Availability: Evidence pack retained for 7 years per record retention policy
>   Independent validators can verify using `glassalpha verify-evidence-pack`."

---

### Section V: Fairness and Discrimination Risk

**Requirement** (Implicit in SR 11-7, explicit in OCC Bulletin 2011-12 and ECOA/Reg B): "Models used in credit decisions must not result in prohibited discrimination."

**GlassAlpha Artifacts:**

| Compliance Area             | Feature                          | Regulatory Standard        | Output     |
| --------------------------- | -------------------------------- | -------------------------- | ---------- |
| **Group fairness**          | Demographic parity, TPR/FPR (E5) | ECOA, Fair Housing Act     | Section 7  |
| **Intersectional analysis** | Multi-way fairness (E5.1)        | Hidden bias detection      | Section 8  |
| **Individual treatment**    | Consistency score (E11)          | Disparate treatment (ECOA) | Section 9  |
| **Statistical confidence**  | Bootstrap CIs (E10)              | Statistically sound claims | Throughout |
| **Adverse action notices**  | Reason codes (E2)                | ECOA Reg B §1002.9         | CLI/API    |
| **Actionable recourse**     | Counterfactuals (E2.5)           | Fair lending best practice | CLI/API    |

**Example fairness analysis**:

```
Group Fairness Analysis (E5):
- gender_male: TPR=0.86, FPR=0.12
- gender_female: TPR=0.82, FPR=0.15
- Difference: ΔTPR=0.04 (4pp), within 5pp tolerance
- Statistical significance: p=0.04 (bootstrap n=1,000)
- Power: 0.94 (adequate to detect 5pp difference)

Intersectional Fairness (E5.1):
- gender_male × age_18-25: TPR=0.78 (n=45, WARNING: small sample)
- gender_female × age_18-25: TPR=0.71 (n=38, WARNING: small sample)
- Hidden bias detected in young female group (ΔTPR=0.15 vs baseline)

Individual Fairness (E11):
- Consistency score: 0.87 (similar applicants treated similarly)
- Matched pairs: 42 pairs identified, avg Δ=0.06 (acceptable)
- Flip test: 3 violations (protected attr changes outcome)
```

**Example citation**:

> "Fairness analysis per SR 11-7 Section V (implicit) and ECOA requirements:
>
> - Group fairness: Max TPR difference = 4pp (within 5pp tolerance), p=0.04
> - Intersectional: Young female group shows 15pp disparity (flagged for review)
> - Individual consistency: 87% score (similar cases treated similarly)
> - Adverse action: Reason codes provided for 100% of denials per Reg B §1002.9
> - Recourse: Actionable recommendations for 94% of denials
>   See Sections 7-9 for complete fairness audit with statistical significance."

---

## Example Examiner Q&A

### Q1: "How do you document model assumptions?"

**A**: "Per SR 11-7 Section III.C.1, all assumptions are documented in three artifacts:

1. **Preprocessing Manifest** (JSON): Documents all data transformations with SHA256 hashes for reproducibility
2. **Model Card** (PDF Section 11): Documents architecture, hyperparameters, and training process
3. **Policy Configuration** (YAML): Codifies business constraints (immutable features, monotone directions, bounds)

These artifacts are generated automatically by GlassAlpha during the audit process. See `audit.manifest.json` for complete preprocessing assumptions."

### Q2: "How do you ensure reproducibility?"

**A**: "Per SR 11-7 Section IV.A, reproducibility is ensured through:

1. **Deterministic execution**: All random processes seeded (seed=42 in config)
2. **Version pinning**: Package versions captured in manifest (sklearn 1.3.0, xgboost 1.7.6)
3. **Data integrity**: Dataset hash (SHA256) verified before each run
4. **Pipeline integrity**: Preprocessing artifact dual-hash (file + params)

We verify reproducibility by generating byte-identical PDF reports on reruns. Independent validators can reproduce using our audit config + manifest."

### Q3: "How do you test for fairness and discrimination?"

**A**: "Per SR 11-7 Section V (implicit) and ECOA requirements, we implement:

1. **Group Fairness** (E5): Demographic parity, TPR/FPR differences with statistical significance (p-values, power analysis)
2. **Intersectional Analysis** (E5.1): Hidden bias in demographic combinations (e.g., race×gender)
3. **Individual Fairness** (E11): Consistency score ensures similar applicants treated similarly
4. **Statistical Confidence** (E10): All fairness metrics include 95% confidence intervals

All analysis documented in Sections 7-9 of the audit PDF with sample size warnings and power calculations."

### Q4: "How do you monitor model performance over time?"

**A**: "Per SR 11-7 Section III.A.3, we implement:

1. **Deployment Gates**: Shift testing in CI/CD (`--check-shift`) blocks deployment if degradation >threshold
2. **Monthly Audits**: Automated audits track calibration (ECE), fairness (TPR/FPR), performance (accuracy)
3. **Quarterly Reviews**: Comprehensive fairness drift analysis with statistical significance testing
4. **Data Quality Monitoring**: Distribution drift tests (KS, χ²) detect training/production divergence

All monitoring automated in GitHub Actions with alerts on degradation. See `.github/workflows/model-monitoring.yml`."

### Q5: "How do you document limitations?"

**A**: "Per SR 11-7 Section III.C.3, limitations documented in Model Card with quantitative bounds:

1. **Performance**: Confidence intervals (e.g., accuracy 87.6% ±3.2%)
2. **Calibration**: ECE with CIs (e.g., ECE=0.032 ±0.006)
3. **Robustness**: Stability thresholds (e.g., stable for Δ<10% noise)
4. **Sample Size**: Power analysis for each protected group
5. **Applicability**: Training data demographics and valid input ranges

All limitations quantified with statistical measures, not qualitative statements. See Model Card (Section 11)."

---

## Summary: GlassAlpha Coverage of SR 11-7

| SR 11-7 Section         | Compliance Level | Key Artifacts                     |
| ----------------------- | ---------------- | --------------------------------- |
| III.A.1 (Documentation) | ✅ Full Coverage | PDF, Manifest, Model Card         |
| III.A.2 (Testing)       | ✅ Full Coverage | Calibration, Fairness, Stability  |
| III.A.3 (Monitoring)    | ✅ Full Coverage | Shift Testing, CI/CD Gates        |
| III.B.1 (Development)   | ✅ Full Coverage | Out-of-sample, Bootstrap CIs      |
| III.B.2 (Sensitivity)   | ✅ Full Coverage | Shift Simulator, Perturbations    |
| III.B.3 (Outcomes)      | ✅ Full Coverage | Reason Codes, Recourse, Fairness  |
| III.C.1 (Assumptions)   | ✅ Full Coverage | Preprocessing Manifest, Policy    |
| III.C.2 (Data Quality)  | ✅ Full Coverage | Dataset Bias Audit (E12)          |
| III.C.3 (Limitations)   | ✅ Full Coverage | Model Card with quantified bounds |
| IV.A (Independence)     | ✅ Full Coverage | Determinism, Reproducibility      |
| IV.B (Evidence)         | ✅ Full Coverage | Evidence Pack, Verification       |
| V (Fairness)            | ✅ Full Coverage | Group, Intersectional, Individual |

**Overall Assessment**: GlassAlpha provides comprehensive SR 11-7 compliance for ML model risk management in banking contexts.

---

## Related Resources

- [Fair Lending Compliance (ECOA)](./ecoa-compliance.md)
- [EU AI Act Mapping](./eu-ai-act-mapping.md)
- [NAIC Model Risk Framework](./naic-mapping.md)
- [GlassAlpha User Guides](../guides/)

---

**Questions?** Contact your compliance officer or [file an issue](https://github.com/glassalpha/glassalpha/issues) for GlassAlpha support.
