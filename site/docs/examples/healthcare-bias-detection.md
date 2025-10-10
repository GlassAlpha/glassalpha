# Healthcare bias detection: medical risk assessment

This example demonstrates using GlassAlpha to detect bias in healthcare AI systems, focusing on medical risk assessment models and regulatory compliance for healthcare applications.

## Use case overview

**Scenario**: Hospital system uses ML to predict patient readmission risk

**Regulatory Context**:

- **HIPAA** compliance for patient data protection
- **FDA** guidance for AI/ML in medical devices
- **CMS** requirements for healthcare AI transparency
- **Joint Commission** standards for patient safety

**Bias Concerns**:

- Racial disparities in healthcare access and outcomes
- Gender bias in medical diagnosis and treatment
- Age-related discrimination in care decisions
- Socioeconomic factors affecting medical AI

**Industry Context**: For healthcare-specific compliance requirements (HIPAA, health equity mandates, clinical validation), see the [Healthcare Compliance Guide](../compliance/healthcare-guide.md).

## Healthcare-specific configuration

```yaml
# healthcare_audit.yaml
audit_profile: healthcare_compliance

# Strict regulatory compliance
strict_mode: true
reproducibility:
  random_seed: 2024
  track_git_sha: true
  require_data_hash: true

# Healthcare data configuration
data:
  path: patient_readmission_data.csv
  target_column: readmission_30_days

  # Protected attributes for healthcare bias detection
  protected_attributes:
    - race_ethnicity
    - gender
    - age_group
    - insurance_type
    - primary_language
    - zip_code_income_quartile

  # Feature categories
  feature_columns:
    - age
    - gender
    - race_ethnicity
    - insurance_type
    - primary_diagnosis
    - comorbidity_count
    - previous_admissions
    - length_of_stay
    - discharge_disposition
    - medications_count
    - lab_abnormalities
    - vital_signs_severity
    - socioeconomic_score

  # Healthcare-specific validation
  schema_validation: strict
  pii_detection: enabled
  clinical_codes_validation: true

# Model configuration for healthcare (LogisticRegression baseline)
model:
  type: logistic_regression
  params:
    # Conservative parameters for healthcare interpretability
    random_state: 42
    max_iter: 1000

  # Healthcare-specific validation
  validation:
    cross_validation: true
    stratified: true # Preserve class distribution
    folds: 10
    metrics: [auc_roc, precision, recall]

# Explainability for healthcare
explainers:
  priority: [treeshap, kernelshap]
  config:
    treeshap:
      max_samples: 500
      include_individual_explanations: true
      clinical_interpretation: true
    kernelshap:
      n_samples: 200
      background_size: 100

# Healthcare-specific metrics
metrics:
  performance:
    - accuracy
    - precision
    - recall
    - f1
    - auc_roc
    - classification_report

  # Critical fairness metrics for healthcare
  fairness:
    - demographic_parity
    - equal_opportunity
    - equalized_odds
    - predictive_parity

  # Healthcare bias thresholds (stricter than general ML)
  fairness_thresholds:
    demographic_parity: 0.02 # 2% maximum disparity
    equal_opportunity: 0.02 # Equal access to care
    statistical_significance: 0.01 # 99% confidence

# Healthcare compliance settings
compliance:
  frameworks:
    - hipaa
    - fda_ai_ml_guidance
    - cms_innovation

  # Documentation requirements
  documentation:
    clinical_validation: true
    bias_assessment: true
    safety_monitoring: true
    performance_monitoring: true

  # Risk assessment
  risk_assessment: high
  human_oversight: required
  continuous_monitoring: enabled

# Healthcare-specific reporting
report:
  template: healthcare_audit

  clinical_sections:
    - executive_summary
    - clinical_performance
    - bias_assessment
    - fairness_analysis
    - individual_explanations
    - safety_considerations
    - regulatory_compliance
    - monitoring_recommendations

  # Healthcare stakeholder focus
  audience:
    - clinical_staff
    - quality_assurance
    - regulatory_affairs
    - risk_management
    - medical_directors

# Security for healthcare data
security:
  encryption_at_rest: true
  audit_logging: comprehensive
  access_controls: rbac
  data_minimization: true
  retention_policy: hipaa_compliant

# Performance monitoring
monitoring:
  model_drift: enabled
  fairness_drift: enabled
  performance_degradation: enabled
  alert_thresholds:
    accuracy_drop: 0.05
    fairness_violation: 0.02
    prediction_drift: 0.1
```

## Running the healthcare audit

```bash
# Quick demo (use --fast for 2-3 second audits)
glassalpha audit \
  --config healthcare_outcomes.yaml \
  --output healthcare_bias_assessment.pdf \
  --fast

# With additional validation
glassalpha validate \
  --config healthcare_outcomes.yaml \
  --profile tabular_compliance \
  --strict

# Multi-model comparison for robustness
glassalpha audit \
  --config healthcare_outcomes.yaml \
  --output xgboost_healthcare.pdf \
  --override '{"model": {"type": "xgboost"}}' \
  --fast

glassalpha audit \
  --config healthcare_outcomes.yaml \
  --output lightgbm_healthcare.pdf \
  --override '{"model": {"type": "lightgbm"}}' \
  --fast

# For production/regulatory: omit --fast and add --strict
```

## Healthcare-specific report sections

### 1. Clinical performance assessment

**Key Metrics:**

- **Sensitivity (Recall)**: 85% - Correctly identifies patients at risk
- **Specificity**: 78% - Correctly identifies low-risk patients
- **Positive Predictive Value**: 42% - Of predicted high-risk, 42% actually readmitted
- **Negative Predictive Value**: 96% - Of predicted low-risk, 96% not readmitted
- **AUC-ROC**: 0.82 - Good discriminative ability

**Clinical Interpretation:**

- High sensitivity ensures most at-risk patients are identified
- High NPV means low-risk predictions are highly reliable
- PPV indicates some false positives (additional interventions to low-risk patients)

### 2. Healthcare bias assessment

**Racial/Ethnic Disparities:**

```
Demographic Parity by Race/Ethnicity:
- White patients: 12.3% predicted high-risk
- Black patients: 18.7% predicted high-risk (52% higher)
- Hispanic patients: 16.1% predicted high-risk (31% higher)
- Asian patients: 10.8% predicted high-risk (12% lower)

Statistical significance: p < 0.001 (highly significant)
```

**Gender Disparities:**

```
Equal Opportunity by Gender:
- Male patients (actual readmissions): 89% correctly identified
- Female patients (actual readmissions): 81% correctly identified

Gender bias detected: 8 percentage point difference
Clinical impact: Female patients more likely to be missed
```

**Age-Related Bias:**

```
Predictive Parity by Age Group:
- 18-40 years: 38% PPV (lower precision)
- 41-65 years: 45% PPV (higher precision)
- 65+ years: 41% PPV (moderate precision)

Age bias: Middle-aged patients receive more accurate predictions
```

### 3. Feature impact analysis

**Most Influential Features (SHAP):**

1. **Previous admissions** (0.23) - Clinical relevance: Strong predictor
2. **Comorbidity count** (0.19) - Clinical relevance: Multiple conditions increase risk
3. **Length of stay** (0.16) - Clinical relevance: Longer stays indicate complexity
4. **Age** (0.12) - **Potential bias source**: Age discrimination concerns
5. **Insurance type** (0.11) - **Bias concern**: Socioeconomic proxy
6. **Primary diagnosis** (0.09) - Clinical relevance: Disease-specific risk
7. **Discharge disposition** (0.08) - Clinical relevance: Care continuity
8. **ZIP code income** (0.06) - **Bias concern**: Socioeconomic proxy

**Bias Source Analysis:**

- **Direct bias**: Age, gender, race/ethnicity features
- **Proxy bias**: Insurance type, ZIP code income, primary language
- **Clinical correlation**: Some bias may reflect actual health disparities

### 4. Individual case examples

**Case 1: Potential Racial Bias**

```
Patient Profile:
- Black, Female, Age 45
- Diabetes, Hypertension
- Medicaid insurance
- Previous admission: 1

Model Prediction: HIGH RISK (0.73 probability)
Actual Outcome: No readmission

SHAP Explanation:
- Race/ethnicity: +0.15 (contributed to high-risk prediction)
- Insurance type: +0.12 (Medicaid increased risk score)
- Previous admission: +0.25 (clinical factor)
- Comorbidities: +0.18 (clinical factor)

Bias Assessment: Racial and socioeconomic factors inflated risk score
```

**Case 2: Missed Female Patient**

```
Patient Profile:
- White, Female, Age 62
- Heart failure, COPD
- Medicare insurance
- Previous admissions: 2

Model Prediction: LOW RISK (0.38 probability)
Actual Outcome: Readmitted within 30 days

SHAP Explanation:
- Gender: -0.08 (reduced risk prediction for females)
- Age: +0.22 (increased risk due to age)
- Comorbidities: +0.31 (strong clinical risk factors)
- Previous admissions: +0.28 (strong clinical predictor)

Bias Assessment: Gender bias led to underestimation of risk
```

## Regulatory compliance analysis

### FDA AI/ML guidance compliance

**Algorithm Transparency:**

- ✅ Complete model documentation provided
- ✅ Training data characteristics described
- ✅ Performance metrics across demographic groups
- ✅ Known limitations and biases identified
- ⚠️ Requires ongoing monitoring plan implementation

**Risk Assessment:**

- **Risk Level**: High (impacts patient care decisions)
- **Bias Impact**: Moderate (disparities detected but not extreme)
- **Clinical Impact**: Significant (affects care allocation)
- **Mitigation Required**: Yes (bias correction needed)

### HIPAA compliance

**Data Protection:**

- ✅ No PHI in model outputs or logs
- ✅ Data encryption at rest and in transit
- ✅ Access controls and audit logging
- ✅ Minimum necessary data principle followed

### CMS innovation requirements

**Health Equity:**

- ⚠️ **Disparities detected**: Requires intervention
- ✅ **Bias measurement**: Comprehensive analysis completed
- ⚠️ **Correction plan needed**: Bias mitigation strategy required
- ✅ **Monitoring plan**: Continuous bias monitoring enabled

## Bias mitigation recommendations

### 1. Immediate actions

**Data Enhancement:**

- Collect more diverse training data, especially from underrepresented groups
- Review feature selection to remove unnecessary socioeconomic proxies
- Implement data quality checks for demographic balance

**Model Adjustments:**

- Apply demographic parity constraints during training
- Use threshold optimization by demographic group
- Implement fairness-aware ensemble methods

**Clinical Workflow:**

- Add human review for high-risk predictions in protected groups
- Implement second opinion protocols for demographic edge cases
- Train clinical staff on bias awareness and mitigation

### 2. Long-term improvements

**Algorithmic Fairness:**

- Develop group-specific models or adjustments
- Implement causal inference methods to separate clinical from social factors
- Use adversarial debiasing techniques during training

**Clinical Integration:**

- Embed fairness metrics into clinical dashboards
- Implement real-time bias monitoring alerts
- Regular bias audits (quarterly recommended)

**Regulatory Alignment:**

- Develop standard operating procedures for bias detection
- Create incident response plan for bias violations
- Establish governance committee for AI fairness oversight

## Monitoring and maintenance

### Continuous monitoring setup

```python
# Healthcare bias monitoring configuration
monitoring_config = {
    "frequency": "daily",
    "metrics": [
        "demographic_parity",
        "equal_opportunity",
        "predictive_parity"
    ],
    "alert_thresholds": {
        "demographic_parity": 0.02,
        "equal_opportunity": 0.02,
        "statistical_significance": 0.01
    },
    "stakeholder_alerts": [
        "chief_medical_officer",
        "quality_director",
        "risk_manager"
    ]
}
```

### Performance tracking

**Monthly Reviews:**

- Model performance by demographic group
- Bias metric trends over time
- Clinical outcome correlation analysis
- Staff feedback on prediction utility

**Quarterly Audits:**

- Comprehensive bias assessment
- Regulatory compliance review
- Clinical effectiveness evaluation
- Stakeholder feedback collection

## Healthcare AI best practices

### Clinical integration

#### 1. Human-AI Collaboration

- AI predictions as decision support, not replacement
- Clear indication of AI involvement in care decisions
- Easy override mechanisms for clinical judgment

#### 2. Transparency

- Explain AI predictions in clinical terms
- Document AI usage in patient records
- Train staff on AI capabilities and limitations

#### 3. Continuous Improvement

- Regular retraining with new data
- Bias monitoring and correction
- Clinical outcome feedback integration

### Regulatory preparedness

#### 1. Documentation

- Maintain complete audit trails
- Document all model changes and rationale
- Keep regulatory submission materials current

#### 2. Risk Management

- Regular risk assessments
- Incident reporting and response
- Continuous safety monitoring

#### 3. Quality Assurance

- Peer review of AI decisions
- Clinical validation studies
- External audit readiness

## Conclusion

Healthcare AI requires heightened attention to bias detection and mitigation due to:

- **High-stakes decisions** affecting patient care and outcomes
- **Complex regulatory environment** with multiple oversight bodies
- **Historical healthcare disparities** that AI can perpetuate or amplify
- **Professional and ethical obligations** to "first, do no harm"

GlassAlpha provides the comprehensive bias detection and audit capabilities needed for responsible healthcare AI deployment, ensuring models serve all patients fairly while maintaining clinical effectiveness.

**Next Steps:**

1. Implement bias mitigation strategies
2. Establish continuous monitoring processes
3. Engage clinical stakeholders in AI governance
4. Prepare regulatory submission materials
5. Plan for ongoing audit and improvement cycles

For additional healthcare AI guidance, consult clinical AI specialists and regulatory experts familiar with your specific healthcare setting and patient population.
