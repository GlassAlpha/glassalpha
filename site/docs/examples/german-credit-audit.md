# German Credit audit tutorial

Complete walkthrough of performing a comprehensive ML audit on the German Credit dataset using GlassAlpha. This tutorial demonstrates credit risk model evaluation with fairness analysis for regulatory compliance.

## Overview

The German Credit dataset is a classic benchmark for credit risk assessment, containing 1,000 loan applications with demographic and financial attributes. This tutorial shows how to audit an XGBoost credit scoring model for bias and compliance violations.

### What you'll learn

- How to configure GlassAlpha for credit risk models
- Interpreting model performance metrics
- Understanding SHAP explanations for credit decisions
- Identifying bias in credit scoring
- Generating regulatory-ready audit reports

### Use case context

Credit scoring models must comply with fair lending laws including:

- **Equal Credit Opportunity Act (ECOA)** - Prohibits discrimination based on protected characteristics
- **Fair Credit Reporting Act (FCRA)** - Requires accuracy and fairness in credit reporting
- **GDPR Article 22** - Right to explanation for automated decision-making

**Industry Context**: For banking-specific compliance requirements (SR 11-7, ECOA, FCRA), see the [Banking Compliance Guide](../compliance/banking-guide.md).

## Prerequisites

- GlassAlpha installed ([Installation Guide](../getting-started/installation.md))
- Basic understanding of credit risk modeling
- Familiarity with bias and fairness concepts

## Step 1: Understanding the dataset

### Dataset characteristics

The German Credit dataset contains:

- **1,000 loan applications**
- **21 features** (financial, demographic, and historical)
- **Binary target**: Good credit risk (70%) vs Bad credit risk (30%)
- **Protected attributes**: Gender, age, foreign worker status

### Key features

**Financial Attributes:**

- `credit_amount` - Loan amount requested
- `duration_months` - Loan duration
- `checking_account_status` - Current account balance
- `savings_account` - Savings account balance
- `employment_duration` - Length of current employment

**Demographic Attributes (Protected):**

- `gender` - Extracted from personal status (Male/Female)
- `age_group` - Categorized age ranges (Young/Middle/Senior)
- `foreign_worker` - Nationality/residency status

**Risk Factors:**

- `credit_history` - Past credit performance
- `purpose` - Loan purpose (car, furniture, etc.)
- `existing_credits_count` - Number of existing credits

## Step 2: Configuration setup

Create a configuration file for the German Credit audit:

```yaml
# german_credit.yaml
audit_profile: tabular_compliance

# Reproducibility for consistent results
reproducibility:
  random_seed: 42

# Data configuration
data:
  # GlassAlpha automatically downloads and processes the dataset
  path: ~/.glassalpha/data/german_credit_processed.csv
  target_column: credit_risk
  protected_attributes:
    - gender
    - age_group
    - foreign_worker

# LogisticRegression model (baseline, always available)
model:
  type: logistic_regression
  params:
    random_state: 42
    max_iter: 1000

# Explanation configuration
explainers:
  strategy: first_compatible
  priority:
    - treeshap # Exact SHAP values for XGBoost
  config:
    treeshap:
      max_samples: 1000
      check_additivity: true

# Comprehensive metrics for credit risk
metrics:
  performance:
    metrics:
      - accuracy
      - precision
      - recall
      - f1
      - auc_roc
      - classification_report

  fairness:
    metrics:
      - demographic_parity
      - equal_opportunity
      - equalized_odds
      - predictive_parity
    config:
      # Stricter thresholds for financial services
      demographic_parity:
        threshold: 0.05 # Maximum 5% difference between groups
      equal_opportunity:
        threshold: 0.05

# Professional audit report
report:
  template: standard_audit
  styling:
    color_scheme: professional
    compliance_statement: true
```

**Want to understand these options better?** See the [Configuration Guide](../getting-started/configuration.md) for detailed explanations of all settings.

**Considering a different model?** The [Model Selection Guide](../reference/model-selection.md) can help you choose between LogisticRegression, XGBoost, and LightGBM.

## Step 3: Running the audit

Execute the audit with regulatory compliance mode enabled:

```bash
# Generate comprehensive audit (use --fast for quick demos)
glassalpha audit \
  --config german_credit.yaml \
  --output german_credit_audit.pdf \
  --fast

# For production/regulatory: omit --fast and add --strict
# glassalpha audit --config german_credit.yaml --output audit.pdf --strict
```

### Expected execution

```
GlassAlpha Audit Generation
========================================
Loading configuration from: configs/german_credit.yaml
Audit profile: tabular_compliance
Strict mode: ENABLED
⚠️ Strict mode enabled - enforcing regulatory compliance

Running audit pipeline...
  Loading data and initializing components...
✓ Audit pipeline completed in 4.23s

📊 Audit Summary:
  ✅ Performance metrics: 6 computed
     ✅ accuracy: 75.2%
  ⚖️ Fairness metrics: 12/12 computed
     ⚠️ Bias detected in: gender.demographic_parity
  🔍 Explanations: ✅ Global feature importance
     Most important: checking_account_status (+0.234)
  📋 Dataset: 1,000 samples, 21 features
  🔧 Components: 3 selected
     Model: xgboost

Generating PDF report: german_credit_audit.pdf
✓ Saved plot to /tmp/plots/shap_importance.png
✓ Saved plot to /tmp/plots/performance_summary.png
✓ Saved plot to /tmp/plots/fairness_analysis.png
✓ Saved plot to /tmp/plots/confusion_matrix.png

🎉 Audit Report Generated Successfully!
==================================================
📁 Output: /path/to/german_credit_audit.pdf
📊 Size: 1,247,832 bytes (1.2 MB)
⏱️ Total time: 5.67s
   • Pipeline: 4.23s
   • PDF generation: 1.44s

🛡️ Strict mode: Report meets regulatory compliance requirements

The audit report is ready for review and regulatory submission.
```

## Step 4: Interpreting the results

### Model performance analysis

**Overall Performance:**

- **Accuracy: 75.2%** - Model correctly classifies 3 out of 4 loan applications
- **AUC-ROC: 0.821** - Strong discriminative ability
- **Precision: 82.1%** - Of approved loans, 82% are actually good risks
- **Recall: 71.4%** - Model identifies 71% of all good credit risks

**Business Interpretation:**

- Model performance is reasonable for credit risk assessment
- Conservative approach with higher precision than recall (safer lending)
- False positive rate (approving bad risks) is 18% - acceptable for many lenders

### SHAP explanations

**Global Feature Importance (Top 5):**

1. **`checking_account_status` (+0.234)**

   - Most predictive feature
   - Higher account balances strongly indicate good credit risk
   - Aligns with traditional banking wisdom

2. **`credit_history` (+0.187)**

   - Past credit performance is highly predictive
   - Good credit history significantly improves approval odds
   - Critical factor in credit underwriting

3. **`duration_months` (-0.156)**

   - Longer loan terms increase risk
   - Model correctly identifies duration as risk factor
   - Consistent with increased default probability over time

4. **`credit_amount` (-0.142)**

   - Larger loan amounts increase risk
   - Higher stakes loans have higher default rates
   - Model appropriately weights loan size

5. **`age_years` (+0.098)**
   - Older applicants generally have lower risk
   - Reflects financial stability with age
   - Note: Age is correlated with protected attribute

**Individual Prediction Example:**

For a 35-year-old male requesting €2,000 for a car:

- **Base probability:** 0.70 (population average)
- **Checking account (positive):** +0.15
- **Good credit history:** +0.12
- **Car purchase purpose:** +0.08
- **Age (35):** +0.04
- **Final probability:** 0.89 (strong approval recommendation)

### Fairness analysis results

**Demographic Parity Analysis:**

**Gender Bias (DETECTED):**

- **Male approval rate:** 72.3%
- **Female approval rate:** 66.1%
- **Difference:** 6.2% (exceeds 5% threshold)
- **Statistical significance:** p < 0.05
- **Conclusion:** Potential gender discrimination

**Age Group Analysis:**

- **Young (18-30):** 68.4% approval rate
- **Middle (31-50):** 74.2% approval rate
- **Senior (51+):** 78.9% approval rate
- **Maximum difference:** 10.5% (Young vs Senior)
- **Conclusion:** Age-based disparities detected

**Foreign Worker Status:**

- **German workers:** 73.1% approval rate
- **Foreign workers:** 71.8% approval rate
- **Difference:** 1.3% (within acceptable range)
- **Conclusion:** No significant bias detected

### Equal opportunity analysis

**True Positive Rate Parity:**

- Measures whether qualified applicants are approved equally across groups
- **Gender:** Males 89.2% vs Females 84.7% (4.5% difference - borderline)
- **Age:** Varies from 82.1% to 91.3% across age groups
- **Foreign worker:** No significant difference

### Risk assessment

**High Risk Findings:**

#### Gender Discrimination Risk

- 6.2% approval rate difference violates ECOA guidelines
- Could result in regulatory action or lawsuits
- Requires immediate model adjustment or feature engineering

#### Age-Based Disparities

- 10.5% difference across age groups may violate age discrimination laws
- Consider removing age-correlated features
- Evaluate business justification for age-related patterns

**Medium Risk Findings:**

#### Correlated Protected Attributes

- Several features correlate with protected characteristics
- May create indirect discrimination
- Consider fairness-aware modeling techniques

**Compliance Assessment:**

- **ECOA Compliance:** ❌ FAIL (gender bias detected)
- **FCRA Accuracy:** ✅ PASS (75% accuracy meets standards)
- **GDPR Article 22:** ⚠️ REVIEW (explanations available but bias concerns)

## Step 5: Regulatory recommendations

### Immediate actions required

#### Address Gender Bias

```python
# Consider preprocessing approaches:
# - Remove gender-correlated features
# - Apply fairness constraints during training
# - Post-processing bias mitigation
```

#### Feature Engineering

- Audit features correlated with protected attributes
- Consider removing or transforming biased features
- Implement fairness-aware feature selection

#### Model Adjustment

- Retrain with fairness constraints
- Consider ensemble methods with bias reduction
- Validate improvements with new audit

### Long-term compliance strategy

#### Ongoing Monitoring

- Regular bias audits on new data
- Statistical tests for demographic parity
- Performance monitoring across protected groups

#### Documentation Requirements

- Maintain complete audit trails
- Document bias mitigation efforts
- Prepare regulatory submission packages

#### Process Improvements

- Establish fairness review boards
- Implement bias testing in model development
- Create remediation procedures for biased decisions

## Step 6: Business impact analysis

### Financial impact

**Current Model:**

- **Approval rate:** 70% overall
- **Expected default rate:** ~25% (based on precision)
- **Revenue impact:** Moderate (typical for conservative lending)

**With Bias Correction:**

- **May increase approvals for underrepresented groups**
- **Could slightly increase default risk if not carefully implemented**
- **Compliance benefits outweigh small performance trade-offs**

### Legal risk mitigation

**Before Correction:**

- High risk of ECOA violations
- Potential for class-action lawsuits
- Regulatory enforcement actions

**After Correction:**

- Compliance with fair lending laws
- Reduced legal exposure
- Improved reputation and stakeholder trust

## Step 7: Next steps and recommendations

### Technical remediation

#### Implement Fairness Constraints

```python
# Example: Add fairness penalty to XGBoost training
# Consider libraries like fairlearn or aif360
```

#### Alternative Modeling Approaches

- Pre-processing: Remove biased features or transform data
- In-processing: Fairness-constrained optimization
- Post-processing: Adjust predictions to achieve parity

#### Validation Strategy

- Cross-validation with fairness metrics
- Holdout testing on diverse populations
- A/B testing for production deployment

### Operational changes

#### Model Governance

- Establish bias testing requirements
- Create fairness review processes
- Implement continuous monitoring

#### Human Oversight

- Manual review for borderline cases
- Appeals process for declined applicants
- Regular expert review of model decisions

#### Stakeholder Engagement

- Train staff on fair lending requirements
- Engage with compliance and legal teams
- Communicate changes to management

## Conclusion

This German Credit audit revealed a technically sound but biased model that requires immediate attention before production deployment. The audit demonstrated:

**Strengths:**

- Strong predictive performance (75% accuracy, 0.82 AUC)
- Interpretable feature importance aligned with domain knowledge
- Comprehensive bias detection and measurement

**Critical Issues:**

- Gender bias exceeding regulatory thresholds
- Age-based disparities requiring investigation
- Non-compliance with fair lending regulations

**Action Plan:**

1. Implement bias mitigation techniques
2. Retrain model with fairness constraints
3. Re-audit improved model
4. Deploy with ongoing monitoring

This tutorial demonstrates how GlassAlpha enables thorough, regulatory-ready ML auditing that identifies both performance strengths and compliance risks, providing the detailed analysis necessary for responsible AI deployment in regulated industries.

## Additional resources

- [Configuration Guide](../getting-started/configuration.md) - Detailed configuration options
- [CLI Reference](../reference/cli.md) - Complete command documentation
- [Trust & Deployment](../reference/trust-deployment.md) - Architecture, licensing, security, and compliance
- [Troubleshooting Guide](../reference/troubleshooting.md) - Common issues and solutions

For questions or support, please visit our [GitHub repository](https://github.com/GlassAlpha/glassalpha) or contact our team.
