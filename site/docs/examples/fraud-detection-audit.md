# Financial fraud detection audit tutorial

Complete walkthrough of performing a comprehensive ML audit on a credit card fraud detection model using GlassAlpha. This tutorial demonstrates how to audit automated fraud detection systems for compliance with payment card industry standards and anti-money laundering regulations.

## Overview

Financial institutions use machine learning models to detect fraudulent credit card transactions in real-time. These models must balance:

- **False Positive Minimization**: Avoid blocking legitimate transactions
- **Fraud Detection Maximization**: Catch actual fraudulent activity
- **Regulatory Compliance**: Meet PCI DSS and AML requirements
- **Fairness**: Avoid bias against protected demographic groups

### What you'll learn

- How to configure GlassAlpha for fraud detection models
- Interpreting performance metrics for fraud detection systems
- Understanding SHAP explanations for transaction risk assessment
- Identifying bias in fraud detection across demographic groups
- Generating regulatory-ready audit reports for financial compliance

### Regulatory context

Fraud detection models must comply with:

- **Payment Card Industry Data Security Standard (PCI DSS)**: Security and fraud monitoring requirements
- **Bank Secrecy Act (BSA)/Anti-Money Laundering (AML)**: Suspicious activity reporting
- **Fair Credit Reporting Act (FCRA)**: Accuracy in consumer reporting
- **Equal Credit Opportunity Act (ECOA)**: Non-discrimination in credit decisions

**Industry Context**: For fraud detection-specific compliance requirements (FCRA adverse action, FTC fairness), see the [Fraud Detection Compliance Guide](../compliance/fraud-guide.md).

## Prerequisites

- GlassAlpha installed ([Installation Guide](../getting-started/installation.md))
- Basic understanding of fraud detection systems and payment processing
- Familiarity with bias and fairness concepts in financial services

## Step 1: Understanding the fraud detection dataset

### Dataset characteristics

The fraud detection dataset contains:

- **50,000 synthetic credit card transactions**
- **20 features** covering transaction details, customer behavior, and risk factors
- **Binary target**: Fraudulent transaction (1) vs Legitimate (0)
- **1% fraud rate** (realistic for credit card fraud)
- **Protected attributes**: Gender, age groups for fairness analysis

### Key features

**Transaction Information:**

- `amount` - Transaction amount ($1 - $10,000)
- `merchant_category` - Merchant type (0-15 categories)
- `transaction_hour` - Hour of transaction (0-23)
- `transaction_day` - Day of week (0-6, Monday-Sunday)
- `transaction_type` - Online, in-store, or ATM transaction

**Customer Behavior:**

- `cardholder_age` - Customer age (18-85)
- `account_age_months` - Account tenure (0-120 months)
- `transaction_count_24h` - Transactions in last 24 hours (0-50)
- `amount_avg_24h` - Average transaction amount in last 24 hours
- `time_since_last_txn` - Minutes since last transaction (0-1440)

**Risk Indicators:**

- `location_distance` - Distance from customer's home location (0-1000 miles)
- `device_fingerprint_risk` - Device identification risk score (0-100)
- `ip_geolocation_risk` - IP address risk assessment (0-100)
- `merchant_country_risk` - Merchant location risk score (0-100)
- `velocity_check_failed` - Failed transaction velocity checks
- `amount_deviation_score` - Unusual amount pattern score (0-100)
- `weekend_transaction` - Weekend vs. weekday indicator
- `international_transaction` - Domestic vs. international

## Step 2: Configuration setup

Create a configuration file for the fraud detection audit:

```yaml
# fraud_detection_audit.yaml
# Direct configuration for fraud detection

# Reproducibility for regulatory compliance
reproducibility:
  random_seed: 42

# Data configuration
data:
  path: ~/.glassalpha/data/fraud_detection.csv
  target_column: is_fraud
  protected_attributes:
    - gender
    - age_group

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
    - treeshap # XGBoost supports TreeSHAP
    - kernelshap # Universal fallback

# Fraud-specific metrics
metrics:
  performance:
    metrics:
      - accuracy
      - precision # Minimize false positives
      - recall # Maximize fraud detection
      - f1 # Balance precision and recall
      - auc_roc # Overall model quality

  fairness:
    metrics:
      - demographic_parity # Equal fraud detection rates
      - equal_opportunity # Equal TPR for fraud detection
      - predictive_parity # Equal precision across groups
    config:
      # Stricter thresholds for fraud detection
      demographic_parity:
        threshold: 0.02 # Maximum 2% difference
      equal_opportunity:
        threshold: 0.02
      predictive_parity:
        threshold: 0.02

# Professional audit report
report:
  template: standard_audit
  styling:
    color_scheme: professional
    compliance_statement: true
```

## Step 3: Running the audit

Execute the fraud detection audit:

```bash
# Generate comprehensive audit (use --fast for quick demos)
glassalpha audit \
  --config fraud_detection_audit.yaml \
  --output fraud_detection_audit.pdf \
  --fast

# For production/regulatory: omit --fast and add --strict
# glassalpha audit --config fraud_detection_audit.yaml --output audit.pdf --strict
```

### Expected execution

```
GlassAlpha Audit Generation
========================================
Loading configuration from: fraud_detection_audit.yaml
Audit profile: fraud_detection
Strict mode: ENABLED
⚠️ Strict mode enabled - enforcing regulatory compliance

Running audit pipeline...
  Loading data and initializing components...
✓ Audit pipeline completed in 4.23s

📊 Audit Summary:
  ✅ Performance metrics: 5 computed
     ✅ accuracy: 98.7%
  ⚖️ Fairness metrics: 9/9 computed
     ✅ demographic_parity: 0.008 (within 2% threshold)
  🔍 Explanations: ✅ Global feature importance
     Most important: amount_deviation_score (+0.342)
  📋 Dataset: 50,000 samples, 20 features
  🔧 Components: 3 selected
     Model: xgboost

Generating PDF report: fraud_detection_audit.pdf
✓ Saved plot to /tmp/plots/shap_importance.png
✓ Saved plot to /tmp/plots/performance_summary.png
✓ Saved plot to /tmp/plots/fairness_analysis.png
✓ Saved plot to /tmp/plots/confusion_matrix.png

🎉 Audit Report Generated Successfully!
==================================================
📁 Output: /path/to/fraud_detection_audit.pdf
📊 Size: 1,287,654 bytes (1.3 MB)
⏱️ Total time: 5.18s
   • Pipeline: 4.23s
   • PDF generation: 0.95s

🛡️ Strict mode: Report meets regulatory compliance requirements
```

## Step 4: Interpreting the results

### Model performance analysis

**Overall Performance:**

- **Accuracy: 98.7%** - Model correctly classifies 99% of transactions
- **AUC-ROC: 0.923** - Excellent discriminative ability for fraud detection
- **Precision: 87.4%** - Of flagged transactions, 87% are actually fraudulent
- **Recall: 79.2%** - Model catches 79% of all fraudulent transactions

**Fraud Detection Business Interpretation:**

- **False Positive Rate**: 13% of legitimate transactions incorrectly flagged
- **False Negative Rate**: 21% of fraudulent transactions missed
- **Business Impact**: Balance between fraud losses and customer friction
- **Regulatory Impact**: Must justify false positive rates to card networks

### SHAP explanations

**Global Feature Importance (Top 5):**

**1. `amount_deviation_score` (+0.342)**

- Most important fraud indicator
- Measures how unusual the transaction amount is for this customer
- Critical for detecting anomalous spending patterns

**2. `velocity_check_failed` (+0.287)**

- Transaction velocity violations (too many transactions too quickly)
- Strong indicator of card testing or account takeover
- Real-time fraud prevention mechanism

**3. `location_distance` (+0.234)**

- Distance from customer's normal geographic area
- Indicates potential card-not-present fraud
- Must be monitored for legitimate travel patterns

**4. `merchant_country_risk` (+0.198)**

- Risk assessment of merchant location
- Higher risk for international or high-risk countries
- Balances security with legitimate international commerce

**5. `transaction_hour` (-0.156)**

- Time of day patterns (negative = unusual timing)
- Fraud often occurs during off-hours
- Must account for legitimate late-night transactions

**Individual Transaction Example:**
For a $2,500 international transaction at 3 AM from 500 miles away:

- **Base fraud probability:** 0.012 (1.2% population rate)
- **Amount deviation (high):** +0.15 probability increase
- **Velocity check (failed):** +0.12 probability increase
- **Location distance (far):** +0.08 probability increase
- **International transaction:** +0.06 probability increase
- **Transaction hour (unusual):** +0.04 probability increase
- **Final fraud probability:** 0.452 (45.2% - flag for review)

### Fairness analysis results

**Demographic Parity Analysis:**

**Age Group Analysis:**

- **Young (18-24):** 1.2% fraud detection rate
- **Young Adult (25-34):** 1.1% fraud detection rate
- **Middle Age (35-49):** 0.9% fraud detection rate
- **Senior (50-64):** 0.8% fraud detection rate
- **Elderly (65+):** 0.7% fraud detection rate
- **Maximum difference:** 0.5% (Young vs. Elderly)
- **Conclusion:** ✅ Within 2% threshold, no significant age bias

**Gender Analysis:**

- **Male:** 1.0% fraud detection rate
- **Female:** 0.9% fraud detection rate
- **Difference:** 0.1% (well within acceptable range)
- **Conclusion:** ✅ No gender bias detected

**Predictive Parity Analysis:**

- **Young:** 86.2% precision (of flagged young customers, 86% actually fraudulent)
- **Middle Age:** 87.8% precision
- **Difference:** 1.6% (within acceptable range)
- **Conclusion:** ✅ Consistent fraud detection accuracy across age groups

### Risk assessment

**Low Risk Findings:**

1. **Model Performance**: Excellent overall accuracy with appropriate precision/recall balance
2. **Fairness Compliance**: All demographic groups within acceptable thresholds
3. **Feature Interpretability**: Clear business logic in top risk factors

**Medium Risk Findings:**

1. **False Positive Impact**: 13% false positive rate may cause customer friction
2. **Geographic Bias**: Location-based features could disadvantage rural customers
3. **International Transactions**: Higher scrutiny may impact legitimate global commerce

**Compliance Assessment:**

- **PCI DSS Compliance:** ✅ PASS - Appropriate fraud monitoring implemented
- **BSA/AML Compliance:** ✅ PASS - Suspicious activity detection in place
- **FCRA Accuracy:** ✅ PASS - 99% accuracy meets standards
- **ECOA Compliance:** ✅ PASS - No demographic discrimination detected

## Step 5: Regulatory recommendations

### Immediate actions required

**1. Monitor False Positive Rates**

- Track customer complaints about blocked legitimate transactions
- Consider customer communication about fraud prevention measures
- Implement appeals process for declined transactions

**2. Geographic Feature Review**

- Analyze location risk factors for potential rural/urban bias
- Consider normalizing by local economic factors
- Validate with diverse geographic test sets

**3. International Transaction Handling**

- Ensure legitimate international commerce isn't overly burdened
- Consider customer travel pattern analysis
- Implement graduated response based on risk level

### Long-term compliance strategy

**1. Ongoing Monitoring**

- Regular bias audits on new transaction patterns
- Performance monitoring across demographic groups
- Model drift detection for changing fraud patterns

**2. Regulatory Reporting**

- Maintain fraud detection effectiveness metrics
- Document bias mitigation measures
- Prepare regulatory examination materials

**3. Model Governance**

- Establish fraud detection accuracy standards
- Implement model validation procedures
- Create change management processes

## Step 6: Business impact analysis

### Financial impact

**Current Model:**

- **Fraud detection rate:** 79% (catches 79% of fraud)
- **False positive rate:** 13% (13% of legitimate transactions flagged)
- **Estimated fraud losses prevented:** $2.3M annually
- **Estimated customer friction costs:** $180K annually

**Optimization Opportunities:**

- **Precision improvement**: Could reduce false positives by 20%
- **Recall improvement**: Could catch additional 5% of fraud
- **Net benefit**: $300K+ annual improvement potential

### Regulatory risk mitigation

**Before Audit:**

- Potential compliance violations if bias undetected
- Regulatory scrutiny without proper documentation
- Customer complaints without clear appeals process

**After Audit:**

- Documented compliance with PCI DSS requirements
- Demonstrated fairness across demographic groups
- Clear audit trail for regulatory examinations

## Step 7: Next steps and recommendations

### Technical improvements

**1. Model Optimization**

- Feature engineering for better fraud signal extraction
- Ensemble methods combining multiple detection approaches
- Real-time model updating based on new fraud patterns

**2. Customer Experience Enhancement**

- Personalized fraud thresholds based on customer history
- Clear communication about why transactions were flagged
- Streamlined appeals process for legitimate transactions

**3. Advanced Analytics**

- Customer segmentation for risk-based approaches
- Network analysis for organized fraud rings
- Temporal pattern analysis for seasonal fraud trends

### Operational changes

**1. Fraud Team Integration**

- Real-time model performance dashboards
- Analyst feedback loop for model improvement
- Investigation workflow integration

**2. Customer Communication**

- Transparent fraud prevention messaging
- Educational materials about fraud indicators
- Clear policies for transaction disputes

**3. Regulatory Engagement**

- Regular reporting to payment card networks
- Participation in industry fraud prevention initiatives
- Collaboration with law enforcement on fraud trends

## Conclusion

This fraud detection audit revealed a highly effective model that successfully balances fraud prevention with customer experience while maintaining regulatory compliance. The audit demonstrated:

**Strengths:**

- Excellent overall performance (99% accuracy, 92% AUC)
- Appropriate precision/recall balance for fraud detection
- No demographic bias detected across protected groups
- Clear interpretability of fraud risk factors

**Areas for Enhancement:**

- Optimization of false positive rates
- Enhanced geographic risk factor analysis
- Improved international transaction handling

**Compliance Status:**

- ✅ PCI DSS compliance requirements met
- ✅ BSA/AML suspicious activity monitoring in place
- ✅ FCRA accuracy standards satisfied
- ✅ ECOA non-discrimination requirements fulfilled

This tutorial demonstrates how GlassAlpha enables thorough, regulatory-ready ML auditing that identifies both technical performance strengths and compliance risks, providing the detailed analysis necessary for responsible AI deployment in regulated financial fraud detection systems.

## Additional resources

- [Configuration Guide](../getting-started/configuration.md) - Detailed configuration options
- [CLI Reference](../reference/cli.md) - Complete command documentation
- [Trust & Deployment](../reference/trust-deployment.md) - Architecture, licensing, security, and compliance
- [Troubleshooting Guide](../reference/troubleshooting.md) - Common issues and solutions

For questions or support, please visit our [GitHub repository](https://github.com/GlassAlpha/glassalpha) or contact our team.
