# ML Manager Workflow

Guide for ML managers, team leads, and AI governance professionals overseeing model development teams and establishing organization-wide compliance policies.

## Overview

This guide is for managers and team leads who need to:

- Establish model governance policies
- Configure compliance gates for teams
- Track audit status across model portfolio
- Train teams on compliance requirements
- Report to executives and regulators
- Evaluate vendor tools and processes

**Not a manager?** For hands-on implementation, see [ML Engineer Workflow](ml-engineer-workflow.md). For compliance review, see [Compliance Officer Workflow](compliance-workflow.md).

## Key Capabilities

### Policy-as-Code Governance

Centralized compliance configuration:

- Define organization-wide fairness thresholds
- Specify required metrics for all models
- Enforce determinism and reproducibility
- Version-control policy changes

### Portfolio-Level Tracking

Monitor multiple models:

- Registry of all audited models
- Compliance status dashboard
- Failed gate tracking
- Trend analysis over time

### Team Enablement

Support model development teams:

- Template configurations for common use cases
- Training materials and best practices
- Self-service audit generation
- Clear escalation paths

## Typical Workflows

### Workflow 1: Establishing Organization-Wide Compliance Policy

**Scenario**: Define baseline compliance requirements for all credit models in your organization.

#### Step 1: Define policy requirements

Work with legal, risk, and compliance teams to establish thresholds:

**Example requirements discussion:**

- Legal: "ECOA requires no disparate impact - what's acceptable threshold?"
- Risk: "What calibration error is acceptable for our risk appetite?"
- Compliance: "SR 11-7 requires monitoring - what metrics should we track?"

**Documented requirements:**

```yaml
# Policy: Credit Models Baseline (v1.0)
# Effective: 2025-01-01
# Applies to: All consumer credit models
# Review frequency: Quarterly

Requirements:
  - Calibration ECE < 5%
  - Demographic parity difference < 10%
  - Equalized odds difference < 15%
  - Minimum group size ≥ 30
  - Statistical power ≥ 0.80
  - Model stability under ±10% demographic shifts
```

#### Step 2: Create policy configuration

Translate requirements into policy-as-code:

```yaml
# configs/policy/org_credit_baseline_v1.yaml
policy_name: "Organization Credit Model Baseline"
version: "1.0"
effective_date: "2025-01-01"
applies_to: ["credit_scoring", "loan_pricing", "risk_assessment"]
citation: "SR 11-7, ECOA, Internal Risk Policy RP-2024-03"

gates:
  # Performance requirements
  - name: "Minimum Model Performance"
    metric: "roc_auc"
    threshold: 0.75
    comparison: "greater_than"
    severity: "error"
    clause: "Internal Risk Policy RP-2024-03 §2.1"
    rationale: "Minimum discriminative ability for credit decisions"

  # Calibration requirements
  - name: "Calibration Quality"
    metric: "expected_calibration_error"
    threshold: 0.05
    comparison: "less_than"
    severity: "error"
    clause: "SR 11-7 §III.B.1"
    rationale: "Predicted probabilities must align with observed outcomes"

  # Fairness requirements
  - name: "Demographic Parity"
    metric: "demographic_parity_difference"
    threshold: 0.10
    comparison: "less_than"
    severity: "error"
    clause: "ECOA fair lending"
    rationale: "Maximum acceptable approval rate difference across protected groups"

  - name: "Equalized Odds"
    metric: "equalized_odds_difference"
    threshold: 0.15
    comparison: "less_than"
    severity: "warning"
    clause: "ECOA fair lending"
    rationale: "TPR/FPR parity across protected groups"

  # Data quality requirements
  - name: "Minimum Statistical Power"
    metric: "min_group_size"
    threshold: 30
    comparison: "greater_than"
    severity: "error"
    clause: "SR 11-7 §III.B.2"
    rationale: "Adequate sample size for statistical significance"

  - name: "Statistical Power for Bias Detection"
    metric: "statistical_power"
    threshold: 0.80
    comparison: "greater_than"
    severity: "warning"
    clause: "SR 11-7 §III.B.2"
    rationale: "Ability to detect 5pp difference with 95% confidence"

  # Robustness requirements
  - name: "Demographic Shift Robustness"
    metric: "max_shift_degradation"
    threshold: 0.10
    comparison: "less_than"
    severity: "error"
    clause: "SR 11-7 §III.A.3"
    rationale: "Model must remain stable under population changes"

reproducibility:
  strict_mode_required: true
  seed_required: true
  manifest_required: true
  git_commit_required: true

review_cycle:
  frequency: "quarterly"
  next_review: "2025-04-01"
  owner: "Chief Risk Officer"
```

#### Step 3: Communicate to teams

Create team-facing documentation:

````markdown
# Credit Model Compliance Policy (v1.0)

## What Changed

Effective January 1, 2025, all credit models must meet baseline compliance gates.

## Required Actions

1. Update your audit config to reference this policy
2. Run audit with `--policy-gates configs/policy/org_credit_baseline_v1.yaml`
3. Address any failed gates before requesting deployment approval

## Policy Gates

- **Calibration**: ECE < 5% (ERROR - blocks deployment)
- **Fairness**: Demographic parity < 10% (ERROR - blocks deployment)
- **Performance**: AUC ≥ 0.75 (ERROR - blocks deployment)
- **Robustness**: Max degradation < 10% under shift (ERROR)
- **Equalized Odds**: < 15% (WARNING - requires documentation)

## How to Use

```bash
glassalpha audit \
  --config your_model_config.yaml \
  --policy-gates configs/policy/org_credit_baseline_v1.yaml \
  --strict \
  --output audit_report.pdf
```
````

## Getting Help

- **Policy questions**: Contact Risk Management Team
- **Technical support**: #ml-compliance Slack channel
- **Approval requests**: Submit audit report to Model Review Board

## Resources

- [ML Engineer Workflow](../guides/ml-engineer-workflow.md)
- [Compliance Readiness Checklist](../compliance/compliance-readiness-checklist.md)
- [Example Configs](https://github.com/GlassAlpha/glassalpha/tree/main/configs) - CCPA, GDPR compliance configs on GitHub

````

#### Step 4: Pilot with one team

Roll out to one team first:

```bash
# Team validates policy on existing models
cd team-credit-scoring/
glassalpha audit \
  --config models/current_model.yaml \
  --policy-gates ../configs/policy/org_credit_baseline_v1.yaml \
  --strict
````

**Pilot checklist:**

- [ ] Policy gates run successfully
- [ ] Failed gates are actionable (clear fix path)
- [ ] Documentation is sufficient for self-service
- [ ] Performance impact is acceptable (<5 second overhead)
- [ ] Team understands remediation process

#### Step 5: Organization-wide rollout

After successful pilot:

1. **Announce**: Email + Slack announcement with 2-week lead time
2. **Train**: Host training session (live + recorded)
3. **Support**: Dedicated support channel for first month
4. **Enforce**: Update CI/CD pipelines to require policy gates
5. **Monitor**: Track adoption and failed gates

### Workflow 2: Model Portfolio Tracking

**Scenario**: Track compliance status across 20+ models in production.

#### Step 1: Establish registry

Create centralized audit registry:

```bash
# Initialize registry (SQLite for simplicity)
glassalpha registry init --database model_registry.db

# Or use shared database
glassalpha registry init --database postgresql://host/model_registry
```

#### Step 2: Configure automated uploads

Teams submit audits to registry:

```yaml
# Add to all model configs: configs/common_settings.yaml
registry:
  enabled: true
  database: "postgresql://registry.company.com/models"
  auto_submit: true
  metadata:
    team: "{{ TEAM_NAME }}"
    cost_center: "{{ COST_CENTER }}"
```

**In CI/CD pipeline:**

```bash
# Audit runs and automatically submits to registry
export TEAM_NAME="credit-risk"
export COST_CENTER="CR-4501"

glassalpha audit \
  --config model_config.yaml \
  --policy-gates org_policy.yaml \
  --output audit.pdf
```

#### Step 3: Query registry for portfolio view

```bash
# List all models with failed gates
glassalpha registry list --failed-gates

# Output:
# model_id          | team        | last_audit | failed_gates | status
# credit_model_v2   | credit-risk | 2025-01-05 | 2            | BLOCKED
# fraud_model_v3    | fraud       | 2025-01-03 | 0            | APPROVED
# loan_pricing_v1   | lending     | 2024-12-20 | 1            | REVIEW
```

#### Step 4: Generate executive dashboard

```python
# scripts/generate_dashboard.py
import glassalpha as ga
import pandas as pd
import matplotlib.pyplot as plt

# Query registry
registry = ga.Registry.connect("postgresql://host/registry")
audits = registry.query_all(last_n_days=30)

# Summary metrics
total_models = len(audits)
passed = sum(1 for a in audits if a.status == "PASSED")
blocked = sum(1 for a in audits if a.status == "BLOCKED")
review = sum(1 for a in audits if a.status == "REVIEW")

# Create summary
summary = pd.DataFrame({
    "Status": ["Passed", "Blocked", "Needs Review"],
    "Count": [passed, blocked, review],
    "Percentage": [
        passed/total_models*100,
        blocked/total_models*100,
        review/total_models*100
    ]
})

# Generate plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Status pie chart
ax1.pie(summary["Count"], labels=summary["Status"], autopct='%1.1f%%')
ax1.set_title("Model Compliance Status")

# Failed gates breakdown
failed_gates = registry.query_failed_gates_summary()
ax2.barh(failed_gates["gate_name"], failed_gates["count"])
ax2.set_xlabel("Number of Models Failing Gate")
ax2.set_title("Most Common Failed Gates")

plt.tight_layout()
plt.savefig("reports/portfolio_dashboard.pdf")
```

#### Step 5: Executive reporting

Monthly report to leadership:

```markdown
# Model Compliance Report - January 2025

## Executive Summary

- **Total Models**: 24 in production
- **Compliance Status**: 21 passed (87.5%), 2 blocked (8.3%), 1 under review (4.2%)
- **Top Issue**: Calibration quality (2 models failing ECE threshold)
- **Action Required**: Credit Model v2 and Loan Pricing v1 need recalibration

## Portfolio Overview

[Include: portfolio_dashboard.pdf]

## Models Requiring Attention

### Credit Model v2 (BLOCKED)

- **Issue**: Calibration ECE = 0.078 (threshold: 0.05)
- **Impact**: Cannot deploy to production
- **Owner**: Credit Risk Team
- **Timeline**: Remediation in progress, reaudit scheduled 2025-01-15
- **Risk**: Low (model not yet in production)

### Loan Pricing v1 (REVIEW)

- **Issue**: Equalized odds difference = 0.17 (warning threshold: 0.15)
- **Impact**: Requires documentation and monitoring
- **Owner**: Lending Team
- **Action**: Document mitigation strategy, increase monitoring frequency
- **Risk**: Medium (model in production, regulatory review possible)

## Trends

- Fairness metrics improving quarter-over-quarter
- Average audit turnaround time: 3.2 days (down from 5.1 days)
- 95% of models meeting calibration requirements (up from 87%)

## Recommendations

1. Increase training on calibration techniques
2. Update policy to require monthly re-audit for models with warning-level gates
3. Invest in automated monitoring for production models
```

### Workflow 3: Team Training and Enablement

**Scenario**: Onboard new team members on model compliance process.

#### Step 1: Create training materials

**Training deck outline:**

1. **Why Model Auditing Matters** (10 min)

   - Regulatory landscape (SR 11-7, ECOA, EU AI Act)
   - Real-world consequences of model bias
   - Company policy and requirements

2. **GlassAlpha Basics** (15 min)

   - Architecture overview
   - Config-driven approach
   - Policy gates and compliance

3. **Hands-On: Your First Audit** (20 min)

   - Live demo with German Credit dataset
   - Interpreting audit reports
   - Addressing failed gates

4. **Production Workflow** (10 min)

   - CI/CD integration
   - Registry submission
   - Approval process

5. **Q&A and Resources** (5 min)

#### Step 2: Create template repository

```bash
# Create org template repo
glassalpha-template-credit/
├── README.md                  # Getting started guide
├── .github/
│   └── workflows/
│       └── model-audit.yml    # Pre-configured CI
├── configs/
│   ├── dev_audit.yaml         # Fast iteration config
│   ├── prod_audit.yaml        # Full audit config
│   └── policy/
│       └── org_baseline.yaml  # Organization policy (symlink)
├── data/
│   └── README.md              # Data requirements
├── models/
│   └── README.md              # Model saving conventions
├── notebooks/
│   └── example_audit.ipynb    # Starter notebook
└── tests/
    └── test_audit.py          # Audit validation tests
```

**README template:**

````markdown
# Credit Model Audit Template

Quick-start template for credit model compliance audits.

## Quick Start

```bash
# 1. Clone this template
# 2. Add your model and data
cp my_model.pkl models/
cp my_data.csv data/

# 3. Update config
vim configs/prod_audit.yaml

# 4. Run audit
glassalpha audit --config configs/prod_audit.yaml --output audit.pdf
```
````

## Configuration

- `dev_audit.yaml`: Fast iteration (reduced samples, skips slow sections)
- `prod_audit.yaml`: Full audit (required for deployment approval)

## Policy Gates

This template uses organization baseline policy (v1.0):

- Calibration ECE < 5%
- Demographic parity < 10%
- AUC ≥ 0.75

See [Policy Gates](#policy-gates) section above for details.

## CI/CD

Push to main triggers automatic audit. Failed gates block merge.

## Getting Help

- Slack: #ml-compliance
- Docs: [ML Engineer Workflow](https://docs.company.com/ml-compliance)
- Support: ml-compliance@company.com

````

#### Step 3: Conduct training sessions

**Session format:**
- Monthly introductory sessions (1 hour)
- Advanced topics quarterly (deep dives)
- Office hours weekly (drop-in support)

**Track attendance:**
```python
# scripts/track_training.py
training_log = {
    "date": "2025-01-15",
    "session": "Introduction to Model Auditing",
    "attendees": 12,
    "teams_represented": ["credit-risk", "fraud", "lending"],
    "feedback_score": 4.6
}
````

#### Step 4: Create self-service resources

**Internal wiki:**

- FAQ: Common audit errors and fixes
- Troubleshooting guide: Step-by-step debugging
- Best practices: Team-specific recommendations
- Example audits: Real (anonymized) audit reports

**Slack bot for quick help:**

```
/audit-help failed-gate demographic_parity

Response:
📘 Failed Gate: Demographic Parity

Issue: Approval rate difference between protected groups exceeds 10% threshold.

Common causes:
1. Imbalanced training data
2. Proxy features correlated with protected attributes
3. Threshold not optimized for fairness

Recommended fixes:
1. Check for dataset bias: `glassalpha detect-bias --data data.csv`
2. Review feature correlations: See [Feature Selection Guide]
3. Try threshold adjustment: See [Threshold Tuning Guide]
4. Consider fairness constraints: See [Fair Training Guide]

Need more help? Post in #ml-compliance or DM @compliance-team
```

### Workflow 4: Vendor and Tool Evaluation

**Scenario**: Evaluate if GlassAlpha meets your organization's needs compared to alternatives.

#### Evaluation criteria matrix

| Criterion             | Weight | GlassAlpha | Vendor A | Vendor B | Internal Build |
| --------------------- | ------ | ---------- | -------- | -------- | -------------- |
| **Functionality**     |        |            |          |          |                |
| Fairness metrics      | 10%    | 9/10       | 8/10     | 9/10     | 6/10           |
| Explainability        | 10%    | 9/10       | 7/10     | 8/10     | 5/10           |
| Calibration testing   | 5%     | 9/10       | 6/10     | 7/10     | 4/10           |
| Custom models         | 5%     | 8/10       | 9/10     | 7/10     | 10/10          |
| **Integration**       |        |            |          |          |                |
| CI/CD integration     | 10%    | 9/10       | 7/10     | 8/10     | 9/10           |
| Existing ML pipeline  | 10%    | 8/10       | 6/10     | 7/10     | 10/10          |
| API quality           | 5%     | 9/10       | 8/10     | 7/10     | 7/10           |
| **Compliance**        |        |            |          |          |                |
| Audit trails          | 10%    | 10/10      | 8/10     | 7/10     | 5/10           |
| Reproducibility       | 10%    | 10/10      | 7/10     | 6/10     | 6/10           |
| Regulatory templates  | 5%     | 8/10       | 9/10     | 8/10     | 3/10           |
| **Operations**        |        |            |          |          |                |
| On-premise deployment | 10%    | 10/10      | 5/10     | 6/10     | 10/10          |
| Maintenance burden    | 5%     | 9/10       | 8/10     | 8/10     | 4/10           |
| Training requirements | 5%     | 8/10       | 7/10     | 6/10     | 5/10           |
| **Total Score**       |        | **90%**    | 73%      | 72%      | 67%            |

#### Pilot program structure

**Week 1-2: Setup**

- Install GlassAlpha
- Configure for 2-3 representative models
- Train 3-5 team members

**Week 3-4: Parallel Run**

- Run GlassAlpha audits alongside existing process
- Compare outputs
- Measure time/effort savings

**Week 5-6: Evaluation**

- Team feedback survey
- Cost-benefit analysis
- Decision presentation to leadership

**Success criteria:**

- 80%+ of audits complete successfully
- 50%+ time savings vs manual process
- 4/5+ satisfaction score from teams
- Regulatory-ready outputs

## Best Practices

### Policy Management

- **Version control**: Track all policy changes in Git
- **Review cycles**: Quarterly policy reviews with stakeholders
- **Communication**: 2-week notice before policy changes
- **Exceptions**: Formal exception process with executive approval
- **Documentation**: Clear rationale for every threshold

### Team Enablement

- **Self-service first**: Provide templates and documentation
- **Support channels**: Slack + office hours + on-demand training
- **Feedback loops**: Regular surveys and retrospectives
- **Champion programs**: Identify power users in each team
- **Success stories**: Share wins to build momentum

### Portfolio Monitoring

- **Automated tracking**: Registry integration in all CI/CD pipelines
- **Regular reporting**: Monthly portfolio dashboard to leadership
- **Proactive intervention**: Alert teams before regulatory review
- **Trend analysis**: Track metrics over time to spot systemic issues
- **Resource allocation**: Prioritize support based on portfolio risk

### Change Management

- **Pilot first**: Test with friendly team before org rollout
- **Incremental adoption**: Start with new models, gradually expand
- **Clear benefits**: Emphasize time savings and risk reduction
- **Executive sponsorship**: Ensure leadership visibility and support
- **Celebrate wins**: Highlight successful audits and approvals

## Common Challenges and Solutions

### Challenge: Team pushback on compliance overhead

**Solution:**

- Show time savings: "10 hours manual → 30 minutes automated"
- Emphasize risk reduction: "Avoid regulatory fines"
- Provide templates: "80% pre-configured, just add your model"
- Share success stories: "Team X deployed in half the time"

### Challenge: Policy too strict, blocking deployments

**Solution:**

- Review gates with stakeholders
- Consider tiered thresholds (warning vs error)
- Allow documented exceptions with approval
- Adjust thresholds based on data (e.g., if 80% models fail, threshold may be unrealistic)

### Challenge: Teams bypassing audits

**Solution:**

- Enforce at CI/CD level (audit required for merge)
- Require audit report for deployment approval
- Track non-compliance in performance reviews
- Provide support to make compliance easy

### Challenge: Inconsistent audit quality

**Solution:**

- Provide templates and standards
- Automated validation in CI (config linting)
- Peer review process for audits
- Training on common pitfalls

## Metrics to Track

### Adoption Metrics

- Number of teams using GlassAlpha
- Audits per month
- Template adoption rate
- Training attendance

### Quality Metrics

- Failed gate rate by team
- Time from audit to approval
- Number of reaudits required
- Audit completeness score

### Impact Metrics

- Time savings per audit (vs manual)
- Deployment velocity (time to production)
- Regulatory findings (should decrease)
- Compliance incidents (should decrease)

### Team Satisfaction

- Survey scores (quarterly)
- Support ticket volume
- Training effectiveness ratings

## Related Resources

### For Managers

- [Compliance Overview](../compliance/index.md) - Regulatory landscape
- [SR 11-7 Mapping](../compliance/sr-11-7-mapping.md) - Banking requirements
- [Trust & Deployment](../reference/trust-deployment.md) - Architecture and security

### For Teams

- [ML Engineer Workflow](ml-engineer-workflow.md) - Implementation guide
- [Data Scientist Workflow](data-scientist-workflow.md) - Exploratory analysis
- [Compliance Officer Workflow](compliance-workflow.md) - Evidence packs

### Policy Resources

- [Policy Configuration Guide](#policy-gates) - Writing policy gates

## Support

For manager-specific questions:

- GitHub Discussions: [GlassAlpha/glassalpha/discussions](https://github.com/GlassAlpha/glassalpha/discussions)
- Email: [enterprise@glassalpha.com](mailto:enterprise@glassalpha.com)
- Documentation: [glassalpha.com](https://glassalpha.com)
