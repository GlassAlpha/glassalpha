# Reason Codes Guide

Generate ECOA-compliant adverse action notices with GlassAlpha's reason code extraction.

## Overview

The reason codes feature extracts top-N negative feature contributions from model predictions to explain adverse decisions. This is a regulatory requirement under the Equal Credit Opportunity Act (ECOA) for credit decisions.

**Key capabilities:**

- Extract top-N most impactful features for denial
- Automatic exclusion of protected attributes
- ECOA-compliant adverse action notice templates
- Deterministic output for regulatory reproducibility
- JSON and text output formats

## Quick Start

Generate a reason code notice for a single instance:

```bash
glassalpha reasons \
  --model models/german_credit.pkl \
  --data models/test_data.csv \
  --instance 42 \
  --output notices/instance_42.txt
```

This generates an ECOA-compliant adverse action notice explaining why instance 42 was denied.

!!! warning "Data Format Requirements"
**Use the preprocessed test data saved during audit generation.**

    - ✅ **Correct**: `--data models/test_data.csv` (already preprocessed, includes one-hot encoding)
    - ❌ **Wrong**: `--data original_data.csv` (needs preprocessing, will cause column mismatch errors)

    The model expects the same feature format it was trained with. The `test_data.csv` file is automatically saved during audit generation and matches the model's expected format exactly.

## How It Works

### 1. SHAP value extraction

GlassAlpha uses SHAP (SHapley Additive exPlanations) to determine feature contributions:

- **Positive contributions**: Features that push toward approval
- **Negative contributions**: Features that push toward denial

Reason codes focus on the **most negative** contributions.

!!! important "Reason codes for denied decisions only"
Reason codes are only generated for **denied** decisions (predictions below threshold). If all SHAP contributions are positive, the decision is approved and no reason codes are generated. This is correct ECOA behavior - adverse action notices are only required for denials.

### 2. Protected Attribute Filtering

Protected attributes are automatically excluded from reason codes per ECOA requirements:

**Default exclusions:**

- age, gender, sex, race, ethnicity
- national_origin, nationality, religion
- marital_status, disability, foreign_worker

### 3. Ranking and Selection

Features are ranked by SHAP magnitude (most negative first). Ties are broken deterministically using a seeded random generator.

**ECOA typical**: 2-4 reason codes (default: 4)

## Configuration

### Basic Config

```yaml
# reason_codes.yaml
reason_codes:
  top_n: 4
  threshold: 0.5
  organization: "Example Bank"
  contact_info: "1-800-555-0199"

data:
  protected_attributes:
    - age
    - gender
    - foreign_worker

reproducibility:
  random_seed: 42
```

### Full Config Options

```yaml
reason_codes:
  # Number of reason codes (ECOA typical: 2-4)
  top_n: 4

  # Decision threshold
  threshold: 0.5

  # Organization info for notice
  organization: "Your Organization Name"
  contact_info: "Contact information for inquiries"

  # Output format: 'text' or 'json'
  format: text

  # Custom template (optional)
  template: "templates/custom_aan.txt"

data:
  # Protected attributes (never in reason codes)
  protected_attributes:
    - age
    - gender
    - race

  # Dataset and target
  dataset: "german_credit"
  target_column: "target"

explainer:
  type: "treeshap"
  max_samples: 1000

reproducibility:
  random_seed: 42
  deterministic: true
```

## CLI usage

### Generate Single Notice

```bash
glassalpha reasons \
  --model models/credit_model.pkl \
  --data data/denied_apps.csv \
  --instance 0 \
  --output notices/app_001.txt
```

### JSON output

```bash
glassalpha reasons \
  --model model.pkl \
  --data test.csv \
  --instance 42 \
  --format json \
  --output reasons_42.json
```

### Custom Configuration

```bash
glassalpha reasons \
  --model model.pkl \
  --data test.csv \
  --instance 10 \
  --config reason_codes.yaml
```

### Custom Threshold and Top-N

```bash
glassalpha reasons \
  --model model.pkl \
  --data test.csv \
  --instance 5 \
  --threshold 0.6 \
  --top-n 3
```

## Python API

### Extract Reason Codes

```python
import numpy as np
import pandas as pd
from glassalpha.explain.reason_codes import extract_reason_codes

# SHAP values for single instance
shap_values = np.array([-0.5, 0.3, -0.2, 0.1])
feature_names = ["debt", "income", "duration", "savings"]
feature_values = pd.Series([5000, 30000, 24, 1000])

result = extract_reason_codes(
    shap_values=shap_values,
    feature_names=feature_names,
    feature_values=feature_values,
    instance_id=42,
    prediction=0.35,
    threshold=0.5,
    top_n=4,
    protected_attributes=["age", "gender"],
    seed=42,
)

print(f"Decision: {result.decision}")
print(f"Top {len(result.reason_codes)} reasons:")
for code in result.reason_codes:
    print(f"  {code.rank}. {code.feature}: {code.contribution:.3f}")
```

### Format Adverse Action Notice

```python
from glassalpha.explain.reason_codes import format_adverse_action_notice

notice = format_adverse_action_notice(
    result=result,
    organization="Example Bank",
    contact_info="1-800-555-0199",
)

print(notice)
```

### Integration with Audit Pipeline

```python
from glassalpha.explain.shap import TreeSHAPExplainer

# Fit explainer
explainer = TreeSHAPExplainer()
explainer.fit(model, X_train)

# Get SHAP values for instance
instance = X_test.iloc[42]
shap_values = explainer.explain(instance.to_frame().T)[0]

# Extract reason codes
result = extract_reason_codes(
    shap_values=shap_values,
    feature_names=X_test.columns.tolist(),
    feature_values=instance,
    instance_id=42,
    prediction=model.predict_proba(instance.to_frame().T)[0, 1],
    seed=42,
)
```

## Output Formats

### Text (ECOA Notice)

Default format suitable for regulatory submission:

```
ADVERSE ACTION NOTICE
Equal Credit Opportunity Act (ECOA) Disclosure

Example Bank
Date: 2025-01-15T10:30:00+00:00
Application ID: 42

DECISION: DENIED
Predicted Score: 35.0%

PRINCIPAL REASONS FOR ADVERSE ACTION:

The following factors most negatively affected your application:

1. Debt: Value of 5000 negatively impacted the decision
2. Duration: Value of 24 negatively impacted the decision
3. Credit History: Value of 2 negatively impacted the decision
4. Savings: Value of 1000 negatively impacted the decision

IMPORTANT RIGHTS UNDER FEDERAL LAW:
...
```

### JSON

Structured data for programmatic processing:

```json
{
  "instance_id": 42,
  "prediction": 0.35,
  "decision": "denied",
  "reason_codes": [
    {
      "rank": 1,
      "feature": "debt",
      "contribution": -0.5,
      "feature_value": 5000
    },
    {
      "rank": 2,
      "feature": "duration",
      "contribution": -0.3,
      "feature_value": 24
    }
  ],
  "excluded_features": ["age", "gender"],
  "timestamp": "2025-01-15T10:30:00+00:00",
  "model_hash": "abc123...",
  "seed": 42
}
```

## Regulatory Compliance

### ECOA Requirements

✅ **Specific reasons**: Not just "credit score" - actual features
✅ **Ranked by importance**: Most negative first
✅ **Typically 2-4 reasons**: Default is 4
✅ **Understandable to applicant**: Human-readable feature names
✅ **No protected attributes**: Automatically excluded

### Reproducibility

All reason code outputs are **deterministic**:

- Same seed → same reason codes
- Same tie-breaking order
- Byte-identical notices across platforms

This is critical for regulatory audits and compliance verification.

### Audit Trail

Every output includes:

- Instance ID
- Prediction score
- Decision threshold
- Timestamp (ISO 8601 UTC)
- Model hash (provenance)
- Random seed (reproducibility)
- Excluded features (protected attributes)

## Advanced Usage

### Custom Template

Create a custom adverse action notice template:

```text
# templates/custom_aan.txt
CREDIT DECISION NOTICE

{organization}
Application: {instance_id}
Date: {timestamp}

DECISION: {decision}
Score: {prediction}

PRIMARY FACTORS:
{reason_codes}

Questions? Contact: {contact_info}

Model ID: {model_hash}
```

Use with `--config`:

```yaml
reason_codes:
  template: "templates/custom_aan.txt"
```

### Batch Processing

Generate notices for all denied applications:

```python
import pandas as pd
from glassalpha.explain.reason_codes import extract_reason_codes, format_adverse_action_notice

# Load denied applications
denied = df[df['decision'] == 'denied']

for idx, row in denied.iterrows():
    # Generate SHAP values
    shap_vals = explainer.explain(row.to_frame().T)[0]

    # Extract reason codes
    result = extract_reason_codes(
        shap_values=shap_vals,
        feature_names=feature_names,
        feature_values=row,
        instance_id=idx,
        prediction=row['prediction'],
        seed=42,
    )

    # Generate notice
    notice = format_adverse_action_notice(result)

    # Save to file
    with open(f"notices/app_{idx}.txt", "w") as f:
        f.write(notice)
```

## Troubleshooting

### "No negative contributions found"

This error occurs when all SHAP values are positive (approved decision).

**Solution**: Reason codes are for **denied** decisions only. Check:

- Prediction is below threshold
- Model predicted denial
- SHAP values have negative contributions

### "All features are protected attributes"

All features are in the protected attribute exclusion list.

**Solution**:

- Verify `protected_attributes` list in config
- Ensure non-protected features exist
- Check feature name matching (case-insensitive)

### "SHAP values don't match feature names"

Array dimension mismatch.

**Solution**:

- Ensure SHAP values are 1D: `shap_values.shape[0] == len(feature_names)`
- For multi-output models, select class: `shap_values = shap_values[1]`
- Flatten if needed: `shap_values = shap_values[0]`

### "Model not TreeSHAP-compatible"

CLI can't generate SHAP values.

**Solution**:

- Use TreeSHAP-compatible models: XGBoost, LightGBM, RandomForest
- For other models, use Python API with custom explainer
- Consider KernelSHAP for non-tree models (slower)

## Best Practices

### 1. Use Standard Number of Reasons

ECOA typical: **4 reason codes**

```bash
--top-n 4  # Default, recommended
```

### 2. Verify Protected Attributes

Always configure protected attributes for your dataset:

```yaml
data:
  protected_attributes:
    - age
    - gender
    - race
    # Add dataset-specific protected features
```

### 3. Maintain Reproducibility

Always set a seed for regulatory audits:

```yaml
reproducibility:
  random_seed: 42 # Fixed seed
```

### 4. Document Custom Templates

If using custom templates, document:

- Why custom template is needed
- What changes were made
- Compliance officer approval

### 5. Archive Generated Notices

Store notices with audit trail:

```
notices/
  2025-01-15/
    app_001.txt
    app_002.txt
    manifest.json  # Metadata
```

## Examples

### German Credit

```bash
glassalpha reasons \
  --model artifacts/german_credit_model.pkl \
  --data artifacts/german_credit_test.csv \
  --instance 42 \
  --config configs/reason_codes_german_credit.yaml
```

### Adult Income

```bash
glassalpha reasons \
  --model models/adult_income.pkl \
  --data data/adult_test.csv \
  --instance 100 \
  --threshold 0.6 \
  --top-n 3
```

## Next Steps

- **[Configuration Guide](../getting-started/configuration.md)** - Full config options
- **[CLI Reference](../reference/cli.md)** - All CLI commands
- **[Quick Start Guide](../getting-started/quickstart.md)** - Generate your first audit
- **[Recourse Guide](recourse.md)** - Actionable recommendations

## Industry-Specific Guidance

- **[Banking Compliance Guide](../compliance/banking-guide.md)** - ECOA adverse action requirements for credit models
- **[Fraud Detection Guide](../compliance/fraud-guide.md)** - FCRA adverse action requirements for fraud models
- **[Compliance Officer Workflow](compliance-workflow.md)** - Evidence packs and regulator communication

## Support

Questions or issues?

- **Documentation**: [glassalpha.com](https://glassalpha.com)
- **Issues**: [github.com/GlassAlpha/glassalpha/issues](https://github.com/GlassAlpha/glassalpha/issues)
- **Contact**: [Contact form](https://glassalpha.com/contact)
