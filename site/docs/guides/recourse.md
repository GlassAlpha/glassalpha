# Counterfactual Recourse Guide

## Overview

**Counterfactual recourse** provides actionable recommendations for individuals who receive adverse decisions from ML models. Instead of just explaining why a decision was made, recourse tells them **what they can change** to achieve a different outcome.

### Why Recourse Matters

**ECOA Compliance**: The Equal Credit Opportunity Act (ECOA) requires lenders to provide not just reasons for adverse actions, but also **actionable guidance** for improving future applications. Recourse directly addresses this requirement.

**Regulatory Alignment**:

- **SR 11-7 (Federal Reserve)**: Emphasizes "clear and understandable" explanations that applicants can act upon
- **CFPB Guidelines**: Recommends providing specific actions consumers can take
- **EU AI Act**: Requires "meaningful information about the logic involved" including paths to favorable outcomes

**User Trust**: Recourse demonstrates good faith by showing individuals how to improve, rather than simply rejecting them.

## Quick Start

### Basic Usage

Generate recourse for a denied credit application:

```bash
glassalpha recourse \
  --model models/german_credit.pkl \
  --data data/test.csv \
  --instance 42 \
  --config recourse_german_credit.yaml \
  --output recourse/instance_42.json
```

### Example Output

```json
{
  "instance_id": 42,
  "original_prediction": 0.35,
  "threshold": 0.5,
  "recommendations": [
    {
      "rank": 1,
      "feature_changes": {
        "savings_balance": {
          "old": 500,
          "new": 2000
        }
      },
      "total_cost": 1200.0,
      "predicted_probability": 0.62,
      "feasible": true
    },
    {
      "rank": 2,
      "feature_changes": {
        "checking_balance": {
          "old": 100,
          "new": 800
        },
        "duration": {
          "old": 24,
          "new": 18
        }
      },
      "total_cost": 1540.0,
      "predicted_probability": 0.58,
      "feasible": true
    }
  ],
  "policy_constraints": {
    "immutable_features": ["age", "gender", "foreign_worker"],
    "monotonic_constraints": {
      "savings_balance": "increase_only",
      "num_existing_credits": "decrease_only"
    }
  },
  "seed": 42,
  "total_candidates": 47,
  "feasible_candidates": 12
}
```

### Interpreting Results

Each recommendation includes:

- **rank**: Lower rank = lower cost (easier to achieve)
- **feature_changes**: Specific features to modify with old/new values
- **total_cost**: Weighted difficulty of making these changes
- **predicted_probability**: Model prediction if changes are made
- **feasible**: Whether recommendation passes the threshold

## Configuration

### Full Configuration Example

```yaml
# Recourse configuration for German Credit
recourse:
  enabled: true

  # Immutable features (cannot be changed)
  immutable_features:
    - age # Cannot change age
    - gender # Protected attribute
    - foreign_worker # Immigration status
    - employment_duration # Historical data

  # Monotonic constraints (directional restrictions)
  monotonic_constraints:
    # Can only increase
    savings_balance: increase_only
    checking_balance: increase_only

    # Can only decrease
    num_existing_credits: decrease_only
    num_dependents: decrease_only

  # Cost function
  cost_function: weighted_l1

  # Maximum iterations
  max_iterations: 100

# Protected attributes to exclude
data:
  protected_attributes:
    - age
    - gender
    - foreign_worker

# Seed for deterministic results
reproducibility:
  random_seed: 42
```

### Policy Constraints

#### 1. Immutable Features

Features that individuals **cannot reasonably change**:

```yaml
immutable_features:
  - age # Cannot change
  - gender # Protected attribute (ECOA)
  - race # Protected attribute (ECOA)
  - national_origin # Protected attribute (ECOA)
  - employment_duration # Historical data
  - present_residence_duration # Historical data
```

**Rationale**: Recourse must be **actionable**. Suggesting someone change their age or gender violates ECOA and is not actionable.

#### 2. Monotonic Constraints

Features that can only change in **one direction**:

```yaml
monotonic_constraints:
  # Features that can only increase
  income: increase_only # Can increase income
  savings_balance: increase_only # Can save more
  checking_balance: increase_only # Can increase balance
  employment_duration: increase_only # Accumulates over time

  # Features that can only decrease
  debt: decrease_only # Can pay off debt
  num_existing_credits: decrease_only # Can close credits
  num_dependents: decrease_only # Unlikely to increase
```

**Rationale**: Realistic recommendations respect real-world constraints. You can increase savings but not decrease them retroactively.

#### 3. Cost Function

**Weighted L1 Distance**: Measures the "difficulty" of making changes.

```python
cost = sum(|old_value - new_value| * feature_weight)
```

**Example feature weights**:

- **Low cost** (easy to change): `checking_balance`, `savings_balance`, `loan_purpose`
- **Medium cost**: `credit_amount`, `duration`, `installment_rate`
- **High cost** (difficult): `property`, `employment_duration`, `other_installment_plans`
- **Infinite cost** (immutable): `age`, `gender`, `foreign_worker`

### Threshold Configuration

```yaml
threshold: 0.5 # Decision threshold
```

Recommendations must achieve `predicted_probability >= threshold` to be considered feasible.

### Top-N Recommendations

```yaml
top_n: 5 # Return top 5 recommendations
```

Returns the `top_n` lowest-cost feasible recommendations.

## CLI reference

### Command: `glassalpha recourse`

Generate counterfactual recourse recommendations for adverse decisions.

#### Required Arguments

- `--model, -m`: Path to trained model file (`.pkl`, `.joblib`)
- `--data, -d`: Path to test data file (CSV)
- `--instance, -i`: Row index of instance to explain (0-based)

#### Optional Arguments

- `--config, -c`: Path to recourse configuration YAML (highly recommended)
- `--output, -o`: Path for output JSON file (defaults to stdout)
- `--threshold, -t`: Decision threshold (default: 0.5)
- `--top-n, -n`: Number of recommendations to generate (default: 5)

#### Examples

**Basic usage with config**:

```bash
glassalpha recourse \
  -m models/credit_model.pkl \
  -d data/test.csv \
  -i 42 \
  -c recourse_config.yaml
```

**With custom threshold and top-N**:

```bash
glassalpha recourse \
  -m models/credit_model.pkl \
  -d data/test.csv \
  -i 10 \
  --threshold 0.6 \
  --top-n 3 \
  -c recourse_config.yaml
```

**Save to file**:

```bash
glassalpha recourse \
  -m models/credit_model.pkl \
  -d data/test.csv \
  -i 5 \
  -c recourse_config.yaml \
  --output recourse/recommendations.json
```

**Without config (no constraints)**:

```bash
glassalpha recourse \
  -m models/credit_model.pkl \
  -d data/test.csv \
  -i 42 \
  --threshold 0.5
```

⚠️ **Warning**: Running without a config file uses no policy constraints, which may produce unrealistic recommendations (e.g., suggesting age changes).

### Exit Codes

- `0`: Success (recommendations generated)
- `1`: Error (validation failure, file not found, etc.)
- `2`: Runtime error (model prediction failed, SHAP computation failed)

## Algorithm Details

### Greedy Search with Policy Constraints

GlassAlpha uses a **greedy search algorithm** that:

1. Identifies **negative contributors** using SHAP values (features pushing toward denial)
2. Filters out **immutable features** from policy constraints
3. For each **mutable feature**, generates candidate changes:
   - Respects **monotonic constraints** (increase/decrease only)
   - Validates **feature bounds** (min/max values)
   - Computes **cost** (weighted L1 distance)
4. Predicts outcome for each candidate
5. Filters **feasible candidates** (prediction >= threshold)
6. Sorts by **cost** (lowest first)
7. Returns **top-N** recommendations

### Why Greedy Search?

**Deterministic**: Same seed → same recommendations (regulatory requirement)

**Gradient-free**: Works with any tabular model (trees, ensembles, linear)

**Interpretable**: Single-feature or small multi-feature changes are easier to explain

**Fast**: Evaluates hundreds of candidates in seconds

### Enterprise Upgrades

**OSS (Current)**:

- Greedy search with policy constraints
- Single-feature and two-feature changes
- Fixed cost function (weighted L1)

**Enterprise (Future)**:

- Multi-objective optimization (cost, feasibility, diversity)
- Batch recourse generation (1000+ instances)
- Custom cost functions (domain-specific weights)
- Catalog mapping (feature names → user-friendly labels)
- PII controls (automatic redaction for compliance)

## Integration with E2 (Reason Codes)

Recourse builds on **E2 (Reason Codes)** by using SHAP values to identify negative contributors:

```python
from glassalpha.explain import extract_reason_codes, generate_recourse

# Step 1: Extract reason codes (E2)
reason_result = extract_reason_codes(
    shap_values=shap_values,
    feature_names=feature_names,
    feature_values=feature_values,
    instance_id=42,
    prediction=0.35,
    threshold=0.5,
)

# Step 2: Generate recourse (E2.5)
recourse_result = generate_recourse(
    model=model,
    feature_values=feature_values,
    shap_values=shap_values,
    feature_names=feature_names,
    instance_id=42,
    original_prediction=0.35,
    threshold=0.5,
    policy_constraints=policy,
)
```

**Workflow**:

1. **Reason codes** explain **why** the decision was made
2. **Recourse** suggests **what to change** to get approval

## Programmatic API

### Python API

```python
from glassalpha.explain import generate_recourse
from glassalpha.explain.policy import PolicyConstraints
import pandas as pd

# Define policy constraints
policy = PolicyConstraints(
    immutable_features=["age", "gender", "foreign_worker"],
    monotonic_constraints={
        "savings_balance": "increase_only",
        "debt": "decrease_only",
    },
    feature_costs={
        "savings_balance": 0.5,  # Easy to change
        "debt": 0.8,  # Harder to change
    },
    feature_bounds={
        "savings_balance": (0, 100000),
        "debt": (0, 50000),
    },
)

# Generate recourse
result = generate_recourse(
    model=trained_model,
    feature_values=pd.Series({"age": 25, "savings_balance": 500, "debt": 5000}),
    shap_values=np.array([-0.1, -0.3, -0.2]),
    feature_names=["age", "savings_balance", "debt"],
    instance_id=42,
    original_prediction=0.35,
    threshold=0.5,
    policy_constraints=policy,
    top_n=5,
    seed=42,
)

# Access recommendations
for rec in result.recommendations:
    print(f"Rank {rec.rank}: Change {rec.feature_changes}")
    print(f"  Cost: {rec.total_cost:.2f}")
    print(f"  New prediction: {rec.predicted_probability:.1%}")
```

### Data Classes

```python
@dataclass(frozen=True)
class RecourseRecommendation:
    """Single counterfactual recommendation."""
    feature_changes: dict[str, tuple[float, float]]
    total_cost: float
    predicted_probability: float
    feasible: bool
    rank: int

@dataclass(frozen=True)
class RecourseResult:
    """Complete recourse generation result."""
    instance_id: str | int
    original_prediction: float
    threshold: float
    recommendations: list[RecourseRecommendation]
    policy_constraints: PolicyConstraints
    seed: int
    total_candidates: int
    feasible_candidates: int
```

## Troubleshooting

### No Feasible Recourse Found

**Problem**: `feasible_candidates: 0` and empty recommendations list

**Causes**:

1. **Too many immutable features**: Most features are locked
2. **Restrictive monotonic constraints**: Valid changes don't improve prediction
3. **High threshold**: Target threshold is unreachable with small changes

**Solutions**:

```yaml
# Option 1: Relax immutable features
immutable_features:
  - age
  - gender
  # Remove: employment_duration (allow changes)

# Option 2: Reduce monotonic constraints
monotonic_constraints:
  savings_balance: increase_only
  # Remove: num_existing_credits (allow increase)

# Option 3: Lower threshold
threshold: 0.45 # Instead of 0.5

# Option 4: Increase feature bounds
feature_bounds:
  savings_balance: [0, 50000] # Instead of [0, 10000]
```

### SHAP computation failed

**Problem**: `TreeExplainer failed` or `KernelSHAP timeout`

**Cause**: Model incompatible with TreeSHAP or too complex for KernelSHAP

**Solution**: Use `PermutationExplainer` (slower but works with any model):

```yaml
explainer:
  type: permutation
  n_samples: 1000
```

### Recommendations Are Unrealistic

**Problem**: Suggests changing age, gender, or other unchangeable features

**Cause**: Missing or incomplete policy constraints

**Solution**: Always use a comprehensive config file:

```yaml
recourse:
  immutable_features:
    - age
    - gender
    - race
    - national_origin
    - employment_duration
    - present_residence_duration
```

### Recommendations Are Too Expensive

**Problem**: All recommendations have very high costs

**Cause**: Default feature costs are too high

**Solution**: Define custom feature costs:

```yaml
recourse:
  feature_costs:
    checking_balance: 0.3 # Easy to change
    savings_balance: 0.5 # Medium difficulty
    employment_duration: 2.0 # Very difficult
```

### Determinism Issues

**Problem**: Different runs produce different recommendations

**Cause**: Missing or different random seeds

**Solution**: Always set explicit seed:

```yaml
reproducibility:
  random_seed: 42
  deterministic: true
```

## Best Practices

### 1. Always Use Policy Constraints

❌ **Bad**: No config file

```bash
glassalpha recourse -m model.pkl -d test.csv -i 42
```

✅ **Good**: Comprehensive config

```bash
glassalpha recourse -m model.pkl -d test.csv -i 42 -c recourse_config.yaml
```

### 2. Document Constraint Rationale

Add comments to config explaining why each constraint exists:

```yaml
immutable_features:
  - age # ECOA protected attribute (cannot change)
  - employment_duration # Historical data (cannot retroactively change)

monotonic_constraints:
  savings_balance: increase_only # Cannot "unsave" money
  debt: decrease_only # Cannot retroactively decrease debt principal
```

### 3. Test with Real Data

Validate recommendations with domain experts:

```bash
# Generate recourse for 10 denied instances
for i in {0..9}; do
  glassalpha recourse \
    -m model.pkl \
    -d denied_instances.csv \
    -i $i \
    -c recourse_config.yaml \
    --output recourse/instance_${i}.json
done
```

### 4. Combine with Reason Codes

Provide **both** reason codes and recourse in adverse action notices:

```bash
# Step 1: Generate reason codes
glassalpha reasons \
  -m model.pkl \
  -d test.csv \
  -i 42 \
  -c config.yaml \
  --output notices/instance_42_reasons.txt

# Step 2: Generate recourse
glassalpha recourse \
  -m model.pkl \
  -d test.csv \
  -i 42 \
  -c config.yaml \
  --output notices/instance_42_recourse.json
```

### 5. Version Control Configs

Track policy evolution over time:

```bash
git add configs/recourse_german_credit.yaml
git commit -m "Add recourse config with ECOA-compliant constraints"
```

## Regulatory Compliance

### ECOA requirements

**15 U.S.C. § 1691(d)**: Creditors must provide "a statement of specific reasons for the action taken"

**Regulation B (12 CFR § 1002.9)**: Reasons must be "specific and indicate the principal reason(s) for the adverse action"

**Recourse Alignment**: Counterfactual recommendations satisfy "specific reasons" by showing:

- **What features** caused denial (via SHAP)
- **What changes** would lead to approval (via recourse)
- **How feasible** those changes are (via cost function)

### SR 11-7 (Federal Reserve)

**Guidance**: "Clear and understandable information" that is "meaningful to the consumer"

**Recourse Alignment**:

- ✅ Specific feature changes (not vague "improve creditworthiness")
- ✅ Realistic constraints (respects immutables and monotonic constraints)
- ✅ Multiple options (top-N recommendations sorted by cost)

### CFPB supervision guidelines

**Recommendation**: Provide "actionable steps consumers can take"

**Recourse Alignment**:

- ✅ Concrete actions (increase savings to $2000)
- ✅ Prioritized by difficulty (lowest cost first)
- ✅ Verifiable (predicted probability shown)

### Audit Trail

Every recourse result includes:

```json
{
  "seed": 42,
  "policy_constraints": {
    "immutable_features": ["age", "gender"],
    "monotonic_constraints": { "savings": "increase_only" }
  },
  "total_candidates": 47,
  "feasible_candidates": 12
}
```

This provides a **complete audit trail** for regulatory review.

## Examples

### Example 1: German Credit

```yaml
# recourse_german_credit.yaml
recourse:
  enabled: true
  immutable_features: [age, gender, foreign_worker]
  monotonic_constraints:
    savings_balance: increase_only
    num_existing_credits: decrease_only
```

```bash
glassalpha recourse \
  -m models/german_credit.pkl \
  -d data/german_credit_test.csv \
  -i 42 \
  -c recourse_german_credit.yaml \
  --output recourse/gc_instance_42.json
```

### Example 2: Adult Income

```yaml
# recourse_adult_income.yaml
recourse:
  enabled: true
  immutable_features: [age, gender, race, native_country]
  monotonic_constraints:
    education_years: increase_only
    capital_gain: increase_only
    hours_per_week: increase_only
```

```bash
glassalpha recourse \
  -m models/adult_income.pkl \
  -d data/adult_income_test.csv \
  -i 100 \
  -c recourse_adult_income.yaml \
  --top-n 3
```

### Example 3: Healthcare Outcomes

```yaml
# recourse_healthcare.yaml
recourse:
  enabled: true
  immutable_features: [patient_age, patient_gender, diagnosis_date]
  monotonic_constraints:
    medication_adherence: increase_only
    exercise_minutes_per_week: increase_only
    bmi: decrease_only
```

```bash
glassalpha recourse \
  -m models/healthcare_risk.pkl \
  -d data/patients_test.csv \
  -i 50 \
  -c recourse_healthcare.yaml \
  --threshold 0.3
```

## Next Steps

- **Integrate with audit reports**: Include recourse in PDF audits
- **Batch generation**: Generate recourse for all denied instances
- **Custom cost functions**: Domain-specific difficulty weights
- **Multi-objective optimization**: Balance cost, diversity, and feasibility

## Related Documentation

- [Reason Codes Guide](reason-codes.md) - Adverse action notices (ECOA requirement)
- [CLI Reference](../reference/cli.md) - Complete CLI documentation

### Industry-Specific Guidance

- **[Banking Compliance Guide](../compliance/banking-guide.md)** - Recourse for credit decisions (SR 11-7, ECOA)
- **[Insurance Compliance Guide](../compliance/insurance-guide.md)** - Recourse for underwriting decisions
- **[Fraud Detection Guide](../compliance/fraud-guide.md)** - Contestability for fraud flags

### Workflow Guides

- **[Compliance Officer Workflow](compliance-workflow.md)** - Evidence generation and regulator communication
- **[ML Engineer Workflow](ml-engineer-workflow.md)** - Integration and debugging

## Support

For questions or issues:

- GitHub Issues: [github.com/GlassAlpha/glassalpha/issues](https://github.com/GlassAlpha/glassalpha/issues)
- Documentation: [glassalpha.com](https://glassalpha.com)
- Examples: [github.com/GlassAlpha/glassalpha/tree/main/examples](https://github.com/GlassAlpha/glassalpha/tree/main/examples)
