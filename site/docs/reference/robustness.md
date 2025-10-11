# Robustness Testing

!!! info "Part of Understanding Section"
Model robustness testing validates stability under adversarial perturbations. Related topics:

    - **[Testing Demographic Shifts](../guides/shift-testing.md)** - Robustness to population changes
    - **[Individual Fairness](fairness-metrics.md#individual-fairness-e11)** - Consistency score (similar concept)
    - **[Configuration Guide](../getting-started/configuration.md)** - How to enable stability metrics

# Robustness Testing

Model robustness testing validates that small input perturbations do not cause large prediction changes. This is critical for regulatory compliance (EU AI Act, NIST AI RMF) and production deployment confidence.

## Overview

GlassAlpha's robustness testing includes:

1. **Adversarial Perturbation Sweeps (E6+)**: Tests model stability under Gaussian noise
2. **Protected Feature Exclusion**: Prevents synthetic bias during testing
3. **Deterministic Execution**: Fully reproducible with random seeds
4. **Automated Gates**: PASS/FAIL/WARNING based on thresholds

## Adversarial Perturbation Sweeps (E6+)

### What It Tests

Adds small Gaussian noise (ε-perturbations) to non-protected features and measures maximum prediction change:

- **ε = 0.01** (1% noise): Minimal perturbation
- **ε = 0.05** (5% noise): Moderate perturbation
- **ε = 0.10** (10% noise): Significant perturbation

**Robustness Score** = Maximum prediction delta (L∞ norm) across all samples and all epsilon values.

### Why It Matters

- **Regulatory requirement**: EU AI Act requires robustness testing for high-risk AI systems
- **Production safety**: Catches models that are sensitive to measurement noise or data errors
- **Fairness**: Ensures similar individuals get similar predictions (consistency)

### Configuration

Enable robustness testing in your audit configuration:

```yaml
metrics:
  stability:
    enabled: true # Enable perturbation sweeps
    epsilon_values: [0.01, 0.05, 0.1] # Noise levels to test (default)
    threshold: 0.15 # Max allowed prediction change (15%)
```

**Parameters:**

- `enabled` (bool): Whether to run perturbation sweeps. Default: `false`
- `epsilon_values` (list[float]): List of epsilon values to test. Default: `[0.01, 0.05, 0.1]`
- `threshold` (float): Maximum allowed prediction delta. Default: `0.15` (15%)

### How It Works

For each epsilon value:

1. **Generate perturbed data**: Add Gaussian noise ~ N(0, ε) to each non-protected feature
2. **Get predictions**: Run model on perturbed data
3. **Compute deltas**: |prediction_original - prediction_perturbed| for each sample
4. **Record max delta**: Maximum prediction change across all samples for this epsilon

**Robustness score** = max(max*delta*ε₁, max*delta*ε₂, ..., max*delta*εₙ)

### Protected Features

**Critical**: Perturbations are NEVER applied to protected attributes (gender, race, age, etc.) to prevent introducing synthetic bias.

The system automatically excludes features marked in `data.protected_attributes` from perturbation.

### Gate Logic

Results are automatically classified:

| Status     | Condition                                      | Interpretation          |
| ---------- | ---------------------------------------------- | ----------------------- |
| ✅ PASS    | robustness_score < threshold                   | Model is stable         |
| ⚠️ WARNING | threshold ≤ robustness_score < 1.5 × threshold | Approaching instability |
| ❌ FAIL    | robustness_score ≥ 1.5 × threshold             | Model is unstable       |

**Default threshold**: 0.15 (15% max prediction change)

**Example thresholds by use case:**

- **High-stakes decisions** (credit, healthcare): 0.05 (5%)
- **Standard compliance**: 0.15 (15%)
- **Research/exploratory**: 0.25 (25%)

### Interpreting Results

**Good robustness (score < 0.10):**

```
Robustness Score: 0.08
Gate: PASS
Max delta at ε=0.01: 0.03
Max delta at ε=0.05: 0.06
Max delta at ε=0.10: 0.08
```

**Interpretation**: Small input changes → small prediction changes. Model is stable.

**Poor robustness (score > 0.20):**

```
Robustness Score: 0.34
Gate: FAIL
Max delta at ε=0.01: 0.12
Max delta at ε=0.05: 0.28
Max delta at ε=0.10: 0.34
```

**Interpretation**: Small input changes → large prediction changes. Model may be:

- Overfitted to noise
- Relying on unstable features
- Sensitive to measurement error

**Actions**: Retrain with regularization, feature selection, or ensemble methods.

### PDF Output

When enabled, robustness results appear in the audit PDF:

- **Section**: "Model Robustness Testing"
- **Content**:
  - Robustness score with gate status (PASS/WARNING/FAIL)
  - Per-epsilon max delta table
  - Protected features exclusion confirmation
  - Sample size and seed information

### JSON Export

All robustness results are exported in the audit manifest JSON:

```json
{
  "stability_analysis": {
    "robustness_score": 0.084,
    "gate_status": "PASS",
    "threshold": 0.15,
    "per_epsilon_deltas": {
      "0.01": 0.031,
      "0.05": 0.064,
      "0.10": 0.084
    },
    "n_samples": 200,
    "n_features": 21,
    "n_protected": 3,
    "seed": 42
  }
}
```

## Deterministic Execution

All robustness tests are fully deterministic:

- Seeded Gaussian noise generation
- Stable sort order for epsilon values
- Byte-identical results across runs with same config

**Requirement**: Set `reproducibility.random_seed` in your config:

```yaml
reproducibility:
  random_seed: 42 # Required for determinism
```

## Examples

### Minimal Configuration

Enable with defaults:

```yaml
metrics:
  stability:
    enabled: true
```

Uses defaults:

- `epsilon_values: [0.01, 0.05, 0.1]`
- `threshold: 0.15`

### Custom Epsilon Values

Test more aggressive perturbations:

```yaml
metrics:
  stability:
    enabled: true
    epsilon_values: [0.05, 0.10, 0.15, 0.20]
    threshold: 0.20
```

### Strict Threshold

For high-stakes models:

```yaml
metrics:
  stability:
    enabled: true
    epsilon_values: [0.01, 0.05] # Test only small perturbations
    threshold: 0.05 # Allow only 5% max change
```

### CI/CD Integration

Fail deployment if robustness fails:

```yaml
# audit_config.yaml
metrics:
  stability:
    enabled: true
    threshold: 0.15

# In CI pipeline
- name: Run audit with robustness gates
  run: |
    glassalpha audit --config audit_config.yaml --strict
    # Exits with code 1 if robustness gate fails
```

## Complete Example

Full configuration with robustness testing:

```yaml
# audit_config.yaml
# Direct configuration

reproducibility:
  random_seed: 42 # Required for deterministic perturbations

data:
  dataset: german_credit
  target_column: credit_risk
  protected_attributes: # Excluded from perturbations
    - gender
    - age_group
    - foreign_worker

model:
  type: xgboost
  params:
    n_estimators: 100
    max_depth: 5
    random_state: 42

metrics:
  stability:
    enabled: true
    epsilon_values: [0.01, 0.05, 0.1]
    threshold: 0.15 # Fail if >15% prediction change
```

**Run audit:**

```bash
glassalpha audit --config audit_config.yaml --output audit.pdf
```

**Expected output:**

```
Running robustness testing...
  ✓ Epsilon 0.01: max_delta = 0.031
  ✓ Epsilon 0.05: max_delta = 0.064
  ✓ Epsilon 0.10: max_delta = 0.084

Robustness Score: 0.084
Gate: PASS (threshold: 0.15)
```

## Troubleshooting

### "Robustness section missing from PDF"

**Cause**: Stability metrics not enabled.

**Fix**: Set `metrics.stability.enabled: true` in config.

### "Gate status: FAIL"

**Cause**: Model predictions change >threshold under perturbations.

**Fixes**:

1. **Increase threshold** (if acceptable for use case)
2. **Add regularization** (L1/L2, dropout)
3. **Feature selection** (remove unstable features)
4. **Ensemble methods** (averaging reduces variance)

### "Robustness score is 0.00"

**Cause**: Model is deterministic (e.g., decision tree with no randomness) or perturbations too small.

**Fix**: Increase epsilon values or check model is probabilistic.

### "Different robustness scores across runs"

**Cause**: Missing or different random seeds.

**Fix**: Set explicit seed in config:

```yaml
reproducibility:
  random_seed: 42
```

## Related Features

- **[Calibration Analysis](calibration.md)**: Validates probability accuracy
- **[Individual Fairness](fairness-metrics.md#individual-fairness-e11)**: Consistency score (similar to robustness)
- **[Shift Testing](../guides/shift-testing.md)**: Tests robustness to demographic distribution changes

## Implementation Details

**Module**: `glassalpha.metrics.stability.perturbation`

**API**:

- `run_perturbation_sweep()`: Main entry point
- `PerturbationResult`: Result dataclass with robustness_score, gate_status, per_epsilon_deltas

**Test Coverage**: 22 contract tests validating determinism, epsilon validation, gate logic, and protected feature exclusion.

## References

- EU AI Act Article 15: Accuracy, robustness and cybersecurity requirements
- NIST AI RMF: Measure and manage AI risks (trustworthiness)
- SR 11-7 Section III.C.3: Model limitations and testing

## Summary

Robustness testing ensures your model is stable under realistic input variations. Enable it for:

- ✅ Regulatory compliance (EU AI Act, NIST)
- ✅ Production confidence (catch overfitting)
- ✅ Fairness validation (consistency across similar individuals)

Configure once, run in every audit. Results are deterministic, byte-stable, and ready for regulatory submission.
