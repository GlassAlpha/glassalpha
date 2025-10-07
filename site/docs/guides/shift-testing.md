# Demographic Shift Testing (E6.5)

## Overview

The **Shift Simulator** tests model robustness by simulating changes in demographic distributions. This helps answer critical questions:

- "What if our user base becomes 10% more female?"
- "How does the model perform if younger users increase by 15%?"
- "Will fairness metrics degrade if demographic shifts occur?"

This feature is essential for:

- **Regulatory compliance**: Stress testing under demographic changes (EU AI Act, NIST AI RMF)
- **Production readiness**: Ensuring model stability as populations evolve
- **CI/CD integration**: Automated gates to catch degradation before deployment

## Quick Start

### Basic Usage

Test a single demographic shift:

```bash
glassalpha audit --config audit.yaml --check-shift gender:+0.1
```

This simulates a 10 percentage point increase in the proportion of one gender group.

### With Degradation Threshold

Fail CI if metrics degrade by more than 5 percentage points:

```bash
glassalpha audit --config audit.yaml \
  --check-shift gender:+0.1 \
  --fail-on-degradation 0.05
```

Exit codes:

- `0`: No violations (PASS)
- `1`: Violations detected (degradation exceeds threshold)

### Multiple Shifts

Test multiple scenarios in one run:

```bash
glassalpha audit --config audit.yaml \
  --check-shift gender:+0.1 \
  --check-shift age:-0.05 \
  --fail-on-degradation 0.05
```

## How It Works

### 1. Post-Stratification Reweighting

The simulator adjusts sample weights to match a target demographic distribution:

```
Original: 40% group A, 60% group B
Shift: +10pp for group A
Result: 50% group A, 50% group B (via reweighting)
```

**Mathematical formula:**

```python
weight[group_A] *= (p_target / p_original)
weight[group_B] *= ((1 - p_target) / (1 - p_original))
```

### 2. Metric Recomputation

All metrics are recomputed with adjusted weights:

- **Fairness**: TPR, FPR, demographic parity, etc.
- **Calibration**: ECE, Brier score
- **Performance**: Accuracy, precision, recall

### 3. Degradation Detection

Degradation = difference between shifted and baseline metrics.

**Example:**

```
Baseline TPR: 0.85
Shifted TPR: 0.78
Degradation: -0.07 (7 percentage points worse)

If threshold = 0.05 → FAIL (exceeds threshold)
```

## Shift Specification Format

### Syntax

```
attribute:shift
```

- `attribute`: Column name in `protected_attributes`
- `shift`: Signed float indicating percentage point change

### Examples

```bash
# Increase proportion by 10pp
--check-shift gender:+0.1

# Decrease proportion by 5pp
--check-shift age:-0.05

# Positive shift (+ is optional)
--check-shift race:0.1
```

### Constraints

- Shifted proportion must be in `[0.01, 0.99]` (1% to 99%)
- Attribute must be binary (multi-class not yet supported)
- Attribute must exist in `protected_attributes`

**Invalid examples:**

```bash
# Would result in >99% (if original is 90%)
--check-shift gender:+0.15  # Error: exceeds 0.99 bound

# Non-existent attribute
--check-shift nonexistent:+0.1  # Error: attribute not found
```

## Configuration

### Audit Config

Ensure your audit config includes the protected attributes you want to test:

```yaml
# audit.yaml
data:
  path: "data/test.csv"
  target: "outcome"
  protected_attributes:
    - gender_male # Binary: 0 or 1
    - age_group # Binary: 0 or 1

model:
  type: xgboost
  path: "models/model.pkl"

metrics:
  fairness:
    threshold: 0.5
  calibration:
    enabled: true

seed: 42
```

### CLI Flags

| Flag                    | Description                                  | Default         |
| ----------------------- | -------------------------------------------- | --------------- |
| `--check-shift`         | Shift specification (can use multiple times) | None            |
| `--fail-on-degradation` | Degradation threshold for CI gates           | None (no gates) |

## Output Format

### Console Output

```
============================================================
DEMOGRAPHIC SHIFT ANALYSIS (E6.5)
============================================================

Loading test data...
Loading model...
Generating predictions...

Analyzing shift: gender_male +0.10 (+10pp)
  Original proportion: 0.387
  Shifted proportion:  0.487
  Gate status: PASS
  ✓ No violations detected

📄 Shift analysis results: audit.shift_analysis.json

✓ Shift analysis complete - no violations detected
```

### JSON Export

Results are exported to `{output}.shift_analysis.json`:

```json
{
  "shift_analysis": {
    "threshold": 0.05,
    "shifts": [
      {
        "shift_specification": {
          "attribute": "gender_male",
          "shift": 0.1,
          "original_proportion": 0.387,
          "shifted_proportion": 0.487
        },
        "baseline_metrics": {
          "fairness": {
            "demographic_parity": 0.123,
            "tpr_difference": 0.045
          },
          "calibration": {
            "ece": 0.032
          },
          "performance": {
            "accuracy": 0.876
          }
        },
        "shifted_metrics": {
          "fairness": {
            "demographic_parity": 0.145,
            "tpr_difference": 0.062
          },
          "calibration": {
            "ece": 0.038
          },
          "performance": {
            "accuracy": 0.869
          }
        },
        "degradation": {
          "fairness": {
            "demographic_parity": 0.022,
            "tpr_difference": 0.017
          },
          "calibration": {
            "ece": 0.006
          },
          "performance": {
            "accuracy": -0.007
          }
        },
        "gate_status": "PASS",
        "violations": []
      }
    ],
    "summary": {
      "total_shifts": 1,
      "violations_detected": false,
      "failed_shifts": 0,
      "warning_shifts": 0
    }
  }
}
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Model Audit with Shift Testing

on: [pull_request]

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install GlassAlpha
        run: pip install glassalpha[all]

      - name: Run audit with shift tests
        run: |
          glassalpha audit \
            --config audit.yaml \
            --check-shift gender:+0.1 \
            --check-shift age:-0.05 \
            --fail-on-degradation 0.05

      - name: Upload shift analysis results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: shift-analysis
          path: "*.shift_analysis.json"
```

### Exit Code Handling

```bash
# In CI script
glassalpha audit --config audit.yaml \
  --check-shift gender:+0.1 \
  --fail-on-degradation 0.05

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "✓ Shift test passed"
elif [ $EXIT_CODE -eq 1 ]; then
  echo "✗ Shift test failed - degradation exceeds threshold"
  exit 1
else
  echo "✗ Error running shift test"
  exit 1
fi
```

## Use Cases

### 1. Regulatory Stress Testing

**Scenario**: EU AI Act requires testing under demographic distribution changes.

```bash
# Test ±10% shifts for each protected attribute
glassalpha audit --config audit.yaml \
  --check-shift gender:+0.1 \
  --check-shift gender:-0.1 \
  --check-shift age:+0.1 \
  --check-shift age:-0.1 \
  --fail-on-degradation 0.05
```

**Interpretation**: Model must maintain fairness/performance within 5pp under realistic demographic shifts.

### 2. Pre-Deployment Validation

**Scenario**: Ensure model is robust before production deployment.

```bash
# Conservative test (±5% shifts, strict 2% threshold)
glassalpha audit --config audit.yaml \
  --check-shift gender:+0.05 \
  --check-shift gender:-0.05 \
  --fail-on-degradation 0.02
```

**Action**: Only deploy if all shifts pass.

### 3. Continuous Monitoring

**Scenario**: Detect if real-world demographic drift impacts model.

```bash
# Monthly check with actual drift values
glassalpha audit --config audit.yaml \
  --check-shift gender:+0.03  # Based on observed drift
  --fail-on-degradation 0.05
```

**Alert**: If shift test fails, investigate root cause and consider retraining.

### 4. Scenario Planning

**Scenario**: Marketing team plans campaign targeting specific demographics.

```bash
# Simulate expected demographic change from campaign
glassalpha audit --config audit.yaml \
  --check-shift age:-0.15  # Expect 15pp increase in younger users
```

**Analysis**: Review degradation to assess if model needs adjustment before campaign.

## Interpretation Guide

### Gate Status

| Status    | Meaning                                    | Action            |
| --------- | ------------------------------------------ | ----------------- |
| `PASS`    | No degradation exceeds threshold           | ✓ Safe to proceed |
| `WARNING` | Some metrics degraded but within tolerance | ⚠ Monitor closely |
| `FAIL`    | Degradation exceeds threshold              | ✗ Do not deploy   |
| `INFO`    | No threshold set (informational only)      | Review manually   |

### Degradation Metrics

**Fairness degradation:**

- **< 2pp**: Minimal impact
- **2-5pp**: Moderate concern (monitor)
- **> 5pp**: Significant concern (investigate)

**Calibration degradation (ECE):**

- **< 0.02**: Minimal impact
- **0.02-0.05**: Moderate impact
- **> 0.05**: Poor calibration

**Performance degradation:**

- **< 1pp**: Negligible
- **1-3pp**: Moderate
- **> 3pp**: Significant

### Common Patterns

**Large fairness degradation + small performance degradation:**

- **Cause**: Shift exposes existing bias in model
- **Action**: Consider fairness constraints in retraining

**Large calibration degradation:**

- **Cause**: Model not well-calibrated across demographics
- **Action**: Apply calibration (Platt scaling, isotonic regression)

**All metrics degrade significantly:**

- **Cause**: Model overfits to original distribution
- **Action**: Collect more diverse training data

## Limitations

### Current Version (E6.5)

**Supported:**

- ✅ Binary protected attributes
- ✅ Single-factor shifts (one attribute at a time)
- ✅ Absolute percentage point shifts
- ✅ Deterministic reweighting

**Not yet supported (Future/Enterprise):**

- ❌ Multi-class protected attributes
- ❌ Multi-factor shifts (e.g., `gender:+0.1,age:-0.05` simultaneously)
- ❌ Relative shifts (e.g., "increase by 10%")
- ❌ Complex scenario files (YAML-based stress tests)

### Statistical Considerations

**Sample size**: Ensure sufficient samples in each group after reweighting. Very extreme shifts may produce unreliable results.

**Reweighting limits**: Shifts are constrained to [1%, 99%] to prevent extreme weight values that would be unrealistic.

**Independence assumption**: Treats demographic attributes as independent. Real-world correlations are not modeled.

## Troubleshooting

### Error: "Shifted proportion must be ≤ 0.99"

**Cause**: Shift would result in >99% or <1% proportion.

**Solution**: Use a smaller shift value.

```bash
# If original is 92%
--check-shift gender:+0.05  # Results in 97% ✓
--check-shift gender:+0.10  # Results in 102% ✗ (exceeds bound)
```

### Error: "Attribute 'X' not found"

**Cause**: Attribute not in `protected_attributes`.

**Solution**: Check audit config:

```yaml
data:
  protected_attributes:
    - gender_male # Must match exactly
```

### No violations detected but metrics degraded

**Cause**: No `--fail-on-degradation` threshold set.

**Solution**: Add threshold to enable gates:

```bash
--fail-on-degradation 0.05
```

### Extremely large degradation values

**Cause**: Insufficient samples in one group, or shift creates very unbalanced distribution.

**Solution**:

1. Check sample sizes in each group
2. Use smaller shift values
3. Collect more data if group is very small

## Best Practices

1. **Start conservative**: Test ±5% shifts before larger values
2. **Set realistic thresholds**: 5-10pp degradation is typical tolerance
3. **Test both directions**: Check positive and negative shifts
4. **Automate in CI**: Make shift testing part of deployment pipeline
5. **Document assumptions**: Note which shifts reflect realistic scenarios
6. **Review regularly**: Update shift values based on observed demographic drift

## Related Features

- **E5.1 Intersectional Fairness**: Analyze multi-way demographic interactions
- **E10 Statistical Confidence**: Add confidence intervals to shift metrics (Enterprise)
- **E12 Dataset Bias**: Detect bias before shift testing
- **E13 Fairness Drift Monitoring**: Track real-world demographic changes (Enterprise)

## References

- Post-stratification: [Lohr (2009) Sampling: Design and Analysis](https://www.wiley.com/en-us/Sampling%3A+Design+and+Analysis%2C+2nd+Edition-p-9780538733526)
- EU AI Act stress testing requirements
- NIST AI Risk Management Framework

---

**Questions?** See [FAQ](../faq.md) or [file an issue](https://github.com/glassalpha/glassalpha/issues).
