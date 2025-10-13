# Using Saved Models with Reasons and Recourse

GlassAlpha allows you to save trained models during audits and use them later for generating reason codes and recourse recommendations. This guide explains how to work with saved models.

## Saving Models During Audits

When running an audit, use the `--save-model` flag to save the trained model:

```bash
# Save model during audit
glassalpha audit --config audit.yaml --save-model model.pkl --output report.html

# Model is automatically saved after training
# Use --fast for quicker development
glassalpha audit --config audit.yaml --save-model model.pkl --fast
```

The saved model file will contain:

- The trained model (wrapped for compatibility)
- Feature metadata for validation
- Training configuration for reproducibility

## Using Saved Models with Reasons

Generate ECOA-compliant reason codes for individual predictions:

```bash
# Basic usage
glassalpha reasons --model model.pkl --data test_data.csv --instance 0 --output reasons.json

# With custom threshold
glassalpha reasons --model model.pkl --data test_data.csv --instance 0 --threshold 0.7 --output reasons.json

# Multiple instances
glassalpha reasons --model model.pkl --data test_data.csv --instances 0,1,2 --output reasons.json
```

**Important**: The saved model expects the same feature preprocessing as was used during training. Use the same data format and preprocessing pipeline.

## Using Saved Models with Recourse

Generate counterfactual recourse recommendations:

```bash
# Basic recourse for denied predictions
glassalpha recourse --model model.pkl --data test_data.csv --instance 0 --output recourse.json

# With policy constraints (immutable features)
glassalpha recourse --model model.pkl --data test_data.csv --instance 0 \
  --immutable age,gender --output recourse.json

# Multiple instances at once
glassalpha recourse --model model.pkl --data test_data.csv --instances 0,1,2 --output recourse.json
```

## Troubleshooting Common Issues

### "Too many missing features" Error

**Problem**: Saved model expects different features than provided data.

**Solution**:

1. **Use preprocessed data**: Models are trained on preprocessed data, not raw CSV
2. **Check feature alignment**: Ensure data has same columns and preprocessing as training data
3. **Verify data source**: Use the same data source and preprocessing pipeline

```bash
# Example workflow:
# 1. Train model with preprocessed data
glassalpha audit --config audit.yaml --save-model model.pkl

# 2. Use same preprocessing for reasons/recourse
# Apply your preprocessing pipeline to new data
preprocessed_data = apply_preprocessing_pipeline(raw_data)
glassalpha reasons --model model.pkl --data preprocessed_data.csv --instance 0
```

### Model File Not Found

**Problem**: Model file doesn't exist or path is incorrect.

**Solution**:

1. **Save model during audit** (required first step)
2. **Use absolute paths** for model files
3. **Check file permissions**

```bash
# Save model first
glassalpha audit --config audit.yaml --save-model /path/to/model.pkl

# Then use it
glassalpha reasons --model /path/to/model.pkl --data data.csv --instance 0
```

### "Model not compatible" Error

**Problem**: Model type not supported by reasons/recourse commands.

**Solution**:

- Use models trained with supported types: XGBoost, LightGBM, LogisticRegression
- Check model was saved correctly during audit
- Verify model file isn't corrupted

## Best Practices

### 1. Consistent Data Pipeline

Always use the same preprocessing for training and inference:

```python
# Apply same preprocessing pipeline
def preprocess_for_model(data):
    # Your preprocessing steps here
    return processed_data

# Training
audit_data = preprocess_for_model(training_data)
glassalpha audit --config audit.yaml --save-model model.pkl

# Inference
inference_data = preprocess_for_model(new_data)
glassalpha reasons --model model.pkl --data inference_data.csv --instance 0
```

### 2. Model Versioning

Track model versions for reproducibility:

```bash
# Include version in model filename
glassalpha audit --config audit.yaml --save-model model_v1.2.pkl

# Document model version in audit config
echo "model_version: v1.2" >> audit.yaml
```

### 3. Batch Processing

For multiple instances, use batch processing:

```bash
# Process multiple instances efficiently
glassalpha reasons --model model.pkl --data test_data.csv \
  --instances 0,1,2,3,4 --output batch_reasons.json

glassalpha recourse --model model.pkl --data test_data.csv \
  --instances 0,1,2,3,4 --output batch_recourse.json
```

## Advanced Usage

### Custom Thresholds

Override default prediction thresholds:

```bash
# Use custom threshold for binary classification
glassalpha reasons --model model.pkl --data data.csv --instance 0 \
  --threshold 0.8 --output reasons.json
```

### Policy Constraints

Apply business rules and constraints:

```bash
# Specify immutable features (cannot be changed in recourse)
glassalpha recourse --model model.pkl --data data.csv --instance 0 \
  --immutable age,gender,credit_history --output recourse.json

# Set feature bounds
glassalpha recourse --model model.pkl --data data.csv --instance 0 \
  --bounds "age:18:65,income:0:100000" --output recourse.json
```

## Integration with CI/CD

Automate model validation in deployment pipelines:

```yaml
# .github/workflows/model-validation.yml
- name: Generate reasons for sample predictions
  run: |
    glassalpha reasons --model model.pkl --data test_data.csv \
      --instances 0,1,2 --output reasons.json

- name: Generate recourse for denied predictions
  run: |
    glassalpha recourse --model model.pkl --data test_data.csv \
      --instances 0,1,2 --output recourse.json
```

## Security Considerations

- **No PII in model files**: Models may contain training data patterns
- **Access control**: Restrict access to saved model files
- **Audit trails**: Track model usage for compliance
- **Version control**: Store models in version control for reproducibility

## Performance Tips

- **Use fast mode** for development: Enable `runtime.fast_mode: true` in config
- **Batch processing** for multiple instances
- **Preprocessed data** avoids redundant feature engineering

## Related Commands

- `glassalpha audit --save-model`: Save model during audit
- `glassalpha models`: List available model types
- `glassalpha datasets info`: Check dataset schema compatibility
- `glassalpha validate`: Validate model and data compatibility

For more examples, see the [examples directory](../examples/index.md) and [API reference](../reference/cli.md).
