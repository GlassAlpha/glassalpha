#!/usr/bin/env python3
"""Quick demo script to verify the core foundation works.

Run this to see the architecture patterns in action:
- Protocols/interfaces
- Registry pattern
- NoOp implementations
- Deterministic selection
- Feature flags
"""

import os
import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

# Import core components
from glassalpha.core import (
    ExplainerRegistry,
    MetricRegistry,
    ModelRegistry,
    NoOpExplainer,
    NoOpMetric,
    PassThroughModel,
    is_enterprise,
    list_components,
    select_explainer,
)


def main():
    print("=" * 60)
    print("GlassAlpha Core Foundation Demo")
    print("=" * 60)

    # 1. Show registered components
    print("\n1. REGISTERED COMPONENTS:")
    components = list_components()
    for comp_type, names in components.items():
        print(f"   {comp_type}: {names}")

    # 2. Test PassThrough Model
    print("\n2. PASSTHROUGH MODEL TEST:")
    model = PassThroughModel(default_value=0.75)
    test_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    predictions = model.predict(test_data)
    print(f"   Input shape: {test_data.shape}")
    print(f"   Predictions: {predictions}")
    print(f"   Model type: {model.get_model_type()}")

    # 3. Test NoOp Explainer
    print("\n3. NOOP EXPLAINER TEST:")
    explainer = NoOpExplainer()
    explanation = explainer.explain(model, test_data)
    print(f"   Explanation status: {explanation['status']}")
    print(f"   SHAP values shape: {explanation['shap_values'].shape}")
    print(f"   Supports model: {explainer.supports_model(model)}")

    # 4. Test NoOp Metric
    print("\n4. NOOP METRIC TEST:")
    metric = NoOpMetric()
    y_true = np.array([0, 1, 0])
    y_pred = predictions > 0.5
    results = metric.compute(y_true, y_pred)
    print(f"   Metric results: {results}")

    # 5. Test Deterministic Selection
    print("\n5. DETERMINISTIC EXPLAINER SELECTION:")
    config = {"explainers": {"priority": ["noop", "nonexistent"]}}
    selected1 = select_explainer("xgboost", config)
    selected2 = select_explainer("xgboost", config)
    print(f"   First selection: {selected1}")
    print(f"   Second selection: {selected2}")
    print(f"   Deterministic: {selected1 == selected2}")

    # 6. Test Feature Flags
    print("\n6. ENTERPRISE FEATURE FLAGS:")
    print(f"   Enterprise mode: {is_enterprise()}")
    print(f"   License env var: {os.environ.get('GLASSALPHA_LICENSE_KEY', 'Not set')}")

    # Test with license
    os.environ["GLASSALPHA_LICENSE_KEY"] = "demo-key"
    print(f"   After setting license: {is_enterprise()}")
    del os.environ["GLASSALPHA_LICENSE_KEY"]  # Clean up

    # 7. Show how plugins work together
    print("\n7. FULL PIPELINE DEMO:")
    print("   Creating pipeline with minimal components...")

    # Get components from registry
    model_cls = ModelRegistry.get("passthrough")
    explainer_cls = ExplainerRegistry.get("noop")
    metric_cls = MetricRegistry.get("noop")

    # Instantiate
    pipeline_model = model_cls()
    pipeline_explainer = explainer_cls()
    pipeline_metric = metric_cls()

    # Run pipeline
    preds = pipeline_model.predict(test_data)
    expl = pipeline_explainer.explain(pipeline_model, test_data)
    metrics = pipeline_metric.compute(y_true, preds > 0.5)

    print(f"   ✓ Model produced {len(preds)} predictions")
    print(f"   ✓ Explainer produced explanation with status: '{expl['status']}'")
    print(f"   ✓ Metric computed {len(metrics)} values")
    print("\n   Pipeline works with NoOp components!")

    print("\n" + "=" * 60)
    print("✅ Core foundation is working correctly!")
    print("=" * 60)


if __name__ == "__main__":
    main()
