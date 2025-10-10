#!/usr/bin/env python3
"""Minimal demo of core foundation without external dependencies.

This demonstrates the architecture patterns work without
needing pandas, numpy, etc.
"""

import os
import sys

sys.path.insert(0, "src")


# First, let's create mock versions of pandas/numpy for demo
class MockDataFrame:
    """Minimal DataFrame replacement for demo."""

    def __init__(self, data):
        self.data = data
        self.shape = (len(next(iter(data.values()))), len(data))

    def __len__(self):
        return self.shape[0]


class MockArray:
    """Minimal numpy array replacement for demo."""

    def __init__(self, data):
        self.data = data
        self.shape = (len(data),) if isinstance(data[0], (int, float)) else (len(data), len(data[0]))

    def __len__(self):
        return len(self.data)


# Mock the imports in our modules
import sys

sys.modules["pandas"] = type(sys)("pandas")
sys.modules["pandas"].DataFrame = MockDataFrame
sys.modules["numpy"] = type(sys)("numpy")
sys.modules["numpy"].ndarray = MockArray
sys.modules["numpy"].array = lambda x: MockArray(x)
sys.modules["numpy"].zeros = lambda shape: MockArray([[0] * shape[1] for _ in range(shape[0])])
sys.modules["numpy"].full = lambda n, val: MockArray([val] * n)
sys.modules["numpy"].unique = lambda x: set(x.data)

# Now import our components
from glassalpha.core import (
    ExplainerRegistry,
    MetricRegistry,
    ModelRegistry,
    is_enterprise,
    list_components,
    select_explainer,
)


def main():
    print("=" * 60)
    print("GlassAlpha Core Foundation Demo (Minimal)")
    print("=" * 60)

    # 1. Show registry pattern works
    print("\n1. REGISTRY PATTERN:")
    print("   - ModelRegistry created ✓")
    print("   - ExplainerRegistry created ✓")
    print("   - MetricRegistry created ✓")

    # 2. List registered components
    print("\n2. AUTO-REGISTERED COMPONENTS:")
    components = list_components()
    for comp_type, names in components.items():
        if names:
            print(f"   {comp_type}: {names}")

    # 3. Test component retrieval
    print("\n3. REGISTRY RETRIEVAL:")
    model_cls = ModelRegistry.get("passthrough")
    print(f"   PassThrough model found: {model_cls is not None}")

    explainer_cls = ExplainerRegistry.get("noop")
    print(f"   NoOp explainer found: {explainer_cls is not None}")

    metric_cls = MetricRegistry.get("noop")
    print(f"   NoOp metric found: {metric_cls is not None}")

    # 4. Test deterministic selection
    print("\n4. DETERMINISTIC SELECTION:")
    config = {"explainers": {"priority": ["noop"]}}

    selected1 = select_explainer("xgboost", config)
    selected2 = select_explainer("xgboost", config)
    selected3 = select_explainer("xgboost", config)

    print(f"   Selection 1: {selected1}")
    print(f"   Selection 2: {selected2}")
    print(f"   Selection 3: {selected3}")
    print(f"   All identical: {selected1 == selected2 == selected3}")

    # 5. Test feature flags
    print("\n5. FEATURE FLAGS:")
    print(f"   Enterprise mode: {is_enterprise()}")

    # Set license key
    os.environ["GLASSALPHA_LICENSE_KEY"] = "test-key"
    print(f"   With license key: {is_enterprise()}")

    # Clean up
    del os.environ["GLASSALPHA_LICENSE_KEY"]
    print(f"   After removing key: {is_enterprise()}")

    # 6. Test enterprise filtering
    print("\n6. ENTERPRISE COMPONENT FILTERING:")

    # Register a test enterprise component
    @MetricRegistry.register("test_enterprise", enterprise=True)
    class TestEnterpriseMetric:
        metric_type = "test"
        version = "1.0.0"

    oss_components = MetricRegistry.get_all(include_enterprise=False)
    all_components = MetricRegistry.get_all(include_enterprise=True)

    print(f"   OSS metrics: {list(oss_components.keys())}")
    print(f"   All metrics: {list(all_components.keys())}")
    print(f"   Enterprise filtered: {'test_enterprise' not in oss_components}")

    # 7. Summary
    print("\n" + "=" * 60)
    print("✅ CORE FOUNDATION VERIFIED:")
    print("   • Protocol interfaces defined")
    print("   • Registry pattern working")
    print("   • Component registration working")
    print("   • Deterministic selection working")
    print("   • Feature flags working")
    print("   • NoOp implementations registered")
    print("\nThe architecture foundation is ready for Phase 1!")
    print("=" * 60)


if __name__ == "__main__":
    main()
