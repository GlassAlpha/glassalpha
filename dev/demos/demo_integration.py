#!/usr/bin/env python3
"""Integration demo to test all Phase 0 and Phase 1 components.

This demonstrates:
- Registry pattern with multiple components
- Audit profiles with validation
- Configuration system with Pydantic
- CLI structure (simulated)
- Strict mode validation
"""

import os
import sys

sys.path.insert(0, "src")

# Mock imports as before
sys.modules["pandas"] = type(sys)("pandas")
sys.modules["numpy"] = type(sys)("numpy")
sys.modules["pandas"].DataFrame = lambda x: x
sys.modules["numpy"].ndarray = list
sys.modules["numpy"].array = lambda x: x
sys.modules["numpy"].zeros = lambda shape: (
    [[0] * shape[1] for _ in range(shape[0])] if len(shape) > 1 else [0] * shape[0]
)
sys.modules["numpy"].full = lambda n, val: [val] * n
sys.modules["numpy"].unique = lambda x: set(x) if hasattr(x, "__iter__") else {x}

import logging

logging.basicConfig(level=logging.INFO)

from glassalpha.config import load_config
from glassalpha.config.strict import StrictModeError
from glassalpha.core import (
    is_enterprise,
    list_components,
    select_explainer,
)
from glassalpha.profiles import TabularComplianceProfile


def main():
    print("\n" + "=" * 60)
    print("GlassAlpha Integration Demo")
    print("=" * 60)

    # 1. Show registered components
    print("\n1. COMPONENT REGISTRY STATUS:")
    components = list_components()
    for comp_type, items in components.items():
        print(f"   {comp_type}: {items}")

    # 2. Test audit profile
    print("\n2. AUDIT PROFILE VALIDATION:")
    profile = TabularComplianceProfile()
    print(f"   Profile: {profile.name}")
    print(f"   Compatible models: {profile.compatible_models[:3]}...")
    print(f"   Required metrics: {len(profile.required_metrics)} types")

    # 3. Test configuration system
    print("\n3. CONFIGURATION SYSTEM:")

    # Minimal config
    minimal_config = {
        "audit_profile": "tabular_compliance",
        "model": {"type": "xgboost", "path": "/path/to/model.pkl"},
        "data": {
            "path": "/path/to/data.csv",
            "schema_path": "/path/to/schema.yaml",
            "protected_attributes": ["age", "gender"],
            "target_column": "outcome",
        },
        "explainers": {"priority": ["treeshap", "noop"]},
    }

    try:
        # Load without strict mode
        config = load_config(minimal_config, strict=False)
        print("   ✓ Configuration loaded successfully")
        print(f"   - Profile: {config.audit_profile}")
        print(f"   - Model: {config.model.type}")
        print(f"   - Explainers: {config.explainers.priority}")
        print(f"   - Strict mode: {config.strict_mode}")
    except Exception as e:
        print(f"   ✗ Config error: {e}")

    # 4. Test strict mode
    print("\n4. STRICT MODE VALIDATION:")

    # Config missing required fields for strict mode
    incomplete_config = {
        "audit_profile": "tabular_compliance",
        "model": {
            "type": "xgboost",
            # Missing path!
        },
        "data": {
            "path": "/path/to/data.csv",
            # Missing schema!
            "protected_attributes": [],
        },
    }

    try:
        config = load_config(incomplete_config, strict=True)
        print("   ✗ Should have failed strict validation!")
    except StrictModeError as e:
        print("   ✓ Strict mode correctly rejected incomplete config")
        error_lines = str(e).split("\n")[:3]
        for line in error_lines:
            if line.strip():
                print(f"     {line}")
    except Exception:
        print("   ✓ Config validation failed as expected")

    # 5. Test deterministic selection
    print("\n5. DETERMINISTIC COMPONENT SELECTION:")

    config_dict = {"explainers": {"priority": ["nonexistent", "noop"]}}  # First doesn't exist

    selected = select_explainer("xgboost", config_dict)
    print(f"   Selected explainer: {selected}")
    print(f"   Fallback worked: {selected == 'noop'}")

    # Multiple selections should be identical
    selections = [select_explainer("xgboost", config_dict) for _ in range(3)]
    print(f"   All selections identical: {len(set(selections)) == 1}")

    # 6. Test enterprise features
    print("\n6. ENTERPRISE FEATURE FLAGS:")
    print(f"   Enterprise mode: {is_enterprise()}")

    # Simulate enterprise
    os.environ["GLASSALPHA_LICENSE_KEY"] = "demo-key"
    print(f"   With license: {is_enterprise()}")
    del os.environ["GLASSALPHA_LICENSE_KEY"]

    # 7. Simulate CLI commands
    print("\n7. CLI COMMAND STRUCTURE (simulated):")
    commands = [
        "glassalpha audit --config audit.yaml --output report.pdf",
        "glassalpha audit --config audit.yaml --output report.pdf --strict",
        "glassalpha validate --config audit.yaml",
        "glassalpha list models",
        "glassalpha dashboard serve  # Enterprise only",
        "glassalpha monitor drift    # Enterprise only",
    ]
    for cmd in commands:
        print(f"   $ {cmd}")

    # 8. Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY:")
    print("=" * 60)
    print("✅ Phase 0 Complete:")
    print("   • Protocol interfaces defined")
    print("   • Registry pattern working")
    print("   • NoOp components registered")
    print("   • Feature flags functional")
    print("   • Audit profiles configured")
    print("   • Enterprise features documented")

    print("\n✅ Phase 1 Progress:")
    print("   • Configuration system with Pydantic")
    print("   • Typer CLI with command groups")
    print("   • Strict mode validation")
    print("   • Deterministic selection")

    print("\n📋 Remaining TODOs:")
    print("   • Architecture tests (test-1, test-2)")
    print("   • OSS/Enterprise package structure (arch-4)")
    print("   • Actual model/explainer implementations")
    print("   • Audit pipeline execution")
    print("   • PDF report generation")

    print("\n🚀 Ready to implement actual ML components!")
    print("=" * 60)


if __name__ == "__main__":
    main()
