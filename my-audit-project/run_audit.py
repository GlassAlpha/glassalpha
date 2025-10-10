#!/usr/bin/env python3
"""Example audit script for german_credit dataset with xgboost model.

This script demonstrates the simplest way to run a GlassAlpha audit programmatically.

Requirements:
    - GlassAlpha must be installed: pip install 'glassalpha[all]'
    - For development from source: pip install -e ".[all]" from project root
"""

import sys
from pathlib import Path

# Check if glassalpha is installed
try:
    from glassalpha.api import run_audit
except ImportError:
    print("Error: GlassAlpha not installed.")
    print()
    print("Install with one of these commands:")
    print("  pip install 'glassalpha[all]'  # From PyPI")
    print("  cd ../packages && pip install -e '.[all]'  # From source")
    print()
    sys.exit(1)

if __name__ == "__main__":
    print("=" * 60)
    print("GlassAlpha Audit - German Credit")
    print("=" * 60)
    print()

    # Configuration paths
    config_path = Path("audit_config.yaml")
    output_path = Path("reports/audit_report.html")

    print(f"Configuration: {config_path}")
    print(f"Output: {output_path}")
    print()
    print("Running audit...")
    print()

    # Run audit using programmatic API
    try:
        report_path = run_audit(
            config_path=config_path,
            output_path=output_path,
        )

        print()
        print("✓ Audit complete!")
        print(f"  Report: {report_path}")
        print()
        print("Next steps:")
        print("  1. Open reports/audit_report.html to view the audit")
        print("  2. Modify audit_config.yaml to customize metrics")
        print("  3. Try CLI for shift testing: glassalpha audit --check-shift gender:+0.1")
        print()

    except Exception as e:
        print(f"✗ Audit failed: {e}")
        raise
