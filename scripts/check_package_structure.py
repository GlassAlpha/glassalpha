#!/usr/bin/env python3
"""Verify AI hasn't broken package structure."""

import sys
from pathlib import Path

REQUIRED_DIRS = [
    "src/glassalpha/",
    "src/glassalpha/api/",
    "src/glassalpha/cli/",
    "src/glassalpha/models/",
    "tests/",
]

FORBIDDEN_PATTERNS = [
    "temp-*.py",
    "test_*.html",
    "*.pkl",  # Models should be in examples/ or artifacts/
]


def main():
    errors = []

    # Check required structure exists
    for dir_path in REQUIRED_DIRS:
        if not Path(dir_path).exists():
            errors.append(f"Missing required directory: {dir_path}")

    # Check for forbidden files in src/
    src = Path("src/glassalpha/")
    for pattern in FORBIDDEN_PATTERNS:
        for file in src.rglob(pattern):
            errors.append(f"Forbidden file in src/: {file}")

    if errors:
        print("❌ Package structure check failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print("✅ Package structure OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
