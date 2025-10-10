"""Test CLI import time budget."""

import subprocess
import sys

import pytest


@pytest.mark.ci
def test_cli_import_time_under_budget():
    """CLI import must be under 300ms (lazy imports working)."""
    # Measure import time using -X importtime
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "importtime",
            "-c",
            "import glassalpha.cli.main",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, f"CLI import failed: {result.stderr}"

    # Parse importtime output to get total time
    lines = result.stderr.strip().split("\n")
    total_time = 0.0
    for line in lines:
        if "import time" in line.lower():
            # Extract time from format like 'import time: 123456 | glassalpha.cli.main'
            parts = line.split("|")
            if len(parts) >= 2:
                time_part = parts[0].strip()
                try:
                    # Time is in microseconds, need to extract number
                    time_str = time_part.split(":")[-1].strip()
                    total_time = float(time_str) / 1000  # Convert to ms
                    break
                except (ValueError, IndexError):
                    continue

    # Gate: Must be under 300ms (lazy imports working)
    IMPORT_BUDGET_MS = 300.0
    assert total_time < IMPORT_BUDGET_MS, (
        f"CLI import time {total_time:.3f}ms exceeds budget {IMPORT_BUDGET_MS}ms. Check for eager imports of heavy dependencies (xgboost, shap, etc.)"
    )

    print(f"✅ CLI import time {total_time:.3f}ms within budget")
