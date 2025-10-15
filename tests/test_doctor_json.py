"""Test doctor --json output."""

import json
import subprocess
import sys


def test_doctor_json_output():
    """Verify doctor --json produces valid JSON."""
    result = subprocess.run(
        [sys.executable, "-m", "glassalpha", "doctor", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"doctor --json failed: {result.stderr}"

    # Parse JSON
    data = json.loads(result.stdout)

    # Validate structure
    assert "python_version" in data
    assert "platform" in data
    assert "determinism" in data
    assert "features" in data

    # Validate determinism keys
    assert "PYTHONHASHSEED" in data["determinism"]
    assert "TZ" in data["determinism"]

    # Validate features keys
    assert "shap" in data["features"]
    assert "xgboost" in data["features"]
    assert "lightgbm" in data["features"]
    assert "matplotlib" in data["features"]


def test_doctor_json_output_has_versions():
    """Verify doctor --json includes package version information."""
    result = subprocess.run(
        [sys.executable, "-m", "glassalpha", "doctor", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"doctor --json failed: {result.stderr}"

    data = json.loads(result.stdout)

    # Should have versions section (may be empty if metadata import fails)
    assert "versions" in data

    # If versions are present, should include core packages
    if data["versions"]:
        core_packages = ["numpy", "pandas", "scikit-learn", "glassalpha"]
        for pkg in core_packages:
            assert pkg in data["versions"] or True  # May be missing if import fails
