"""Smoke tests for end-to-end audit functionality.

This module provides smoke tests that verify the complete audit pipeline
works without crashing, with focus on explainer selection and PDF generation.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_audit_german_credit_simple_works(tmp_path):
    """Smoke test: full audit should complete without crashing.

    This test validates that the Phase 2.5 explainer fixes allow audits
    to complete successfully without TypeError or signature errors.
    """
    pdf = tmp_path / "audit.pdf"

    # Find the config file - try packages/configs first, then root configs
    packages_config = Path(__file__).parent.parent / "configs" / "german_credit_simple.yaml"
    root_config = Path(__file__).parent.parent.parent / "configs" / "german_credit_simple.yaml"

    if packages_config.exists():
        config_path = packages_config
    elif root_config.exists():
        config_path = root_config
    else:
        pytest.skip(f"german_credit_simple.yaml not found at {packages_config} or {root_config}")

    # Run audit command
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "glassalpha",
            "audit",
            "--config",
            str(config_path),
            "--output",
            str(pdf),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,  # 2 minute timeout
    )

    # Check for signature errors in output
    if "TypeError" in result.stderr or "is_compatible" in result.stderr:
        pytest.fail(
            f"Explainer signature error detected:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

    # Should succeed
    if result.returncode != 0:
        pytest.fail(
            f"Audit failed with code {result.returncode}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

    # PDF should exist and have reasonable size
    assert pdf.exists(), f"PDF was not generated at {pdf}"
    file_size = pdf.stat().st_size
    assert file_size > 10_000, f"PDF seems too small: {file_size} bytes"

    # Manifest should also be generated
    manifest_path = pdf.with_suffix(".manifest.json")
    if manifest_path.exists():
        # Verify manifest is valid JSON
        import json

        with manifest_path.open() as f:
            manifest = json.load(f)
            # Check for configuration key (actual key used in manifests)
            assert "configuration" in manifest or "audit_config" in manifest or "config" in manifest, (
                f"Manifest missing config. Keys found: {list(manifest.keys())}"
            )


def test_audit_stderr_no_explainer_errors(tmp_path):
    """Verify audit output contains no explainer-related errors."""
    pdf = tmp_path / "audit_check.pdf"

    # Find config
    packages_config = Path(__file__).parent.parent / "configs" / "german_credit_simple.yaml"
    root_config = Path(__file__).parent.parent.parent / "configs" / "german_credit_simple.yaml"

    if packages_config.exists():
        config_path = packages_config
    elif root_config.exists():
        config_path = root_config
    else:
        pytest.skip("german_credit_simple.yaml not found")

    # Run audit
    result = subprocess.run(
        [sys.executable, "-m", "glassalpha", "audit", "--config", str(config_path), "--output", str(pdf)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )

    # Check for specific error patterns
    error_patterns = [
        "TypeError",
        "has wrong signature",
        "is_compatible.*incompatible",
        "explainer.*failed",
    ]

    combined_output = result.stdout + result.stderr

    for pattern in error_patterns:
        if pattern.lower() in combined_output.lower():
            pytest.fail(
                f"Found error pattern '{pattern}' in output:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )


@pytest.mark.skipif(
    not Path("configs/quickstart.yaml").exists() and not Path("packages/configs/quickstart.yaml").exists(),
    reason="quickstart.yaml not found",
)
def test_quickstart_audit_works(tmp_path):
    """Test that quickstart config also works (if available)."""
    pdf = tmp_path / "quickstart.pdf"

    # Find quickstart config
    packages_config = Path(__file__).parent.parent / "configs" / "quickstart.yaml"
    root_config = Path(__file__).parent.parent.parent / "configs" / "quickstart.yaml"

    if packages_config.exists():
        config_path = packages_config
    elif root_config.exists():
        config_path = root_config
    else:
        pytest.skip("quickstart.yaml not found")

    result = subprocess.run(
        [sys.executable, "-m", "glassalpha", "audit", "--config", str(config_path), "--output", str(pdf)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )

    # Should complete (though may have warnings)
    if result.returncode != 0:
        # Check if it's just missing data (acceptable for smoke test)
        if "FileNotFoundError" in result.stderr or "No such file" in result.stderr:
            pytest.skip(f"Data file not available for quickstart: {result.stderr}")

        pytest.fail(
            f"Quickstart audit failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

    assert pdf.exists(), "Quickstart PDF not generated"
