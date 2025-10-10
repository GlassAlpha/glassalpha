"""Test dataset CLI commands."""

import subprocess

import pytest


@pytest.mark.ci
def test_datasets_list_command():
    """Test 'glassalpha datasets list' command."""
    result = subprocess.run(
        ["glassalpha", "datasets", "list"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "german_credit" in result.stdout


@pytest.mark.ci
def test_datasets_cache_dir_command():
    """Test 'glassalpha datasets cache-dir' command."""
    result = subprocess.run(
        ["glassalpha", "datasets", "cache-dir"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "glassalpha" in result.stdout


@pytest.mark.ci
def test_datasets_info_command():
    """Test 'glassalpha datasets info' command.

    Exit code 1 is acceptable if dataset not cached (expected behavior).
    Exit code 0 is acceptable if dataset is cached.
    """
    result = subprocess.run(
        ["glassalpha", "datasets", "info", "german_credit"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",  # Handle emoji in output
    )

    # Either success (cached) or informational failure (not cached) is fine
    assert result.returncode in [
        0,
        1,
    ], f"Command failed with unexpected exit code {result.returncode}: {result.stderr}"

    assert "german_credit" in result.stdout
