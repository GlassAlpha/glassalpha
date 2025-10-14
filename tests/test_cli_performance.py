"""Performance regression tests for CLI commands.

Ensures CLI responsiveness for user-facing commands.
"""

import os
import subprocess
import sys
import time

import pytest


class TestCLIPerformance:
    """Test CLI command performance to prevent regressions."""

    @pytest.mark.slow
    def test_help_command_performance(self) -> None:
        """Help commands must be <300ms (user perception of instant).

        User experience research shows:
        - <100ms: Feels instant
        - 100-300ms: Acceptable
        - >300ms: Noticeable delay

        We target <300ms for help commands (no heavy imports).
        """
        start = time.time()
        # Ensure UTF-8 encoding for subprocess to handle Unicode in help text
        env = os.environ.copy()
        env["LC_ALL"] = "C.UTF-8"
        env["LANG"] = "C.UTF-8"
        result = subprocess.run(
            [sys.executable, "-m", "glassalpha", "--help"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        elapsed = time.time() - start

        assert result.returncode == 0, "Help command should succeed"
        assert elapsed < 0.3, (
            f"--help took {elapsed:.3f}s (expected <0.3s).\n"
            f"This suggests heavy imports during CLI initialization.\n"
            f"Use lazy imports for numpy, pandas, sklearn, etc."
        )

    @pytest.mark.slow
    def test_audit_help_performance(self) -> None:
        """Audit help command should be fast (<300ms)."""
        start = time.time()
        # Ensure UTF-8 encoding for subprocess to handle Unicode in help text
        env = os.environ.copy()
        env["LC_ALL"] = "C.UTF-8"
        env["LANG"] = "C.UTF-8"
        result = subprocess.run(
            [sys.executable, "-m", "glassalpha", "audit", "--help"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        elapsed = time.time() - start

        assert result.returncode == 0, "Audit help should succeed"
        assert elapsed < 0.3, (
            f"audit --help took {elapsed:.3f}s (expected <0.3s).\nCommand-specific help should be fast."
        )

    @pytest.mark.slow
    def test_doctor_command_performance(self) -> None:
        """Doctor command should complete reasonably fast (<2 seconds)."""
        start = time.time()
        # Ensure UTF-8 encoding for subprocess to handle Unicode in output
        env = os.environ.copy()
        env["LC_ALL"] = "C.UTF-8"
        env["LANG"] = "C.UTF-8"
        result = subprocess.run(
            [sys.executable, "-m", "glassalpha", "doctor"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        elapsed = time.time() - start

        assert result.returncode == 0, "Doctor command should succeed"
        assert elapsed < 2.0, f"doctor took {elapsed:.3f}s (expected <2.0s).\nEnvironment check should be quick."

    @pytest.mark.slow
    def test_version_command_performance(self) -> None:
        """Version command should be instant (<100ms)."""
        start = time.time()
        # Ensure UTF-8 encoding for subprocess to handle Unicode in output
        env = os.environ.copy()
        env["LC_ALL"] = "C.UTF-8"
        env["LANG"] = "C.UTF-8"
        result = subprocess.run(
            [sys.executable, "-m", "glassalpha", "--version"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        elapsed = time.time() - start

        assert result.returncode == 0, "Version command should succeed"
        assert elapsed < 0.1, (
            f"--version took {elapsed:.3f}s (expected <0.1s).\nVersion should be instant (no imports needed)."
        )
