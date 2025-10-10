"""Tests to prevent matplotlib backend regressions.

These tests ensure that matplotlib uses non-interactive backends in CI/test
environments, preventing crashes on macOS and headless Linux systems.
"""

import os

import matplotlib


def test_backend_is_non_interactive_in_ci():
    """Verify matplotlib uses Agg backend in CI environments.

    This prevents the macOS Cocoa backend crash that occurs when matplotlib
    tries to instantiate a GUI backend in headless or non-main-thread contexts.
    """
    # In CI, we always set MPLBACKEND=Agg
    if os.environ.get("CI"):
        backend = matplotlib.get_backend()
        assert backend.lower() == "agg", f"Expected Agg backend in CI, got {backend}"


def test_backend_respects_explicit_setting():
    """Verify that explicit MPLBACKEND settings are respected."""
    # This test verifies that our guard functions respect user overrides
    backend = matplotlib.get_backend()
    expected = os.environ.get("MPLBACKEND", "").lower()

    if expected:
        assert backend.lower() == expected, f"Expected {expected} backend, got {backend}"


def test_plots_module_forces_headless_on_macos():
    """Verify that importing plots.py forces Agg backend on macOS without DISPLAY.

    This test ensures the guard in plots.py works correctly.
    """
    import sys

    # Only run on macOS without explicit backend override
    if sys.platform == "darwin" and not os.environ.get("MPLBACKEND"):
        # Import plots module which should force Agg
        from glassalpha.report import plots

        backend = matplotlib.get_backend()
        assert backend.lower() == "agg", f"Expected Agg backend on macOS, got {backend}"
