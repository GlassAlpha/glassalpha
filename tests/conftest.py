"""Pytest configuration and fixtures for GlassAlpha test suite.

This module provides session-level configuration and fixtures that enforce
deterministic behavior, prevent zombie processes, and provide test utilities.

Critical functions:
- pytest_sessionstart: Sets deterministic environment for reproducibility
- _cleanup_threads: Prevents zombie processes from tqdm/threading
"""

import atexit
import locale
import os
import random
import signal
import sys
import threading

import numpy as np
import pytest

# ============================================================================
# Thread and Process Cleanup (Zombie Prevention)
# ============================================================================


def _cleanup_threads():
    """Clean up any leftover threads before exit.

    Prevents zombie processes from tqdm progress bars and other threading operations.
    Called automatically via atexit and signal handlers.
    """
    for thread in threading.enumerate():
        if thread is not threading.main_thread() and thread.is_alive():
            if hasattr(thread, "_stop"):
                thread._stop()


def _signal_handler(signum, frame):
    """Signal handler for graceful shutdown."""
    _cleanup_threads()
    # Re-raise signal for normal handling
    os.kill(os.getpid(), signum)


# Register cleanup handlers
atexit.register(_cleanup_threads)
signal.signal(signal.SIGINT, _signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, _signal_handler)  # Kill signal


# ============================================================================
# Session-Level Deterministic Environment Setup
# ============================================================================


def pytest_sessionstart(session):
    """Set up deterministic environment for all tests.

    This enforces reproducible test behavior by pinning all sources of entropy
    and platform-specific behavior. Runs once per test session before any tests.

    Critical for:
    - Byte-identical PDFs across runs
    - Reproducible SHAP values
    - Stable floating point operations
    - Cross-platform consistency

    Environment variables set:
    - PYTHONHASHSEED: Pin dict ordering and hash values
    - TZ: Fix timezone for timestamp reproducibility
    - MPLBACKEND: Force non-interactive matplotlib backend
    - GLASSALPHA_NO_PROGRESS: Disable progress bars (prevents tqdm hangs)
    - Threading controls: Single-threaded BLAS/LAPACK for deterministic numerical ops
    - Locale: Fix string sorting and number formatting
    """
    # ========================================================================
    # Core Determinism Controls
    # ========================================================================

    # Pin Python hash seed for dict ordering, set hashing
    # CRITICAL: Must be set before any Python imports that use dicts/sets
    os.environ["PYTHONHASHSEED"] = "0"

    # Fix timezone for timestamp reproducibility
    # All datetime operations will use UTC
    os.environ["TZ"] = "UTC"

    # Force non-interactive matplotlib backend
    # Prevents display-dependent rendering variations
    os.environ["MPLBACKEND"] = "Agg"

    # Disable progress bars during tests (prevents tqdm thread cleanup hangs)
    # Progress bars can cause zombie processes in CI environments
    os.environ["GLASSALPHA_NO_PROGRESS"] = "1"

    # ========================================================================
    # Numerical Determinism (BLAS/LAPACK Threading Control)
    # ========================================================================

    # CRITICAL: Disable BLAS/LAPACK threading for deterministic numerical ops
    # Use assignment (not setdefault) to override any existing values
    # This prevents SHAP from spawning OpenMP threads that become zombies
    #
    # Why this matters:
    # - Multi-threaded BLAS operations have non-deterministic floating point behavior
    # - Thread scheduling is non-deterministic, causing order-dependent results
    # - OpenMP threads can become zombies when not properly cleaned up
    #
    # Performance impact: ~2-3x slower tests, but ensures determinism
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate framework
    os.environ["BLIS_NUM_THREADS"] = "1"  # Alternative BLAS implementation

    # ========================================================================
    # Locale and String Handling
    # ========================================================================

    # Fix locale for string sorting and number formatting
    # Prevents locale-dependent string comparisons and number parsing
    os.environ.setdefault("LC_ALL", "C")
    try:
        locale.setlocale(locale.LC_ALL, "C")
    except locale.Error:
        # Some systems don't support "C" locale, try "en_US.UTF-8"
        try:
            locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
        except locale.Error:
            # If both fail, continue (better than blocking tests)
            # Tests may have minor locale-dependent variations
            pass

    # ========================================================================
    # Random Seed Initialization
    # ========================================================================

    # Seed Python's random module globally
    # NumPy will be seeded in individual tests using fixtures
    # (importing numpy here would slow down test collection)
    random.seed(0)

    # ========================================================================
    # Additional Test Environment Controls
    # ========================================================================

    # Silence tqdm monitor threads in tests (prevents noisy shutdowns)
    os.environ.setdefault("TQDM_DISABLE", "1")

    # Disable any telemetry or network calls during tests
    os.environ.setdefault("GLASSALPHA_TELEMETRY", "0")

    # ========================================================================
    # Platform-Specific Adjustments
    # ========================================================================

    # macOS-specific: Prevent framework issues with multiprocessing
    if sys.platform == "darwin":
        # Set fork method for multiprocessing (safer on macOS)
        os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

    # ========================================================================
    # CI Environment Detection and Adjustments
    # ========================================================================

    # Detect if running in CI and apply additional controls
    is_ci = any(
        [
            os.environ.get("CI") == "true",
            os.environ.get("GITHUB_ACTIONS") == "true",
            os.environ.get("JENKINS_HOME"),
            os.environ.get("GITLAB_CI"),
        ],
    )

    if is_ci:
        # CI-specific: Use even stricter determinism
        os.environ["SOURCE_DATE_EPOCH"] = os.environ.get("SOURCE_DATE_EPOCH", "1577836800")
        # CI-specific: Disable any caching that might cause flakiness
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Register custom markers and configure pytest behavior."""
    # Register custom markers (prevents pytest warnings)
    config.addinivalue_line("markers", "slow: Marks tests as slow (run with -m slow)")
    config.addinivalue_line("markers", "integration: Marks integration tests")
    config.addinivalue_line("markers", "contract: Marks contract tests (test public APIs)")
    config.addinivalue_line("markers", "ci: Marks CI-specific tests")


# ============================================================================
# Session-Level Fixtures
# ============================================================================


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip conditions."""
    for item in items:
        # Auto-mark integration tests based on path
        if "integration" in str(item.fspath):
            item.add_marker("integration")

        # Auto-mark slow tests based on naming convention
        if "slow" in item.name or "e2e" in item.name:
            item.add_marker("slow")


# ============================================================================
# Test Isolation Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def isolate_determinism_state():
    """Ensure each test starts with clean determinism state.

    This fixture runs automatically before each test to ensure:
    1. Random states are reset to known values
    2. No state pollution between tests
    3. Deterministic environment for reproducibility

    Best effort cleanup - restores original state after test.
    """
    import os

    # Save original state
    orig_seed = os.environ.get("PYTHONHASHSEED")
    orig_np_state = np.random.get_state()
    orig_py_state = random.getstate()

    # Reset to known state
    np.random.seed(42)
    random.seed(42)

    yield

    # Restore original state (best effort)
    if orig_seed:
        os.environ["PYTHONHASHSEED"] = orig_seed
    np.random.set_state(orig_np_state)
    random.setstate(orig_py_state)


@pytest.fixture
def deterministic_env():
    """Fixture for tests requiring deterministic environment.

    Returns a copy of the current environment with deterministic settings
    applied. Useful for tests that need to run subprocesses in a controlled
    environment.

    Returns:
        Dict with deterministic environment variables set

    """
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = "1577836800"
    env["PYTHONHASHSEED"] = "42"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["GLASSALPHA_DETERMINISTIC"] = "1"
    return env
