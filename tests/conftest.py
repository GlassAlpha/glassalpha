# packages/tests/conftest.py
import atexit
import locale
import os
import random
import signal
import sys
import threading

# MUST set BEFORE any imports that might use it
# Use assignment (not setdefault) to override any existing values
os.environ["GLASSALPHA_NO_PROGRESS"] = "1"
os.environ["GLASSALPHA_DETERMINISTIC"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TZ"] = "UTC"
os.environ["MPLBACKEND"] = "Agg"
# CRITICAL: Force single-threaded to prevent zombie SHAP/BLAS threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pytest

# Track active threads for cleanup
_test_threads: list[threading.Thread] = []


def _cleanup_threads():
    """Force cleanup of any lingering test threads."""
    for thread in _test_threads:
        if thread.is_alive():
            # Give thread 1 second to finish gracefully
            thread.join(timeout=1.0)
    _test_threads.clear()


def _signal_handler(signum, frame):  # noqa: ARG001
    """Handle interrupts by cleaning up threads before exit."""
    _cleanup_threads()
    # Re-raise the signal to allow normal termination
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Register cleanup handlers
atexit.register(_cleanup_threads)
signal.signal(signal.SIGINT, _signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, _signal_handler)  # Kill signal


def pytest_sessionstart(session):  # noqa: ARG001
    """Set up deterministic environment for all tests.

    This enforces reproducible test behavior by pinning all sources of entropy
    and platform-specific behavior. Runs once per test session before any tests.

    Critical for:
    - Byte-identical PDFs across runs
    - Reproducible SHAP values
    - Stable floating point operations
    - Cross-platform consistency
    """
    # Pin Python hash seed for dict ordering, set hashing
    os.environ["PYTHONHASHSEED"] = "0"

    # Fix timezone for timestamp reproducibility
    os.environ["TZ"] = "UTC"

    # Force non-interactive matplotlib backend
    os.environ["MPLBACKEND"] = "Agg"

    # Disable progress bars during tests (prevents tqdm thread cleanup hangs)
    os.environ["GLASSALPHA_NO_PROGRESS"] = "1"

    # Disable BLAS/LAPACK threading (prevents non-deterministic floating point ops)
    # CRITICAL: Use assignment (not setdefault) to override any existing values
    # This prevents SHAP from spawning OpenMP threads that become zombies
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Fix locale for string sorting and number formatting
    os.environ.setdefault("LC_ALL", "C")
    try:
        locale.setlocale(locale.LC_ALL, "C")
    except locale.Error:
        # Some systems don't support "C" locale, try "en_US.UTF-8"
        try:
            locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
        except locale.Error:
            # If both fail, continue (better than blocking tests)
            pass

    # Seed random number generators
    random.seed(0)

    # NumPy will be seeded in individual tests using fixtures
    # (importing numpy here would slow down test collection)

    # Silence tqdm monitor threads in tests (prevents noisy shutdowns)
    os.environ.setdefault("TQDM_DISABLE", "1")


# Pytest markers for platform-specific tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "linux_only: marks tests that should only run on Linux (e.g., PDF rendering with WeasyPrint)",
    )


# Skip decorator for Linux-only tests
linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="PDF rendering via WeasyPrint is verified on Linux in CI.",
)


@pytest.fixture(autouse=True)
def isolate_tests(request):
    """Ensure test isolation by cleaning up between tests."""
    import gc
    import tempfile

    # Skip isolation for tests that explicitly test module loading
    if "lazy_load" in request.node.name or "import" in request.node.name.lower():
        yield
        return

    # Save original temp directory
    original_tempdir = tempfile.gettempdir()

    yield  # Run the test

    # Force garbage collection to clean up any cached objects
    gc.collect()

    # Restore temp directory
    tempfile.tempdir = original_tempdir

    # Clear any test-specific environment variables
    test_env_vars = [k for k in os.environ if k.startswith("TEST_") or k.startswith("GLASSALPHA_TEST_")]
    for var in test_env_vars:
        os.environ.pop(var, None)
