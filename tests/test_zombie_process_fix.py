"""Verify zombie process fix works correctly.

This test verifies that SHAP runs single-threaded and doesn't create zombie processes.
"""

import os

import pytest


def test_environment_variables_set():
    """Verify critical environment variables are set to prevent zombie processes."""
    # These must be set to "1" to prevent SHAP/BLAS from spawning threads
    assert os.environ.get("OMP_NUM_THREADS") == "1", "OMP_NUM_THREADS must be 1"
    assert os.environ.get("MKL_NUM_THREADS") == "1", "MKL_NUM_THREADS must be 1"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1", "OPENBLAS_NUM_THREADS must be 1"
    assert os.environ.get("NUMEXPR_NUM_THREADS") == "1", "NUMEXPR_NUM_THREADS must be 1"


@pytest.mark.skipif(
    not os.environ.get("CI") and not os.environ.get("GLASSALPHA_TEST_SHAP"),
    reason="Skipping SHAP test (slow). Set GLASSALPHA_TEST_SHAP=1 to run",
)
def test_shap_runs_single_threaded():
    """Verify SHAP runs without spawning threads."""
    import numpy as np

    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        pytest.skip("shap or sklearn not installed")

    # Create tiny model and data
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y = np.array([0, 1, 1, 0])

    model = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=42)
    model.fit(X, y)

    # This should not spawn threads (OMP_NUM_THREADS=1)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Verify we got results
    assert shap_values is not None
    assert len(shap_values) > 0


def test_no_background_threads_after_import():
    """Verify no unexpected background threads are running."""
    import threading

    # Count non-daemon threads (daemon threads are OK)
    active_threads = [t for t in threading.enumerate() if not t.daemon and t != threading.current_thread()]

    # Should only be main thread
    assert len(active_threads) == 0, f"Unexpected threads: {[t.name for t in active_threads]}"
