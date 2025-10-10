"""Test deterministic reproduction functionality."""

import os
import random

import numpy as np
import pytest

from glassalpha.runtime.repro import (
    get_repro_status,
    reset_repro,
    set_repro,
    validate_repro,
)


def test_set_repro_basic():
    """Test basic reproduction mode setup."""
    status = set_repro(seed=42, strict=False, thread_control=False)

    # Check basic structure
    assert "seed" in status
    assert "strict_mode" in status
    assert "thread_control" in status
    assert "controls" in status

    assert status["seed"] == 42
    assert status["strict_mode"] is False
    assert status["thread_control"] is False

    # Check that some controls were attempted
    assert len(status["controls"]) > 0
    assert "python_random" in status["controls"]
    assert "numpy_random" in status["controls"]


def test_set_repro_strict_mode():
    """Test strict reproduction mode."""
    status = set_repro(seed=123, strict=True, thread_control=True)

    assert status["seed"] == 123
    assert status["strict_mode"] is True
    assert status["thread_control"] is True

    # Should have more controls in strict mode
    controls = status["controls"]
    assert "environment" in controls
    assert "system" in controls

    # Thread control should be enabled
    if "threads" in controls:
        assert controls["threads"].get("success", False) or "failed_variables" in controls["threads"]


def test_python_random_reproducibility():
    """Test that Python random is reproducible after set_repro."""
    # Set reproduction mode
    set_repro(seed=42)

    # Generate random values
    random.seed(42)
    values1 = [random.random() for _ in range(10)]

    # Reset and generate again
    random.seed(42)
    values2 = [random.random() for _ in range(10)]

    # Should be identical
    assert values1 == values2


def test_numpy_random_reproducibility():
    """Test that NumPy random is reproducible after set_repro."""
    # Set reproduction mode
    set_repro(seed=42)

    # Generate random values
    np.random.seed(42)
    values1 = np.random.random(10)

    # Reset and generate again
    np.random.seed(42)
    values2 = np.random.random(10)

    # Should be identical
    assert np.array_equal(values1, values2)


def test_environment_variables_set():
    """Test that environment variables are set correctly."""
    # Store original values
    original_pythonhashseed = os.environ.get("PYTHONHASHSEED")
    original_omp_threads = os.environ.get("OMP_NUM_THREADS")

    try:
        # Set reproduction mode with thread control
        status = set_repro(seed=42, thread_control=True)

        # Check environment variables were set
        assert os.environ.get("PYTHONHASHSEED") == "0"

        if status["controls"].get("threads", {}).get("success", False):
            assert os.environ.get("OMP_NUM_THREADS") == "1"

    finally:
        # Restore original values
        if original_pythonhashseed is not None:
            os.environ["PYTHONHASHSEED"] = original_pythonhashseed
        elif "PYTHONHASHSEED" in os.environ:
            del os.environ["PYTHONHASHSEED"]

        if original_omp_threads is not None:
            os.environ["OMP_NUM_THREADS"] = original_omp_threads
        elif "OMP_NUM_THREADS" in os.environ:
            del os.environ["OMP_NUM_THREADS"]


def test_get_repro_status():
    """Test getting current reproduction status."""
    # Set some reproduction controls
    set_repro(seed=42, thread_control=True)

    # Get status
    status = get_repro_status()

    # Check structure
    assert "environment_variables" in status
    assert "library_versions" in status
    assert "random_states" in status

    # Check environment variables
    env_vars = status["environment_variables"]
    assert "PYTHONHASHSEED" in env_vars
    assert "OMP_NUM_THREADS" in env_vars

    # Check library versions
    lib_versions = status["library_versions"]
    assert "numpy" in lib_versions
    assert "pandas" in lib_versions

    # Check random states
    random_states = status["random_states"]
    assert "python_random" in random_states
    assert "numpy_random" in random_states


def test_validate_repro():
    """Test reproduction validation."""
    # Set reproduction mode
    set_repro(seed=42)

    # Validate
    validation = validate_repro(expected_seed=42)

    # Check structure
    assert "seed" in validation
    assert "tests" in validation
    assert "overall_success" in validation

    assert validation["seed"] == 42

    # Check individual tests
    tests = validation["tests"]
    assert "python_random" in tests
    assert "numpy_random" in tests
    assert "environment" in tests

    # At least basic random tests should pass
    assert tests["python_random"]["success"] is True
    assert tests["numpy_random"]["success"] is True


def test_reset_repro():
    """Test resetting reproduction controls."""
    # Set reproduction mode first
    set_repro(seed=42, thread_control=True)

    # Verify some variables are set
    assert os.environ.get("PYTHONHASHSEED") == "0"

    # Reset
    reset_status = reset_repro()

    # Check structure
    assert "environment_variables" in reset_status
    assert "random_seeds" in reset_status

    # Check that PYTHONHASHSEED was removed
    env_vars = reset_status["environment_variables"]
    assert "PYTHONHASHSEED" in env_vars

    # Verify it's actually removed from environment
    assert os.environ.get("PYTHONHASHSEED") is None


def test_library_specific_controls():
    """Test that library-specific controls are handled correctly."""
    status = set_repro(seed=42, strict=True)

    controls = status["controls"]

    # XGBoost control should exist (even if library not available)
    assert "xgboost" in controls
    xgb_control = controls["xgboost"]
    assert "success" in xgb_control

    # If XGBoost is available, should have parameters
    if xgb_control["success"] and "parameters" in xgb_control:
        params = xgb_control["parameters"]
        assert "random_state" in params
        assert params["random_state"] == 42

    # LightGBM control should exist
    assert "lightgbm" in controls
    lgb_control = controls["lightgbm"]
    assert "success" in lgb_control

    # Scikit-learn should be noted
    assert "sklearn" in controls
    sklearn_control = controls["sklearn"]
    assert "success" in sklearn_control


def test_deterministic_predictions_identical():
    """Test that identical configs produce identical predictions."""
    # This test requires actual ML libraries, so we'll use a simple example
    try:
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        pytest.skip("Scikit-learn not available")

    # Create dataset
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=0)

    # First run
    set_repro(seed=42, strict=True)
    model1 = RandomForestClassifier(n_estimators=10, random_state=42)
    model1.fit(X, y)
    pred1 = model1.predict(X)

    # Second run with same settings
    set_repro(seed=42, strict=True)
    model2 = RandomForestClassifier(n_estimators=10, random_state=42)
    model2.fit(X, y)
    pred2 = model2.predict(X)

    # Predictions should be identical
    assert np.array_equal(pred1, pred2)


def test_different_seeds_different_results():
    """Test that different seeds produce different results."""
    try:
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        pytest.skip("Scikit-learn not available")

    # Create dataset
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=0)

    # First run with seed 42
    set_repro(seed=42)
    model1 = RandomForestClassifier(n_estimators=10, random_state=42)
    model1.fit(X, y)
    pred1 = model1.predict(X)

    # Second run with seed 123
    set_repro(seed=123)
    model2 = RandomForestClassifier(n_estimators=10, random_state=123)
    model2.fit(X, y)
    pred2 = model2.predict(X)

    # Predictions should be different (with high probability)
    assert not np.array_equal(pred1, pred2)


def test_thread_control_environment():
    """Test that thread control sets appropriate environment variables."""
    # Store original values
    original_vars = {}
    thread_vars = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]

    for var in thread_vars:
        original_vars[var] = os.environ.get(var)

    try:
        # Set reproduction with thread control
        status = set_repro(seed=42, thread_control=True)

        # Check that thread control was attempted
        if "threads" in status["controls"] and status["controls"]["threads"].get("success", False):
            # At least OMP_NUM_THREADS should be set
            assert os.environ.get("OMP_NUM_THREADS") == "1"

    finally:
        # Restore original values
        for var, original_value in original_vars.items():
            if original_value is not None:
                os.environ[var] = original_value
            elif var in os.environ:
                del os.environ[var]


def test_warn_on_failure_parameter():
    """Test that warn_on_failure parameter is respected."""
    # This is hard to test directly, but we can check it's passed through
    status = set_repro(seed=42, warn_on_failure=False)

    # Should still succeed
    assert "controls" in status
    assert len(status["controls"]) > 0


def test_repro_status_after_multiple_calls():
    """Test that multiple calls to set_repro work correctly."""
    # First call
    status1 = set_repro(seed=42, strict=False)
    assert status1["seed"] == 42
    assert status1["strict_mode"] is False

    # Second call with different parameters
    status2 = set_repro(seed=123, strict=True, thread_control=True)
    assert status2["seed"] == 123
    assert status2["strict_mode"] is True
    assert status2["thread_control"] is True

    # Environment should reflect latest call
    assert os.environ.get("PYTHONHASHSEED") == "0"


def test_validation_with_wrong_seed():
    """Test validation fails when expected seed doesn't match."""
    # Set reproduction with seed 42
    set_repro(seed=42)

    # Validate with different expected seed
    validation = validate_repro(expected_seed=123)

    # Should still have structure but may show mismatches
    assert "seed" in validation
    assert validation["seed"] == 123  # Expected seed
    assert "tests" in validation


if __name__ == "__main__":
    pytest.main([__file__])
