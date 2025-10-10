"""Contract tests for constants and resource packaging.

These tests ensure that exact contract strings never drift and that
wheel packaging correctly includes template resources.
"""

import pytest


def test_exact_contract_strings():
    """Ensure contract strings never drift from expected values."""
    from glassalpha.constants import NO_EXPLAINER_MSG, NO_MODEL_MSG

    # Exact strings required by tests and error handling
    assert NO_MODEL_MSG == "Model not loaded. Load a model first."
    assert NO_EXPLAINER_MSG == "No compatible explainer found"


def test_wheel_template_resources_available():
    """Ensure wheel packaging includes template resources."""
    try:
        from importlib.resources import files
    except ImportError:
        pytest.skip("importlib.resources not available")

    # Check that standard audit template is packaged
    template_files = files("glassalpha.report.templates")
    standard_template = template_files.joinpath("standard_audit.html")

    assert standard_template.is_file(), (
        "standard_audit.html template not found in wheel. Check pyproject.toml package-data and MANIFEST.in"
    )


def test_constants_module_exports():
    """Ensure all required constants are properly exported."""
    from glassalpha.constants import BINARY_CLASSES, BINARY_THRESHOLD, INIT_LOG_MESSAGE

    # Check that key constants exist and have expected types
    assert isinstance(INIT_LOG_MESSAGE, str)
    assert BINARY_CLASSES == 2
    assert BINARY_THRESHOLD == 0.5


def test_backward_compatible_aliases():
    """Ensure backward-compatible aliases are maintained."""
    from glassalpha.constants import NO_EXPLAINER_MSG, NO_MODEL_MSG

    # These aliases should still work for existing code
    assert isinstance(NO_MODEL_MSG, str)
    assert isinstance(NO_EXPLAINER_MSG, str)


def test_import_contract_critical_modules():
    """Ensure contract-critical modules import cleanly and basic functions work."""
    import numpy as np
    import pandas as pd

    from glassalpha.constants import NO_MODEL_MSG
    from glassalpha.models._features import align_features

    # Import base wrapper and feature alignment (covered by contract tests)
    from glassalpha.models.tabular.base import BaseTabularWrapper, _ensure_fitted

    # Test _ensure_fitted functionality
    class MockModel:
        def __init__(self):
            self.model = None
            self._is_fitted = False

    mock = MockModel()
    try:
        _ensure_fitted(mock)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == NO_MODEL_MSG

    # Test align_features with basic DataFrame
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    feature_names = ["a", "b"]
    aligned = align_features(df, feature_names)
    assert list(aligned.columns) == ["a", "b"]

    # Test align_features with renamed columns (same width)
    df_renamed = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    aligned_renamed = align_features(df_renamed, feature_names)
    assert list(aligned_renamed.columns) == ["a", "b"]
    assert aligned_renamed.values.tolist() == [[1, 3], [2, 4]]

    # Test with non-DataFrame (should pass through)
    arr = np.array([[1, 2], [3, 4]])
    aligned_arr = align_features(arr, feature_names)
    assert np.array_equal(aligned_arr, arr)

    # Verify base wrapper can be imported (class exists)
    assert BaseTabularWrapper is not None
