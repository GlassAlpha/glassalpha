"""Test that all public modules can be imported successfully.

This ensures the package structure is correct and prevents
import regressions from build configuration issues.
"""


def test_relative_imports():
    """Test that internal relative imports work correctly."""
    # This tests the specific import that was failing in german_credit.py
    from glassalpha.datasets.german_credit import load_german_credit

    # Verify the function can be called (minimal test)
    assert callable(load_german_credit)
