"""Test the source_loader utility functions."""

import pytest

from .source_loader import get_module_source


def test_get_module_source_with_inspect():
    """Test that get_module_source works with inspect.getsource for loaded modules."""
    import os  # Use a standard library module as test case

    source = get_module_source(os)
    assert "import" in source  # Should contain import statements
    assert "def" in source  # Should contain function definitions
    assert len(source) > 0  # Should not be empty


def test_get_module_source_with_fallback():
    """Test that get_module_source handles fallback gracefully."""
    # For standard library modules, the fallback won't work as expected
    # because they're not in the package structure that importlib.resources expects
    import inspect
    import os
    from unittest.mock import patch

    with patch.object(inspect, "getsource", side_effect=OSError("Mocked error")):
        # This should fail gracefully since "os" is not a package with files we can access
        with pytest.raises(FileNotFoundError):
            get_module_source(os, pkg_fallback="os", filename="os.py")


def test_get_module_source_no_fallback_fails():
    """Test that get_module_source fails gracefully when no fallback is available."""
    import inspect
    import os
    from unittest.mock import patch

    with patch.object(inspect, "getsource", side_effect=OSError("Mocked error")):
        with pytest.raises(FileNotFoundError, match="Could not load source"):
            get_module_source(os)  # No pkg_fallback or filename provided


def test_get_module_source_glassalpha_pipeline():
    """Test that get_module_source works with glassalpha pipeline modules."""
    from glassalpha.pipeline import audit

    source = get_module_source(audit, pkg_fallback="glassalpha.pipeline", filename="audit.py")
    assert len(source) > 0
    assert "class AuditPipeline" in source  # Should contain the main class
    assert "def run" in source  # Should contain the main method
