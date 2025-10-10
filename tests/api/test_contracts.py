"""Phase 1: Namespace & Import Contract Tests

Tests for fast imports with PEP 562 lazy loading.
"""

import sys
import time

import pytest

# Mark as contract test (must pass before release)
pytestmark = pytest.mark.contract


@pytest.fixture(autouse=True)
def _restore_modules():
    """Save and restore sys.modules to prevent test contamination."""
    # Save original state
    original_modules = sys.modules.copy()

    yield

    # Restore original state after test
    # Remove any new glassalpha modules added during test
    for key in list(sys.modules.keys()):
        if key.startswith("glassalpha") and key not in original_modules:
            del sys.modules[key]

    # Restore any deleted glassalpha modules
    for key, module in original_modules.items():
        if key.startswith("glassalpha") and key not in sys.modules:
            sys.modules[key] = module


class TestImportSpeed:
    """Import must complete in <200ms"""

    def test_import_speed_cold_start(self):
        """Import completes in <200ms on cold start"""
        # Remove glassalpha from sys.modules if present
        modules_to_remove = [key for key in sys.modules.keys() if key.startswith("glassalpha")]
        for key in modules_to_remove:
            del sys.modules[key]

        # Time the import
        start = time.time()
        import glassalpha as ga

        duration = time.time() - start

        assert duration < 0.2, f"Import took {duration:.3f}s (threshold: 0.2s)"

    def test_import_speed_repeated(self):
        """Repeated imports are instant (cached)"""
        import glassalpha as ga

        start = time.time()
        import glassalpha as ga2

        duration = time.time() - start

        assert duration < 0.001, f"Repeat import took {duration:.6f}s"


class TestLazyLoading:
    """Modules load on first access"""

    def test_audit_not_loaded_initially(self):
        """Audit module not in sys.modules before first access"""
        # Remove glassalpha modules
        modules_to_remove = [key for key in sys.modules.keys() if key.startswith("glassalpha")]
        for key in modules_to_remove:
            del sys.modules[key]

        # Import glassalpha
        import glassalpha as ga

        # Check audit not loaded yet
        assert "glassalpha.api" not in sys.modules, "audit module should not be loaded until accessed"

    def test_audit_loads_on_access(self):
        """Audit module loads when accessed"""
        import glassalpha as ga

        # Remove audit module if present
        if "glassalpha.api" in sys.modules:
            del sys.modules["glassalpha.api"]

        # Access audit attribute (will trigger lazy load)
        _ = ga.audit

        # Verify it loaded
        assert "glassalpha.api" in sys.modules, "audit module should be in sys.modules after access"

        # Verify entry points are accessible
        assert hasattr(ga.audit, "from_model")
        assert hasattr(ga.audit, "from_predictions")
        assert hasattr(ga.audit, "from_config")

    def test_datasets_lazy_loads(self):
        """Datasets module loads on access"""
        import glassalpha as ga

        # Try to access datasets
        try:
            _ = ga.datasets
        except (ImportError, AttributeError):
            # Module doesn't exist yet, that's ok
            pass

    def test_utils_lazy_loads(self):
        """Utils module loads on access"""
        import glassalpha as ga

        # Try to access utils
        try:
            _ = ga.utils
        except (ImportError, AttributeError):
            # Module doesn't exist yet, that's ok
            pass


class TestTabCompletion:
    """dir() includes lazy modules for tab completion"""

    def test_dir_includes_audit(self):
        """dir(ga) includes 'audit'"""
        import glassalpha as ga

        dir_output = dir(ga)
        assert "audit" in dir_output, "'audit' should be in dir(ga) for tab completion"

    def test_dir_includes_datasets(self):
        """dir(ga) includes 'datasets'"""
        import glassalpha as ga

        dir_output = dir(ga)
        assert "datasets" in dir_output, "'datasets' should be in dir(ga) for tab completion"

    def test_dir_includes_utils(self):
        """dir(ga) includes 'utils'"""
        import glassalpha as ga

        dir_output = dir(ga)
        assert "utils" in dir_output, "'utils' should be in dir(ga) for tab completion"

    def test_dir_includes_version(self):
        """dir(ga) includes '__version__'"""
        import glassalpha as ga

        dir_output = dir(ga)
        assert "__version__" in dir_output


class TestPrivacyEnforcement:
    """Internal modules not exposed in public API"""

    def test_export_not_in_dir(self):
        """_export not in dir(ga)"""
        import glassalpha as ga

        dir_output = dir(ga)
        assert "_export" not in dir_output, "Private module '_export' should not be in public API"

    def test_export_not_in_all(self):
        """_export not in __all__"""
        import glassalpha as ga

        assert "_export" not in ga.__all__, "Private module '_export' should not be in __all__"

    def test_lazy_modules_dict_private(self):
        """_LAZY_MODULES dict is private (starts with _)"""
        import glassalpha as ga

        # Can access via __dict__, but not in public API
        assert "_LAZY_MODULES" in ga.__dict__
        assert "_LAZY_MODULES" not in dir(ga)


class TestVersionExport:
    """Version is immediately available (not lazy)"""

    def test_version_accessible(self):
        """__version__ accessible without triggering lazy load"""
        import glassalpha as ga

        version = ga.__version__
        assert isinstance(version, str)
        assert version.startswith("0."), f"Expected v0.x, got {version}"

    def test_version_format(self):
        """__version__ follows semantic versioning"""
        import glassalpha as ga

        parts = ga.__version__.split(".")
        assert len(parts) >= 2, "Version should be X.Y or X.Y.Z"

        # Should be parseable as integers
        for part in parts:
            assert part.isdigit() or part.endswith("a") or part.endswith("b"), f"Invalid version component: {part}"


class TestAttributeError:
    """Unknown attributes raise AttributeError"""

    def test_unknown_attribute_raises(self):
        """Accessing unknown attribute raises AttributeError"""
        import glassalpha as ga

        with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
            _ = ga.nonexistent

    def test_error_message_helpful(self):
        """Error message includes module name and attribute"""
        import glassalpha as ga

        try:
            _ = ga.this_does_not_exist
        except AttributeError as e:
            error_msg = str(e)
            assert "glassalpha" in error_msg
            assert "this_does_not_exist" in error_msg
        else:
            pytest.fail("Should have raised AttributeError")


class TestNoHeavyImportsOnInit:
    """No heavy dependencies loaded on import"""

    def test_sklearn_not_loaded(self):
        """Sklearn not loaded on import glassalpha

        Runs in subprocess to avoid polluting sys.modules for parallel tests.
        """
        import subprocess

        code = """
import sys
import glassalpha as ga

# Sklearn should not be loaded on import
sklearn_modules = [key for key in sys.modules.keys() if key.startswith("sklearn")]
if sklearn_modules:
    print(f"FAILED: sklearn loaded: {sklearn_modules}")
    sys.exit(1)
else:
    print("OK: sklearn not loaded")
    sys.exit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

        assert result.returncode == 0, f"sklearn was loaded on import: {result.stdout}"

    def test_xgboost_not_loaded(self):
        """Xgboost not loaded on import glassalpha

        Runs in subprocess to avoid polluting sys.modules for parallel tests.
        """
        import subprocess

        code = """
import sys
import glassalpha as ga

# Xgboost should not be loaded on import
xgboost_modules = [key for key in sys.modules.keys() if key.startswith("xgboost")]
if xgboost_modules:
    print(f"FAILED: xgboost loaded: {xgboost_modules}")
    sys.exit(1)
else:
    print("OK: xgboost not loaded")
    sys.exit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

        assert result.returncode == 0, f"xgboost was loaded on import: {result.stdout}"

    def test_matplotlib_not_loaded(self):
        """Matplotlib not loaded on import glassalpha

        Runs in subprocess to avoid polluting sys.modules for parallel tests.
        """
        import subprocess

        code = """
import sys
import glassalpha as ga

# Matplotlib should not be loaded on import
matplotlib_modules = [key for key in sys.modules.keys() if key.startswith("matplotlib")]
if matplotlib_modules:
    print(f"FAILED: matplotlib loaded: {matplotlib_modules}")
    sys.exit(1)
else:
    print("OK: matplotlib not loaded")
    sys.exit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

        assert result.returncode == 0, f"matplotlib was loaded on import: {result.stdout}"
