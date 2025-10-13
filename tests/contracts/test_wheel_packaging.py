"""Regression tests for wheel packaging compliance.

Prevents template packaging failures that cause runtime import errors.
"""

import os
import tempfile
import zipfile
from pathlib import Path

import pytest


def _assert_using_pypa_build():
    """Guard against local 'build' package shadowing PyPA build tool."""
    import pathlib

    import build

    # build.__file__ can be None for namespace packages
    if build.__file__ is None:
        # Check if this is the local build/ directory shadowing PyPA build
        # This happens when running tests from a directory with build artifacts
        if hasattr(build, "__path__"):
            build_paths = [str(p) for p in build.__path__]
            # If any path is in the project directory (not site-packages), skip test
            project_root = pathlib.Path(__file__).parent.parent.parent
            for path in build_paths:
                if str(project_root) in path and "site-packages" not in path:
                    pytest.skip("Local build/ directory shadowing PyPA build package. Run 'rm -rf build/' to fix.")

        # If __file__ is None but it has ProjectBuilder, it's the real package
        if not hasattr(build, "ProjectBuilder"):
            pytest.skip("build module doesn't have ProjectBuilder (wrong package or not installed)")
        return

    build_path = pathlib.Path(build.__file__)
    if "site-packages" not in str(build_path):
        pytest.skip(f"Local 'build' package shadowing PyPA build at {build_path}. Run 'rm -rf build/' to fix.")


class TestWheelPackaging:
    """Test wheel packaging contract compliance."""

    def test_templates_packaged_in_wheel(self) -> None:
        """Ensure templates are properly packaged in built wheel.

        Prevents the "Template not packaged in wheel" flake where tests
        find wrong wheels or templates are missing from built packages.
        """
        # Guard against local build package shadowing PyPA build
        _assert_using_pypa_build()

        # Build wheel in temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Build wheel from project root
            import subprocess
            import sys

            project_root = Path(__file__).parent.parent.parent
            result = subprocess.run(
                [sys.executable, "-m", "build", "--wheel", "--outdir", str(temp_path)],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )

            # Fail test if build tools not available
            assert result.returncode == 0, f"Wheel build failed: {result.stderr}"

            # Find the built wheel
            wheel_files = list(temp_path.glob("*.whl"))
            assert len(wheel_files) == 1, f"Expected 1 wheel, found {len(wheel_files)}: {wheel_files}"

            wheel_path = wheel_files[0]

            # Open wheel as zip and check contents
            with zipfile.ZipFile(wheel_path, "r") as wheel_zip:
                namelist = wheel_zip.namelist()

                # Contract: Template must be in wheel
                template_path = "glassalpha/report/templates/standard_audit.html"
                assert template_path in namelist, (
                    f"Template {template_path} not found in wheel. Available files: {sorted(namelist)}"
                )

                # Verify template content is not empty
                template_content = wheel_zip.read(template_path).decode("utf-8")
                assert len(template_content) > 100, "Template file appears to be empty or too small"
                assert "<html" in template_content.lower(), "Template doesn't appear to be HTML"

    def test_template_loading_via_importlib_resources(self) -> None:
        """Test that templates can be loaded via importlib.resources.

        Prevents failures where files exist but loader breaks under zipimport.
        """
        from importlib.resources import files

        # This should work whether installed from wheel or running from source
        template_files = files("glassalpha.report.templates")
        standard_audit = template_files / "standard_audit.html"

        # Contract: Template must be accessible and readable
        assert standard_audit.is_file(), "standard_audit.html not accessible via importlib.resources"

        content = standard_audit.read_text(encoding="utf-8")
        assert len(content) > 100, "Template content too small"
        assert "<html" in content.lower(), "Template content doesn't appear to be HTML"

    def test_templates_package_has_init(self) -> None:
        """Ensure templates directory has __init__.py for package recognition."""

        # If this import succeeds, the package is properly configured
        # The __init__.py file makes setuptools treat it as a package
        # Verify the module path
        import glassalpha.report.templates as templates_pkg

        assert hasattr(templates_pkg, "__file__"), "Templates package should have __file__ attribute"

        templates_path = Path(templates_pkg.__file__).parent
        init_file = templates_path / "__init__.py"
        assert init_file.exists(), "Templates directory should have __init__.py"

    @pytest.mark.skipif(
        os.getenv("CI") != "true",
        reason="Dist directory test only relevant in CI",
    )
    def test_clean_dist_directory_in_ci(self) -> None:
        """Verify dist directory is clean before wheel build in CI.

        Prevents tests from finding wrong wheels due to dist/ contamination.
        """
        project_root = Path(__file__).parent.parent.parent
        dist_dir = project_root / "dist"

        if dist_dir.exists():
            wheel_files = list(dist_dir.glob("*.whl"))
            # In CI, we expect at most one wheel (the one we just built)
            assert len(wheel_files) <= 1, (
                f"Too many wheels in dist/: {wheel_files}. CI should clean dist/ before building."
            )
