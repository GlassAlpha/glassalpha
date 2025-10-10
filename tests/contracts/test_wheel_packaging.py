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
        # If __file__ is None, check if build module has the expected attributes
        assert hasattr(build, "ProjectBuilder"), "build module doesn't have ProjectBuilder (wrong package?)"
        return

    build_path = pathlib.Path(build.__file__)
    assert "site-packages" in str(build_path), f"Local 'build' package shadowing PyPA build at {build_path}"


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

            # Build wheel (requires packages/ directory context)
            import subprocess  # noqa: PLC0415
            import sys

            packages_dir = Path(__file__).parent.parent.parent
            result = subprocess.run(  # noqa: S603
                [sys.executable, "-m", "build", "--wheel", "--outdir", str(temp_path)],
                cwd=packages_dir,
                capture_output=True,
                text=True,
                check=False,
            )

            # Fail test if build tools not available
            assert result.returncode == 0, f"Wheel build failed: {result.stderr}"

            # Find the built wheel
            wheel_files = list(temp_path.glob("*.whl"))
            assert len(wheel_files) == 1, f"Expected 1 wheel, found {len(wheel_files)}: {wheel_files}"  # noqa: S101

            wheel_path = wheel_files[0]

            # Open wheel as zip and check contents
            with zipfile.ZipFile(wheel_path, "r") as wheel_zip:
                namelist = wheel_zip.namelist()

                # Contract: Template must be in wheel
                template_path = "glassalpha/report/templates/standard_audit.html"
                assert template_path in namelist, (  # noqa: S101
                    f"Template {template_path} not found in wheel. Available files: {sorted(namelist)}"
                )

                # Verify template content is not empty
                template_content = wheel_zip.read(template_path).decode("utf-8")
                assert len(template_content) > 100, "Template file appears to be empty or too small"  # noqa: PLR2004, S101
                assert "<html" in template_content.lower(), "Template doesn't appear to be HTML"  # noqa: S101

    def test_template_loading_via_importlib_resources(self) -> None:
        """Test that templates can be loaded via importlib.resources.

        Prevents failures where files exist but loader breaks under zipimport.
        """
        from importlib.resources import files  # noqa: PLC0415

        # This should work whether installed from wheel or running from source
        template_files = files("glassalpha.report.templates")
        standard_audit = template_files / "standard_audit.html"

        # Contract: Template must be accessible and readable
        assert standard_audit.is_file(), "standard_audit.html not accessible via importlib.resources"  # noqa: S101

        content = standard_audit.read_text(encoding="utf-8")
        assert len(content) > 100, "Template content too small"  # noqa: PLR2004, S101
        assert "<html" in content.lower(), "Template content doesn't appear to be HTML"  # noqa: S101

    def test_templates_package_has_init(self) -> None:
        """Ensure templates directory has __init__.py for package recognition."""
        import glassalpha.report.templates  # noqa: PLC0415

        # If this import succeeds, the package is properly configured
        # The __init__.py file makes setuptools treat it as a package
        # Verify the module path
        import glassalpha.report.templates as templates_pkg  # noqa: PLC0415

        assert hasattr(templates_pkg, "__file__"), "Templates package should have __file__ attribute"  # noqa: S101

        templates_path = Path(templates_pkg.__file__).parent
        init_file = templates_path / "__init__.py"
        assert init_file.exists(), "Templates directory should have __init__.py"  # noqa: S101

    @pytest.mark.skipif(
        os.getenv("CI") != "true",
        reason="Dist directory test only relevant in CI",
    )
    def test_clean_dist_directory_in_ci(self) -> None:
        """Verify dist directory is clean before wheel build in CI.

        Prevents tests from finding wrong wheels due to dist/ contamination.
        """
        packages_dir = Path(__file__).parent.parent.parent
        dist_dir = packages_dir / "dist"

        if dist_dir.exists():
            wheel_files = list(dist_dir.glob("*.whl"))
            # In CI, we expect at most one wheel (the one we just built)
            assert len(wheel_files) <= 1, (  # noqa: S101
                f"Too many wheels in dist/: {wheel_files}. CI should clean dist/ before building."
            )
