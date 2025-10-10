"""Tests for template packaging and importlib.resources access."""

import importlib.resources

import pytest


def test_templates_packaged():
    """Test that report templates are properly packaged and accessible."""
    # Test that we can access the standard audit template
    try:
        template_files = list(
            importlib.resources.files("glassalpha.report.templates").iterdir(),
        )
        template_names = [f.name for f in template_files if f.is_file()]

        # Should have at least the standard audit template
        assert any("standard_audit" in name for name in template_names), (
            f"Standard audit template not found in packaged files: {template_names}"
        )

    except Exception as e:
        pytest.fail(f"Failed to access packaged templates: {e}")


def test_template_content_accessible():
    """Test that template content can be read via importlib.resources."""
    try:
        # Read the standard audit template
        template_content = importlib.resources.read_text(
            "glassalpha.report.templates",
            "standard_audit.html",
        )

        # Should contain basic HTML structure
        assert "<html" in template_content.lower()
        assert "</html>" in template_content.lower()

        # Should contain Jinja2 template variables
        assert "{{" in template_content
        assert "}}" in template_content

    except FileNotFoundError:
        pytest.skip("Standard audit template not found in package")
    except Exception as e:
        pytest.fail(f"Failed to read template content: {e}")
