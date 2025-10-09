"""Test documentation path references and configuration file accessibility.

This test validates that all configuration files referenced in documentation
actually exist and are accessible, preventing user confusion like path errors.
"""

import re
from pathlib import Path

import pytest


def test_all_documented_config_files_exist():
    """Test that all configuration files referenced in documentation actually exist."""
    project_root = Path(__file__).parent.parent.parent

    # Explicitly test known config file references in documentation
    config_references = [
        # (documentation_file, config_file_name)
        ("site/docs/index.md", "german_credit_simple.yaml"),
        ("site/docs/getting-started/quickstart.md", "german_credit_simple.yaml"),
        ("site/docs/getting-started/installation.md", "german_credit_simple.yaml"),
        ("site/docs/getting-started/custom-data.md", "custom_template.yaml"),
        ("site/docs/getting-started/data-sources.md", "adult_income.yaml"),
        ("site/docs/getting-started/data-sources.md", "compas_recidivism.yaml"),
        ("site/docs/getting-started/data-sources.md", "credit_card_fraud_template.yaml"),
        ("site/docs/getting-started/data-sources.md", "folktables_income_template.yaml"),
        ("site/docs/reference/faq.md", "custom_template.yaml"),
        ("site/docs/reference/troubleshooting.md", "fraud_detection.yaml"),
        ("site/docs/getting-started/configuration.md", "minimal.yaml"),
        ("site/docs/getting-started/configuration.md", "fairness_focused.yaml"),
        ("site/docs/getting-started/configuration.md", "production.yaml"),
        ("site/docs/getting-started/quickstart.md", "custom_template.yaml"),
        ("site/docs/reference/troubleshooting.md", "custom_template.yaml"),
        ("site/docs/getting-started/custom-data.md", "custom_template.yaml"),
        ("site/docs/getting-started/data-sources.md", "custom_template.yaml"),
    ]

    missing_files = []

    for doc_path, config_name in config_references:
        config_path = project_root / "packages" / "configs" / config_name
        if not config_path.exists():
            missing_files.append((doc_path, config_name))

    # Report any missing files
    if missing_files:
        error_msg = "Documentation references non-existent config files:\n"
        for doc_path, config_name in missing_files:
            error_msg += f"  {doc_path} → {config_name}\n"
        pytest.fail(error_msg)


def test_config_files_are_valid_yaml():
    """Test that all configuration files in configs/ are valid YAML."""
    import yaml

    project_root = Path(__file__).parent.parent.parent
    configs_dir = project_root / "packages" / "configs"

    for config_file in configs_dir.glob("*.yaml"):
        if config_file.name in ["custom_template.yaml"]:  # Skip template files that may have template syntax
            continue

        try:
            with open(config_file, encoding="utf-8") as f:
                yaml.safe_load(f)
        except (yaml.YAMLError, UnicodeDecodeError) as e:
            pytest.fail(f"Config file {config_file.name} is not valid YAML: {e}")


def test_config_files_have_required_fields():
    """Test that configuration files have basic required structure."""
    import yaml

    project_root = Path(__file__).parent.parent.parent
    configs_dir = project_root / "packages" / "configs"

    required_fields = ["audit_profile"]

    for config_file in configs_dir.glob("*.yaml"):
        if config_file.name in ["custom_template.yaml"]:  # Skip template files
            continue

        try:
            with open(config_file, encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except (yaml.YAMLError, UnicodeDecodeError) as e:
            pytest.skip(f"Config file {config_file.name} cannot be parsed: {e}")

        for field in required_fields:
            assert field in config, f"Config file {config_file.name} missing required field: {field}"


def test_example_notebooks_reference_existing_configs():
    """Test that example notebooks reference existing configuration files."""
    import re

    project_root = Path(__file__).parent.parent.parent
    examples_dir = project_root / "examples" / "notebooks"

    # Common patterns for config file references in notebooks
    config_patterns = [
        r'configs/([^"]+\.yaml)',
        r'packages/configs/([^"]+\.yaml)',
    ]

    for notebook in examples_dir.glob("*.ipynb"):
        try:
            content = notebook.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            pytest.skip(f"Notebook {notebook.name} has encoding issues")

        for pattern in config_patterns:
            matches = re.findall(pattern, content)
            for config_name in matches:
                config_path = project_root / "packages" / "configs" / config_name
                assert config_path.exists(), f"Notebook {notebook.name} references non-existent config: {config_name}"
