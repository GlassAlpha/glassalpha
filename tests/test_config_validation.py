"""Test that all example configuration files are valid.

This ensures users can rely on example configs without validation errors.
"""

from pathlib import Path

import pytest

from glassalpha.config.loader import load_config_from_file


def get_all_config_files():
    """Get all YAML config files from configs directory."""
    configs_dir = Path(__file__).parent.parent / "configs"
    return list(configs_dir.glob("*.yaml"))


@pytest.mark.parametrize("config_file", get_all_config_files(), ids=lambda p: p.name)
def test_config_file_validates(config_file: Path) -> None:
    """Ensure each example config file is valid.

    Args:
        config_file: Path to configuration file to validate

    This test ensures that:
    1. Config file can be loaded without errors
    2. Schema validation passes
    3. No critical validation errors exist

    Note: Warnings about missing data files are acceptable since
    these are example configs pointing to user-specific paths.

    """
    try:
        config = load_config_from_file(config_file)
        assert config is not None, f"Config {config_file.name} failed to load"

        # Verify basic required fields exist
        assert hasattr(config, "audit_profile"), f"{config_file.name} missing audit_profile"
        assert hasattr(config, "data"), f"{config_file.name} missing data section"
        assert hasattr(config, "model"), f"{config_file.name} missing model section"

        # Verify data section has required fields
        assert hasattr(config.data, "dataset"), f"{config_file.name} data missing dataset field"
        assert hasattr(config.data, "target_column"), f"{config_file.name} data missing target_column"

    except Exception as e:
        pytest.fail(f"Config {config_file.name} validation failed: {e}")


def test_all_configs_found() -> None:
    """Verify that config files were found for testing."""
    config_files = get_all_config_files()
    assert len(config_files) > 0, "No config files found in configs/ directory"
    print(f"\nFound {len(config_files)} config files to validate:")
    for config_file in config_files:
        print(f"  - {config_file.name}")


def test_specific_config_examples_exist() -> None:
    """Verify that key example configs exist."""
    configs_dir = Path(__file__).parent.parent / "configs"

    required_examples = [
        "quickstart.yaml",
        "german_credit_simple.yaml",
        "adult_income_simple.yaml",
    ]

    for example in required_examples:
        config_path = configs_dir / example
        assert config_path.exists(), f"Required example config missing: {example}"


def test_quickstart_config_works() -> None:
    """Verify the quickstart config specifically (most important for new users)."""
    configs_dir = Path(__file__).parent.parent / "configs"
    quickstart_path = configs_dir / "quickstart.yaml"

    config = load_config_from_file(quickstart_path)

    # Verify quickstart uses safe defaults
    assert config.audit_profile == "tabular_compliance"
    assert config.model.type == "logistic_regression"  # Always available
    assert config.data.dataset == "german_credit"  # Built-in dataset
    assert config.reproducibility.random_seed is not None  # Deterministic
