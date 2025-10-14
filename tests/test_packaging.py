"""Test packaging configuration to catch missing files before CI.

This test validates that:
1. All required config files are in the correct location
2. MANIFEST.in correctly includes necessary files
3. Scripts reference correct paths
4. No hardcoded references to old paths exist

Run this before committing packaging changes to avoid CI failures.
"""

import sys
from pathlib import Path

import pytest


def test_config_files_location():
    """Verify config files are in src/glassalpha/configs/ not src/glassalpha/data/configs/."""
    # Get the repo root
    repo_root = Path(__file__).parent.parent

    # Verify correct location exists
    correct_configs_path = repo_root / "src" / "glassalpha" / "configs"
    assert correct_configs_path.exists(), f"Config directory not found: {correct_configs_path}"

    # Verify german_credit config exists in correct location
    german_credit_config = correct_configs_path / "german_credit.yaml"
    assert german_credit_config.exists(), f"german_credit.yaml not found in {correct_configs_path}"

    # Verify old location doesn't exist
    old_configs_path = repo_root / "src" / "glassalpha" / "data" / "configs"
    assert not old_configs_path.exists(), (
        f"Old config path still exists: {old_configs_path}. Configs should be in {correct_configs_path}"
    )


def test_manifest_includes_configs():
    """Verify MANIFEST.in includes config files from correct location."""
    repo_root = Path(__file__).parent.parent
    manifest_path = repo_root / "MANIFEST.in"

    assert manifest_path.exists(), "MANIFEST.in not found"

    manifest_content = manifest_path.read_text()

    # Verify correct path is referenced
    assert "src/glassalpha/configs" in manifest_content, "MANIFEST.in should include src/glassalpha/configs"

    # Verify old path is NOT referenced
    assert "glassalpha/data/configs" not in manifest_content, (
        "MANIFEST.in should NOT reference old path glassalpha/data/configs"
    )


def test_scripts_reference_correct_paths():
    """Verify scripts reference correct config paths."""
    repo_root = Path(__file__).parent.parent
    scripts_to_check = [
        repo_root / "scripts" / "create_test_datasets.py",
    ]

    for script_path in scripts_to_check:
        if not script_path.exists():
            continue  # Skip if script doesn't exist

        script_content = script_path.read_text(encoding="utf-8")

        # Check for old path references
        if "glassalpha" in script_content and "data/configs" in script_content:
            assert "glassalpha/data/configs" not in script_content, (
                f"{script_path.name} contains reference to old path 'glassalpha/data/configs'. "
                f"Should be 'glassalpha/configs'"
            )


def test_ci_workflow_references_correct_paths():
    """Verify CI workflows reference correct config paths."""
    repo_root = Path(__file__).parent.parent
    ci_workflow = repo_root / ".github" / "workflows" / "ci.yml"

    if not ci_workflow.exists():
        pytest.skip("CI workflow not found")

    ci_content = ci_workflow.read_text(encoding="utf-8")

    # Check for correct path in CI verification
    if "german_credit_simple.yaml" in ci_content:
        # If we're checking for german_credit_simple.yaml, ensure it's the right path
        assert "glassalpha/configs/german_credit_simple.yaml" in ci_content, (
            "CI workflow should reference glassalpha/configs/german_credit_simple.yaml"
        )

        # Ensure old path is not referenced
        assert "glassalpha/data/configs/german_credit_simple.yaml" not in ci_content, (
            "CI workflow should NOT reference old path glassalpha/data/configs/"
        )


def test_no_legacy_path_references():
    """Scan common files for references to old config path structure."""
    repo_root = Path(__file__).parent.parent
    files_to_check = [
        repo_root / "README.md",
        repo_root / "site" / "docs" / "getting-started" / "quickstart.md",
    ]

    for file_path in files_to_check:
        if not file_path.exists():
            continue

        content = file_path.read_text(encoding="utf-8")

        # Allow CHANGELOG.md to reference old paths (historical record)
        if file_path.name == "CHANGELOG.md":
            continue

        # Check for problematic references
        if "glassalpha/data/configs" in content:
            pytest.fail(
                f"{file_path.relative_to(repo_root)} contains reference to old path "
                f"'glassalpha/data/configs'. Should be 'glassalpha/configs'"
            )


@pytest.mark.skipif(sys.platform == "win32", reason="Package installation check unreliable on Windows")
def test_installed_package_includes_configs():
    """Verify that installed package includes config files (requires package to be installed)."""
    try:
        import glassalpha
    except ImportError:
        pytest.skip("glassalpha not installed")

    # Get the installation path
    import glassalpha

    glassalpha_path = Path(glassalpha.__file__).parent

    # Verify configs directory exists
    configs_path = glassalpha_path / "configs"
    assert configs_path.exists(), (
        f"Configs directory not found in installed package: {configs_path}. Check MANIFEST.in and pyproject.toml"
    )

    # Verify key config files exist
    required_configs = [
        "german_credit.yaml",
        "german_credit_simple.yaml",
        "minimal.yaml",
    ]

    for config_name in required_configs:
        config_path = configs_path / config_name
        assert config_path.exists(), (
            f"Required config '{config_name}' not found in installed package at {config_path}. "
            f"Check MANIFEST.in includes src/glassalpha/configs/*.yaml"
        )


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
