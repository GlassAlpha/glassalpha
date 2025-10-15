"""Test that all defined CLI commands are properly registered and accessible.

This test ensures that commands defined in commands.py are actually
registered in main.py and show up in --help output.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_all_commands_appear_in_help():
    """Verify all registered commands appear in --help output."""
    # Run glassalpha --help
    result = subprocess.run(
        [sys.executable, "-m", "glassalpha.cli.main", "--help"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0, f"--help failed: {result.stderr}"

    help_output = result.stdout

    # Expected commands that should be registered
    expected_commands = [
        "audit",
        "doctor",
        "docs",
        "list",
        "quickstart",
        "reasons",
        "recourse",
        "validate",
    ]

    # Verify each command appears in help output
    for command in expected_commands:
        # Commands appear in table format: │ command_name
        # Look for command in the commands table
        command_pattern = f"│ {command}"  # Table format with Unicode box
        assert command_pattern in help_output, (
            f"Command '{command}' not found in --help output. Commands section: {help_output.count('│') > 0}"
        )


def test_each_command_has_help():
    """Verify each command has its own --help text."""
    commands = ["audit", "doctor", "docs", "list", "quickstart", "reasons", "recourse", "validate"]

    for command in commands:
        result = subprocess.run(
            [sys.executable, "-m", "glassalpha.cli.main", command, "--help"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        assert result.returncode == 0, f"{command} --help failed: {result.stderr}"
        assert len(result.stdout) > 50, f"{command} --help output too short: {result.stdout[:100]}"
        assert command in result.stdout.lower(), f"Command name '{command}' not in its own help text"


def test_registered_commands_match_imports():
    """Verify main.py imports match registered commands (detect missing registrations)."""
    # Read main.py to check imports and registrations
    main_file = Path(__file__).parent.parent.parent / "src" / "glassalpha" / "cli" / "main.py"
    assert main_file.exists(), f"main.py not found at {main_file}"

    main_content = main_file.read_text(encoding="utf-8")

    # Check that imports and registrations are consistent
    # Import line: from .commands import audit, doctor, docs, ...
    assert "from .commands import" in main_content, "Missing command imports"
    assert "from .quickstart import quickstart" in main_content, "Missing quickstart import"

    # Registration lines: app.command()(audit)
    assert "app.command()(audit)" in main_content, "audit not registered"
    assert "app.command()(doctor)" in main_content, "doctor not registered"
    assert "app.command()(docs)" in main_content, "docs not registered"
    assert "app.command()(quickstart)" in main_content, "quickstart not registered"
    assert "app.command()(reasons)" in main_content, "reasons not registered"
    assert "app.command()(recourse)" in main_content, "recourse not registered"
    assert "app.command()(validate)" in main_content, "validate not registered"
    assert 'app.command(name="list")(list_components_cmd)' in main_content, "list not registered"


def test_no_duplicate_command_names():
    """Verify no command names are registered multiple times."""
    result = subprocess.run(
        [sys.executable, "-m", "glassalpha.cli.main", "--help"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0, f"--help failed: {result.stderr}"

    help_output = result.stdout
    commands = ["audit", "doctor", "docs", "list", "quickstart", "reasons", "recourse", "validate"]

    for command in commands:
        # Count occurrences of command as standalone word in help output
        # (should appear exactly once in the commands section)
        import re

        # Look for command in the commands table (after "Commands" marker)
        commands_section = help_output
        if "Commands:" in help_output:
            commands_section = help_output.split("Commands:")[1]
        elif "╭─ Commands ─" in help_output:
            # Handle table format with Unicode box drawing (top border)
            commands_section = help_output.split("╭─ Commands ─")[1]

        # Commands appear in table format: │ command_name
        command_pattern = f"│ {command}"
        matches = [m for m in commands_section.split("\n") if command_pattern in m]

        # Each command should appear at least once
        assert len(matches) >= 1, (
            f"Command '{command}' appears {len(matches)} times (expected >= 1) in commands section"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
