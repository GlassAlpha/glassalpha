"""Regression tests to prevent obsolete command references.

Ensures code doesn't reference commands that have been renamed or removed.
"""

import subprocess
from pathlib import Path

import pytest


class TestObsoleteCommandReferences:
    """Prevent references to obsolete or renamed commands."""

    def test_no_init_command_references(self):
        """Ensure no references to obsolete 'glassalpha init' command.

        The command was renamed to 'glassalpha quickstart'.
        This test prevents the reference from creeping back in.
        """
        # Search for "glassalpha init" in source and docs
        result = subprocess.run(
            ["git", "grep", "-i", "-n", "glassalpha init", "src/", "docs/", "site/"],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,  # Repo root
        )

        if result.returncode == 0:
            # Found references - fail with helpful message
            matches = result.stdout.strip()
            pytest.fail(
                f"Found references to obsolete 'glassalpha init' command.\n"
                f"This command was renamed to 'glassalpha quickstart'.\n\n"
                f"Found in:\n{matches}\n\n"
                f"Replace all instances with 'glassalpha quickstart'."
            )

        # returncode != 0 means no matches found (success)
        assert result.returncode != 0, "Should not find 'glassalpha init' references"

    def test_no_deprecated_cli_patterns(self):
        """Check for other deprecated CLI patterns."""
        # Add more checks here as needed
        deprecated_patterns = [
            # Add future deprecated commands/patterns here
            # Example: ("old_command", "new_command", ["src/", "docs/"])
        ]

        for pattern, replacement, search_paths in deprecated_patterns:
            result = subprocess.run(
                ["git", "grep", "-i", "-n", pattern] + search_paths,
                check=False,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )

            if result.returncode == 0:
                matches = result.stdout.strip()
                pytest.fail(
                    f"Found references to deprecated pattern: '{pattern}'\n"
                    f"Should use: '{replacement}'\n\n"
                    f"Found in:\n{matches}"
                )
