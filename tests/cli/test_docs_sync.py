"""Tests for CLI documentation synchronization.

This module ensures that the generated CLI documentation stays in sync
with the actual CLI implementation, preventing the CI issue where docs
become stale during development.
"""


def test_cli_documentation_is_current():
    """Test that CLI documentation is current and in sync with CLI implementation.

    This test runs the CLI documentation generation script in check mode.
    If the documentation is out of date, the script will exit with code 1,
    causing this test to fail.

    This ensures that documentation drift is caught during local development
    and testing, not just in CI.
    """
    import subprocess
    import sys
    from pathlib import Path

    # Path to the documentation generation script
    script_path = Path(__file__).parent.parent.parent / "scripts" / "generate_cli_docs.py"

    # Run the script in check mode
    result = subprocess.run(
        [sys.executable, str(script_path), "--check"],
        check=False,
        capture_output=True,
        text=True,
        cwd=script_path.parent,  # Run from packages directory
        encoding="utf-8",
    )

    # If docs are out of date, the script exits with code 1
    assert result.returncode == 0, (
        "CLI documentation is out of date.\n"
        f"Script output: {result.stdout}\n"
        f"Script errors: {result.stderr}\n"
        "Run: python packages/scripts/generate_cli_docs.py --output site/docs/reference/cli.md"
    )
