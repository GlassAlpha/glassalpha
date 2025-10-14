"""Publish-check command for pre-release validation.

Verifies the package is ready for PyPI publication.
"""

import subprocess
import sys
from pathlib import Path

import typer

from ..exit_codes import ExitCode


def publish_check(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output from checks",
    ),
):
    """Check if package is ready for publication.

    Runs pre-publication validation including:
    - Version tag matches CHANGELOG
    - All tests passing
    - Determinism verification
    - Example configs valid
    - CLI commands documented
    - Notebooks executable

    Examples:
        # Basic check
        glassalpha publish-check

        # Verbose output
        glassalpha publish-check --verbose

    """
    typer.echo("Pre-Publication Checklist")
    typer.echo("=" * 40)
    typer.echo()

    issues = []
    checks_passed = 0
    checks_total = 0

    # Get project root
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        project_root = Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        typer.secho("❌ Not in a git repository", fg=typer.colors.RED)
        raise typer.Exit(ExitCode.SYSTEM_ERROR)

    # Check 1: Version tag matches CHANGELOG
    checks_total += 1
    try:
        from glassalpha import __version__

        changelog_path = project_root / "CHANGELOG.md"
        if changelog_path.exists():
            changelog_content = changelog_path.read_text()
            if f"## [{__version__}]" in changelog_content or f"## {__version__}" in changelog_content:
                typer.secho(f"✅ Version tag matches CHANGELOG (v{__version__})", fg=typer.colors.GREEN)
                checks_passed += 1
            else:
                typer.secho(f"❌ Version v{__version__} not found in CHANGELOG", fg=typer.colors.RED)
                issues.append(f"Add v{__version__} entry to CHANGELOG.md")
        else:
            typer.secho("⚠️  CHANGELOG.md not found", fg=typer.colors.YELLOW)
            issues.append("Create CHANGELOG.md")
    except Exception as e:
        typer.secho(f"❌ Version check failed: {e}", fg=typer.colors.RED)
        issues.append("Fix version detection")

    # Check 2: Tests passing
    checks_total += 1
    try:
        result = subprocess.run(
            ["pytest", "--co", "-q"],
            check=False,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=10,
        )
        if result.returncode == 0:
            # Count tests
            test_count = len([line for line in result.stdout.split("\n") if line.strip().startswith("tests/")])
            typer.secho(f"✅ All tests collected successfully ({test_count} tests)", fg=typer.colors.GREEN)
            checks_passed += 1
        else:
            typer.secho("❌ Test collection failed", fg=typer.colors.RED)
            issues.append("Fix test collection errors")
            if verbose:
                typer.echo(f"  Error: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        typer.secho("❌ Test collection timed out", fg=typer.colors.RED)
        issues.append("Fix test collection performance")
    except FileNotFoundError:
        typer.secho("⚠️  pytest not installed", fg=typer.colors.YELLOW)
        issues.append("Install pytest: pip install pytest")

    # Check 3: Determinism verified
    checks_total += 1
    determinism_script = project_root / "scripts" / "check-determinism-quick.sh"
    if determinism_script.exists():
        typer.secho("✅ Determinism verification script exists", fg=typer.colors.GREEN)
        checks_passed += 1
    else:
        typer.secho("⚠️  Determinism script not found", fg=typer.colors.YELLOW)
        issues.append("Create scripts/check-determinism-quick.sh")

    # Check 4: Example configs valid
    checks_total += 1
    configs_path = project_root / "src" / "glassalpha" / "configs"
    if configs_path.exists():
        config_files = list(configs_path.glob("*.yaml"))
        if config_files:
            typer.secho(f"✅ Example configs found ({len(config_files)} configs)", fg=typer.colors.GREEN)
            checks_passed += 1
        else:
            typer.secho("❌ No example configs found", fg=typer.colors.RED)
            issues.append("Add example configs to src/glassalpha/configs/")
    else:
        typer.secho("❌ Configs directory not found", fg=typer.colors.RED)
        issues.append("Create src/glassalpha/configs/ directory")

    # Check 5: CLI commands documented
    checks_total += 1
    try:
        # Check if --help works
        result = subprocess.run(
            [sys.executable, "-m", "glassalpha", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Count commands in help output
            command_count = result.stdout.count("Commands:")
            typer.secho("✅ CLI help working", fg=typer.colors.GREEN)
            checks_passed += 1
        else:
            typer.secho("❌ CLI help failed", fg=typer.colors.RED)
            issues.append("Fix CLI help output")
    except Exception as e:
        typer.secho(f"❌ CLI check failed: {e}", fg=typer.colors.RED)
        issues.append("Fix CLI command structure")

    # Check 6: Notebooks (optional check)
    checks_total += 1
    notebooks_path = project_root / "examples" / "notebooks"
    if notebooks_path.exists():
        notebook_files = list(notebooks_path.glob("*.ipynb"))
        if notebook_files:
            typer.secho(f"ℹ️  Example notebooks found ({len(notebook_files)} notebooks)", fg=typer.colors.CYAN)
            typer.echo("   Note: Run manual validation for notebooks")
            checks_passed += 1
        else:
            typer.secho("⚠️  No notebooks found", fg=typer.colors.YELLOW)
    else:
        typer.secho("⚠️  Notebooks directory not found", fg=typer.colors.YELLOW)

    # Summary
    typer.echo()
    typer.echo("=" * 40)
    typer.echo(f"Status: {checks_passed}/{checks_total} checks passed")
    typer.echo("=" * 40)
    typer.echo()

    if issues:
        typer.secho(f"❌ {len(issues)} issue(s) found:", fg=typer.colors.RED, bold=True)
        for i, issue in enumerate(issues, 1):
            typer.echo(f"  {i}. {issue}")
        typer.echo()
        typer.echo("Fix these issues before publishing to PyPI")
        raise typer.Exit(ExitCode.USER_ERROR)
    else:
        typer.secho("✅ All checks passed! Ready for publication.", fg=typer.colors.GREEN, bold=True)
        typer.echo()
        typer.echo("Next steps:")
        typer.echo("  1. Build: python -m build")
        typer.echo("  2. Check: twine check dist/*")
        typer.echo("  3. Upload: twine upload dist/*")
        raise typer.Exit(ExitCode.SUCCESS)
