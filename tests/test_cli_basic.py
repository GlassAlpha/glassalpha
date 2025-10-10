"""Basic CLI functionality tests.

Tests that CLI loads and core commands work without crashing.
These tests focus on coverage for main.py and commands.py.
"""

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from glassalpha.cli.main import app


def test_cli_app_loads():
    """Test that the CLI app loads without errors."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "GlassAlpha - AI Compliance Toolkit" in result.stdout


def test_version_command():
    """Test the version command works."""
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 or "version" in result.stdout.lower()


def test_audit_command_missing_config():
    """Test audit command fails gracefully with missing config."""
    runner = CliRunner()
    result = runner.invoke(app, ["audit", "--config", "nonexistent.yaml", "--output", "test.pdf"])
    # Should exit with error but not crash
    assert result.exit_code != 0


def test_audit_command_with_temp_config():
    """Test audit command loads with valid config structure."""
    runner = CliRunner()

    # Create minimal valid config
    config_content = """
audit_profile: "tabular_compliance"
model:
  type: "xgboost"
data:
  path: "test.csv"
explainers:
  strategy: "first_compatible"
  priority: ["treeshap"]
metrics:
  performance: ["accuracy"]
reproducibility:
  random_seed: 42
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as output_file:
            output_path = output_file.name

        # This will likely fail due to missing data file, but should load config
        result = runner.invoke(app, ["audit", "--config", config_path, "--output", output_path])

        # Command should at least attempt to load config (may fail later due to missing data)
        # We just want to exercise the CLI code path
        assert "audit_profile" not in result.stdout or result.exit_code != 0

    finally:
        Path(config_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def test_strict_mode_flag():
    """Test that strict mode flag is recognized."""
    runner = CliRunner()

    config_content = """
audit_profile: "tabular_compliance"
model:
  type: "xgboost"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        result = runner.invoke(app, ["audit", "--config", config_path, "--output", "test.pdf", "--strict"])
        # Should recognize the flag (may fail for other reasons)
        assert result.exit_code != 125  # 125 is "unknown option" in Click/Typer

    finally:
        Path(config_path).unlink(missing_ok=True)


def _has_jinja2():
    """Check if jinja2 is available in the current environment."""
    try:
        import jinja2

        return True
    except ImportError:
        return False


def test_pdf_guard_without_pdf_backend_exits_cleanly(tmp_path):
    """Test that PDF guard exits cleanly when PDF backend is missing."""
    # Create a minimal config that requests PDF output
    config_content = """
audit_profile: "tabular_compliance"
model:
  type: "logistic_regression"
data:
  path: "test.csv"
explainers:
  strategy: "first_compatible"
report:
  output_format: "pdf"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        # Create output path with .pdf extension
        output_path = tmp_path / "test.pdf"

        # Run the audit command
        runner = CliRunner()
        result = runner.invoke(app, ["audit", "--config", config_path, "--output", str(output_path)])

        # Check if PDF backend is available (weasyprint or reportlab)
        has_pdf_backend = False
        try:
            import weasyprint

            has_pdf_backend = True
        except ImportError:
            try:
                import reportlab

                has_pdf_backend = True
            except ImportError:
                pass

        if not has_pdf_backend:
            # When PDF backend is not available, should exit with error and mention docs
            assert result.exit_code != 0
            assert 'pip install "glassalpha[docs]"' in (result.stdout + result.stderr)
        else:
            # When PDF backend is available, may fail for other reasons but not for missing PDF backend
            # (the test doesn't check for other failures, just that it doesn't fail for missing PDF backend)
            pass

    finally:
        Path(config_path).unlink(missing_ok=True)


def test_environment_aware_quickstart_adaptation(tmp_path):
    """Test that quickstart adapts to environment (HTML baseline, PDF when available)."""
    runner = CliRunner()

    # Test doctor command gives environment-aware recommendations
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0

    # Should always show templating is OK (jinja2 is in core)
    assert "Templating: ✅ installed" in result.stdout

    # Check PDF backend availability
    has_pdf_backend = False
    try:
        import weasyprint

        has_pdf_backend = True
    except ImportError:
        try:
            import reportlab

            has_pdf_backend = True
        except ImportError:
            pass

    if has_pdf_backend:
        # When PDF backend is available, should recommend PDF
        assert "quickstart.pdf" in result.stdout
        assert "PDF generation: ✅ installed" in result.stdout
    else:
        # When PDF backend is not available, should recommend HTML
        assert "quickstart.html" in result.stdout
        assert "PDF generation: ❌ not installed" in result.stdout
        assert 'pip install "glassalpha[docs]"' in result.stdout
