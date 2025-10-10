"""Tests for smart context-aware defaults."""

from pathlib import Path

import pytest

from glassalpha.cli.defaults import (
    get_smart_defaults,
    infer_config_path,
    infer_output_path,
    should_enable_repro_mode,
    should_enable_strict_mode,
)


class TestInferConfigPath:
    """Tests for config path inference."""

    def test_finds_glassalpha_yaml_first(self, tmp_path, monkeypatch):
        """Should prioritize glassalpha.yaml over other names."""
        monkeypatch.chdir(tmp_path)

        # Create multiple config files
        (tmp_path / "glassalpha.yaml").write_text("# config")
        (tmp_path / "audit.yaml").write_text("# config")
        (tmp_path / "config.yaml").write_text("# config")

        result = infer_config_path()
        assert result == Path("glassalpha.yaml")

    def test_finds_audit_yaml_second(self, tmp_path, monkeypatch):
        """Should find audit.yaml if glassalpha.yaml doesn't exist."""
        monkeypatch.chdir(tmp_path)

        (tmp_path / "audit.yaml").write_text("# config")
        (tmp_path / "config.yaml").write_text("# config")

        result = infer_config_path()
        assert result == Path("audit.yaml")

    def test_finds_config_yaml_third(self, tmp_path, monkeypatch):
        """Should find config.yaml as fallback."""
        monkeypatch.chdir(tmp_path)

        (tmp_path / "config.yaml").write_text("# config")

        result = infer_config_path()
        assert result == Path("config.yaml")

    def test_finds_hidden_glassalpha_yaml_last(self, tmp_path, monkeypatch):
        """Should find .glassalpha.yaml as last resort."""
        monkeypatch.chdir(tmp_path)

        (tmp_path / ".glassalpha.yaml").write_text("# config")

        result = infer_config_path()
        assert result == Path(".glassalpha.yaml")

    def test_returns_none_if_no_config_found(self, tmp_path, monkeypatch):
        """Should return None if no config files exist."""
        monkeypatch.chdir(tmp_path)

        result = infer_config_path()
        assert result is None


class TestInferOutputPath:
    """Tests for output path inference."""

    def test_replaces_yaml_with_html(self):
        """Should replace .yaml extension with .html."""
        config = Path("audit.yaml")
        result = infer_output_path(config)
        assert result == Path("audit.html")

    def test_replaces_yml_with_html(self):
        """Should replace .yml extension with .html."""
        config = Path("config.yml")
        result = infer_output_path(config)
        assert result == Path("config.html")

    def test_preserves_directory_structure(self):
        """Should preserve parent directories."""
        config = Path("configs/production.yaml")
        result = infer_output_path(config)
        assert result == Path("configs/production.html")

    def test_handles_no_extension(self):
        """Should add .html to files without extension."""
        config = Path("my_audit")
        result = infer_output_path(config)
        assert result == Path("my_audit.html")


class TestShouldEnableStrictMode:
    """Tests for strict mode auto-detection."""

    def test_enabled_for_production_filename(self):
        """Should enable strict mode for production configs."""
        config = Path("production.yaml")
        assert should_enable_strict_mode(config) is True

    def test_enabled_for_prod_filename(self):
        """Should enable strict mode for prod configs."""
        config = Path("prod.yaml")
        assert should_enable_strict_mode(config) is True

    def test_enabled_for_strict_filename(self):
        """Should enable strict mode for strict configs."""
        config = Path("strict_audit.yaml")
        assert should_enable_strict_mode(config) is True

    def test_case_insensitive(self):
        """Should detect patterns case-insensitively."""
        config = Path("PRODUCTION.yaml")
        assert should_enable_strict_mode(config) is True

    def test_disabled_for_normal_filename(self):
        """Should not enable strict mode for normal configs."""
        config = Path("audit.yaml")
        assert should_enable_strict_mode(config) is False

    def test_enabled_by_environment_variable(self, monkeypatch):
        """Should enable strict mode if GLASSALPHA_STRICT=1."""
        monkeypatch.setenv("GLASSALPHA_STRICT", "1")
        config = Path("audit.yaml")
        assert should_enable_strict_mode(config) is True

    def test_environment_variable_true(self, monkeypatch):
        """Should enable strict mode if GLASSALPHA_STRICT=true."""
        monkeypatch.setenv("GLASSALPHA_STRICT", "true")
        config = Path("audit.yaml")
        assert should_enable_strict_mode(config) is True


class TestShouldEnableReproMode:
    """Tests for repro mode auto-detection."""

    def test_enabled_for_test_filename(self):
        """Should enable repro mode for test configs."""
        config = Path("test.yaml")
        assert should_enable_repro_mode(config) is True

    def test_enabled_for_ci_filename(self):
        """Should enable repro mode for CI configs."""
        config = Path("ci_audit.yaml")
        assert should_enable_repro_mode(config) is True

    def test_enabled_in_ci_environment(self, monkeypatch):
        """Should enable repro mode if CI=true."""
        monkeypatch.setenv("CI", "true")
        assert should_enable_repro_mode() is True

    def test_enabled_by_environment_variable(self, monkeypatch):
        """Should enable repro mode if GLASSALPHA_REPRO=1."""
        monkeypatch.setenv("GLASSALPHA_REPRO", "1")
        assert should_enable_repro_mode() is True

    def test_disabled_for_normal_filename(self):
        """Should not enable repro mode for normal configs."""
        config = Path("audit.yaml")
        assert should_enable_repro_mode(config) is False


class TestGetSmartDefaults:
    """Tests for complete smart defaults system."""

    def test_uses_explicit_values_when_provided(self, tmp_path):
        """Should not override explicitly provided values."""
        config = tmp_path / "my_config.yaml"
        config.write_text("# config")
        output = tmp_path / "my_output.html"

        defaults = get_smart_defaults(
            config=config,
            output=output,
            strict=True,
            repro=False,
        )

        assert defaults["config"] == config
        assert defaults["output"] == output
        assert defaults["strict"] is True
        assert defaults["repro"] is False

    def test_infers_all_defaults(self, tmp_path, monkeypatch):
        """Should infer all defaults when none provided."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "audit.yaml").write_text("# config")

        defaults = get_smart_defaults()

        assert defaults["config"] == Path("audit.yaml")
        assert defaults["output"] == Path("audit.html")
        assert defaults["strict"] is False
        assert defaults["repro"] is False

    def test_infers_output_from_config(self, tmp_path):
        """Should infer output path from config path."""
        config = tmp_path / "my_audit.yaml"
        config.write_text("# config")

        defaults = get_smart_defaults(config=config)

        assert defaults["output"] == tmp_path / "my_audit.html"

    def test_enables_strict_for_production_config(self, tmp_path):
        """Should auto-enable strict mode for production configs."""
        config = tmp_path / "production.yaml"
        config.write_text("# config")

        defaults = get_smart_defaults(config=config)

        assert defaults["strict"] is True

    def test_enables_repro_for_test_config(self, tmp_path):
        """Should auto-enable repro mode for test configs."""
        config = tmp_path / "test.yaml"
        config.write_text("# config")

        defaults = get_smart_defaults(config=config)

        assert defaults["repro"] is True

    def test_raises_if_no_config_found(self, tmp_path, monkeypatch):
        """Should raise ValueError if no config can be found."""
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="No configuration file found"):
            get_smart_defaults()

    def test_mixed_explicit_and_inferred(self, tmp_path):
        """Should handle mix of explicit and inferred values."""
        config = tmp_path / "prod.yaml"
        config.write_text("# config")

        # Provide config explicitly, let others be inferred
        defaults = get_smart_defaults(config=config, repro=False)

        assert defaults["config"] == config
        assert defaults["output"] == tmp_path / "prod.html"
        assert defaults["strict"] is True  # Inferred from "prod" in filename
        assert defaults["repro"] is False  # Explicitly set
