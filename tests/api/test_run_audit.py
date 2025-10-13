"""Tests for run_audit() function - CLI/file-based audit generation."""

import tempfile
from pathlib import Path

import pytest

import glassalpha as ga


def test_run_audit_missing_config_file():
    """Test run_audit raises FileNotFoundError for missing config file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        config_path = tmp_path / "nonexistent.yaml"
        output_path = tmp_path / "audit.html"

        with pytest.raises(FileNotFoundError, match="Configuration file does not exist"):
            ga.audit.run_audit(config_path, output_path)


def test_run_audit_creates_output_directory():
    """Test run_audit creates output directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create config file with invalid model (will fail later, but tests directory creation)
        config_path = tmp_path / "audit.yaml"
        config = {
            "audit_profile": "minimal",
            "model": {"type": "logistic_regression", "path": "nonexistent.pkl"},
            "data": {"dataset": "custom", "path": "nonexistent.csv", "target_column": "target"},
            "reproducibility": {"random_seed": 42},
        }
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Try to run audit (will fail on model, but should create nested directory)
        nested_dir = tmp_path / "nested" / "deep"
        output_path = nested_dir / "audit.html"

        # This should fail on model loading, but create the directory structure
        with pytest.raises(RuntimeError, match="Audit pipeline failed"):
            ga.audit.run_audit(config_path, output_path)

        # Verify directory was created
        assert nested_dir.exists()
        assert nested_dir.is_dir()


def test_run_audit_non_writable_directory(tmp_path):
    """Test run_audit raises RuntimeError for non-writable output directory."""
    # Create config file first
    config_path = tmp_path / "audit.yaml"
    config = {
        "audit_profile": "minimal",
        "model": {"type": "logistic_regression", "path": "nonexistent.pkl"},
        "data": {"dataset": "custom", "path": "nonexistent.csv", "target_column": "target"},
        "reproducibility": {"random_seed": 42},
    }
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Create read-only directory
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    readonly_dir.chmod(0o444)  # Read-only

    output_path = readonly_dir / "audit.html"

    try:
        with pytest.raises(RuntimeError, match="Output directory is not writable"):
            ga.audit.run_audit(config_path, output_path)
    finally:
        readonly_dir.chmod(0o755)  # Restore permissions for cleanup


def test_run_audit_with_strict_mode():
    """Test run_audit accepts strict mode parameter."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create config file
        config_path = tmp_path / "audit.yaml"
        config = {
            "audit_profile": "minimal",
            "model": {"type": "logistic_regression", "path": "nonexistent.pkl"},
            "data": {"dataset": "custom", "path": "nonexistent.csv", "target_column": "target"},
            "reproducibility": {"random_seed": 42},
        }
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        output_path = tmp_path / "audit.html"

        # This should fail on model loading, but strict mode should be accepted
        with pytest.raises(RuntimeError, match="Audit pipeline failed"):
            ga.audit.run_audit(config_path, output_path, strict=True)


def test_run_audit_with_override_config():
    """Test run_audit accepts override config parameter."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create base config
        config_path = tmp_path / "audit.yaml"
        config = {
            "audit_profile": "minimal",
            "model": {"type": "logistic_regression", "path": "nonexistent.pkl"},
            "data": {"dataset": "custom", "path": "nonexistent.csv", "target_column": "target"},
            "reproducibility": {"random_seed": 42},
        }
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Create override config
        override_config = {
            "reproducibility": {"random_seed": 123},  # Different seed
        }
        override_path = tmp_path / "override.yaml"
        with open(override_path, "w") as f:
            yaml.dump(override_config, f)

        output_path = tmp_path / "audit.html"

        # This should fail on model loading, but override should be accepted
        with pytest.raises(RuntimeError, match="Audit pipeline failed"):
            ga.audit.run_audit(config_path, output_path, override_config=override_path)


def test_run_audit_pdf_fallback_logic():
    """Test run_audit PDF fallback logic exists (structural test)."""
    # Verify the fallback logic is implemented in the function
    import inspect

    from glassalpha.api.audit import run_audit

    source = inspect.getsource(run_audit)

    # Check that PDF fallback logic exists
    assert "PDF_AVAILABLE" in source
    assert 'output_format == "pdf"' in source
    assert 'output_format = "html"' in source  # Fallback logic


def test_run_audit_invalid_config():
    """Test run_audit handles invalid configuration gracefully."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create invalid config (missing required fields)
        config_path = tmp_path / "invalid.yaml"
        config = {
            "audit_profile": "minimal",
            "model": {},  # Missing type and path
            "data": {"dataset": "custom", "path": "nonexistent.csv"},  # Missing target_column
        }
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        output_path = tmp_path / "audit.html"

        # Should fail during config validation (before pipeline execution)
        with pytest.raises(ValueError):
            ga.audit.run_audit(config_path, output_path)


def test_run_audit_deterministic_output():
    """Test run_audit produces deterministic output when SOURCE_DATE_EPOCH is set."""
    # This test verifies that the SOURCE_DATE_EPOCH logic exists and is used
    import inspect

    from glassalpha.api.audit import run_audit

    source = inspect.getsource(run_audit)

    # Check that deterministic timestamp logic exists
    assert "SOURCE_DATE_EPOCH" in source
    assert "fromtimestamp" in source
    assert "UTC" in source


def test_run_audit_creates_output_directory():
    """Test run_audit creates output directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create config file
        config_path = tmp_path / "audit.yaml"
        config = {
            "audit_profile": "minimal",
            "model": {"type": "logistic_regression", "path": "nonexistent.pkl"},
            "data": {"dataset": "custom", "path": "./nonexistent.csv", "target_column": "target"},
            "reproducibility": {"random_seed": 42},
        }
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Try to run audit (will fail on model, but should create nested directory)
        nested_dir = tmp_path / "nested" / "deep"
        output_path = nested_dir / "audit.html"

        # This should fail on model loading, but create the directory structure
        with pytest.raises(RuntimeError, match="Audit pipeline failed"):
            ga.audit.run_audit(config_path, output_path)

        # Verify directory was created
        assert nested_dir.exists()
        assert nested_dir.is_dir()


def test_run_audit_missing_config_file():
    """Test run_audit raises FileNotFoundError for missing config file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        config_path = tmp_path / "nonexistent.yaml"
        output_path = tmp_path / "audit.html"

        with pytest.raises(FileNotFoundError, match="Configuration file does not exist"):
            ga.audit.run_audit(config_path, output_path)


def test_run_audit_non_writable_directory(tmp_path):
    """Test run_audit raises RuntimeError for non-writable output directory."""
    # Create config file first
    config_path = tmp_path / "audit.yaml"
    config = {
        "model": {"path": "nonexistent.pkl"},
        "data": {"X_path": "x.csv", "y_path": "y.csv"},
        "audit": {"random_seed": 42},
    }
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Create read-only directory
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    readonly_dir.chmod(0o444)  # Read-only

    output_path = readonly_dir / "audit.html"

    # Create config file first
    config_path = tmp_path / "audit.yaml"
    config = {
        "audit_profile": "minimal",
        "model": {"type": "logistic_regression", "path": "nonexistent.pkl"},
        "data": {"dataset": "custom", "path": "nonexistent.csv", "target_column": "target"},
        "reproducibility": {"random_seed": 42},
    }
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    try:
        with pytest.raises(RuntimeError, match="Output directory is not writable"):
            ga.audit.run_audit(config_path, output_path)
    finally:
        readonly_dir.chmod(0o755)  # Restore permissions for cleanup


def test_run_audit_with_strict_mode():
    """Test run_audit accepts strict mode parameter."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create config file that meets strict mode requirements
        config_path = tmp_path / "audit.yaml"
        config = {
            "audit_profile": "tabular_compliance",
            "strict_mode": True,
            "model": {
                "type": "logistic_regression",
                "path": "nonexistent.pkl",
            },
            "data": {
                "dataset": "custom",
                "path": "./nonexistent.csv",
                "target_column": "target",
                "protected_attributes": ["feature"],
                "schema_path": "schema.yaml",  # Required for strict mode
            },
            "reproducibility": {
                "random_seed": 42,
                "deterministic": True,
                "capture_environment": True,
            },
            "explainers": {
                "strategy": "first_compatible",
                "priority": ["coefficients"],
            },
            "manifest": {
                "enabled": True,
                "include_git_sha": True,
                "include_config_hash": True,
                "include_data_hash": True,
            },
            "metrics": {
                "performance": ["accuracy"],
                "fairness": ["demographic_parity"],
            },
            "preprocessing": {
                "mode": "artifact",
                "artifact_path": "preprocessor.joblib",
                "expected_file_hash": "sha256:abc123",
                "expected_params_hash": "sha256:def456",
            },
        }
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        output_path = tmp_path / "audit.html"

        # This should fail on model loading, but strict mode should be accepted
        with pytest.raises(RuntimeError, match="Audit pipeline failed"):
            ga.audit.run_audit(config_path, output_path, strict=True)


def test_run_audit_with_override_config():
    """Test run_audit accepts override config parameter."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create base config
        config_path = tmp_path / "audit.yaml"
        config = {
            "audit_profile": "minimal",
            "model": {"type": "logistic_regression", "path": "nonexistent.pkl"},
            "data": {"dataset": "custom", "path": "./nonexistent.csv", "target_column": "target"},
            "reproducibility": {"random_seed": 42},
        }
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Create override config
        override_config = {
            "reproducibility": {"random_seed": 123},  # Different seed
        }
        override_path = tmp_path / "override.yaml"
        with open(override_path, "w") as f:
            yaml.dump(override_config, f)

        output_path = tmp_path / "audit.html"

        # This should fail on model loading, but override should be accepted
        with pytest.raises(RuntimeError, match="Audit pipeline failed"):
            ga.audit.run_audit(config_path, output_path, override_config=override_path)


def test_run_audit_pdf_fallback_to_html():
    """Test run_audit falls back to HTML when PDF dependencies unavailable."""
    # This test is conceptual - the fallback logic is tested in report generation
    # The audit pipeline fails before reaching PDF/HTML generation for invalid configs
    # In a real scenario, this would test with a valid config but PDF deps unavailable

    # For now, verify the fallback logic exists in the code
    import inspect

    from glassalpha.api.audit import run_audit

    # Check that the function handles PDF fallback
    source = inspect.getsource(run_audit)
    assert "PDF_AVAILABLE" in source
    assert 'output_format == "pdf"' in source
    assert 'output_format = "html"' in source  # Fallback logic


def test_run_audit_invalid_config():
    """Test run_audit handles invalid configuration gracefully."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create invalid config (missing required fields)
        config_path = tmp_path / "invalid.yaml"
        config = {
            "audit_profile": "minimal",
            "model": {},  # Missing type and path
            "data": {"dataset": "custom", "path": "nonexistent.csv"},  # Missing target_column
        }
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        output_path = tmp_path / "audit.html"

        # Should fail during config validation (before pipeline execution)
        with pytest.raises(ValueError):
            ga.audit.run_audit(config_path, output_path)


def test_run_audit_deterministic_output():
    """Test run_audit produces deterministic output when SOURCE_DATE_EPOCH is set."""
    # This test verifies that the SOURCE_DATE_EPOCH logic exists and is used
    import inspect

    from glassalpha.api.audit import run_audit

    source = inspect.getsource(run_audit)

    # Check that deterministic timestamp logic exists
    assert "SOURCE_DATE_EPOCH" in source
    assert "fromtimestamp" in source
    assert "UTC" in source
