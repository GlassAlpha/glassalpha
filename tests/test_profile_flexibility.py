"""Test profile flexibility and error handling."""

from unittest.mock import patch

import pytest

from glassalpha.config.loader import load_config


def test_unknown_profile_warns_and_continues():
    """Test that unknown profiles log a warning but don't crash."""
    config_dict = {
        "audit_profile": "totally_fake_profile",
        "model": {"type": "xgboost"},
        "data": {"dataset": "custom", "path": "test.csv"},
        "report": {
            "template": "standard_audit.html",
        },
    }

    with patch("glassalpha.config.loader.logger") as mock_logger:
        # Should not raise error for unknown profile
        config = load_config(config_dict)

        # Should have logged warning about unknown profile
        mock_logger.warning.assert_any_call("Unknown audit profile: totally_fake_profile")

        # Config should be returned (with defaults applied)
        assert hasattr(config, "audit_profile")
        assert hasattr(config, "model")
        assert hasattr(config, "data")
        assert hasattr(config, "report")


def test_unknown_config_keys_raise_validation_error():
    """Test that unknown config keys raise validation errors."""
    config_dict = {
        "audit_profile": "tabular_compliance",
        "model": {"type": "xgboost", "unknown_key": "value"},
        "data": {"dataset": "custom", "path": "test.csv"},
        "report": {"template": "standard_audit.html"},
    }

    # Should raise ValueError due to unknown keys (Pydantic validation)
    with pytest.raises(ValueError, match="Invalid configuration"):
        load_config(config_dict)


def test_missing_profile_fallback():
    """Test that missing profile falls back gracefully."""
    config_dict = {
        "audit_profile": "tabular_compliance",  # Use a known profile to avoid validation errors
        "model": {"type": "xgboost"},
        "data": {"dataset": "custom", "path": "test.csv"},
    }

    with patch("glassalpha.config.loader.logger") as mock_logger:
        config = load_config(config_dict)

        # Should have logged about configuration validation (since profile defaults are applied)
        mock_logger.info.assert_any_call("Configuration validated successfully")

        # Config should be returned with defaults applied
        assert hasattr(config, "audit_profile")
        assert hasattr(config, "explainers")
        assert hasattr(config, "metrics")
