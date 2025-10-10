"""Configuration loading and validation.

This module handles loading configuration from YAML files and
validating against the schema and audit profile requirements.
"""

import logging
import os
import re
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any

import yaml

from ..profiles.registry import ProfileRegistry
from .schema import AuditConfig

logger = logging.getLogger(__name__)


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute environment variables in configuration values.

    Supports ${VAR_NAME} syntax for environment variable substitution.

    Args:
        value: Value to process (string, dict, list, or other)

    Returns:
        Value with environment variables substituted

    """
    if isinstance(value, str):
        # Substitute ${VAR_NAME} patterns
        def replace_env_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))  # Return original if not found

        return re.sub(r"\$\{([^}]+)\}", replace_env_var, value)

    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]

    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML content

    Raises:
        FileNotFoundError: If file doesn't exist locally or in package
        yaml.YAMLError: If YAML is invalid

    """
    path = Path(path)
    text_content = None

    if path.exists():
        # Load from local file system
        try:
            text_content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to read configuration file {path}: {e}") from e
    else:
        # Fall back to packaged built-ins by filename
        try:
            candidate = pkg_files("glassalpha.data.configs") / path.name
            text_content = candidate.read_text(encoding="utf-8")
        except Exception as e:
            raise FileNotFoundError(f"Configuration file not found: {path}") from e

    try:
        return yaml.safe_load(text_content) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration (deep merge)

    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursive merge for nested dicts
            result[key] = merge_configs(result[key], value)
        else:
            # Direct override
            result[key] = value

    return result


def validate_config_completeness(_: "AuditConfig") -> None:
    """Hook for additional completeness checks (patched in tests)."""
    return


def apply_profile_defaults(config: dict[str, Any], profile_name: str | None = None) -> dict[str, Any]:
    """Apply audit profile defaults to configuration.

    Args:
        config: Configuration dictionary
        profile_name: Override profile name (default: from config)

    Returns:
        Configuration with profile defaults applied

    """
    profile_name = profile_name or config.get("audit_profile")
    if not profile_name:
        logger.warning("No audit profile specified, using raw config")
        return config

    # Get profile from registry
    try:
        profile_cls = ProfileRegistry.get(profile_name)
    except KeyError:
        logger.warning(f"Unknown audit profile: {profile_name}")
        return config

    # Apply profile defaults
    try:
        if hasattr(profile_cls, "apply_defaults"):
            config = profile_cls.apply_defaults(config)
            logger.info(f"Applied defaults from profile '{profile_name}'")
        elif hasattr(profile_cls, "get_default_config"):
            default_config = profile_cls.get_default_config()
            # Merge with user config (user config takes precedence)
            config = merge_configs(default_config, config)
            logger.info(f"Applied defaults from profile '{profile_name}'")
        else:
            logger.warning(f"Profile '{profile_name}' does not provide defaults")
    except Exception as e:
        logger.warning(f"Failed to apply profile '{profile_name}' defaults: {e}")
        return config

    # Validate against profile requirements
    if hasattr(profile_cls, "validate_config"):
        try:
            profile_cls.validate_config(config)
            logger.info(f"Configuration validated against profile '{profile_name}'")
        except ValueError as e:
            logger.error(f"Configuration invalid for profile '{profile_name}': {e}")
            raise

    return config


def validate_config(config: dict[str, Any] | AuditConfig) -> AuditConfig:
    """Validate configuration against schema.

    Args:
        config: Configuration dictionary or AuditConfig object

    Returns:
        Validated AuditConfig object

    Raises:
        ValueError: If configuration is invalid

    """
    if isinstance(config, AuditConfig):
        return config

    try:
        # Profile defaults are applied in load_config, not here

        # Create and validate config object
        audit_config = AuditConfig(**config)

        # Call completeness hook (can be patched by tests)
        validate_config_completeness(audit_config)

        logger.info("Configuration validated successfully")
        return audit_config

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise ValueError(f"Invalid configuration: {e}") from e


def load_config(
    config_dict: dict[str, Any],
    profile_name: str | None = None,
    strict: bool = False,
    strict_full: bool = False,
) -> AuditConfig:
    """Load configuration from dictionary.

    Args:
        config_dict: Configuration dictionary
        profile_name: Override audit profile
        strict: Enable strict mode validation
        strict_full: Enable full strict mode (no built-in datasets allowed)

    Returns:
        Validated AuditConfig object

    """
    # Override profile if specified
    if profile_name:
        config_dict["audit_profile"] = profile_name

    # Override strict mode if specified
    if strict:
        config_dict["strict_mode"] = True

    # Apply defaults and validate
    config_dict = apply_profile_defaults(config_dict, config_dict.get("audit_profile"))

    # Substitute environment variables
    logger.info("Substituting environment variables in configuration")
    config_dict = _substitute_env_vars(config_dict)

    audit_config = validate_config(config_dict)

    # Auto-detect target column for built-in datasets if not specified
    if audit_config.data.dataset and audit_config.data.dataset != "custom" and not audit_config.data.target_column:
        # Hardcoded target columns for known built-in datasets
        known_target_columns = {
            "german_credit": "credit_risk",
            "adult_income": "income",
        }

        if audit_config.data.dataset in known_target_columns:
            audit_config.data.target_column = known_target_columns[audit_config.data.dataset]
            logger.info(
                f"Auto-detected target column '{audit_config.data.target_column}' for built-in dataset '{audit_config.data.dataset}'",
            )

    # Apply strict mode validation if enabled
    if audit_config.strict_mode or strict or strict_full:
        from .strict import validate_strict_mode  # noqa: PLC0415

        # Determine quick mode: use if built-in dataset specified AND not in full strict mode AND not explicitly disabled
        # The CLI strict flag should override the automatic quick mode behavior
        quick_mode = bool(
            audit_config.data.dataset
            and audit_config.data.dataset != "custom"
            and not strict_full
            and strict is not False,  # Explicit False means don't use quick mode
        )
        validate_strict_mode(audit_config, quick_mode=quick_mode)

    logger.info("Configuration validated successfully")
    return audit_config


def load_config_from_file(
    path: str | Path,
    override_path: str | Path | None = None,
    profile_name: str | None = None,
    strict: bool = False,
    strict_full: bool = False,
) -> AuditConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to main configuration file
        override_path: Optional path to override configuration
        profile_name: Override audit profile
        strict: Enable strict mode validation
        strict_full: Enable full strict mode (no built-in datasets allowed)

    Returns:
        Validated AuditConfig object

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration is invalid

    """
    logger.info(f"Loading configuration from {path}")

    # Load main config
    config_dict = load_yaml(path)

    # Load and merge override config if provided
    if override_path:
        logger.info(f"Loading override configuration from {override_path}")
        override_dict = load_yaml(override_path)
        config_dict = merge_configs(config_dict, override_dict)

    # Load and validate
    return load_config(config_dict, profile_name, strict, strict_full)


def save_config(config: AuditConfig, path: str | Path, include_defaults: bool = False) -> None:
    """Save audit configuration to YAML file for reproducible compliance audits.

    Serializes the complete audit configuration including model settings,
    data parameters, and compliance policies to enable audit trail documentation
    and configuration version control.

    Args:
        config: Complete audit configuration object with validated settings
        path: Target file path for configuration storage
        include_defaults: Include default values for complete audit trail documentation

    Side Effects:
        - Creates or overwrites YAML file at specified path
        - May create parent directories if they don't exist
        - File written with UTF-8 encoding for international compliance

    Raises:
        IOError: If path is not writable or insufficient disk space
        ValidationError: If configuration contains invalid settings

    Note:
        For regulatory compliance, always save with include_defaults=True to
        maintain complete audit trail of configuration decisions.

    """
    path = Path(path)

    # Convert to dictionary - use ternary for cleaner code
    config_dict = config.model_dump() if include_defaults else config.model_dump(exclude_defaults=True)

    # Convert PosixPath objects to strings for YAML serialization
    def convert_paths(obj):
        if isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    config_dict = convert_paths(config_dict)

    # Write YAML
    with open(path, "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Configuration saved to {path}")


def load_builtin_config(name: str) -> AuditConfig:
    """Load a packaged config by base name, e.g. 'german_credit_simple.yaml' or 'german_credit_simple'.

    Args:
        name: Config name, optionally with .yaml extension

    Returns:
        Validated AuditConfig object

    Raises:
        FileNotFoundError: If config doesn't exist in package
        ValueError: If config is invalid

    """
    if not name.endswith(".yaml"):
        name += ".yaml"

    try:
        text = (pkg_files("glassalpha.data.configs") / name).read_text(encoding="utf-8")
        config_dict = yaml.safe_load(text) or {}
        return load_config(config_dict)
    except Exception as e:
        raise ValueError(f"Failed to load builtin config '{name}': {e}") from e
