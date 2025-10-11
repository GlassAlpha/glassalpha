"""Simplified configuration system for GlassAlpha v0.1.

Single-file Pydantic config with essential validation only.
Maintains determinism via canonical_json() for manifest hashing.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Model configuration."""

    type: str = Field(..., description="Model type: logistic_regression, xgboost, or lightgbm")
    path: Path | None = Field(None, description="Path to saved model file")
    save_path: Path | None = Field(None, description="Path to save trained model")
    params: dict[str, Any] = Field(default_factory=dict, description="Model parameters")

    @field_validator("type")
    @classmethod
    def normalize_type(cls, v: str) -> str:
        """Ensure model type is lowercase."""
        return v.lower()


class DataConfig(BaseModel):
    """Data configuration."""

    dataset: str | None = Field(None, description="Dataset key or 'custom' for external files")
    path: str | None = Field(None, description="Path to data file (when dataset='custom')")
    target_column: str | None = Field(None, description="Name of target column")
    protected_attributes: list[str] = Field(
        default_factory=list,
        description="Protected attributes for fairness analysis",
    )
    feature_columns: list[str] | None = Field(None, description="Feature columns to use")

    @field_validator("protected_attributes")
    @classmethod
    def lowercase_attributes(cls, v: list[str]) -> list[str]:
        """Normalize attribute names to lowercase."""
        return [attr.lower() for attr in v]


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""

    artifact_path: Path | None = Field(None, description="Path to preprocessing artifact (.joblib)")


class ReproducibilityConfig(BaseModel):
    """Reproducibility configuration for determinism."""

    random_seed: int = Field(42, description="Random seed for reproducibility")


class ReportConfig(BaseModel):
    """Report generation configuration."""

    output_format: str = Field("html", description="Output format: html or pdf")
    output_path: Path | None = Field(None, description="Path for output report")


class GAConfig(BaseModel):
    """GlassAlpha configuration - main config model.

    Provides canonical_json() method for deterministic manifest hashing.
    """

    model: ModelConfig
    data: DataConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    def canonical_json(self) -> str:
        """Generate deterministic JSON for manifest hashing.

        Uses sorted keys, excludes None values, ensures consistent ordering.
        Critical for byte-identical audit outputs.
        """
        return self.model_dump_json(
            sort_keys=True,
            exclude_none=True,
            by_alias=True,
        )


def load_config(config_path: str | Path) -> GAConfig:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated GAConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails

    Examples:
        >>> config = load_config("audit.yaml")
        >>> config.model.type
        'xgboost'

    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Run 'glassalpha quickstart' to create a template configuration.",
        )

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    if not raw_config:
        raise ValueError(f"Configuration file is empty: {config_path}")

    return GAConfig(**raw_config)


# Backwards compatibility aliases
load_config_from_file = load_config
AuditConfig = GAConfig  # Alias for backwards compatibility
load_config_from_file = load_config  # Alias for backwards compatibility
load_yaml = load_config  # Alias for backwards compatibility


# Stub classes for test compatibility
class ExplainerConfig(BaseModel):
    """Stub for test compatibility - explainer config removed."""

    priority: list[str] = Field(default_factory=list)


class MetricsConfig(BaseModel):
    """Stub for test compatibility - metrics config simplified."""

    compute_fairness: bool = True
    compute_calibration: bool = True


def apply_profile_defaults(config: dict, profile: str = "default") -> dict:
    """Stub for test compatibility - profile system removed."""
    return config


def merge_configs(base: dict, override: dict) -> dict:
    """Stub for test compatibility - merge system simplified."""
    result = base.copy()
    result.update(override)
    return result


def save_config(config: GAConfig | dict, path: str | Path) -> None:
    """Stub for test compatibility - save config to YAML."""
    import yaml

    data = config.model_dump() if isinstance(config, GAConfig) else config
    with open(path, "w") as f:
        yaml.dump(data, f)


def validate_config(config: GAConfig) -> GAConfig:
    """Validate configuration (pass-through for compatibility).

    Pydantic validation happens automatically on construction.
    """
    return config


__all__ = [
    "AuditConfig",  # Backwards compatibility
    "DataConfig",
    "ExplainerConfig",  # Stub for test compatibility
    "GAConfig",
    "MetricsConfig",  # Stub for test compatibility
    "ModelConfig",
    "PreprocessingConfig",
    "ReportConfig",
    "ReproducibilityConfig",
    "apply_profile_defaults",  # Stub for test compatibility
    "load_config",
    "load_config_from_file",  # Backwards compatibility
    "load_yaml",  # Backwards compatibility
    "merge_configs",  # Stub for test compatibility
    "save_config",  # Stub for test compatibility
    "validate_config",
]
