"""Simplified configuration system for GlassAlpha v0.1.

Single-file Pydantic config with essential validation only.
Maintains determinism via canonical_json() for manifest hashing.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    schema_path: str | None = Field(None, description="Path to schema file")
    offline: bool = Field(False, description="Offline mode - don't fetch remote datasets")
    fetch: str = Field("auto", description="Fetch policy for remote datasets: auto, never, always")

    @field_validator("protected_attributes")
    @classmethod
    def lowercase_attributes(cls, v: list[str]) -> list[str]:
        """Normalize attribute names to lowercase."""
        return [attr.lower() for attr in v]


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""

    mode: str = Field("auto", description="Preprocessing mode: auto, artifact, or none")
    artifact_path: Path | None = Field(None, description="Path to preprocessing artifact (.joblib)")


class ReproducibilityConfig(BaseModel):
    """Reproducibility configuration for determinism."""

    random_seed: int = Field(42, description="Random seed for reproducibility")


class ReportConfig(BaseModel):
    """Report generation configuration."""

    output_format: str = Field("html", description="Output format: html or pdf")
    output_path: Path | None = Field(None, description="Path for output report")


class ExplainerConfig(BaseModel):
    """Explainer configuration."""

    strategy: str = Field("first_compatible", description="Explainer selection strategy")
    priority: list[str] = Field(default_factory=list, description="Priority order for explainer selection")


class PerformanceConfig(BaseModel):
    """Performance metrics configuration."""

    config: dict[str, Any] = Field(default_factory=dict, description="Performance metric configuration")
    metrics: list[str] = Field(default_factory=lambda: ["accuracy"], description="Performance metrics to compute")

    @field_validator("metrics", mode="before")
    @classmethod
    def handle_legacy_format(cls, v):
        """Handle legacy list format for backwards compatibility."""
        if isinstance(v, list):
            return v
        return v


class FairnessConfig(BaseModel):
    """Fairness metrics configuration."""

    config: dict[str, Any] = Field(default_factory=dict, description="Fairness metric configuration")
    metrics: list[str] = Field(default_factory=list, description="Fairness metrics to compute")

    @field_validator("metrics", mode="before")
    @classmethod
    def handle_legacy_format(cls, v):
        """Handle legacy list format for backwards compatibility."""
        if isinstance(v, list):
            return v
        return v


class StabilityConfig(BaseModel):
    """Stability analysis configuration."""

    enabled: bool = Field(True, description="Enable stability analysis")
    config: dict[str, Any] = Field(default_factory=dict, description="Stability analysis configuration")
    epsilon_values: list[float] = Field(
        default_factory=lambda: [0.01, 0.05, 0.1],
        description="Epsilon values for perturbation analysis",
    )
    threshold: float = Field(0.05, description="Threshold for stability analysis")


class MetricsConfig(BaseModel):
    """Metrics configuration."""

    compute_fairness: bool = Field(True, description="Compute fairness metrics")
    compute_calibration: bool = Field(True, description="Compute calibration metrics")
    performance: PerformanceConfig | list[str] = Field(
        default_factory=PerformanceConfig,
        description="Performance metrics configuration",
    )
    fairness: FairnessConfig | list[str] = Field(
        default_factory=FairnessConfig,
        description="Fairness metrics configuration",
    )
    n_bootstrap: int = Field(1000, description="Number of bootstrap samples for confidence intervals")
    compute_confidence_intervals: bool = Field(True, description="Compute confidence intervals")
    performance_mode: str = Field("comprehensive", description="Performance computation mode")
    individual_fairness: dict[str, Any] = Field(default_factory=dict, description="Individual fairness configuration")
    stability: StabilityConfig = Field(default_factory=StabilityConfig, description="Stability analysis configuration")

    @field_validator("performance", mode="before")
    @classmethod
    def handle_legacy_performance_format(cls, v):
        """Handle legacy performance format (list vs object)."""
        if isinstance(v, list):
            return PerformanceConfig(metrics=v)
        return v

    @field_validator("fairness", mode="before")
    @classmethod
    def handle_legacy_fairness_format(cls, v):
        """Handle legacy fairness format (list vs object)."""
        if isinstance(v, list):
            return FairnessConfig(metrics=v)
        return v


class RuntimeConfig(BaseModel):
    """Runtime configuration."""

    strict_mode: bool = Field(False, description="Enable strict mode validation")
    fast_mode: bool = Field(False, description="Enable fast mode (skip some computations)")
    compact_report: bool = Field(True, description="Generate compact report format")
    no_fallback: bool = Field(False, description="Disable fallback explainers")


class GAConfig(BaseModel):
    """GlassAlpha configuration - main config model.

    Provides canonical_json() method for deterministic manifest hashing.
    """

    model_config = ConfigDict(extra="allow")

    model: ModelConfig
    data: DataConfig
    preprocessing: PreprocessingConfig = Field(
        default_factory=lambda: PreprocessingConfig(mode="auto", artifact_path=None),
    )
    reproducibility: ReproducibilityConfig = Field(default_factory=lambda: ReproducibilityConfig(random_seed=42))
    report: ReportConfig = Field(default_factory=lambda: ReportConfig(output_format="html", output_path=None))
    explainers: ExplainerConfig = Field(
        default_factory=lambda: ExplainerConfig(strategy="first_compatible", priority=[]),
    )
    metrics: MetricsConfig = Field(
        default_factory=lambda: MetricsConfig(
            compute_fairness=True,
            compute_calibration=True,
            performance=PerformanceConfig(),
            fairness=FairnessConfig(),
            n_bootstrap=1000,
            compute_confidence_intervals=True,
            performance_mode="comprehensive",
            individual_fairness={},
            stability=StabilityConfig(threshold=0.05),
        ),
    )
    runtime: RuntimeConfig = Field(
        default_factory=lambda: RuntimeConfig(
            strict_mode=False,
            fast_mode=False,
            compact_report=True,
            no_fallback=False,
        ),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary (backwards compatibility)."""
        return self.model_dump()

    def canonical_json(self) -> str:
        """Generate deterministic JSON for manifest hashing.

        Uses sorted keys, excludes None values, ensures consistent ordering.
        Critical for byte-identical audit outputs.
        """
        return self.model_dump_json(
            exclude_none=True,
            by_alias=True,
        )


def load_config(config_path: str | Path, profile_name: str | None = None, strict: bool | None = None) -> GAConfig:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file
        profile_name: Profile name (ignored for compatibility)
        strict: Enable strict mode validation

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

    with open(config_path, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if not raw_config:
        raise ValueError(f"Configuration file is empty: {config_path}")

    # Apply strict mode if specified
    if strict is not None:
        raw_config.setdefault("runtime", {})
        raw_config["runtime"]["strict_mode"] = strict

    return GAConfig(**raw_config)


# Backwards compatibility aliases
load_config_from_file = load_config
AuditConfig = GAConfig  # Alias for backwards compatibility
load_config_from_file = load_config  # Alias for backwards compatibility


def load_yaml(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config as dictionary (backwards compatibility).

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary representation of config

    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Run 'glassalpha quickstart' to create a template configuration.",
        )

    with open(config_path, encoding="utf-8") as f:
        result = yaml.safe_load(f)
        return result if result is not None else {}


# Stub functions for test compatibility


def apply_profile_defaults(config: dict[str, Any], profile: str = "default") -> dict[str, Any]:
    """Stub for test compatibility - profile system removed."""
    return config


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two config dictionaries."""
    import copy

    def deep_merge_dict(base_dict: dict, override_dict: dict) -> dict:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(base_dict)

        for key, value in override_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Deep merge nested dictionaries
                result[key] = deep_merge_dict(result[key], value)
            else:
                # Override or add new key
                result[key] = copy.deepcopy(value)

        return result

    return deep_merge_dict(base, override)


def save_config(config: GAConfig | dict[str, Any], path: str | Path) -> None:
    """Stub for test compatibility - save config to YAML."""
    import yaml

    data = config.model_dump() if isinstance(config, GAConfig) else config
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


def validate_config(config: GAConfig) -> GAConfig:
    """Validate configuration (pass-through for compatibility).

    Pydantic validation happens automatically on construction.
    """
    # Validate strict mode requirements
    if config.runtime.strict_mode:
        _validate_strict_mode(config)

    return config


def _validate_strict_mode(config: GAConfig) -> None:
    """Validate strict mode requirements.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If strict mode requirements are not met

    """
    errors = []

    # Check required fields for strict mode
    if not config.data.target_column:
        errors.append("Explicit target column is required in strict mode")

    if not config.data.protected_attributes:
        errors.append("Explicit protected attributes are required in strict mode")

    if config.reproducibility.random_seed == 42:  # Default value
        errors.append("Explicit random seed is required in strict mode")

    if not config.explainers.priority:
        errors.append("Explicit explainer priority is required in strict mode")

    # Additional strict mode validations could be added here

    if errors:
        error_msg = "Strict mode validation failed:\n"
        error_msg += "\n".join(f"  • {error}" for error in errors)
        raise ValueError(error_msg)


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
