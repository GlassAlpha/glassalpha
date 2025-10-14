"""Simplified configuration system for GlassAlpha v0.2.

Single-file Pydantic config with essential validation only.
Maintains determinism via canonical_json() for manifest hashing.

ARCHITECTURE NOTES:

1. **Configuration Explicitness is Required for Compliance**:
   - Regulatory audits require explicit configuration (no hidden defaults)
   - Users must understand what they're auditing (transparency requirement)
   - DO NOT add "smart profiles" that hide settings - breaks auditability

2. **Why No Profile System**:
   - Removed in v0.2.0 architectural simplification
   - Profiles hid important compliance decisions
   - Current approach: provide example configs at different complexity levels
     - minimal.yaml (8 lines) - quickstart
     - german_credit_simple.yaml (50 lines) - common use case
     - german_credit.yaml (242 lines) - all features documented

3. **Enhanced Error Messages**:
   - Pydantic validation errors are caught and enhanced with suggestions
   - Common missing fields get specific fix examples
   - Errors include references to config templates

4. **Canonical JSON for Determinism**:
   - canonical_json() method ensures byte-identical output
   - Critical for regulatory verification and audit trails
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
    def validate_model_type(cls, v: str) -> str:
        """Validate model type is supported."""
        v = v.lower()
        valid_types = ["xgboost", "lightgbm", "logistic_regression", "random_forest"]
        if v not in valid_types:
            raise ValueError(
                f"Invalid model type: '{v}'\n\n"
                f"Supported types: {', '.join(valid_types)}\n\n"
                f"Run 'glassalpha list' to see available models with current dependencies."
            )
        return v


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


class ReportConfig(BaseModel):
    """Report generation configuration."""

    output_format: str = Field("html", description="Output format: html or pdf")
    output_path: Path | None = Field(None, description="Path for output report")

    # PDF export options
    pdf_profile: str = Field("fast", description="PDF export profile: fast | print | strict")
    pdf_cache: bool = Field(True, description="Enable PDF export caching")


class ExplainerConfig(BaseModel):
    """Explainer configuration."""

    strategy: str = Field("first_compatible", description="Explainer selection strategy")
    priority: list[str] = Field(default_factory=list, description="Priority order for explainer selection")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate explainer selection strategy."""
        valid_strategies = ["first_compatible", "all", "manual"]
        if v not in valid_strategies:
            raise ValueError(
                f"Invalid explainer strategy: '{v}'\n\n"
                f"Supported strategies:\n"
                f"  - first_compatible: Use first compatible explainer (recommended)\n"
                f"  - all: Generate all compatible explanations\n"
                f"  - manual: Use explicitly specified explainer\n\n"
                f"Valid values: {', '.join(valid_strategies)}"
            )
        return v


class MetricsConfig(BaseModel):
    """Metrics configuration - consolidated for simplicity.

    Combines performance, fairness, and stability metrics into a single
    configuration class. Uses nested dicts for flexibility while reducing
    the number of config classes users need to understand.
    """

    # High-level toggles
    compute_fairness: bool = Field(True, description="Compute fairness metrics")
    compute_calibration: bool = Field(True, description="Compute calibration metrics")
    compute_confidence_intervals: bool = Field(True, description="Compute confidence intervals")
    n_bootstrap: int = Field(1000, description="Number of bootstrap samples for confidence intervals")

    # Performance metrics (dict or list for backward compatibility)
    performance: dict[str, Any] | list[str] = Field(
        default_factory=lambda: {"metrics": ["accuracy"]},
        description="Performance metrics: dict with 'metrics' key or list of metric names",
    )

    # Fairness metrics (dict or list for backward compatibility)
    fairness: dict[str, Any] | list[str] = Field(
        default_factory=lambda: {},
        description="Fairness metrics: dict with 'metrics' key or list of metric names",
    )

    # Stability analysis
    stability: dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "epsilon_values": [0.01, 0.05, 0.1],
            "threshold": 0.05,
        },
        description="Stability analysis: dict with 'enabled', 'epsilon_values', 'threshold' keys",
    )

    # Other metric configurations
    performance_mode: str = Field("comprehensive", description="Performance computation mode")
    individual_fairness: dict[str, Any] = Field(
        default_factory=dict,
        description="Individual fairness configuration",
    )

    @field_validator("performance", mode="before")
    @classmethod
    def normalize_performance(cls, v: Any) -> dict[str, Any]:
        """Normalize performance to dict format."""
        if isinstance(v, list):
            return {"metrics": v}
        if isinstance(v, dict) and "metrics" not in v:
            return {"metrics": ["accuracy"]}
        return v

    @field_validator("fairness", mode="before")
    @classmethod
    def normalize_fairness(cls, v: Any) -> dict[str, Any]:
        """Normalize fairness to dict format."""
        if isinstance(v, list):
            return {"metrics": v}
        return v if isinstance(v, dict) else {}


class RuntimeConfig(BaseModel):
    """Runtime configuration."""

    strict_mode: bool = Field(False, description="Enable strict mode validation")
    fast_mode: bool = Field(False, description="Enable fast mode (skip some computations)")
    compact_report: bool = Field(True, description="Generate compact report format")
    no_fallback: bool = Field(False, description="Disable fallback explainers")


class AuditConfig(BaseModel):
    """GlassAlpha audit configuration - main config model.

    Provides canonical_json() method for deterministic manifest hashing.
    Incorporates reproducibility settings directly for simplicity.
    """

    model_config = ConfigDict(extra="allow")

    # Optional audit profile name (defaults to "default" if not specified)
    audit_profile: str = Field(default="default")

    # Core configuration
    model: ModelConfig
    data: DataConfig
    preprocessing: PreprocessingConfig = Field(
        default_factory=lambda: PreprocessingConfig(mode="auto", artifact_path=None),
    )
    report: ReportConfig = Field(
        default_factory=lambda: ReportConfig(
            output_format="html",
            output_path=None,
            pdf_profile="fast",
            pdf_cache=True,
        ),
    )
    explainers: ExplainerConfig = Field(
        default_factory=lambda: ExplainerConfig(strategy="first_compatible", priority=[]),
    )
    metrics: MetricsConfig = Field(default_factory=lambda: MetricsConfig())
    runtime: RuntimeConfig = Field(
        default_factory=lambda: RuntimeConfig(
            strict_mode=False,
            fast_mode=False,
            compact_report=True,
            no_fallback=False,
        ),
    )

    # Reproducibility settings (incorporated directly for simplicity)
    random_seed: int = Field(42, description="Random seed for reproducibility")
    strict: bool = Field(
        False,
        description="Enable strict determinism controls",
    )
    thread_control: bool = Field(
        True,
        description="Control thread counts for deterministic parallel processing",
    )
    warn_on_failure: bool = Field(
        True,
        description="Warn if some determinism controls fail to apply",
    )

    # Backward compatibility: accept 'reproducibility' dict and flatten it
    @field_validator("random_seed", mode="before")
    @classmethod
    def extract_from_reproducibility(cls, v: Any, info: Any) -> int:
        """Extract random_seed from reproducibility dict if present."""
        if isinstance(info.data.get("reproducibility"), dict):
            return info.data["reproducibility"].get("random_seed", 42)
        return v if v is not None else 42

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


def load_config(
    config_path: str | Path,
    profile_name: str | None = None,
    strict: bool | None = None,
) -> AuditConfig:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file
        profile_name: Profile name (ignored for compatibility)
        strict: Enable strict mode validation

    Returns:
        Validated AuditConfig instance

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
            f"Configuration file not found: {config_path}\n\n"
            f"💡 Get started quickly:\n\n"
            f"  # Generate project with example config\n"
            f"  glassalpha quickstart\n\n"
            f"  # Or copy built-in example config\n"
            f"  glassalpha config-template german_credit > audit.yaml\n\n"
            f"Available templates:\n"
            f"  - german_credit       (credit risk, recommended first audit)\n"
            f"  - adult_income        (fairness-focused)\n"
            f"  - minimal             (8-line quickstart)\n"
            f"  - custom_template     (blank template for your data)\n\n"
            f"List all: glassalpha config-list"
        )

    try:
        with open(config_path, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"Invalid YAML syntax in {config_path}\n\n"
            f"Error: {e!s}\n\n"
            "💡 Common YAML issues:\n"
            "  - Incorrect indentation (use spaces, not tabs)\n"
            "  - Missing quotes around values with colons\n"
            "  - Unclosed brackets or quotes\n\n"
            "Validate your YAML: https://www.yamllint.com/"
        ) from None

    if not raw_config:
        raise ValueError(f"Configuration file is empty: {config_path}")

    # Apply strict mode if specified
    if strict is not None:
        raw_config.setdefault("runtime", {})
        raw_config["runtime"]["strict_mode"] = strict

    try:
        return AuditConfig(**raw_config)
    except Exception as e:
        # Enhance Pydantic validation errors with helpful suggestions
        error_msg = str(e)

        # Add suggestions for common missing fields
        if "model" in error_msg and "Field required" in error_msg:
            raise ValueError(
                f"{error_msg}\n\n"
                f"💡 Suggestion: Add model configuration to your config:\n\n"
                f"model:\n"
                f"  type: logistic_regression  # or xgboost, lightgbm\n"
                f"  params:\n"
                f"    random_state: 42\n\n"
                f"See: glassalpha config-template minimal"
            ) from None
        elif "data" in error_msg and "Field required" in error_msg:
            raise ValueError(
                f"{error_msg}\n\n"
                f"💡 Suggestion: Add data configuration to your config:\n\n"
                f"data:\n"
                f"  dataset: german_credit     # Use built-in dataset\n"
                f"  # OR\n"
                f"  # path: data/my_data.csv   # Use custom CSV file\n"
                f"  target_column: credit_risk\n"
                f"  protected_attributes:\n"
                f"    - gender\n"
                f"    - age_group\n\n"
                f"See: glassalpha config-template minimal"
            ) from None
        elif "target_column" in error_msg:
            raise ValueError(
                f"{error_msg}\n\n"
                f"💡 Suggestion: Specify your target column name:\n\n"
                f"data:\n"
                f'  target_column: "your_target_column_name"\n\n'
                f"Common target column names: credit_risk, outcome, label, target, y"
            ) from None
        else:
            # Re-raise original error if we don't have a specific suggestion
            raise


# Backwards compatibility alias
load_config_from_file = load_config


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


def apply_profile_defaults(
    config: dict[str, Any],
    profile: str = "default",
) -> dict[str, Any]:
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


def save_config(config: AuditConfig | dict[str, Any], path: str | Path) -> None:
    """Stub for test compatibility - save config to YAML."""
    import yaml

    data = config.model_dump() if isinstance(config, AuditConfig) else config
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


def validate_config(config: AuditConfig) -> AuditConfig:
    """Validate configuration (pass-through for compatibility).

    Pydantic validation happens automatically on construction.
    """
    # Validate strict mode requirements
    if config.runtime.strict_mode:
        _validate_strict_mode(config)

    return config


def _validate_strict_mode(config: AuditConfig) -> None:
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

    if config.random_seed == 42:  # Default value
        errors.append("Explicit random seed is required in strict mode")

    if not config.explainers.priority:
        errors.append("Explicit explainer priority is required in strict mode")

    # Additional strict mode validations could be added here

    if errors:
        error_msg = "Strict mode validation failed:\n"
        error_msg += "\n".join(f"  • {error}" for error in errors)
        raise ValueError(error_msg)


__all__ = [
    "AuditConfig",
    "DataConfig",
    "ExplainerConfig",
    "MetricsConfig",
    "ModelConfig",
    "PreprocessingConfig",
    "ReportConfig",
    "RuntimeConfig",
    "apply_profile_defaults",  # Stub for test compatibility
    "load_config",
    "load_config_from_file",  # Backwards compatibility
    "load_yaml",  # Backwards compatibility
    "merge_configs",  # Stub for test compatibility
    "save_config",  # Stub for test compatibility
    "validate_config",
]
