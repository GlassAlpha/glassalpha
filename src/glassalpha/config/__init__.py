"""Configuration system for GlassAlpha.

This module provides YAML-based configuration with Pydantic validation,
supporting audit profiles, plugin priorities, and strict mode.
"""

from .loader import (
    load_config,
    load_config_from_file,
    validate_config,
)
from .schema import (
    AuditConfig,
    CalibrationConfig,
    DataConfig,
    ExplainerConfig,
    ManifestConfig,
    MetricsConfig,
    ModelConfig,
    RecourseConfig,
    ReportConfig,
    ReproducibilityConfig,
    ThresholdConfig,
)
from .strict import (
    StrictModeError,
    validate_strict_mode,
)

__all__ = [
    # Config schemas
    "AuditConfig",
    "CalibrationConfig",
    "ModelConfig",
    "DataConfig",
    "ExplainerConfig",
    "MetricsConfig",
    "RecourseConfig",
    "ReportConfig",
    "ReproducibilityConfig",
    "ThresholdConfig",
    "ManifestConfig",
    # Loaders
    "load_config",
    "load_config_from_file",
    "validate_config",
    # Strict mode
    "validate_strict_mode",
    "StrictModeError",
]
