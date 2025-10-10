"""Strict mode validation for regulatory compliance.

This module enforces additional requirements when strict mode is enabled,
ensuring all necessary fields are present and determinism is guaranteed.
"""

import logging
import warnings
from typing import Any

from .schema import AuditConfig

logger = logging.getLogger(__name__)


class StrictModeError(ValueError):
    """Raised when strict mode validation fails."""


def validate_strict_mode(config: AuditConfig, quick_mode: bool = False) -> None:
    """Validate configuration meets strict mode requirements.

    Strict mode enforces:
    - Explicit random seeds (no defaults)
    - Locked data schema (no inference)
    - Full manifest generation
    - Deterministic plugin selection
    - No configuration defaults
    - All optional fields become required
    - Warnings become errors

    Quick mode is a relaxed variant that:
    - Allows built-in datasets (bypasses data.path requirement)
    - Skips model.path requirement (allows training from config)
    - Still enforces reproducibility and determinism

    Args:
        config: Configuration to validate
        quick_mode: If True, use relaxed validation for testing/development

    Raises:
        StrictModeError: If validation fails

    """
    mode_desc = "quick strict mode" if quick_mode else "strict mode"
    logger.info(f"Validating configuration for {mode_desc} compliance")
    errors: list[str] = []

    # Check reproducibility settings (always required)
    if config.reproducibility.random_seed is None:
        errors.append(f"Explicit random seed is required in {mode_desc}")

    if not config.reproducibility.deterministic:
        errors.append(f"Deterministic mode must be enabled in {mode_desc}")

    if not config.reproducibility.capture_environment:
        errors.append(f"Environment capture must be enabled in {mode_desc}")

    # Check data configuration (relaxed in quick mode for built-in datasets)
    if not quick_mode:
        if not config.data.path:
            errors.append("Data path must be specified in strict mode")
    # Quick mode: allow built-in datasets
    elif not config.data.path and not config.data.dataset:
        errors.append("Data path or dataset must be specified in quick strict mode")

    if not config.data.schema_path and not config.data.data_schema:
        # In quick mode, built-in datasets have implicit schemas
        if not (quick_mode and config.data.dataset and config.data.dataset != "custom"):
            errors.append(f"Data schema must be specified (either path or inline) in {mode_desc}")

    if not config.data.protected_attributes:
        errors.append(f"Protected attributes must be specified for fairness analysis in {mode_desc}")

    if not config.data.target_column:
        errors.append(f"Target column must be explicitly specified in {mode_desc}")

    # Check model configuration (relaxed in quick mode)
    if not quick_mode:
        if not config.model.path:
            errors.append("Model path must be specified in strict mode")
    # Quick mode allows training from config

    # Check explainer configuration
    if not config.explainers.priority:
        errors.append(f"Explainer priority list must be specified in {mode_desc}")

    if config.explainers.strategy != "first_compatible":
        errors.append(f"Explainer strategy must be 'first_compatible' for determinism in {mode_desc}")

    # Check manifest configuration
    if not config.manifest.enabled:
        errors.append(f"Manifest generation must be enabled in {mode_desc}")

    if not config.manifest.include_git_sha:
        errors.append(f"Git SHA must be included in manifest in {mode_desc}")

    if not config.manifest.include_config_hash:
        errors.append(f"Config hash must be included in manifest in {mode_desc}")

    if not config.manifest.include_data_hash:
        errors.append(f"Data hash must be included in manifest in {mode_desc}")

    # Check audit profile
    if not config.audit_profile:
        errors.append(f"Audit profile must be specified in {mode_desc}")

    # Check metrics are specified (handle both list and MetricCategory)
    performance_metrics = (
        config.metrics.performance
        if isinstance(config.metrics.performance, list)
        else config.metrics.performance.metrics
    )
    if not performance_metrics:
        errors.append(f"Performance metrics must be specified in {mode_desc}")

    fairness_metrics = (
        config.metrics.fairness if isinstance(config.metrics.fairness, list) else config.metrics.fairness.metrics
    )
    if not fairness_metrics:
        errors.append(f"Fairness metrics must be specified in {mode_desc}")

    # Check recourse if enabled
    if config.recourse.enabled and not config.recourse.immutable_features:
        errors.append(f"Immutable features must be specified when recourse is enabled in {mode_desc}")

    # Check preprocessing configuration (relaxed in quick mode)
    if not quick_mode:
        if config.preprocessing.mode != "artifact":
            errors.append(
                "Preprocessing mode must be 'artifact' in strict mode. "
                "Auto preprocessing is not valid for regulatory compliance.",
            )

        if config.preprocessing.mode == "artifact":
            if not config.preprocessing.artifact_path:
                errors.append("Preprocessing artifact_path must be specified when mode='artifact' in strict mode")

            if not config.preprocessing.expected_file_hash:
                errors.append("Preprocessing expected_file_hash must be specified in strict mode")

            if not config.preprocessing.expected_params_hash:
                errors.append("Preprocessing expected_params_hash must be specified in strict mode")
    # Quick mode allows auto preprocessing for testing
    elif config.preprocessing.mode == "artifact":
        if not config.preprocessing.artifact_path:
            errors.append("Preprocessing artifact_path must be specified when mode='artifact'")

    # Convert warnings to errors (only in full strict mode)
    if not quick_mode:
        warnings.simplefilter("error")

    # Report all errors
    if errors:
        error_msg = f"{mode_desc.capitalize()} validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)

        if quick_mode:
            error_msg += "\n\n💡 Tip: Quick strict mode is for testing. Use full strict mode for production."

        raise StrictModeError(error_msg)

    logger.info(f"{mode_desc.capitalize()} validation passed")

    if quick_mode:
        logger.warning(
            "Quick strict mode is enabled - suitable for testing but NOT for production audits. "
            "See https://glassalpha.com/guides/strict-mode/ for full strict mode requirements.",
        )


def validate_deterministic_config(config: dict[str, Any]) -> bool:
    """Validate that configuration ensures deterministic behavior.

    Args:
        config: Configuration dictionary

    Returns:
        True if configuration is deterministic

    """
    deterministic = True

    # Check for random elements
    if config.get("reproducibility", {}).get("random_seed") is None:
        logger.warning("No random seed specified - results may vary")
        deterministic = False

    # Check for sorted operations
    explainer_priority = config.get("explainers", {}).get("priority", [])
    if not explainer_priority:
        logger.warning("No explainer priority specified - selection may vary")
        deterministic = False

    # Check for unordered collections
    metrics = config.get("metrics", {})
    for category in ["performance", "fairness", "drift"]:
        if category in metrics and isinstance(metrics[category], set):
            logger.warning(f"Metrics in '{category}' use set - order may vary")
            deterministic = False

    return deterministic


def validate_reproducible_environment() -> bool:
    """Check if environment supports reproducible execution.

    Returns:
        True if environment is reproducible

    """
    import platform

    reproducible = True
    issues = []

    # Check Python version

    # Check for known non-deterministic libraries
    try:
        import tensorflow

        issues.append("TensorFlow detected - may introduce randomness")
        reproducible = False
    except ImportError:
        pass

    # Check platform
    if platform.system() == "Windows":
        logger.warning("Windows platform may have path handling differences")

    if issues:
        logger.warning("Environment issues for reproducibility: " + ", ".join(issues))

    return reproducible


def enforce_strict_defaults(config: AuditConfig) -> AuditConfig:
    """Apply strict mode defaults to configuration.

    This modifies the configuration to use the most conservative
    settings for regulatory compliance.

    Args:
        config: Configuration to modify

    Returns:
        Modified configuration

    """
    # Force deterministic settings
    config.reproducibility.deterministic = True
    config.reproducibility.capture_environment = True

    # Force manifest generation
    config.manifest.enabled = True
    config.manifest.include_git_sha = True
    config.manifest.include_config_hash = True
    config.manifest.include_data_hash = True
    config.manifest.include_model_hash = True

    # Force deterministic explainer selection
    config.explainers.strategy = "first_compatible"

    # Enable all auditing features
    config.report.include_sections = [
        "lineage",
        "data_schema",
        "global_explanations",
        "local_explanations",
        "fairness",
        "drift",
        "recourse",
        "assumptions",
    ]

    logger.info("Applied strict mode defaults")
    return config
