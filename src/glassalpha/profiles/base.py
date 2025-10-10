"""Base audit profile implementation.

This provides the foundation for all audit profiles, defining the
interface and common functionality.
"""

from typing import Any


class BaseAuditProfile:
    """Base implementation of audit profile.

    Subclasses should define the specific component sets for their audit type.
    """

    # These should be overridden by subclasses
    name: str = "base"
    version: str = "1.0.0"
    compatible_models: list[str] = []
    required_metrics: list[str] = []
    optional_metrics: list[str] = []
    report_template: str = "standard_audit"
    explainer_priority: list[str] = []

    @classmethod
    def get_compatible_models(cls) -> list[str]:
        """Return list of compatible model types."""
        return cls.compatible_models

    @classmethod
    def get_required_metrics(cls) -> list[str]:
        """Return list of required metric types."""
        return cls.required_metrics

    @classmethod
    def get_optional_metrics(cls) -> list[str]:
        """Return list of optional metric types."""
        return cls.optional_metrics

    @classmethod
    def get_report_template(cls) -> str:
        """Return name of report template to use."""
        return cls.report_template

    @classmethod
    def get_explainer_priority(cls) -> list[str]:
        """Return ordered list of preferred explainers."""
        return cls.explainer_priority

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> bool:
        """Validate configuration for this profile.

        Args:
            config: Configuration to validate

        Returns:
            True if valid for this profile

        Raises:
            ValueError: If configuration is invalid

        """
        # Check model type is compatible
        model_type = config.get("model", {}).get("type")
        if model_type and model_type not in cls.compatible_models:
            raise ValueError(
                f"Model type '{model_type}' not compatible with profile '{cls.name}'. Allowed: {cls.compatible_models}",
            )

        # Check required fields
        if "model" not in config:
            raise ValueError(f"Profile '{cls.name}' requires 'model' configuration")

        if "data" not in config:
            raise ValueError(f"Profile '{cls.name}' requires 'data' configuration")

        return True

    @classmethod
    def get_profile_info(cls) -> dict[str, Any]:
        """Get profile metadata for logging/debugging.

        Returns:
            Dictionary with profile information

        """
        return {
            "name": cls.name,
            "version": cls.version,
            "compatible_models": cls.compatible_models,
            "required_metrics": cls.required_metrics,
            "optional_metrics": cls.optional_metrics,
            "report_template": cls.report_template,
            "explainer_priority": cls.explainer_priority,
        }
