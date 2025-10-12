"""Preflight checks for CLI commands.

These functions validate dependencies and model availability before running
expensive operations like audits.
"""

import logging

logger = logging.getLogger(__name__)


def preflight_check_dependencies() -> bool:
    """Check if required dependencies are available.

    Returns:
        True if all dependencies are available, False otherwise.

    For now, we assume dependencies are available since we're not doing
    strict dependency checking in this simplified version.

    """
    # In a more complete implementation, this would check for:
    # - Required ML libraries (sklearn, xgboost, lightgbm, shap)
    # - PDF generation libraries (weasyprint or reportlab)
    # - Jinja2 for templating

    return True


def preflight_check_model(audit_config, allow_fallback: bool = True):
    """Check model availability and apply fallbacks if needed.

    Args:
        audit_config: Audit configuration to validate
        allow_fallback: Whether to allow fallback to simpler models

    Returns:
        Tuple of (validated_config, requested_model_type)

    For now, this is a simple pass-through since model validation
    is handled elsewhere in the pipeline.

    """
    return audit_config, None
