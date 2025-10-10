"""Centralized constants and contract strings for GlassAlpha.

This module contains all exact strings used in error messages, logging,
and contract assertions to prevent drift and ensure consistency.
"""

# Exact contract strings - primary names used in tests
NO_MODEL_MSG = "Model not loaded. Load a model first."
NO_EXPLAINER_MSG = "No compatible explainer found"
INIT_LOG_MESSAGE = "Initialized audit pipeline with profile: {profile}"
ERR_NOT_FITTED = "Model not loaded. Load a model first."
INIT_LOG_TEMPLATE = INIT_LOG_MESSAGE

# Status values
STATUS_CLEAN = "clean"
STATUS_DIRTY = "dirty"
STATUS_NO_GIT = "no_git"

# Binary classification constants
BINARY_CLASSES = 2
BINARY_THRESHOLD = 0.5

# Template names
STANDARD_AUDIT_TEMPLATE = "standard_audit.html"

# Package paths for resources
TEMPLATES_PACKAGE = "glassalpha.report.templates"

# Export the primary contract strings
__all__ = [
    "BINARY_CLASSES",
    "BINARY_THRESHOLD",
    "ERR_NOT_FITTED",
    "INIT_LOG_MESSAGE",
    "INIT_LOG_TEMPLATE",
    "NO_EXPLAINER_MSG",
    "NO_MODEL_MSG",
    "STANDARD_AUDIT_TEMPLATE",
    "STATUS_CLEAN",
    "STATUS_DIRTY",
    "STATUS_NO_GIT",
    "TEMPLATES_PACKAGE",
]
