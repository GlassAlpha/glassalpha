"""Logging utilities to enforce contract compliance.

This module provides standardized logging functions that prevent
regression to printf-style logging and ensure exact string matching
for contract tests.
"""

import logging
from typing import Any

from .constants import INIT_LOG_TEMPLATE


def log_pipeline_init(logger: logging.Logger, profile: str) -> None:
    """Log pipeline initialization with exact contract compliance.

    Args:
        logger: Logger instance to use
        profile: Audit profile name

    Note:
        Uses single-argument f-string call to match contract test expectations.

    """
    logger.info(INIT_LOG_TEMPLATE.format(profile=profile))


def log_info_single_arg(logger: logging.Logger, message: str, **kwargs: Any) -> None:  # noqa: ANN401
    """Log info message with single argument to prevent printf drift.

    Args:
        logger: Logger instance to use
        message: Pre-formatted message string
        **kwargs: Additional context for formatting (applied to message)

    """
    formatted_msg = message.format(**kwargs) if kwargs else message
    logger.info(formatted_msg)


def log_debug_single_arg(logger: logging.Logger, message: str, **kwargs: Any) -> None:  # noqa: ANN401
    """Log debug message with single argument to prevent printf drift.

    Args:
        logger: Logger instance to use
        message: Pre-formatted message string
        **kwargs: Additional context for formatting (applied to message)

    """
    formatted_msg = message.format(**kwargs) if kwargs else message
    logger.debug(formatted_msg)
