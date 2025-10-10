"""Standardized exit codes for GlassAlpha CLI.

This module defines consistent exit codes across all CLI commands to enable
reliable scripting and CI/CD integration.

Exit Code Schema:
    0: Success - Command completed successfully
    1: User Error - Configuration issues, missing files, invalid inputs
    2: System Error - Permissions, resources, environment issues
    3: Validation Error - Strict mode or validation failures

Examples:
    Success case:
        >>> raise typer.Exit(ExitCode.SUCCESS)

    User made a mistake:
        >>> raise typer.Exit(ExitCode.USER_ERROR)

    System problem (permission denied):
        >>> raise typer.Exit(ExitCode.SYSTEM_ERROR)

    Validation failed in strict mode:
        >>> raise typer.Exit(ExitCode.VALIDATION_ERROR)

"""

from enum import IntEnum


class ExitCode(IntEnum):
    """Standard exit codes for GlassAlpha CLI commands.

    These codes follow Unix conventions and enable reliable scripting:
    - 0 indicates success
    - 1-127 indicate various error conditions

    Attributes:
        SUCCESS: Command completed successfully (exit code 0)
        USER_ERROR: User input or configuration error (exit code 1)
        SYSTEM_ERROR: System-level error like permissions or resources (exit code 2)
        VALIDATION_ERROR: Validation or compliance check failure (exit code 3)

    """

    SUCCESS = 0
    """Command completed successfully."""

    USER_ERROR = 1
    """User error: bad configuration, missing file, invalid input.

    Examples:
        - Configuration file not found
        - Invalid YAML syntax
        - Missing required fields
        - Invalid model type
        - Dataset schema mismatch
    """

    SYSTEM_ERROR = 2
    """System error: permissions, resources, environment.

    Examples:
        - Permission denied writing output
        - Out of memory
        - Disk full
        - Required system library missing
    """

    VALIDATION_ERROR = 3
    """Validation or compliance check failure.

    Examples:
        - Strict mode validation failed
        - Required component unavailable (with --no-fallback)
        - Reproducibility check failed
        - Audit profile requirements not met
    """


# Convenience constants for direct import
SUCCESS = ExitCode.SUCCESS
USER_ERROR = ExitCode.USER_ERROR
SYSTEM_ERROR = ExitCode.SYSTEM_ERROR
VALIDATION_ERROR = ExitCode.VALIDATION_ERROR


def get_exit_code_description(code: int) -> str:
    """Get human-readable description of an exit code.

    Args:
        code: Exit code integer (0-3)

    Returns:
        Human-readable description of what the exit code means

    Examples:
        >>> get_exit_code_description(0)
        'Success'
        >>> get_exit_code_description(1)
        'User Error'
        >>> get_exit_code_description(2)
        'System Error'
        >>> get_exit_code_description(3)
        'Validation Error'

    """
    descriptions = {
        ExitCode.SUCCESS: "Success",
        ExitCode.USER_ERROR: "User Error",
        ExitCode.SYSTEM_ERROR: "System Error",
        ExitCode.VALIDATION_ERROR: "Validation Error",
    }
    return descriptions.get(code, f"Unknown Error ({code})")


def is_error(code: int) -> bool:
    """Check if exit code indicates an error.

    Args:
        code: Exit code to check

    Returns:
        True if code indicates error (non-zero), False if success

    Examples:
        >>> is_error(ExitCode.SUCCESS)
        False
        >>> is_error(ExitCode.USER_ERROR)
        True

    """
    return code != ExitCode.SUCCESS
