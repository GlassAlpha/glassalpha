"""Process utilities for GlassAlpha.

This module provides safe subprocess wrappers that enforce text mode
and prevent decode/encode issues that have caused CI failures.
"""

import subprocess


def run_text(*args: str) -> str | None:
    """Run subprocess command with text mode only - no decode() needed.

    Contract compliance: Uses text=True exclusively and catches FileNotFoundError
    to prevent the decode issues that caused git info test failures.

    Args:
        *args: Command arguments to pass to subprocess.run

    Returns:
        Stripped stdout string if successful, None if command not found

    Note:
        Never use decode method - text=True returns strings directly.

    """
    try:
        process = subprocess.run(  # noqa: S603
            args,
            capture_output=True,
            check=False,
            text=True,  # Critical: ensures stdout is str, not bytes
            encoding="utf-8",  # Explicit encoding for safety
        )
        return (process.stdout or "").strip()
    except FileNotFoundError:
        # Command not found (e.g., git not installed)
        return None
    except OSError:
        # Other OS-level errors (permissions, etc.)
        return None


def run_text_with_success(*args: str) -> tuple[str | None, bool]:
    """Run subprocess command and return output with success status.

    Args:
        *args: Command arguments to pass to subprocess.run

    Returns:
        Tuple of (output_or_none, success_flag)

    """
    try:
        process = subprocess.run(  # noqa: S603
            args,
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
        )
        output = (process.stdout or "").strip()
        success = process.returncode == 0
        return output, success
    except (FileNotFoundError, OSError):
        return None, False


def check_command_available(command: str) -> bool:
    """Check if a command is available in the system PATH.

    Args:
        command: Command name to check (e.g., 'git', 'python')

    Returns:
        True if command is available, False otherwise

    """
    result, success = run_text_with_success(command, "--version")
    return success and result is not None


def run_command_safely(*args: str, **kwargs) -> subprocess.CompletedProcess[str]:
    """Run subprocess command safely with text mode and error handling.

    Args:
        *args: Command arguments to pass to subprocess.run
        **kwargs: Additional keyword arguments for subprocess.run

    Returns:
        CompletedProcess object with stdout/stderr as strings

    Raises:
        subprocess.SubprocessError: If the command fails and check=True

    """
    # Ensure text mode is enabled for safety
    run_kwargs = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        **kwargs,
    }

    return subprocess.run(args, **run_kwargs)
