"""Safe output helpers for CLI commands.

Handles BrokenPipeError when users pipe output to head/tail or cancel with Ctrl+C.
This is standard Unix CLI behavior - commands should exit gracefully without stack traces.
"""

import sys
from typing import Any

import typer


def safe_echo(message: Any = "", **kwargs) -> None:
    """Echo message, handling BrokenPipeError gracefully.

    Args:
        message: Message to echo
        **kwargs: Additional arguments passed to typer.echo

    This wrapper catches BrokenPipeError which occurs when:
    - User pipes output to head/tail: `glassalpha audit | head`
    - User cancels with Ctrl+C while output is streaming
    - Output is redirected to a closed file descriptor

    In all these cases, the command should exit silently (standard Unix behavior).
    """
    try:
        typer.echo(message, **kwargs)
    except BrokenPipeError:
        # Standard Unix behavior: exit silently when pipe breaks
        # Disable further output to avoid "Exception ignored in __del__" messages
        devnull = open("/dev/null", "w")
        sys.stdout = devnull
        sys.stderr = devnull
        pass


def safe_secho(message: Any = "", **kwargs) -> None:
    """Echo styled message, handling BrokenPipeError gracefully.

    Args:
        message: Message to echo
        **kwargs: Additional arguments passed to typer.secho (fg, bg, bold, etc.)
    """
    try:
        typer.secho(message, **kwargs)
    except BrokenPipeError:
        devnull = open("/dev/null", "w")
        sys.stdout = devnull
        sys.stderr = devnull
        pass
