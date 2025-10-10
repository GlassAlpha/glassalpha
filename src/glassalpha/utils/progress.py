"""Progress bar utilities for long-running operations.

Provides tqdm-based progress bars that auto-detect Jupyter notebook vs terminal
environment and respect configuration settings for strict mode and environment variables.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

# Try to import tqdm, but gracefully degrade if unavailable
try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def get_progress_bar(
    iterable: Iterable[Any] | None = None,
    desc: str | None = None,
    total: int | None = None,
    disable: bool = False,
    leave: bool = True,
    **kwargs: Any,
) -> Any:
    """Get a progress bar for iterables or manual updates.

    Auto-detects Jupyter notebook vs terminal environment and respects
    configuration settings for strict mode and environment variables.

    Args:
        iterable: Optional iterable to wrap with progress bar
        desc: Description to display
        total: Total number of iterations (if iterable not provided)
        disable: Force disable progress bar
        leave: Whether to leave progress bar on screen after completion
        **kwargs: Additional arguments to pass to tqdm

    Returns:
        tqdm progress bar if available, otherwise a passthrough wrapper

    Environment Variables:
        GLASSALPHA_NO_PROGRESS: Set to "1" to disable all progress bars

    Examples:
        >>> # Wrap an iterable
        >>> for item in get_progress_bar(items, desc="Processing"):
        ...     process(item)
        >>>
        >>> # Manual updates
        >>> pbar = get_progress_bar(total=100, desc="Computing")
        >>> for i in range(100):
        ...     compute()
        ...     pbar.update(1)
        >>> pbar.close()

    """
    # Check environment variable override
    env_disabled = os.environ.get("GLASSALPHA_NO_PROGRESS", "0") == "1"

    # Combine all disable conditions
    should_disable = disable or env_disabled or not TQDM_AVAILABLE

    if should_disable:
        # Return passthrough wrapper that does nothing
        return _PassthroughProgressBar(iterable)

    # Create tqdm progress bar with explicit cleanup settings
    # mininterval=1.0 reduces monitor thread wake-ups
    # delay=0.5 prevents spawning monitor for very short operations
    return tqdm(
        iterable=iterable,
        desc=desc,
        total=total,
        disable=False,
        leave=leave,
        mininterval=1.0,  # Update display at most once per second
        delay=0.5,  # Don't create monitor thread for operations <0.5s
        **kwargs,
    )


def is_progress_enabled(strict_mode: bool = False) -> bool:
    """Check if progress bars should be enabled.

    Args:
        strict_mode: Whether running in strict regulatory mode

    Returns:
        True if progress bars should be shown

    """
    # Disable in strict mode (professional audit output)
    if strict_mode:
        return False

    # Check environment variable
    if os.environ.get("GLASSALPHA_NO_PROGRESS", "0") == "1":
        return False

    # Check tqdm availability
    return TQDM_AVAILABLE


class _PassthroughProgressBar:
    """Passthrough wrapper when tqdm is unavailable or disabled.

    Provides same interface as tqdm but does nothing.
    """

    def __init__(self, iterable: Iterable[Any] | None = None) -> None:
        """Initialize passthrough progress bar.

        Args:
            iterable: Optional iterable to wrap

        """
        self.iterable = iterable
        self.n = 0

    def __iter__(self):
        """Iterate over wrapped iterable."""
        if self.iterable is not None:
            yield from self.iterable

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""

    def update(self, n: int = 1) -> None:
        """Update progress (no-op).

        Args:
            n: Number of iterations to increment

        """
        self.n += n

    def close(self) -> None:
        """Close progress bar (no-op)."""

    def set_description(self, desc: str) -> None:
        """Set description (no-op).

        Args:
            desc: Description text

        """

    def refresh(self) -> None:
        """Refresh progress bar display (no-op)."""
