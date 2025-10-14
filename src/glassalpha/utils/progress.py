"""Progress utilities for long-running operations.

Provides simple progress indicators that respect configuration settings
for strict mode and environment variables. Uses logging for progress updates.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


def get_progress_bar(
    iterable: Iterable[Any] | None = None,
    desc: str | None = None,
    total: int | None = None,
    disable: bool = False,
    leave: bool = True,
    **kwargs: Any,
) -> Any:
    """Get a progress indicator for iterables or manual updates.

    Provides simple progress updates via logging that respect
    configuration settings for strict mode and environment variables.

    Args:
        iterable: Optional iterable to wrap with progress indicator
        desc: Description to display
        total: Total number of iterations (if iterable not provided)
        disable: Force disable progress updates
        leave: Ignored (for API compatibility)
        **kwargs: Additional arguments (ignored, for API compatibility)

    Returns:
        Progress wrapper that logs periodic updates

    Environment Variables:
        GLASSALPHA_NO_PROGRESS: Set to "1" to disable all progress updates

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

    # Return simple progress wrapper
    return _ProgressWrapper(
        iterable=iterable,
        desc=desc,
        total=total,
        disabled=disable or env_disabled,
    )


def is_progress_enabled(strict_mode: bool = False) -> bool:
    """Check if progress updates should be enabled.

    Args:
        strict_mode: Whether running in strict regulatory mode

    Returns:
        True if progress updates should be shown

    """
    # Disable in strict mode (professional audit output)
    if strict_mode:
        return False

    # Check environment variable
    if os.environ.get("GLASSALPHA_NO_PROGRESS", "0") == "1":
        return False

    return True


class _ProgressWrapper:
    """Simple progress wrapper with logging-based updates.

    Provides same interface as tqdm for API compatibility but uses
    simple logging instead of terminal progress bars.
    """

    def __init__(
        self,
        iterable: Iterable[Any] | None = None,
        desc: str | None = None,
        total: int | None = None,
        disabled: bool = False,
    ) -> None:
        """Initialize progress wrapper.

        Args:
            iterable: Optional iterable to wrap
            desc: Description for progress updates
            total: Total number of iterations expected
            disabled: Whether to disable all updates

        """
        self.iterable = iterable
        self.desc = desc or "Progress"
        self.total = total
        self.disabled = disabled
        self.n = 0
        self._log_interval = 10  # Log every 10% or 10 items

    def __iter__(self) -> Any:
        """Iterate over wrapped iterable with progress logging."""
        if self.iterable is None:
            return

        for i, item in enumerate(self.iterable, 1):
            self.n = i
            self._maybe_log()
            yield item

        if not self.disabled and self.total and self.n == self.total:
            logger.info("%s: %d/%d (100%%)", self.desc, self.n, self.total)

    def __enter__(self) -> _ProgressWrapper:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    def update(self, n: int = 1) -> None:
        """Update progress.

        Args:
            n: Number of iterations to increment

        """
        self.n += n
        self._maybe_log()

    def close(self) -> None:
        """Close progress and log final state."""
        if not self.disabled and self.total and self.n > 0:
            pct = int(100 * self.n / self.total)
            logger.info("%s: %d/%d (%d%%)", self.desc, self.n, self.total, pct)

    def set_description(self, desc: str) -> None:
        """Set description.

        Args:
            desc: Description text

        """
        self.desc = desc

    def refresh(self) -> None:
        """Refresh progress display (no-op for compatibility)."""

    def _maybe_log(self) -> None:
        """Log progress if at interval threshold."""
        if self.disabled or not self.total:
            return

        # Log at 10% intervals or every 10 items if total < 100
        if self.total >= 100:
            interval = self.total // 10
        else:
            interval = max(1, self.total // 10)

        if self.n % interval == 0 or self.n == self.total:
            pct = int(100 * self.n / self.total)
            logger.info("%s: %d/%d (%d%%)", self.desc, self.n, self.total, pct)
