"""Cross-platform file locking utilities.

This module provides file locking functionality that works across different
operating systems using atomic file creation for coordination.
"""

import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def file_lock(lock_path: Path, timeout_s: float = 60.0, retry_ms: int = 100) -> Generator[None, None, None]:
    """Context manager for cross-platform file locking.

    Uses atomic file creation (O_CREAT | O_EXCL) to coordinate access between
    processes. This approach works reliably across Unix-like systems and Windows.

    Args:
        lock_path: Path to the lock file to create
        timeout_s: Maximum time to wait for lock acquisition in seconds (default: 60.0)
                  Use 0 for non-blocking try-lock
        retry_ms: Milliseconds to wait between retry attempts (default: 100ms)

    Raises:
        TimeoutError: If lock cannot be acquired within timeout_s seconds
        OSError: If lock file cannot be created or removed

    Example:
        >>> with file_lock(Path("/tmp/my_lock"), timeout_s=30.0):
        ...     # Critical section - only one process can execute this
        ...     do_something_important()

    Note:
        Uses time.monotonic() for reliable timeouts that aren't affected by
        system clock adjustments.

    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + float(timeout_s)
    fd = None

    try:
        while True:
            try:
                # O_CREAT | O_EXCL gives us atomic "create if not exists"
                # This works on both Unix and Windows
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)

                # Optional: write PID into lock file for debugging
                try:
                    os.write(fd, f"{os.getpid()}".encode())
                except OSError:
                    pass  # Not critical if this fails

                break  # Lock acquired successfully

            except FileExistsError:
                # Lock is held by another process
                if timeout_s == 0 or time.monotonic() >= deadline:
                    raise TimeoutError(f"Lock timed out: {lock_path}")

                # Wait before retrying
                time.sleep(retry_ms / 1000.0)

        # Lock acquired, execute critical section
        yield

    finally:
        # Best-effort cleanup: always try to release the lock
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass  # Already closed or invalid

        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass  # Lock file already removed


def get_lock_path(target_path: Path) -> Path:
    """Get the appropriate lock file path for a target file.

    Creates a lock file with the same name as the target but with .lock extension.

    Args:
        target_path: Path to the file being locked

    Returns:
        Path to the corresponding lock file

    """
    return target_path.with_suffix(target_path.suffix + ".lock")
