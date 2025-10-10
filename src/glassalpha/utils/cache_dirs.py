"""OS-aware cache directory resolution for GlassAlpha datasets.

This module provides platform-specific cache directory resolution that follows
OS conventions and respects environment variable overrides.

Cache Path Resolution
---------------------
GlassAlpha canonicalizes all cache directory paths to absolute real paths
(symlinks resolved). This applies to:
- Environment variable overrides (GLASSALPHA_DATA_DIR)
- Platform-specific defaults (via platformdirs when available)
- Manual OS-specific fallbacks

The canonical path is determined once during resolution and logged to help
users understand path mappings on systems like macOS where /tmp maps to
/private/tmp.

Example:
    >>> import os
    >>> from glassalpha.utils.cache_dirs import resolve_data_root
    >>> os.environ['GLASSALPHA_DATA_DIR'] = '/tmp/my-cache'
    >>> resolve_data_root()
    PosixPath('/private/tmp/my-cache')  # On macOS

The system logs both requested and effective paths for transparency:
    Cache dir requested via GLASSALPHA_DATA_DIR: /tmp/my-cache | effective: /private/tmp/my-cache

"""

import os
import platform
from pathlib import Path

# Try to use platformdirs for better OS path resolution
try:
    from platformdirs import user_data_dir
except ImportError:
    user_data_dir = None

APP_NAME = "glassalpha"


def resolve_data_root() -> Path:
    """Return a canonical absolute path for the cache root.
    No filesystem writes here.
    """
    raw_env = os.getenv("GLASSALPHA_DATA_DIR")
    if raw_env:
        # Expand and canonicalize even if the final directory doesn't exist yet.
        return Path(raw_env).expanduser().resolve()

    if user_data_dir:
        base = Path(user_data_dir(appname=APP_NAME, appauthor=False, roaming=True))
        return (base / "data").resolve()

    # Fallback without platformdirs
    home = Path.home()
    sys = platform.system()
    if sys == "Darwin":
        return (home / "Library" / "Application Support" / APP_NAME / "data").resolve()
    if sys == "Windows":
        appdata = os.getenv("APPDATA")
        base = Path(appdata) if appdata else (home / "AppData" / "Roaming")
        return (base / APP_NAME / "data").resolve()
    xdg = os.getenv("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else (home / ".local" / "share")
    return (base / APP_NAME / "data").resolve()


def ensure_dir_writable(path: Path, mode: int = 0o700) -> Path:
    """Ensure a directory exists and is writable.

    Args:
        path: Directory path to create and test
        mode: Directory permissions (default: 0o700 for security)

    Returns:
        The path (now guaranteed to exist and be writable)

    Raises:
        RuntimeError: If the directory cannot be created or written to

    """
    import time

    path = Path(path).resolve()

    # Retry logic for concurrent directory creation
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Create directory with parents
            path.mkdir(parents=True, exist_ok=True)

            # Small delay to let filesystem settle (helps with concurrent creation)
            if attempt > 0:
                time.sleep(0.01)

            # Verify directory actually exists after creation
            if not path.exists():
                if attempt < max_attempts - 1:
                    time.sleep(0.05)
                    continue
                raise RuntimeError(f"Failed to create directory: {path}")

            if not path.is_dir():
                raise RuntimeError(f"Path exists but is not a directory: {path}")

            # Set restrictive permissions on POSIX systems
            if os.name != "nt":  # Not Windows
                try:
                    path.chmod(mode)
                except OSError:
                    # Ignore permission errors on some systems
                    pass

            # Test writability by creating a temporary file
            test_file = path / ".write_test"
            test_file.write_text("ok")
            test_file.unlink()

            # Success!
            return path

        except FileNotFoundError as e:
            # Transient error during concurrent creation
            if attempt < max_attempts - 1:
                time.sleep(0.05 * (attempt + 1))
                continue
            raise RuntimeError(f"Cannot create or write to cache directory {path}: {e}") from e
        except (OSError, PermissionError) as e:
            raise RuntimeError(f"Cannot create or write to cache directory {path}: {e}") from e

    # Should never reach here
    raise RuntimeError(f"Failed to create writable directory after {max_attempts} attempts: {path}")


def get_cache_path(dataset_key: str, filename: str) -> Path:
    """Get the full cache path for a dataset file.

    Args:
        dataset_key: The dataset identifier (e.g., "german_credit")
        filename: The filename within the dataset (e.g., "processed.csv")

    Returns:
        Full path to the cached file (directory may not exist yet)

    Note:
        This function only resolves the path. Use ensure_dir_writable() to create
        the directory if needed.

    """
    cache_root = resolve_data_root()
    return (cache_root / dataset_key / filename).resolve()
