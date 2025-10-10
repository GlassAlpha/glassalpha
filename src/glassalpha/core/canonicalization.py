"""Canonical representation for deterministic hashing.

Phase 4: Byte-identical reproducibility via strict canonicalization.

Key principles:
- NaN → "NaN" (string)
- Infinity → "Infinity" / "-Infinity" (strings)
- -0.0 → 0.0 (normalized)
- Floats → 17 decimal places (full precision)
- Dicts → sorted by key
- Arrays → lists with shape metadata
- Datetimes → ISO 8601 UTC
- Bytes → base64 strings
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd


def canonicalize(obj: Any) -> Any:
    """Convert object to canonical form for deterministic hashing.

    Rules:
    - float: NaN → "NaN", Inf → "Infinity", -Inf → "-Infinity", -0.0 → 0.0
    - np.ndarray: Convert to list with {"_type": "ndarray", "data": [...], "shape": [...], "dtype": str}
    - dict: Sort keys recursively
    - list/tuple: Recursively canonicalize elements
    - datetime: ISO 8601 UTC string (naive treated as UTC)
    - bytes: Base64 string with {"_type": "bytes", "data": "..."}
    - pd.Timestamp: Convert to datetime then ISO 8601
    - None, bool, int, str: Pass through

    Args:
        obj: Object to canonicalize

    Returns:
        Canonical representation (JSON-serializable)

    Examples:
        >>> canonicalize(np.nan)
        "NaN"
        >>> canonicalize(np.inf)
        "Infinity"
        >>> canonicalize(-0.0)
        0.0
        >>> canonicalize(np.array([1, 2, 3]))
        {"_type": "ndarray", "data": [1, 2, 3], "shape": [3], "dtype": "int64"}

    """
    # None, bool (before int check since bool is subclass of int)
    if obj is None or isinstance(obj, bool):
        return obj

    # Float (includes np.float64, etc.)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj):
            return "NaN"
        if np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        # Normalize -0.0 to 0.0
        if obj == 0.0 and np.signbit(obj):
            return 0.0
        # Round to 17 decimal places for consistency
        return round(float(obj), 17)

    # Int (includes np.int64, etc.)
    if isinstance(obj, (int, np.integer)):
        return int(obj)

    # String
    if isinstance(obj, str):
        return obj

    # Bytes
    if isinstance(obj, bytes):
        return {
            "_type": "bytes",
            "data": base64.b64encode(obj).decode("ascii"),
        }

    # NumPy array
    if isinstance(obj, np.ndarray):
        return {
            "_type": "ndarray",
            "data": [canonicalize(x) for x in obj.ravel()],
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }

    # Pandas Timestamp
    if isinstance(obj, pd.Timestamp):
        dt = obj.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.isoformat()

    # Python datetime
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=UTC)
        return obj.isoformat()

    # Dict (recursively canonicalize and sort keys)
    if isinstance(obj, dict):
        return {k: canonicalize(v) for k, v in sorted(obj.items())}

    # List/tuple (recursively canonicalize)
    if isinstance(obj, (list, tuple)):
        return [canonicalize(x) for x in obj]

    # Mapping (convert to dict then canonicalize)
    if isinstance(obj, Mapping):
        return {k: canonicalize(v) for k, v in sorted(obj.items())}

    # Sequence (convert to list then canonicalize)
    if isinstance(obj, Sequence):
        return [canonicalize(x) for x in obj]

    # Fallback: try to convert to string
    return str(obj)


def compute_result_id(result_dict: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of result dictionary.

    Uses canonical JSON representation (sorted keys, NaN→"NaN", Inf→"Infinity").
    Excludes 'manifest' key (provenance metadata, not part of result identity).

    Args:
        result_dict: Result dictionary with performance, fairness, calibration, etc.

    Returns:
        64-character hex SHA-256 hash

    Examples:
        >>> result_dict = {"performance": {"accuracy": 0.847}, "fairness": {}}
        >>> result_id = compute_result_id(result_dict)
        >>> len(result_id)
        64
        >>> result_id.startswith("a")  # Deterministic
        True

    """
    import hashlib
    import json

    # Exclude manifest (not part of result identity)
    hashable_dict = {k: v for k, v in result_dict.items() if k != "manifest"}

    # Canonicalize
    canonical = canonicalize(hashable_dict)

    # Serialize to JSON (sorted keys, no whitespace)
    json_bytes = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")

    # SHA-256 hash
    return hashlib.sha256(json_bytes).hexdigest()


def hash_data_for_manifest(
    data: pd.DataFrame | pd.Series | np.ndarray,
    *,
    prefix: bool = True,
) -> str:
    """Compute SHA-256 hash of data with dtype awareness.

    Uses streaming hash for memory efficiency. Includes column names,
    dtypes, and data values in hash computation.

    Handles:
    - Categorical dtypes (categories + codes)
    - String dtypes (utf-8 encoded)
    - Boolean dtypes (uint8)
    - Datetime dtypes (int64 view)
    - Timedelta dtypes (int64 view)
    - Numeric dtypes (raw bytes)
    - RangeIndex (reconstructed deterministically)

    Args:
        data: DataFrame, Series, or ndarray to hash
        prefix: If True, prepend "sha256:" to hash (default: True)

    Returns:
        SHA-256 hash, optionally with "sha256:" prefix

    Raises:
        ValueError: If MultiIndex detected (not supported)

    Examples:
        >>> df = pd.DataFrame({"a": [1, 2, 3]})
        >>> hash_data_for_manifest(df)
        "sha256:abc123..."
        >>> hash_data_for_manifest(df, prefix=False)
        "abc123..."

    """
    import hashlib

    hasher = hashlib.sha256()

    # Convert Series to DataFrame for uniform handling
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Convert ndarray to DataFrame
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = pd.DataFrame({"values": data})
        elif data.ndim == 2:
            data = pd.DataFrame(data, columns=[f"col_{i}" for i in range(data.shape[1])])
        else:
            msg = f"ndarray with ndim={data.ndim} not supported (max 2D)"
            raise ValueError(msg)

    # Check for MultiIndex
    if isinstance(data.index, pd.MultiIndex):
        msg = "MultiIndex not supported. Use reset_index() to flatten."
        raise ValueError(msg)

    # Hash column names (order matters)
    for col in data.columns:
        hasher.update(str(col).encode("utf-8"))

    # Hash each column (dtype-aware)
    for col in data.columns:
        series = data[col]
        dtype = series.dtype

        # Hash dtype string
        hasher.update(str(dtype).encode("utf-8"))

        # Categorical: hash categories + codes
        if isinstance(dtype, pd.CategoricalDtype):
            categories = dtype.categories.to_numpy()
            codes = series.cat.codes.to_numpy()

            # Hash categories
            for cat in categories:
                hasher.update(str(cat).encode("utf-8"))

            # Hash codes
            hasher.update(codes.tobytes())

        # String: hash as utf-8
        elif pd.api.types.is_string_dtype(dtype):
            for val in series:
                if pd.isna(val):
                    hasher.update(b"__NA__")
                else:
                    hasher.update(str(val).encode("utf-8"))

        # Boolean: convert to uint8
        elif pd.api.types.is_bool_dtype(dtype):
            hasher.update(series.astype("uint8").to_numpy().tobytes())

        # Datetime: use int64 representation
        elif pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
            hasher.update(series.astype("int64").to_numpy().tobytes())

        # Numeric: hash raw bytes
        else:
            hasher.update(series.to_numpy().tobytes())

    # Hash index (unless RangeIndex with default start/stop/step)
    if isinstance(data.index, pd.RangeIndex):
        # Only hash if non-default (start, stop, step)
        if data.index.start != 0 or data.index.step != 1:
            hasher.update(f"RangeIndex({data.index.start},{data.index.stop},{data.index.step})".encode())
    else:
        # Hash index values
        for val in data.index:
            hasher.update(str(val).encode("utf-8"))

    hash_hex = hasher.hexdigest()
    return f"sha256:{hash_hex}" if prefix else hash_hex


def _atomic_write(path: str, content: bytes) -> None:
    """Write file atomically using temp file + rename.

    This prevents corrupted files if write fails mid-operation.
    Uses os.replace() which is atomic on POSIX systems.

    Args:
        path: Target file path
        content: Bytes to write

    Raises:
        OSError: If write fails

    """
    import os
    import tempfile
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (ensures same filesystem)
    fd, temp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        os.write(fd, content)
        os.close(fd)
        # Atomic rename (POSIX guarantees atomicity)
        os.replace(temp_path, target)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
