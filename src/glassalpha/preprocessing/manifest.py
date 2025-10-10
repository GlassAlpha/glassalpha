"""Preprocessing manifest schema and canonicalization."""

import hashlib
import json
from typing import Any

MANIFEST_SCHEMA_VERSION = "1.0.0"


def params_hash(manifest: dict[str, Any]) -> str:
    """Compute deterministic hash over manifest parameters.

    This produces a stable hash that doesn't change with pickle metadata,
    operating system, or environment differences.

    Args:
        manifest: Extracted preprocessing manifest

    Returns:
        Hash string (e.g., "sha256:abc123...")

    """
    # Canonical JSON: sorted keys, no whitespace
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    hash_value = hashlib.sha256(payload.encode()).hexdigest()
    return f"sha256:{hash_value}"
