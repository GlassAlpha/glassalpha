"""Dataset registry for automatic fetching and caching.

This module provides a registry system for datasets that can be automatically
downloaded, processed, and cached for reproducible ML auditing.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """Specification for a registered dataset.

    Attributes:
        key: Unique identifier for the dataset (e.g., "german_credit")
        default_relpath: Default relative path for the processed file (e.g., "german_credit_processed.csv")
        fetch_fn: Function that downloads/processes the dataset and returns the processed file path
        schema_version: Version of the dataset schema/format (for compatibility checking)
        checksum: Optional SHA256 checksum for integrity verification (Phase 3)

    """

    key: str
    default_relpath: str
    fetch_fn: Callable[[], Path]
    schema_version: str
    checksum: str | None = None


# Global registry of available datasets
# Maps dataset key to DatasetSpec
REGISTRY: dict[str, DatasetSpec] = {}
