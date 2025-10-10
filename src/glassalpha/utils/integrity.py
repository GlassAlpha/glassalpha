"""Dataset integrity and metadata management for reproducible audits.

This module provides utilities for creating and validating dataset integrity
metadata, including checksums, schema versions, and provenance information.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import __version__


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata for a dataset including integrity information.

    Attributes:
        dataset_key: Identifier for the dataset (e.g., "german_credit")
        schema_version: Version of the dataset schema/format
        source_url: URL where the dataset was originally obtained
        timestamp: When the dataset was processed/cached
        glassalpha_version: Version of GlassAlpha that processed the dataset
        row_count: Number of rows in the dataset
        column_count: Number of columns in the dataset
        columns: List of column names
        sha256: SHA256 checksum of the dataset file
        processing_notes: Optional notes about processing steps

    """

    dataset_key: str
    schema_version: str
    source_url: str
    timestamp: datetime
    glassalpha_version: str
    row_count: int
    column_count: int
    columns: list[str]
    sha256: str
    processing_notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO format
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetMetadata":
        """Create from dictionary."""
        # Convert timestamp back from ISO format
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file.

    Args:
        file_path: Path to the file to checksum

    Returns:
        SHA256 checksum as hexadecimal string

    """
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def create_dataset_metadata(
    dataset_key: str,
    schema_version: str,
    source_url: str,
    file_path: Path,
    columns: list[str],
    processing_notes: str | None = None,
) -> DatasetMetadata:
    """Create metadata for a processed dataset.

    Args:
        dataset_key: Dataset identifier
        schema_version: Schema version
        source_url: Source URL
        file_path: Path to the processed dataset file
        columns: Column names in the dataset
        processing_notes: Optional processing notes

    Returns:
        DatasetMetadata object with integrity information

    """
    # Calculate checksum
    sha256 = calculate_sha256(file_path)

    # Get row count by reading the CSV
    import pandas as pd

    df = pd.read_csv(file_path)
    row_count = len(df)

    return DatasetMetadata(
        dataset_key=dataset_key,
        schema_version=schema_version,
        source_url=source_url,
        timestamp=datetime.now(UTC),
        glassalpha_version=__version__,
        row_count=row_count,
        column_count=len(columns),
        columns=columns,
        sha256=sha256,
        processing_notes=processing_notes,
    )


def save_metadata(metadata: DatasetMetadata, file_path: Path) -> None:
    """Save dataset metadata to a sidecar JSON file.

    Args:
        metadata: DatasetMetadata to save
        file_path: Path to the dataset file (metadata saved as .meta.json)

    """
    meta_path = file_path.with_suffix(file_path.suffix + ".meta.json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)


def load_metadata(file_path: Path) -> DatasetMetadata | None:
    """Load dataset metadata from sidecar JSON file.

    Args:
        file_path: Path to the dataset file

    Returns:
        DatasetMetadata if found, None otherwise

    """
    meta_path = file_path.with_suffix(file_path.suffix + ".meta.json")

    if not meta_path.exists():
        return None

    try:
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        return DatasetMetadata.from_dict(data)
    except (json.JSONDecodeError, KeyError):
        # Invalid metadata file
        return None


def verify_dataset_integrity(file_path: Path, expected_metadata: DatasetMetadata | None = None) -> tuple[bool, str]:
    """Verify dataset integrity against metadata.

    Args:
        file_path: Path to the dataset file
        expected_metadata: Expected metadata to verify against (optional)

    Returns:
        Tuple of (is_valid, reason)

    """
    if not file_path.exists():
        return False, f"Dataset file not found: {file_path}"

    # Load existing metadata
    metadata = load_metadata(file_path)
    if metadata is None:
        return False, "No metadata file found"

    # If expected metadata provided, verify it matches
    if expected_metadata:
        if metadata.sha256 != expected_metadata.sha256:
            return False, f"Checksum mismatch: expected {expected_metadata.sha256}, got {metadata.sha256}"

        if metadata.schema_version != expected_metadata.schema_version:
            return (
                False,
                f"Schema version mismatch: expected {expected_metadata.schema_version}, got {metadata.schema_version}",
            )

        if metadata.row_count != expected_metadata.row_count:
            return False, f"Row count mismatch: expected {expected_metadata.row_count}, got {metadata.row_count}"

        if metadata.column_count != expected_metadata.column_count:
            return (
                False,
                f"Column count mismatch: expected {expected_metadata.column_count}, got {metadata.column_count}",
            )

        if metadata.columns != expected_metadata.columns:
            return False, f"Column names mismatch: expected {expected_metadata.columns}, got {metadata.columns}"

    # Verify current checksum matches stored checksum
    current_sha256 = calculate_sha256(file_path)
    if current_sha256 != metadata.sha256:
        return False, f"File has been modified: checksum changed from {metadata.sha256} to {current_sha256}"

    return True, "Dataset integrity verified"
