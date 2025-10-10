"""Audit manifest generation for complete audit trail.

This module creates comprehensive audit manifests that capture every aspect
of the audit process for regulatory compliance and reproducibility.
"""

import json
import logging
import os
import platform
import struct
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from .hashing import hash_config, hash_dataframe, hash_file, hash_object
from .seeds import get_seeds_manifest

logger = logging.getLogger(__name__)


def _safe_platform() -> str:
    """Get platform information safely, avoiding subprocess decode issues."""
    try:
        return platform.platform()
    except (AttributeError, OSError, UnicodeDecodeError):
        # Fallback to basic system info when platform.platform() fails
        return f"{platform.system()} {platform.release()}"


class EnvironmentInfo(BaseModel):
    """System and environment information."""

    python_version: str
    platform: str
    architecture: str
    processor: str
    hostname: str
    user: str
    working_directory: str
    environment_variables: dict[str, str] = Field(default_factory=dict)


class GitInfo(BaseModel):
    """Git repository information."""

    commit_sha: str | None = None  # Contract compliance: use commit_sha
    branch: str | None = None
    status: str = "clean"  # Contract compliance: "clean" or "dirty"
    is_dirty: bool = False
    remote_url: str | None = None
    commit_message: str | None = None
    commit_timestamp: str | None = None

    # Backward compatibility
    @property
    def commit_hash(self) -> str | None:
        """Alias for commit_sha for backward compatibility."""
        return self.commit_sha


class ComponentInfo(BaseModel):
    """Information about selected components."""

    name: str
    type: str  # 'model', 'explainer', 'metric'
    version: str | None = None
    priority: int | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class DataInfo(BaseModel):
    """Information about datasets used."""

    path: str | None = None
    hash: str | None = None
    shape: tuple[int, int] | None = None
    columns: list[str] = Field(default_factory=list)
    missing_values: dict[str, int] = Field(default_factory=dict)
    target_column: str | None = None
    sensitive_features: list[str] = Field(default_factory=list)


class ExecutionInfo(BaseModel):
    """Information about execution."""

    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    duration_seconds: float | None = None
    status: str = "pending"  # pending, running, completed, failed - tests expect "pending" default
    error_message: str | None = None


class ManifestComponent(BaseModel):
    """Component information for manifest."""

    name: str
    type: str
    details: dict[str, Any] = Field(default_factory=dict)


class AuditManifest(BaseModel):
    """Complete audit manifest with all lineage information."""

    # Basic metadata
    manifest_version: str = "1.0"
    audit_id: str
    version: str = "1.0.0"  # Tests expect this field to exist
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())  # String for test compatibility
    creation_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Configuration
    config: dict[str, Any] = Field(default_factory=dict)
    config_hash: str | None = None
    data_hash: str | None = None  # Root-level data hash for e2e tests
    audit_profile: str | None = None
    strict_mode: bool = False

    # Root-level status fields (tests expect these)
    status: str = "pending"
    error_message: str | None = None

    # Environment with defaults (tests expect this to be populated)
    environment: EnvironmentInfo = Field(
        default_factory=lambda: EnvironmentInfo(
            python_version=platform.python_version(),
            platform=_safe_platform(),
            architecture=f"{8 * struct.calcsize('P')}-bit/{platform.machine() or 'unknown'}",
            processor=platform.processor() or "unknown",
            hostname=platform.node(),
            user=Path.home().name,
            working_directory=str(Path.cwd()),
            environment_variables={},
            installed_packages={},
        ),
    )
    execution: ExecutionInfo = Field(default_factory=ExecutionInfo)
    execution_info: ExecutionInfo = Field(default_factory=ExecutionInfo)  # Alias for tests
    git: GitInfo | None = None

    # Seeds and reproducibility
    seeds: dict[str, Any] = Field(default_factory=dict)
    deterministic_validation: dict[str, bool | None] = Field(default_factory=dict)

    # Components (tests expect this field name)
    components: dict[str, dict[str, Any]] = Field(default_factory=dict)  # Changed to dict for test compatibility
    selected_components: dict[str, dict[str, Any]] = Field(default_factory=dict)  # Support nested structure

    # Data
    datasets: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
    )  # Changed to dict for test compatibility with 'name' field

    # Execution (made optional for minimal construction)

    # Results hashes
    result_hashes: dict[str, str] = Field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        """Export manifest as JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation

        """
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """Export manifest as dictionary.

        Returns:
            Dictionary representation

        """
        return self.model_dump()

    def save(self, path: Path) -> None:
        """Save audit manifest to JSON file for regulatory compliance and reproducibility.

        Creates a permanent record of the audit execution including all component
        selections, data hashes, environment details, and configuration parameters.
        This manifest enables full audit trail documentation required for regulatory
        submissions and reproducible audit verification.

        Args:
            path: Target file path for manifest storage (typically .json extension)

        Side Effects:
            - Creates or overwrites JSON file at specified path
            - File contains sensitive environment information (review before sharing)
            - Timestamps record exact audit execution time in UTC
            - File size typically 5-50KB depending on configuration complexity

        Raises:
            IOError: If path is not writable or insufficient disk space
            JSONEncodeError: If manifest contains non-serializable data

        Note:
            This manifest is required for regulatory audit trails and must be
            preserved with the corresponding audit report for compliance verification.

        """
        path = Path(path)

        with path.open("w") as f:
            f.write(self.to_json())

        logger.info(f"Audit manifest saved to {path}")


class ManifestGenerator:
    """Generator for comprehensive audit manifests."""

    def __init__(self, audit_id: str | None = None, seed: int | None = None) -> None:
        """Initialize manifest generator.

        Args:
            audit_id: Unique audit identifier (auto-generated if None)
            seed: Random seed for deterministic audit ID generation

        """
        self.audit_id = audit_id or self._generate_audit_id(seed=seed)

        # Direct attributes for test compatibility
        self.status: str = "initialized"
        self.error: str | None = None
        self.completed_at: datetime | None = None

        # Attributes tests expect to exist
        self.seeds: dict[str, Any] | None = None
        self.config: dict[str, Any] | None = None
        self.config_hash: str | None = None
        self.components: dict[str, ComponentInfo] = {}
        self.datasets: dict[str, Any] = {}
        self.result_hashes: dict[str, str] = {}

        # Determine creation time (use deterministic timestamp for reproducibility)
        from glassalpha.utils.determinism import get_deterministic_timestamp

        creation_time = get_deterministic_timestamp(seed=seed)
        created_at = creation_time.isoformat()
        self.start_time = creation_time  # Use deterministic time for start_time too

        # Initialize empty manifest (don't collect platform/git on init to avoid crashes)
        self.manifest = AuditManifest(
            audit_id=self.audit_id,
            creation_time=creation_time,
            created_at=created_at,
        )

        # Sync start_time between execution and execution_info fields
        self.manifest.execution.start_time = self.start_time
        self.manifest.execution_info.start_time = self.start_time

        logger.info(f"Initialized audit manifest: {self.audit_id}")

    def add_config(self, config: dict[str, Any]) -> None:
        """Add configuration to manifest.

        Args:
            config: Configuration dictionary

        """
        # Update direct attributes for test compatibility
        self.config = config
        self.config_hash = hash_config(config)

        # Update manifest
        self.manifest.config = config
        self.manifest.config_hash = self.config_hash
        self.manifest.audit_profile = config.get("audit_profile")
        self.manifest.strict_mode = config.get("strict_mode", False)

        logger.debug("Added configuration to manifest")

    def add_seeds(self) -> None:
        """Add seed information to manifest."""
        from .seeds import validate_deterministic_environment  # noqa: PLC0415

        # Update direct attribute for test compatibility
        seeds_data = get_seeds_manifest()
        self.seeds = seeds_data

        # Update manifest
        self.manifest.seeds = seeds_data
        self.manifest.deterministic_validation = validate_deterministic_environment()

        logger.debug("Added seed information to manifest")

    def add_component(
        self,
        name: str,
        implementation: str,
        obj: Any = None,  # noqa: ANN401
        *,
        details: dict | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Add component to manifest with flexible details and kwargs.

        Args:
            name: Component role (e.g. 'model', 'explainer')
            implementation: Component implementation (e.g. "xgboost", "treeshap")
            obj: Optional component object for version extraction
            details: Additional details dictionary
            **kwargs: Additional keyword arguments to store in details

        """
        # Build details dict from all sources
        component_details = details or {}
        component_details.update(kwargs)
        component_details.update(
            {
                "implementation": implementation,
                "version": getattr(obj, "version", "1.0.0"),
            },
        )

        # components: detailed catalog - friend's spec format
        self.manifest.components[name] = {
            "name": name,
            "type": implementation,
            "details": component_details,
        }

        # selected_components: compact summary - friend's spec format
        self.manifest.selected_components[name] = {
            "name": implementation,
            "type": name,
        }

        # Store in generator components for backward compatibility
        if hasattr(self, "components"):
            component = ManifestComponent(
                name=name,
                type=implementation,
                details=component_details,
            )
            self.components[name] = component

        logger.debug(f"Added component to manifest: {name} ({implementation})")

    def add_dataset(
        self,
        dataset_name: str,
        file_path: Path | None = None,
        data: pd.DataFrame | None = None,
        target_column: str | None = None,
        sensitive_features: list[str] | None = None,
    ) -> None:
        """Add dataset information to manifest.

        Args:
            dataset_name: Name of dataset
            data: DataFrame (for shape/hash info)
            file_path: Original file path
            target_column: Target column name
            sensitive_features: List of sensitive feature names

        """
        dataset_info = DataInfo(
            path=str(file_path) if file_path is not None else None,
            target_column=target_column,
            sensitive_features=sensitive_features or [],
        )

        # Add data information if available - friend's spec: use explicit None checks
        if data is not None:
            # Treat data as DataFrame, never inspect .shape on file_path
            dataset_info.hash = hash_dataframe(data)
            # Also set root-level data_hash for e2e tests (use the main dataset's hash)
            if dataset_name == "main" or self.manifest.data_hash is None:
                self.manifest.data_hash = dataset_info.hash
            dataset_info.shape = data.shape
            dataset_info.columns = list(data.columns)
            dataset_info.missing_values = data.isna().sum().to_dict()
        elif file_path is not None and file_path.exists():  # Treat path as path only
            dataset_info.hash = hash_file(file_path)
            # Also set root-level data_hash for e2e tests (use the main dataset's hash)
            if dataset_name == "main" or self.manifest.data_hash is None:
                self.manifest.data_hash = dataset_info.hash

        # Update both manifest and direct attribute for test compatibility
        # Ensure datasets include 'name' field as expected by tests
        dataset_dict = dataset_info.model_dump() if hasattr(dataset_info, "model_dump") else vars(dataset_info)
        dataset_dict["name"] = dataset_name  # Add name field for test compatibility

        self.manifest.datasets[dataset_name] = dataset_dict
        self.datasets[dataset_name] = dataset_dict
        logger.debug(f"Added dataset to manifest: {dataset_name}")

    def add_result_hash(self, result_name: str, result_hash: str) -> None:
        """Add hash of result/output to manifest.

        Args:
            result_name: Name of result (e.g., 'report', 'explanations')
            result_hash: Hash of result

        """
        # Update both manifest and direct attribute for test compatibility
        self.manifest.result_hashes[result_name] = result_hash
        self.result_hashes[result_name] = result_hash
        logger.debug(f"Added result hash: {result_name}")

    def mark_completed(self, status: str = "completed", error: str | None = None) -> None:
        """Mark audit as completed.

        Args:
            status: Final status ('completed', 'failed', 'success')
            error: Error message if failed

        """
        # Validate status
        if status not in {"completed", "failed", "success"}:  # Added "success" for test compatibility
            msg = "status must be 'completed', 'failed', or 'success'"
            raise ValueError(msg)

        end_time = datetime.now(UTC)

        # Update direct attributes for test compatibility
        self.status = status
        self.error = error
        self.completed_at = end_time

        # Calculate duration (fixed in deterministic mode)
        if os.environ.get("GLASSALPHA_DETERMINISTIC"):
            duration_seconds = 0.0  # Fixed duration for deterministic testing
        else:
            duration_seconds = (end_time - self.start_time).total_seconds()

        # Keep manifest dict in sync
        self.manifest.execution.end_time = end_time
        self.manifest.execution.duration_seconds = duration_seconds
        self.manifest.execution.status = status

        # Also update execution_info alias for test compatibility
        self.manifest.execution_info.end_time = end_time
        self.manifest.execution_info.duration_seconds = duration_seconds
        self.manifest.execution_info.status = status

        # Friend's spec: Set root-level status and error_message in manifest
        self.manifest.status = status
        self.manifest.error_message = error

        if error:
            self.manifest.execution.error_message = error
            self.manifest.execution_info.error_message = error

        logger.info(f"Audit marked as {status} (duration: {self.manifest.execution.duration_seconds:.2f}s)")

    def finalize(self) -> AuditManifest:
        """Finalize and return the completed manifest.

        Returns:
            Complete audit manifest

        """
        if self.manifest.execution.status == "running":
            self.mark_completed()

        return self.manifest

    def _generate_audit_id(self, seed: int | None = None) -> str:
        """Generate unique audit ID.

        When SOURCE_DATE_EPOCH is set or a seed is provided, generates a deterministic
        audit ID based on the fixed timestamp for reproducibility.

        Args:
            seed: Random seed for deterministic timestamp generation

        Returns:
            Unique audit ID string

        """
        from glassalpha.utils.determinism import get_deterministic_timestamp

        # Use deterministic timestamp (respects SOURCE_DATE_EPOCH and seed)
        dt = get_deterministic_timestamp(seed=seed)
        timestamp = dt.strftime("%Y%m%d_%H%M%S")

        # For audit hash, use deterministic hash if we have a seed
        if seed is not None:
            hash_input = f"deterministic_{seed}_{timestamp}"
        else:
            # Check if SOURCE_DATE_EPOCH is set
            source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
            if source_date_epoch:
                hash_input = f"deterministic_{source_date_epoch}"
            else:
                # Only in non-deterministic mode include process-specific info
                hash_input = f"{timestamp}_{platform.node()}_{os.getpid()}"

        audit_hash = hash_object(hash_input)[:8]
        return f"audit_{timestamp}_{audit_hash}"

    def _collect_environment_info(self) -> EnvironmentInfo:
        """Collect system and environment information.

        Returns:
            Environment information

        """
        # Collect key environment variables (excluding sensitive ones)
        sensitive_keys = {"PASSWORD", "TOKEN", "SECRET", "KEY", "CREDENTIAL"}

        # Use dictionary comprehension for better performance
        safe_env_vars = {
            key: value
            for key, value in os.environ.items()
            if (key.startswith(("PYTHON", "GLASSALPHA", "PATH")) or key in {"USER", "HOME", "PWD"})
            and not any(sensitive in key.upper() for sensitive in sensitive_keys)
        }

        return EnvironmentInfo(
            python_version=sys.version,
            platform=platform.platform(),
            architecture=f"{8 * struct.calcsize('P')}-bit/{platform.machine()}",
            processor=platform.processor(),
            hostname=platform.node(),
            user=os.environ.get("USER", "unknown"),
            working_directory=str(Path.cwd()),
            environment_variables=safe_env_vars,
        )

    def _collect_git_info(self) -> GitInfo | None:
        """Collect Git repository information.

        Returns:
            Git information if available, None otherwise

        """
        import subprocess  # noqa: PLC0415

        def _run_git(*args: str) -> str | None:
            """Run git command and return output or None if unavailable."""
            try:
                result = subprocess.run(
                    ["git", *args],
                    capture_output=True,
                    check=False,
                    text=True,
                    encoding="utf-8",
                )
                return (result.stdout or "").strip()
            except (FileNotFoundError, OSError):
                return None

        # Collect git info with proper commands
        info = {
            "commit_sha": _run_git("rev-parse", "HEAD"),
            "branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
            "porcelain_status": _run_git("status", "--porcelain"),
            "remote_url": _run_git("config", "--get", "remote.origin.url"),
            "last_commit_message": _run_git("log", "-1", "--pretty=%B"),
            "last_commit_date": _run_git("log", "-1", "--pretty=%ci"),
        }

        # Return None only if git not available
        if any(v is None for v in info.values()):
            return None

        # Git available - normalize status: clean if porcelain empty, else dirty
        status = "clean" if not info["porcelain_status"] else "dirty"
        is_dirty = status == "dirty"

        return GitInfo(
            commit_sha=info["commit_sha"],
            branch=info["branch"],
            status=status,
            is_dirty=is_dirty,
            remote_url=info["remote_url"] or None,
            commit_message=info["last_commit_message"],
            commit_timestamp=info["last_commit_date"],
        )


def create_manifest(audit_id: str | None = None, seed: int | None = None) -> ManifestGenerator:
    """Create a new audit manifest generator.

    Args:
        audit_id: Optional audit ID (auto-generated if None)
        seed: Random seed for deterministic audit ID generation

    Returns:
        Manifest generator instance

    """
    return ManifestGenerator(audit_id, seed=seed)


def load_manifest(path: Path) -> AuditManifest:
    """Load existing audit manifest from file.

    Args:
        path: Path to manifest file

    Returns:
        Loaded audit manifest

    Raises:
        FileNotFoundError: If manifest file doesn't exist
        ValueError: If manifest is invalid

    """
    path = Path(path)

    if not path.exists():
        msg = f"Manifest file not found: {path}"
        raise FileNotFoundError(msg)

    try:
        with path.open() as f:
            manifest_data = json.load(f)

        return AuditManifest(**manifest_data)

    except Exception as e:
        msg = f"Invalid manifest file {path}: {e}"
        raise ValueError(msg) from e


def compare_manifests(manifest1: AuditManifest, manifest2: AuditManifest) -> dict[str, Any]:
    """Compare two audit manifests for differences.

    Args:
        manifest1: First manifest
        manifest2: Second manifest

    Returns:
        Dictionary with comparison results

    """
    comparison = {
        "identical": True,
        "differences": [],
        "config_hash_match": manifest1.config_hash == manifest2.config_hash,
        "seed_match": manifest1.seeds == manifest2.seeds,
        "components_match": manifest1.selected_components == manifest2.selected_components,
        "data_hash_match": True,
    }

    # Compare data hashes
    for dataset_name, dataset1 in manifest1.datasets.items():
        if dataset_name in manifest2.datasets:
            dataset2 = manifest2.datasets[dataset_name]
            if dataset1.hash != dataset2.hash:
                comparison["data_hash_match"] = False
                comparison["differences"].append(f"Data hash mismatch for {dataset_name}")
        else:
            comparison["differences"].append(f"Dataset {dataset_name} missing in second manifest")

    # Overall comparison
    if not all(
        [
            comparison["config_hash_match"],
            comparison["seed_match"],
            comparison["components_match"],
            comparison["data_hash_match"],
        ],
    ):
        comparison["identical"] = False

    return comparison
