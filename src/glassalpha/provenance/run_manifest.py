"""Comprehensive provenance tracking for audit reproducibility.

This module provides complete audit trail generation including git SHA,
configuration hashes, dataset fingerprints, model information, library
versions, and environment details. Essential for regulatory compliance
and reproducible research.
"""

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from glassalpha.utils.determinism import get_deterministic_timestamp

logger = logging.getLogger(__name__)


def generate_run_manifest(
    config: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    dataset_df: pd.DataFrame | None = None,
    model_info: dict[str, Any] | None = None,
    selected_components: dict[str, Any] | None = None,
    execution_info: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate comprehensive run manifest for audit reproducibility.

    Args:
        config: Complete audit configuration
        dataset_path: Path to dataset file (if applicable)
        dataset_df: Dataset DataFrame for fingerprinting
        model_info: Model metadata and parameters
        selected_components: Selected explainers, metrics, etc.
        execution_info: Runtime execution details

    Returns:
        Complete manifest dictionary with all provenance information

    """
    logger.info("Generating comprehensive run manifest")

    # Handle None config
    if config is None:
        config = {
            "audit_profile": "unknown",
            "model": {"type": "unknown"},
            "data": {"path": "unknown"},
        }

    manifest = {
        "manifest_version": "1.0",
        "schema_version": "1.0.0",
        "generated_at": get_deterministic_timestamp(seed=seed).isoformat(),
        "glassalpha_version": _get_glassalpha_version(),
    }

    # Git provenance
    manifest["git"] = _get_git_provenance()

    # Configuration provenance
    manifest["configuration"] = _get_config_provenance(config)

    # Dataset provenance
    if dataset_df is not None or dataset_path is not None:
        manifest["dataset"] = _get_dataset_provenance(dataset_path, dataset_df)

    # Model provenance
    if model_info:
        manifest["model"] = _get_model_provenance(model_info)

    # Component selection provenance
    if selected_components:
        manifest["components"] = _get_component_provenance(selected_components)

    # Environment provenance
    manifest["environment"] = _get_environment_provenance()

    # Library versions
    manifest["dependencies"] = _get_dependency_provenance()

    # Execution provenance (always include for compliance)
    if execution_info:
        manifest["execution"] = _get_execution_provenance(execution_info)
    else:
        manifest["execution"] = {
            "start_time": None,
            "end_time": None,
            "duration_seconds": None,
            "success": None,
            "error_message": None,
            "random_seed": None,
            "deterministic_mode": False,
        }

    # Constraints hash (if constraints.txt exists)
    manifest["constraints"] = _get_constraints_provenance()

    # Overall manifest hash (excluding timestamps and this field for determinism)
    manifest_for_hash = _prepare_manifest_for_hash(manifest)
    manifest["manifest_hash"] = _compute_dict_hash(manifest_for_hash)

    logger.info(f"Run manifest generated with {len(manifest)} sections")
    return manifest


def _get_glassalpha_version() -> str:
    """Get GlassAlpha version information."""
    try:
        # Try to get version from package metadata
        import importlib.metadata

        return importlib.metadata.version("glassalpha")
    except Exception:
        # Fallback to git describe if in development
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--dirty", "--always"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"


def _get_git_provenance() -> dict[str, Any]:
    """Get git repository provenance information."""
    git_info = {
        "available": False,
        "sha": None,
        "branch": None,
        "dirty": None,
        "remote_url": None,
        "last_commit_date": None,
        "last_commit_message": None,
    }

    try:
        # Check if we're in a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode != 0:
            logger.debug("Not in a git repository")
            return git_info

        git_info["available"] = True

        # Get commit SHA
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            git_info["sha"] = result.stdout.strip()

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()

        # Check if working directory is dirty
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        git_info["dirty"] = result.returncode != 0

        # Get remote URL
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            git_info["remote_url"] = result.stdout.strip()

        # Get last commit info
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            git_info["last_commit_date"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            git_info["last_commit_message"] = result.stdout.strip()

    except Exception as e:
        logger.debug(f"Failed to get git provenance: {e}")

    return git_info


def _get_config_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """Get configuration provenance information."""
    # Create a clean config copy for hashing (remove sensitive info)
    clean_config = _sanitize_config_for_hash(config)

    return {
        "hash": _compute_dict_hash(clean_config),
        "size_bytes": len(json.dumps(clean_config, sort_keys=True)),
        "sections": list(clean_config.keys()),
        "audit_profile": config.get("audit_profile"),
        "strict_mode": config.get("strict_mode", False),
    }


def _get_dataset_provenance(dataset_path: str | Path | None, dataset_df: pd.DataFrame | None) -> dict[str, Any]:
    """Get dataset provenance information."""
    dataset_info = {
        "path": str(dataset_path) if dataset_path else None,
        "hash": None,
        "size_bytes": None,
        "shape": None,
        "dtypes": None,
        "memory_usage_mb": None,
        "null_counts": None,
        "sample_hash": None,  # Hash of first few rows for verification
    }

    if dataset_df is not None:
        # Dataset shape and basic info
        dataset_info["shape"] = list(dataset_df.shape)
        dataset_info["dtypes"] = {col: str(dtype) for col, dtype in dataset_df.dtypes.items()}
        dataset_info["memory_usage_mb"] = round(dataset_df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        dataset_info["null_counts"] = dataset_df.isnull().sum().to_dict()

        # Compute dataset hash (deterministic)
        dataset_info["hash"] = _compute_dataframe_hash(dataset_df)

        # Sample hash for verification (first 100 rows)
        sample_df = dataset_df.head(100)
        dataset_info["sample_hash"] = _compute_dataframe_hash(sample_df)

    if dataset_path and Path(dataset_path).exists():
        # File size
        dataset_info["size_bytes"] = Path(dataset_path).stat().st_size

        # File hash if not already computed from DataFrame
        if dataset_info["hash"] is None:
            dataset_info["hash"] = _compute_file_hash(dataset_path)

    return dataset_info


def _get_model_provenance(model_info: dict[str, Any]) -> dict[str, Any]:
    """Get model provenance information."""
    return {
        "type": model_info.get("type"),
        "parameters": model_info.get("parameters", {}),
        "calibration": model_info.get("calibration"),
        "training_time": model_info.get("training_time"),
        "feature_count": model_info.get("feature_count"),
        "class_count": model_info.get("class_count"),
        "model_size_estimate": model_info.get("model_size_estimate"),
    }


def _get_component_provenance(selected_components: dict[str, Any]) -> dict[str, Any]:
    """Get component selection provenance."""
    return {
        "explainer": selected_components.get("explainer"),
        "metrics": selected_components.get("metrics", []),
        "threshold_policy": selected_components.get("threshold_policy"),
        "selection_strategy": selected_components.get("selection_strategy"),
        "fallbacks_used": selected_components.get("fallbacks_used", []),
    }


def _get_environment_provenance() -> dict[str, Any]:
    """Get environment provenance information."""
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "user": os.environ.get("USER", "unknown"),
        "working_directory": str(Path.cwd()),
        "environment_variables": _get_relevant_env_vars(),
    }


def _get_dependency_provenance() -> dict[str, Any]:
    """Get dependency version information."""
    dependencies = {}

    # Core dependencies
    core_packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "shap",
        "pydantic",
        "typer",
        "pyyaml",
        "weasyprint",
    ]

    for package in core_packages:
        try:
            import importlib.metadata

            dependencies[package] = importlib.metadata.version(package)
        except Exception:
            dependencies[package] = "unknown"

    # Gather all installed packages and their versions
    installed_packages = {}
    try:
        import importlib.metadata

        installed_packages = {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
            if dist.metadata and "Name" in dist.metadata and dist.version is not None
        }
    except Exception:
        installed_packages = {}

    return {
        "core_packages": dependencies,
        "total_packages": len(dependencies),
        "installed_packages": installed_packages,
    }


def _get_execution_provenance(execution_info: dict[str, Any]) -> dict[str, Any]:
    """Get execution provenance information."""
    return {
        "start_time": execution_info.get("start_time"),
        "end_time": execution_info.get("end_time"),
        "duration_seconds": execution_info.get("duration_seconds"),
        "success": execution_info.get("success"),
        "error_message": execution_info.get("error_message"),
        "random_seed": execution_info.get("random_seed"),
        "deterministic_mode": execution_info.get("deterministic_mode", False),
    }


def _get_constraints_provenance() -> dict[str, Any]:
    """Get constraints.txt provenance if available."""
    constraints_info = {
        "available": False,
        "path": None,
        "hash": None,
        "package_count": None,
    }

    # Look for constraints.txt in common locations
    possible_paths = [
        Path("constraints.txt"),
        Path("../constraints.txt"),
    ]

    for path in possible_paths:
        if path.exists():
            constraints_info["available"] = True
            constraints_info["path"] = str(path.resolve())
            constraints_info["hash"] = _compute_file_hash(path)

            # Count packages
            try:
                with path.open() as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                    constraints_info["package_count"] = len(lines)
            except Exception:
                pass
            break

    return constraints_info


def _get_relevant_env_vars() -> dict[str, str]:
    """Get relevant environment variables for reproducibility."""
    relevant_vars = [
        "PYTHONPATH",
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "GLASSALPHA_TELEMETRY",
    ]

    return {var: os.environ.get(var, "") for var in relevant_vars if os.environ.get(var)}


def _sanitize_config_for_hash(config: dict[str, Any]) -> dict[str, Any]:
    """Remove sensitive information from config before hashing."""
    # Create deep copy and remove sensitive paths/keys
    import copy

    clean_config = copy.deepcopy(config)

    # Recursively convert all Path objects to strings
    def convert_paths(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        return obj

    clean_config = convert_paths(clean_config)

    # Remove or sanitize sensitive information
    if "data" in clean_config and "path" in clean_config["data"] and clean_config["data"]["path"] is not None:
        # Keep only filename, not full path
        path = Path(clean_config["data"]["path"])
        clean_config["data"]["path"] = path.name

    # Remove any license keys or tokens
    for key in list(clean_config.keys()):
        if "license" in key.lower() or "token" in key.lower() or "key" in key.lower():
            clean_config.pop(key, None)

    return clean_config


def _prepare_manifest_for_hash(manifest: dict[str, Any]) -> dict[str, Any]:
    """Prepare manifest for hashing by removing non-deterministic fields.

    Removes timestamps and other time-dependent fields that don't affect
    reproducibility. The hash should verify audit configuration and results
    are identical, not that they ran at the same time.

    Args:
        manifest: Complete manifest dictionary

    Returns:
        Manifest copy with timestamps removed

    """
    import copy

    manifest_copy = copy.deepcopy(manifest)

    # Remove top-level timestamp
    manifest_copy.pop("generated_at", None)

    # Remove execution timestamps (keep seed and success status)
    if "execution" in manifest_copy:
        exec_info = manifest_copy["execution"]
        exec_info.pop("start_time", None)
        exec_info.pop("end_time", None)
        exec_info.pop("duration_seconds", None)

    return manifest_copy


def _compute_dict_hash(data: dict[str, Any]) -> str:
    """Compute deterministic hash of dictionary."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


def _compute_dataframe_hash(df: pd.DataFrame) -> str:
    """Compute deterministic hash of DataFrame."""
    # Sort by all columns to ensure deterministic order
    try:
        sorted_df = df.sort_values(by=list(df.columns))
    except Exception:
        # If sorting fails, use original order
        sorted_df = df

    # Convert to string representation and hash
    df_str = sorted_df.to_csv(index=False)
    return hashlib.sha256(df_str.encode()).hexdigest()


def _compute_file_hash(file_path: str | Path) -> str:
    """Compute SHA-256 hash of file."""
    hash_sha256 = hashlib.sha256()
    with Path(file_path).open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def write_manifest_sidecar(manifest: dict[str, Any], output_path: str | Path) -> Path:
    """Write manifest as JSON sidecar file.

    Args:
        manifest: Complete manifest dictionary
        output_path: Path to main output file (sidecar will be created alongside)

    Returns:
        Path to created manifest file

    """
    output_path = Path(output_path)
    manifest_path = output_path.with_suffix(".manifest.json")

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info(f"Wrote manifest sidecar: {manifest_path}")
    return manifest_path


def get_manifest_summary(manifest: dict[str, Any]) -> str:
    """Get human-readable summary of manifest for PDF embedding.

    Args:
        manifest: Complete manifest dictionary

    Returns:
        Formatted summary string for PDF appendix

    """
    lines = []
    lines.append("AUDIT PROVENANCE SUMMARY")
    lines.append("=" * 50)
    lines.append("")

    # Basic info
    lines.append(f"Generated: {manifest.get('generated_at', 'unknown')}")
    lines.append(f"GlassAlpha Version: {manifest.get('glassalpha_version', 'unknown')}")
    lines.append(f"Manifest Hash: {manifest.get('manifest_hash', 'unknown')[:16]}...")
    lines.append("")

    # Git info
    git_info = manifest.get("git", {})
    if git_info.get("available"):
        lines.append("Git Repository:")
        lines.append(f"  SHA: {git_info.get('sha', 'unknown')[:12]}...")
        lines.append(f"  Branch: {git_info.get('branch', 'unknown')}")
        lines.append(f"  Dirty: {git_info.get('dirty', 'unknown')}")
        if git_info.get("remote_url"):
            lines.append(f"  Remote: {git_info['remote_url']}")
    else:
        lines.append("Git Repository: Not available")
    lines.append("")

    # Configuration
    config_info = manifest.get("configuration", {})
    lines.append("Configuration:")
    lines.append(f"  Hash: {config_info.get('hash', 'unknown')[:16]}...")
    lines.append(f"  Profile: {config_info.get('audit_profile', 'unknown')}")
    lines.append(f"  Strict Mode: {config_info.get('strict_mode', False)}")
    lines.append("")

    # Dataset
    dataset_info = manifest.get("dataset", {})
    if dataset_info:
        lines.append("Dataset:")
        lines.append(f"  Hash: {dataset_info.get('hash', 'unknown')[:16]}...")
        if dataset_info.get("shape"):
            lines.append(f"  Shape: {dataset_info['shape']}")
        if dataset_info.get("size_bytes"):
            size_mb = dataset_info["size_bytes"] / 1024 / 1024
            lines.append(f"  Size: {size_mb:.1f} MB")
    lines.append("")

    # Model
    model_info = manifest.get("model", {})
    if model_info:
        lines.append("Model:")
        lines.append(f"  Type: {model_info.get('type', 'unknown')}")
        if model_info.get("calibration"):
            lines.append(f"  Calibration: {model_info['calibration']}")
        if model_info.get("feature_count"):
            lines.append(f"  Features: {model_info['feature_count']}")
    lines.append("")

    # Environment
    env_info = manifest.get("environment", {})
    if env_info:
        lines.append("Environment:")
        python_version = env_info.get("python_version", "").split()[0]
        lines.append(f"  Python: {python_version}")
        lines.append(f"  Platform: {env_info.get('platform', 'unknown')}")
    lines.append("")

    # Dependencies
    deps_info = manifest.get("dependencies", {})
    if deps_info and deps_info.get("core_packages"):
        lines.append("Key Dependencies:")
        core_packages = deps_info["core_packages"]
        for pkg in ["numpy", "pandas", "scikit-learn", "xgboost"]:
            if pkg in core_packages:
                lines.append(f"  {pkg}: {core_packages[pkg]}")
    lines.append("")

    lines.append("This manifest ensures complete audit reproducibility.")
    lines.append("For full details, see the accompanying .manifest.json file.")

    return "\n".join(lines)
