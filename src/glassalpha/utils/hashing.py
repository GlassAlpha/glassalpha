"""Deterministic hashing utilities for audit reproducibility.

This module provides consistent hashing functions for different data types
to ensure byte-identical audit manifests across runs.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _json_serializer(obj: Any) -> Any:  # noqa: ANN401
    """Custom JSON serializer for common non-serializable objects.

    Args:
        obj: Object to serialize

    Returns:
        Serializable representation

    Raises:
        TypeError: If object cannot be serialized

    """
    # Handle Path objects
    if isinstance(obj, Path):
        return str(obj)

    # Handle callable objects with stable string representation
    if callable(obj):
        name = getattr(obj, "__qualname__", getattr(obj, "__name__", type(obj).__name__))
        return f"<callable:{name}>"

    # Handle numpy types
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    # Handle other common types that have string representations
    if hasattr(obj, "__dict__"):
        # For objects with __dict__, try to serialize their state
        try:
            return obj.__dict__
        except Exception:  # noqa: BLE001, S110
            pass

    # Safe fallback for common non-serializables
    return repr(obj)


def hash_object(obj: Any, algorithm: str = "sha256") -> str:  # noqa: ANN401
    """Generate deterministic hash of any Python object.

    Args:
        obj: Object to hash
        algorithm: Hash algorithm to use

    Returns:
        Hex string hash

    Raises:
        ValueError: If object cannot be serialized

    """
    try:
        # Convert to deterministic JSON representation - no default to ensure strict serialization
        json_str = json.dumps(obj, sort_keys=True, default=_json_serializer)

        # Generate hash
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(json_str.encode("utf-8"))
        return hash_obj.hexdigest()

    except (TypeError, ValueError):
        # Contract compliance: Exact error message required by tests
        raise ValueError("Cannot hash object")


def hash_config(config: dict[str, Any], algorithm: str = "sha256") -> str:
    """Generate deterministic hash of configuration dictionary.

    Args:
        config: Configuration dictionary
        algorithm: Hash algorithm to use

    Returns:
        Hex string hash

    """
    logger.debug("Hashing configuration")

    # Create clean config copy without timestamps or non-deterministic fields
    clean_config = _clean_config_for_hashing(config)

    # Sort recursively for deterministic JSON
    sorted_config = _sort_dict_recursively(clean_config)

    # Generate hash
    config_hash = hash_object(sorted_config, algorithm)
    logger.debug(f"Configuration hash: {config_hash[:12]}...")

    return config_hash


def hash_dataframe(df: pd.DataFrame, algorithm: str = "sha256") -> str:
    """Generate deterministic hash of pandas DataFrame.

    Args:
        df: DataFrame to hash
        algorithm: Hash algorithm to use

    Returns:
        Hex string hash

    """
    logger.debug(f"Hashing DataFrame with shape {df.shape}")

    # Sort by columns and index for consistency
    df_sorted = df.sort_index(axis=1).sort_index(axis=0)

    # Convert to consistent CSV representation
    csv_str = df_sorted.to_csv(index=False, float_format="%.10g")

    # Generate hash
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(csv_str.encode("utf-8"))
    data_hash = hash_obj.hexdigest()

    logger.debug(f"DataFrame hash: {data_hash[:12]}...")
    return data_hash


def hash_array(arr: np.ndarray, algorithm: str = "sha256") -> str:
    """Generate deterministic hash of numpy array.

    Args:
        arr: NumPy array to hash
        algorithm: Hash algorithm to use

    Returns:
        Hex string hash

    """
    logger.debug(f"Hashing array with shape {arr.shape}")

    # Convert to bytes for hashing
    # Use tobytes() for consistent byte representation
    array_bytes = arr.tobytes()

    # Include shape and dtype for complete hash
    metadata = f"shape={arr.shape},dtype={arr.dtype}"
    full_bytes = metadata.encode("utf-8") + array_bytes

    # Generate hash
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(full_bytes)
    array_hash = hash_obj.hexdigest()

    logger.debug(f"Array hash: {array_hash[:12]}...")
    return array_hash


def hash_file(file_path: Path, algorithm: str = "sha256", chunk_size: int = 8192) -> str:
    """Generate hash of file contents.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm to use
        chunk_size: Bytes to read at once for large files

    Returns:
        Hex string hash

    Raises:
        FileNotFoundError: If file doesn't exist

    """
    file_path = Path(file_path)

    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)

    logger.debug(f"Hashing file: {file_path}")

    hash_obj = hashlib.new(algorithm)

    with Path(file_path).open("rb") as f:
        # Read in chunks for memory efficiency
        while chunk := f.read(chunk_size):
            hash_obj.update(chunk)

    file_hash = hash_obj.hexdigest()
    logger.debug(f"File hash: {file_hash[:12]}...")

    return file_hash


def hash_model(model: Any, algorithm: str = "sha256") -> str:  # noqa: ANN401, C901, PLR0912
    """Generate hash of ML model.

    Args:
        model: Model object to hash
        algorithm: Hash algorithm to use

    Returns:
        Hex string hash

    Note:
        This attempts multiple strategies to hash model state

    """
    logger.debug(f"Hashing model of type: {type(model)}")

    # Strategy 1: Use model's built-in serialization if available
    try:
        if hasattr(model, "get_booster"):  # XGBoost
            # Get model dump for XGBoost
            model_dump = model.get_booster().get_dump()
            return hash_object(model_dump, algorithm)

        if hasattr(model, "dump_model"):  # LightGBM
            # Get model dump for LightGBM
            model_dump = model.booster_.dump_model()
            return hash_object(model_dump, algorithm)

        if hasattr(model, "get_params"):  # Sklearn-style models
            # Hash parameters and fitted attributes
            params = model.get_params()

            # Add key fitted attributes
            model_state = {"params": params}

            if hasattr(model, "coef_"):
                model_state["coef_"] = model.coef_.tolist()
            if hasattr(model, "intercept_"):
                model_state["intercept_"] = (
                    model.intercept_.tolist() if hasattr(model.intercept_, "tolist") else model.intercept_
                )
            if hasattr(model, "feature_importances_"):
                model_state["feature_importances_"] = model.feature_importances_.tolist()

            return hash_object(model_state, algorithm)

    except Exception as e:  # noqa: BLE001
        logger.warning(f"Model-specific hashing failed: {e}, falling back to object hash")

    # Strategy 2: Hash object attributes
    try:
        # Get all non-callable attributes
        attributes = {}
        for attr_name in dir(model):
            if not attr_name.startswith("_") and not callable(getattr(model, attr_name)):
                try:
                    attr_value = getattr(model, attr_name)
                    if isinstance(attr_value, np.ndarray):
                        attributes[attr_name] = attr_value.tolist()
                    else:
                        attributes[attr_name] = attr_value
                except Exception:  # noqa: BLE001, S112
                    continue

        return hash_object(attributes, algorithm)

    except Exception as e:  # noqa: BLE001
        logger.warning(f"Attribute hashing failed: {e}, using basic object hash")

    # Strategy 3: Basic object hash
    return hash_object(str(model), algorithm)


def hash_multiple(*objects: Any, algorithm: str = "sha256") -> str:  # noqa: ANN401
    """Generate combined hash of multiple objects.

    Args:
        *objects: Objects to hash together
        algorithm: Hash algorithm to use

    Returns:
        Hex string hash of all objects combined

    """
    logger.debug(f"Hashing {len(objects)} objects together")

    # Hash each object individually, then combine
    individual_hashes = []
    for i, obj in enumerate(objects):
        try:
            if isinstance(obj, pd.DataFrame):
                obj_hash = hash_dataframe(obj, algorithm)
            elif isinstance(obj, np.ndarray):
                obj_hash = hash_array(obj, algorithm)
            elif isinstance(obj, dict):
                obj_hash = hash_config(obj, algorithm)
            else:
                obj_hash = hash_object(obj, algorithm)

            individual_hashes.append(obj_hash)

        except Exception as e:
            # Don't silently fall back - let the error bubble up for unsupported types
            msg = f"Unsupported type for hashing: {type(obj)} at position {i}"
            raise ValueError(msg) from e

    # Combine all hashes
    combined_str = "".join(individual_hashes)

    hash_obj = hashlib.new(algorithm)
    hash_obj.update(combined_str.encode("utf-8"))
    combined_hash = hash_obj.hexdigest()

    logger.debug(f"Combined hash: {combined_hash[:12]}...")
    return combined_hash


def _clean_config_for_hashing(config: dict[str, Any]) -> dict[str, Any]:
    """Remove non-deterministic fields from config before hashing.

    Args:
        config: Original configuration

    Returns:
        Cleaned configuration for deterministic hashing

    """
    # Fields to exclude from hashing
    exclude_fields = {"timestamp", "execution_time", "run_id", "output_path", "temp_dir"}

    def clean_dict(d: dict[str, Any]) -> dict[str, Any]:
        cleaned = {}
        for key, value in d.items():
            if key in exclude_fields:
                continue

            if isinstance(value, dict):
                cleaned[key] = clean_dict(value)
            elif isinstance(value, list):
                # Clean list elements
                cleaned_list = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_list.append(clean_dict(item))
                    else:
                        cleaned_list.append(item)
                cleaned[key] = cleaned_list
            else:
                cleaned[key] = value

        return cleaned

    return clean_dict(config)


def _sort_dict_recursively(d: dict[str, Any]) -> dict[str, Any]:
    """Sort dictionary recursively for deterministic serialization.

    Args:
        d: Dictionary to sort

    Returns:
        Recursively sorted dictionary

    """
    if not isinstance(d, dict):
        return d

    sorted_dict = {}
    for key in sorted(d.keys()):
        value = d[key]

        if isinstance(value, dict):
            sorted_dict[key] = _sort_dict_recursively(value)
        elif isinstance(value, list):
            # Sort list elements if they are dictionaries
            sorted_list = []
            for item in value:
                if isinstance(item, dict):
                    sorted_list.append(_sort_dict_recursively(item))
                else:
                    sorted_list.append(item)
            sorted_dict[key] = sorted_list
        else:
            sorted_dict[key] = value

    return sorted_dict


def verify_hash_consistency(obj: Any, iterations: int = 3) -> bool:  # noqa: ANN401
    """Verify that hashing is deterministic across multiple runs.

    Args:
        obj: Object to test
        iterations: Number of hash iterations to compare

    Returns:
        True if all hashes are identical

    """
    hashes = []

    for i in range(iterations):
        try:
            if isinstance(obj, pd.DataFrame):
                obj_hash = hash_dataframe(obj)
            elif isinstance(obj, np.ndarray):
                obj_hash = hash_array(obj)
            elif isinstance(obj, dict):
                obj_hash = hash_config(obj)
            else:
                obj_hash = hash_object(obj)

            hashes.append(obj_hash)

        except Exception:
            logger.exception(f"Hash iteration {i} failed")
            return False

    # Check if all hashes are identical
    is_consistent = len(set(hashes)) == 1

    if is_consistent:
        logger.info(f"Hash consistency verified: {hashes[0][:12]}...")
    else:
        logger.error(f"Hash inconsistency detected: {hashes}")

    return is_consistent
