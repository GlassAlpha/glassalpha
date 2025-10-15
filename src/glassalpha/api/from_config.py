"""from_config() entry point - audit from YAML configuration.

Loads config, dataset, model from paths specified in YAML.
Used for reproducible audits in CI/CD pipelines.

LAZY IMPORTS: numpy and pandas are imported inside functions to enable
basic module imports without scientific dependencies installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from glassalpha.api.result import AuditResult
from glassalpha.exceptions import (
    DataHashMismatchError,
    ResultIDMismatchError,
)


def from_config(config_path: str | Path) -> AuditResult:
    """Generate audit from YAML config file.

    Loads config, dataset, model from paths specified in YAML.
    Used for reproducible audits in CI/CD pipelines.

    Args:
        config_path: Path to YAML config file

    Returns:
        AuditResult matching the config specification

    Raises:
        GlassAlphaError (GAE2002): Result ID mismatch (if expected_result_id provided)
        GlassAlphaError (GAE2003): Data hash mismatch
        FileNotFoundError: Config or referenced files not found

    Config schema:

        Training config (model.type):
            model:
              type: xgboost  # Train new model
              params:
                random_state: 42
                n_estimators: 100

            data:
              dataset: german_credit  # Built-in dataset
              target_column: credit_risk
              protected_attributes:
                - gender
                - age_group

            reproducibility:
              random_seed: 42

        Inference config (model.path):
            model:
              path: "models/xgboost.pkl"  # Pre-trained model
              type: "xgboost.XGBClassifier"  # For verification

            data:
              X_path: "data/X_test.parquet"
              y_path: "data/y_test.parquet"
              protected_attributes:
                gender: "data/gender.parquet"
                race: "data/race.parquet"
              expected_hashes:
                X: "sha256:abc123..."
                y: "sha256:def456..."

            audit:
              random_seed: 42
              explain: true
              recourse: false
              calibration: true

            validation:
              expected_result_id: "abc123..."  # Optional: fail if mismatch

    Examples:
        Basic usage:
        >>> result = ga.audit.from_config("audit.yaml")
        >>> result.to_pdf("report.pdf")

        Verify reproducibility:
        >>> result1 = ga.audit.from_config("audit.yaml")
        >>> result2 = ga.audit.from_config("audit.yaml")
        >>> assert result1.id == result2.id  # Byte-identical

    """
    import pickle
    from pathlib import Path as PathLib

    import pandas as pd
    import yaml

    from glassalpha.utils.canonicalization import hash_data_for_manifest

    # Load YAML config
    config_path_obj = PathLib(config_path) if isinstance(config_path, str) else config_path

    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path_obj) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {config_path_obj}\n\nError: {e}") from e

    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path_obj}")

    # Get base directory for relative paths
    base_dir = config_path_obj.parent

    # Detect config type: inference (model.path) vs training (model.type)
    model_config = config.get("model", {})
    data_config = config.get("data", {})

    if "path" in model_config:
        # Inference config: Load pre-trained model
        model_path = base_dir / model_config["path"]
        with open(model_path, "rb") as f:
            model = pickle.load(f)  # nosec: B301
    elif "type" in model_config:
        # Training config: Use run_audit_pipeline CLI workflow instead
        # This delegates to the full audit pipeline which handles dataset loading and training
        from glassalpha.config import load_config
        from glassalpha.pipeline.audit import run_audit_pipeline

        audit_config = load_config(config_path_obj)
        result = run_audit_pipeline(audit_config)
        return result
    else:
        raise ValueError(
            "Config must have either model.path (for inference) or model.type (for training). "
            f"Found: {list(model_config.keys())}",
        )

    # Load data (inference config path)
    X_path = base_dir / data_config["X_path"]
    y_path = base_dir / data_config["y_path"]

    # Load X and y (support CSV, parquet, etc.)
    if str(X_path).endswith(".parquet"):
        X = pd.read_parquet(X_path)
    elif str(X_path).endswith(".csv"):
        X = pd.read_csv(X_path)
    else:
        raise ValueError(f"Unsupported file format: {X_path}")

    if str(y_path).endswith(".parquet"):
        y = pd.read_parquet(y_path)
        if isinstance(y, pd.DataFrame) and len(y.columns) == 1:
            y = y.iloc[:, 0]
    elif str(y_path).endswith(".csv"):
        y = pd.read_csv(y_path, header=None).iloc[:, 0]
    else:
        raise ValueError(f"Unsupported file format: {y_path}")

    # Validate data hashes (if provided)
    if "expected_hashes" in config.get("data", {}):
        expected_hashes = config["data"]["expected_hashes"]

        if "X" in expected_hashes:
            actual_hash = hash_data_for_manifest(X)
            if actual_hash != expected_hashes["X"]:
                raise DataHashMismatchError("X", expected_hashes["X"], actual_hash)

        if "y" in expected_hashes:
            actual_hash = hash_data_for_manifest(y)
            if actual_hash != expected_hashes["y"]:
                raise DataHashMismatchError("y", expected_hashes["y"], actual_hash)

    # Load protected attributes (if provided)
    protected_attributes = None
    if "protected_attributes" in config.get("data", {}):
        protected_attributes = {}
        for attr_name, attr_path in config["data"]["protected_attributes"].items():
            attr_full_path = base_dir / attr_path
            if str(attr_full_path).endswith(".parquet"):
                attr_data = pd.read_parquet(attr_full_path)
                if isinstance(attr_data, pd.DataFrame) and len(attr_data.columns) == 1:
                    attr_data = attr_data.iloc[:, 0]
            elif str(attr_full_path).endswith(".csv"):
                attr_data = pd.read_csv(attr_full_path, header=None).iloc[:, 0]
            else:
                raise ValueError(f"Unsupported file format: {attr_full_path}")

            protected_attributes[attr_name] = attr_data

    # Get audit parameters
    audit_config = config.get("audit", {})
    random_seed = audit_config.get("random_seed", 42)
    explain = audit_config.get("explain", True)
    recourse = audit_config.get("recourse", False)
    calibration = audit_config.get("calibration", True)

    # Import from_model here to avoid circular imports
    from glassalpha.api.from_model import from_model

    # Run audit via from_model
    result = from_model(
        model=model,
        X=X,
        y=y,
        protected_attributes=protected_attributes,
        random_seed=random_seed,
        explain=explain,
        recourse=recourse,
        calibration=calibration,
    )

    # Validate result ID (if provided)
    if "validation" in config and "expected_result_id" in config["validation"]:
        expected_id = config["validation"]["expected_result_id"]
        if result.id != expected_id:
            raise ResultIDMismatchError(expected_id, result.id)

    return result


__all__ = ["from_config"]
