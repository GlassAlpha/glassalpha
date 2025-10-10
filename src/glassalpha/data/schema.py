"""Schema validation for datasets and configurations.

This module provides first-class schema validation to ensure configuration
and data alignment before training and fairness analysis.
"""

import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class DatasetConfig(BaseModel):
    """Configuration schema for dataset validation."""

    model_config = {"extra": "forbid"}

    features: list[str] = Field(
        description="List of feature column names expected in the dataset",
    )
    target: str = Field(
        description="Target column name",
    )
    protected_attributes: list[str] = Field(
        default_factory=list,
        description="List of protected/sensitive attribute column names for fairness analysis",
    )


def validate_config_schema(df: pd.DataFrame, cfg: dict[str, Any]) -> DatasetConfig:
    """Validate that dataset schema matches configuration expectations.

    Args:
        df: Loaded dataset DataFrame
        cfg: Configuration dictionary containing 'data' section

    Returns:
        Validated DatasetConfig object

    Raises:
        ValueError: If schema validation fails with actionable error message

    """
    # Extract data configuration
    data_cfg = cfg.get("data", {})

    # Build expected schema from config
    expected_schema = {
        "features": [],
        "target": data_cfg.get("target_column", ""),
        "protected_attributes": data_cfg.get("protected_attributes", []),
    }

    # Infer features as all columns except target and metadata
    target_col = expected_schema["target"]
    protected_attrs = expected_schema["protected_attributes"]

    # Common metadata columns to exclude from features
    metadata_cols = [
        "id",
        "customer_id",
        "patient_id",
        "consumer_id",
        "user_id",
        "date",
        "timestamp",
        "created_at",
        "updated_at",
        "consent_date",
        "treatment_date",
        "data_collection_date",
        "last_consent_update",
        "last_data_processing",
        "automated_decision_subject",
        "profiling_category",
        "data_portability_requested",
        "data_deletion_requested",
        "do_not_sell_requested",
    ]

    # Features are all columns except target, protected attributes, and metadata
    all_cols = set(df.columns)
    exclude_cols = {target_col} | set(protected_attrs) | set(metadata_cols)
    expected_schema["features"] = sorted(list(all_cols - exclude_cols))

    # Validate schema
    try:
        dataset_config = DatasetConfig(**expected_schema)
    except ValidationError as e:
        raise ValueError(f"Invalid dataset configuration: {e}") from e

    # Check for missing columns
    missing_features = [col for col in dataset_config.features if col not in df.columns]
    missing_target = target_col not in df.columns if target_col else False
    missing_protected = [col for col in dataset_config.protected_attributes if col not in df.columns]

    # Build actionable error message
    errors = []
    hints = []

    if missing_target:
        errors.append(f"Missing target column: '{target_col}'")
        available_targets = [col for col in df.columns if col.lower() in ["target", "label", "outcome", "class"]]
        if available_targets:
            hints.append(f"Available target-like columns: {available_targets}")
        else:
            # Always show available columns when target is missing, even if no obvious target-like columns
            hints.append(f"Available columns: {sorted(df.columns.tolist())}")

    if missing_features:
        errors.append(f"Missing feature columns: {missing_features}")
        hints.append(f"Available columns: {sorted(df.columns.tolist())}")

    if missing_protected:
        errors.append(f"Missing protected attribute columns: {missing_protected}")
        available_protected = [
            col
            for col in df.columns
            if any(attr in col.lower() for attr in ["age", "gender", "race", "ethnicity", "region", "income"])
        ]
        if available_protected:
            hints.append(f"Available protected-like columns: {available_protected}")
        else:
            # Always show available columns when protected attributes are missing, even if no obvious protected-like columns
            hints.append(f"Available columns: {sorted(df.columns.tolist())}")

    if errors:
        error_msg = "Schema validation failed:\n"
        error_msg += "\n".join(f"  • {error}" for error in errors)
        if hints:
            error_msg += "\n\nSuggestions:"
            error_msg += "\n".join(f"  • {hint}" for hint in hints)
        error_msg += f"\n\nDataset shape: {df.shape}"
        raise ValueError(error_msg)

    logger.info(
        f"Schema validation passed: {len(dataset_config.features)} features, "
        f"1 target, {len(dataset_config.protected_attributes)} protected attributes",
    )

    return dataset_config


def validate_data_quality(df: pd.DataFrame, schema: DatasetConfig) -> None:
    """Validate data quality after schema validation.

    Args:
        df: Dataset DataFrame
        schema: Validated schema configuration

    Raises:
        ValueError: If data quality issues are found

    """
    issues = []

    # Check for missing values in critical columns
    critical_cols = [schema.target] + schema.protected_attributes
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_pct = (null_count / len(df)) * 100
                issues.append(f"Column '{col}' has {null_count} null values ({null_pct:.1f}%)")

    # Check target column distribution
    if schema.target in df.columns:
        target_counts = df[schema.target].value_counts()
        if len(target_counts) < 2:
            issues.append(f"Target column '{schema.target}' has only {len(target_counts)} unique value(s)")

        # Check for severe class imbalance
        min_class_pct = (target_counts.min() / len(df)) * 100
        if min_class_pct < 1.0:
            issues.append(f"Severe class imbalance: smallest class is {min_class_pct:.1f}% of data")

    # Check for constant features
    constant_features = []
    for col in schema.features:
        if col in df.columns and df[col].nunique() <= 1:
            constant_features.append(col)

    if constant_features:
        issues.append(f"Constant features (no variation): {constant_features}")

    if issues:
        warning_msg = "Data quality warnings:\n"
        warning_msg += "\n".join(f"  • {issue}" for issue in issues)
        logger.warning(warning_msg)


def get_schema_summary(df: pd.DataFrame, schema: DatasetConfig) -> dict[str, Any]:
    """Get a summary of the validated schema.

    Args:
        df: Dataset DataFrame
        schema: Validated schema configuration

    Returns:
        Dictionary with schema summary information

    """
    summary = {
        "dataset_shape": df.shape,
        "n_features": len(schema.features),
        "n_protected_attributes": len(schema.protected_attributes),
        "target_column": schema.target,
        "feature_columns": schema.features,
        "protected_columns": schema.protected_attributes,
    }

    # Add target distribution if available
    if schema.target in df.columns:
        target_dist = df[schema.target].value_counts().to_dict()
        summary["target_distribution"] = target_dist
        summary["n_classes"] = len(target_dist)

    return summary
