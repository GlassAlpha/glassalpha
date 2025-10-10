"""Configuration builder for programmatic API.

Generates AuditConfig from in-memory models and data without YAML files.
Enables 3-line audits in notebooks and scripts.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from .schema import AuditConfig

logger = logging.getLogger(__name__)


def build_config_from_model(
    model: Any,  # noqa: ANN401
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    protected_attributes: list[str],
    *,
    random_seed: int = 42,
    audit_profile: str = "tabular_compliance",
    explainer: str | None = None,
    fairness_threshold: float | None = None,
    recourse_config: dict | None = None,
    feature_names: list[str] | None = None,
    target_name: str | None = None,
    **config_overrides: Any,
) -> tuple[AuditConfig, dict[str, Any]]:
    """Build AuditConfig from in-memory model and data.

    Args:
        model: Fitted model instance
        X_test: Test features (DataFrame or array)
        y_test: Test labels (Series or array)
        protected_attributes: Protected attribute column names
        random_seed: Random seed for reproducibility
        audit_profile: Audit profile name
        explainer: Explainer to use (None = auto-select)
        fairness_threshold: Classification threshold (None = use policy)
        recourse_config: Recourse configuration dict
        feature_names: Feature names if X_test is array
        target_name: Target column name
        **config_overrides: Additional config overrides

    Returns:
        Tuple of (AuditConfig, runtime_context)
        runtime_context contains model and data for pipeline execution

    Raises:
        ValueError: If model type cannot be detected
        ValueError: If protected attributes not in features
        ValueError: If data shapes are incompatible

    """
    from ..models.detection import detect_model_type  # noqa: PLC0415

    # 1. Detect model type
    model_type = detect_model_type(model)
    logger.info(f"Detected model type: {model_type}")

    # 2. Validate data shapes
    if hasattr(X_test, "shape"):
        n_samples_X = X_test.shape[0]  # noqa: N806
    else:
        n_samples_X = len(X_test)  # noqa: N806

    if hasattr(y_test, "shape"):
        n_samples_y = y_test.shape[0]
    else:
        n_samples_y = len(y_test)

    if n_samples_X != n_samples_y:
        raise ValueError(
            f"X_test and y_test have different sample counts: X_test={n_samples_X}, y_test={n_samples_y}",
        )

    logger.info(f"Validated data shapes: {n_samples_X} samples")

    # 3. Extract feature names
    if isinstance(X_test, pd.DataFrame):
        feature_cols = list(X_test.columns)
    elif feature_names is not None:
        feature_cols = feature_names
    else:
        # Generate feature names for arrays
        n_features = X_test.shape[1] if len(X_test.shape) > 1 else 1
        feature_cols = [f"feature_{i}" for i in range(n_features)]
        logger.warning(
            f"No feature names provided for array input, using generated names: "
            f"{feature_cols[:3]}{'...' if len(feature_cols) > 3 else ''}",
        )

    # 4. Validate protected attributes
    for attr in protected_attributes:
        if attr not in feature_cols:
            raise ValueError(
                f"Protected attribute '{attr}' not found in features.\n"
                f"Available features: {feature_cols}\n"
                f"Ensure protected_attributes match column names in X_test.",
            )

    logger.info(f"Validated {len(protected_attributes)} protected attributes: {protected_attributes}")

    # 5. Detect target type (binary/multiclass)
    unique_values = np.unique(y_test)
    n_classes = len(unique_values)
    problem_type = "binary" if n_classes == 2 else "multiclass"
    logger.info(f"Detected {problem_type} classification ({n_classes} classes)")

    # 6. Build minimal config dict
    config_dict = {
        "audit_profile": audit_profile,
        "model": {
            "type": model_type,
            "path": None,  # In-memory model, no path
        },
        "data": {
            "dataset": "custom",  # Flag as in-memory data
            "path": ":memory:",  # Special sentinel for in-memory
            "protected_attributes": protected_attributes,
            "target_column": target_name or "target",
            "feature_columns": feature_cols,
        },
        "reproducibility": {
            "random_seed": random_seed,
            "deterministic": True,
        },
    }

    # 7. Add optional configs
    if explainer:
        config_dict["explainers"] = {
            "strategy": "first_compatible",
            "priority": [explainer],
        }
        logger.info(f"Configured explainer: {explainer}")
    # If no explainer specified, let the registry handle auto-selection

    if fairness_threshold is not None:
        config_dict["report"] = {
            "threshold": {
                "policy": "fixed",
                "threshold": fairness_threshold,
            },
        }
        logger.info(f"Configured fixed threshold: {fairness_threshold}")

    if recourse_config:
        config_dict["recourse"] = recourse_config
        logger.info("Configured recourse generation")

    # 8. Apply config overrides (advanced users)
    if config_overrides:
        from .loader import merge_configs  # noqa: PLC0415

        logger.info(f"Applying {len(config_overrides)} config overrides")
        config_dict = merge_configs(config_dict, config_overrides)

    # 9. Load and validate config
    from .loader import load_config  # noqa: PLC0415

    audit_config = load_config(config_dict, profile_name=audit_profile)
    logger.info("Configuration built and validated successfully")

    # 10. Infer which features the model was trained on
    # For sklearn models, check n_features_in_ attribute
    model_n_features = None
    if hasattr(model, "n_features_in_"):
        model_n_features = model.n_features_in_
        logger.info(f"Model was trained on {model_n_features} features")

    # If model has fewer features than provided, it was trained on a subset
    # This happens when protected attributes are in X_test but not used for training
    if model_n_features is not None and model_n_features < len(feature_cols):
        logger.warning(
            f"Model trained on {model_n_features} features but X_test has {len(feature_cols)} columns. "
            f"Model will use first {model_n_features} features for predictions.",
        )

    # 10. Build runtime context (model + data)
    runtime_context = {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_cols,
        "target_name": target_name or "target",
        "model_n_features": model_n_features,  # Track how many features model was trained on
    }

    return audit_config, runtime_context
