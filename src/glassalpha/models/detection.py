"""Model type detection for programmatic API.

Automatically detects model types from instances to enable
config-free audit workflows in notebooks and scripts.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Model type mapping: module path -> registry key
MODEL_TYPE_MAP = {
    "sklearn.linear_model": "logistic_regression",
    "sklearn.ensemble": "random_forest",
    "xgboost.sklearn": "xgboost",
    "xgboost.core": "xgboost",
    "lightgbm.sklearn": "lightgbm",
    "lightgbm.basic": "lightgbm",
}


def detect_model_type(model: Any) -> str:  # noqa: ANN401
    """Detect model type from instance.

    Args:
        model: Fitted model instance

    Returns:
        Model type string for registry (e.g., "xgboost", "logistic_regression")

    Raises:
        ValueError: If model type cannot be detected

    Examples:
        >>> from sklearn.linear_model import LogisticRegression
        >>> model = LogisticRegression()
        >>> detect_model_type(model)
        'logistic_regression'

        >>> from xgboost import XGBClassifier
        >>> model = XGBClassifier()
        >>> detect_model_type(model)
        'xgboost'

    """
    model_class = type(model).__name__
    model_module = type(model).__module__

    logger.debug(f"Detecting model type: {model_module}.{model_class}")

    # Check by module path (most reliable)
    for module_prefix, registry_key in MODEL_TYPE_MAP.items():
        if model_module.startswith(module_prefix):
            logger.info(f"Detected model type '{registry_key}' from module '{model_module}'")
            return registry_key

    # Check by class name patterns (fallback)
    class_lower = model_class.lower()

    if "xgb" in class_lower:
        logger.info(f"Detected model type 'xgboost' from class name '{model_class}'")
        return "xgboost"

    if "lgbm" in class_lower or "lightgbm" in class_lower:
        logger.info(f"Detected model type 'lightgbm' from class name '{model_class}'")
        return "lightgbm"

    if "logistic" in class_lower:
        logger.info(f"Detected model type 'logistic_regression' from class name '{model_class}'")
        return "logistic_regression"

    if "randomforest" in class_lower or "forest" in class_lower:
        logger.info(f"Detected model type 'random_forest' from class name '{model_class}'")
        return "random_forest"

    if "tree" in class_lower and "sklearn" in model_module:
        logger.info(f"Detected model type 'decision_tree' from class name '{model_class}'")
        return "decision_tree"

    # Fallback for sklearn models
    if "sklearn" in model_module:
        logger.warning(f"Using fallback 'sklearn' for unknown sklearn model '{model_class}'")
        return "sklearn"

    # Fallback for any model with predict() method
    if hasattr(model, "predict"):
        logger.warning(f"Using fallback 'unknown' for model '{model_module}.{model_class}'")
        return "unknown"

    # Detection failed - provide helpful error
    raise ValueError(
        f"Cannot detect model type for {model_module}.{model_class}. "
        f"Supported: sklearn (LogisticRegression, RandomForest), xgboost (XGBClassifier), "
        f"lightgbm (LGBMClassifier). "
        f"For custom models, pass model type explicitly: "
        f"from_model(..., model_type='your_type', **config_overrides)",
    )
