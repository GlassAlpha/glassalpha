"""from_model() entry point - audit from fitted model.

Primary API for auditing ML models. Extracts predictions and probabilities
from the model, computes fairness metrics, and generates reproducible results.

LAZY IMPORTS: numpy and pandas are imported inside functions to enable
basic module imports without scientific dependencies installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

from glassalpha.api.result import AuditResult
from glassalpha.exceptions import (
    CategoricalDataError,
    InvalidProtectedAttributesError,
    MultiIndexNotSupportedError,
    NonBinaryClassificationError,
    NoPredictProbaError,
)


def _validate_sklearn_compatible(X: pd.DataFrame, model: Any) -> None:
    """Validate that DataFrame is compatible with sklearn models.

    Checks for categorical columns that would cause "could not convert string to float" errors.
    Skips validation for sklearn Pipelines (they handle preprocessing internally).

    Args:
        X: Feature DataFrame to validate
        model: Model instance to check if it's a Pipeline

    Raises:
        CategoricalDataError: If categorical columns are found and model is not a Pipeline

    """
    # Skip validation if model is a sklearn Pipeline (handles preprocessing internally)
    try:
        from sklearn.pipeline import Pipeline

        if isinstance(model, Pipeline):
            return  # Pipeline will handle categorical encoding
    except ImportError:
        pass

    # Check for categorical columns only for non-Pipeline models
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if categorical_cols:
        raise CategoricalDataError(categorical_cols)


def _convert_protected_attributes(
    protected_attributes: Mapping[str, pd.Series | np.ndarray] | Sequence[str] | None,
    X: pd.DataFrame | np.ndarray,
) -> Mapping[str, pd.Series | np.ndarray] | None:
    """Convert protected_attributes from list format to dict format if needed.

    If protected_attributes is a list of attribute names, auto-detect one-hot encoded columns
    in X and extract the original categorical values.

    Args:
        protected_attributes: Either a list of attribute names or dict mapping names to arrays
        X: Feature matrix (used to auto-detect one-hot columns if protected_attributes is a list)

    Returns:
        Dict mapping attribute names to arrays, or None if protected_attributes is None

    Raises:
        InvalidProtectedAttributesError: If auto-detection fails

    """
    import numpy as np
    import pandas as pd

    if protected_attributes is None:
        return None

    # If already a dict, return as-is
    if isinstance(protected_attributes, Mapping):
        return protected_attributes

    # If list, auto-detect one-hot encoded columns
    if not isinstance(X, pd.DataFrame):
        raise InvalidProtectedAttributesError(
            "When using list format for protected_attributes, X must be a DataFrame with column names. "
            "Got numpy array instead."
        )

    result = {}
    for attr_name in protected_attributes:
        # Find columns starting with attr_name_
        prefix = f"{attr_name}_"
        matching_cols = [col for col in X.columns if col.startswith(prefix)]

        if not matching_cols:
            raise InvalidProtectedAttributesError(
                f"Could not find one-hot encoded columns for protected attribute '{attr_name}'. "
                f"Expected columns like '{prefix}*' in X. "
                f"Available columns: {list(X.columns[:10])}{'...' if len(X.columns) > 10 else ''}"
            )

        # Convert one-hot encoding back to categorical
        # Find which column has value 1 for each row
        one_hot_df = X[matching_cols]

        # Extract category names (remove prefix)
        category_names = [col[len(prefix) :] for col in matching_cols]

        # For each row, find which column is 1 (argmax across one-hot columns)
        category_indices = np.argmax(one_hot_df.values, axis=1)
        categorical_values = np.array([category_names[idx] for idx in category_indices])

        result[attr_name] = pd.Series(categorical_values, index=X.index if hasattr(X, "index") else None)

    return result


def _extract_feature_names(X: pd.DataFrame | np.ndarray) -> list[str]:
    """Extract feature names from DataFrame or create generic names for ndarray."""
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        return X.columns.tolist()
    # For numpy arrays, create generic feature names
    return [f"feature_{i}" for i in range(X.shape[1])]


def _to_numpy(arr: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert pandas objects to numpy arrays safely."""
    import numpy as np
    import pandas as pd

    if isinstance(arr, (pd.Series, pd.DataFrame)):
        return arr.to_numpy()
    return np.asarray(arr)


def _validate_no_multiindex(df: pd.DataFrame) -> None:
    """Validate that DataFrame doesn't have MultiIndex (not supported)."""
    import pandas as pd

    if isinstance(df.index, pd.MultiIndex):
        raise MultiIndexNotSupportedError()


def from_model(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    protected_attributes: Mapping[str, pd.Series | np.ndarray] | Sequence[str] | None = None,
    sample_weight: pd.Series | np.ndarray | None = None,
    random_seed: int = 42,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    explain: bool = True,
    recourse: bool = False,
    calibration: bool = True,
    stability: bool = False,
) -> AuditResult:
    """Generate audit from fitted model.

    Primary entry point for auditing ML models. Automatically extracts
    predictions and probabilities from the model, computes fairness metrics,
    and generates a byte-identical reproducible result.

    **Important**: Binary classification only. Multi-class models not yet supported.

    Args:
        model: Fitted sklearn-compatible model with predict() method
        X: Feature matrix (n_samples, n_features)
        y: True labels (n_samples,). Must be binary (2 unique classes)
        protected_attributes: Protected attributes for fairness analysis. Can be:
            - List of attribute names (e.g., ["gender", "age_group"]) - auto-detects one-hot encoded columns
            - Dict mapping attribute names to arrays (e.g., {"gender": gender_array, "race": race_array})
            Missing values (NaN) are mapped to "Unknown" category.
        sample_weight: Sample weights for weighted metrics (optional)
        random_seed: Random seed for deterministic SHAP sampling
        feature_names: Feature names (inferred from X if DataFrame)
        class_names: Class names for classification (e.g., ["Denied", "Approved"])
        explain: Generate SHAP explanations (default: True)
        recourse: Generate recourse recommendations (default: False)
        calibration: Compute calibration metrics (default: True, requires predict_proba)
        stability: Run stability tests (default: False, slower)

    Returns:
        AuditResult with performance, fairness, calibration, explanations

    Raises:
        GlassAlphaError (GAE1001): Invalid protected_attributes format
        GlassAlphaError (GAE1003): Length mismatch between X, y, protected_attributes
        GlassAlphaError (GAE1004): Non-binary classification (not yet supported)
        GlassAlphaError (GAE1012): MultiIndex not supported

    Examples:
        Basic audit (binary classification):
        >>> result = ga.audit.from_model(model, X_test, y_test)
        >>> result.performance.accuracy
        0.847

        With protected attributes:
        >>> result = ga.audit.from_model(
        ...     model, X_test, y_test,
        ...     protected_attributes={"gender": gender, "race": race}
        ... )
        >>> result.fairness.demographic_parity_max_diff
        0.023

        Missing values in protected attributes:
        >>> gender_with_nan = pd.Series([0, 1, np.nan, 1, 0])  # NaN → "Unknown"
        >>> result = ga.audit.from_model(model, X, y, protected_attributes={"gender": gender_with_nan})

        Without explanations (faster):
        >>> result = ga.audit.from_model(model, X, y, explain=False)

    """
    import numpy as np
    import pandas as pd

    # Import from_predictions here to avoid circular imports
    from glassalpha.api.from_predictions import from_predictions
    from glassalpha.models.detection import detect_model_type
    from glassalpha.utils.canonicalization import hash_data_for_manifest

    # Convert inputs to numpy arrays for metrics computation
    X_arr = _to_numpy(X)
    y_arr = _to_numpy(y)

    # Validate binary classification early (before any model operations)
    n_classes = len(np.unique(y_arr))
    if n_classes != 2:
        raise NonBinaryClassificationError(n_classes)

    # Validate inputs before any model operations
    if isinstance(X, pd.DataFrame):
        if hasattr(X.index, "__class__") and X.index.__class__.__name__ == "MultiIndex":
            raise MultiIndexNotSupportedError("X")
        _validate_sklearn_compatible(X, model)

    # Extract feature names
    if feature_names is None:
        feature_names_list = _extract_feature_names(X)
    else:
        feature_names_list = list(feature_names)

    # Detect model type
    model_type = detect_model_type(model)

    # Generate predictions
    # IMPORTANT: Pass original X (not X_arr) to support sklearn pipelines with ColumnTransformer
    # These pipelines require DataFrame input with column names
    y_pred = model.predict(X)

    # Try to get probabilities
    y_proba = None
    if calibration or explain:
        # Check for predict_proba
        if hasattr(model, "predict_proba"):
            y_proba_raw = model.predict_proba(X)
            # Extract positive class for binary classification
            if y_proba_raw.ndim == 2 and y_proba_raw.shape[1] == 2:
                y_proba = y_proba_raw[:, 1]
            else:
                y_proba = y_proba_raw
        elif calibration:
            raise NoPredictProbaError(model_type)

    # Compute model fingerprint (hash of model type + parameters)
    model_fingerprint = f"{model_type}:{hash(str(type(model).__name__))}"

    # Convert protected_attributes from list to dict if needed
    protected_attributes_converted = _convert_protected_attributes(protected_attributes, X)

    # Delegate to from_predictions
    result = from_predictions(
        y_true=y_arr,
        y_pred=y_pred,
        y_proba=y_proba,
        protected_attributes=protected_attributes_converted,
        sample_weight=sample_weight,
        random_seed=random_seed,
        class_names=class_names,
        model_fingerprint=model_fingerprint,
        calibration=calibration,
    )

    # Compute explanations if requested
    explanations_dict = None
    if explain:
        try:
            from glassalpha.explain import compute_explanations

            explanations_dict = compute_explanations(
                model=model,
                X=X,
                feature_names=feature_names_list,
                random_seed=random_seed,
            )
        except Exception as e:
            import logging
            import traceback

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to compute explanations: {e}")
            logger.debug(f"Explanation error traceback: {traceback.format_exc()}")
            # Continue without explanations rather than failing the entire audit

    # Update manifest with model-specific info
    result_dict = {
        "id": result.id,
        "schema_version": result.schema_version,
        "manifest": {
            **dict(result.manifest),
            "model_type": model_type,
            "n_features": X_arr.shape[1],
            "feature_names": feature_names_list,
            "data_hash_X": hash_data_for_manifest(X_arr),
        },
        "performance": dict(result.performance),
        "fairness": dict(result.fairness),
        "calibration": dict(result.calibration),
        "stability": dict(result.stability),
        "explanations": explanations_dict,
        "recourse": result.recourse,
    }

    # Create new AuditResult with updated manifest
    return AuditResult(**result_dict)


__all__ = ["from_model"]
