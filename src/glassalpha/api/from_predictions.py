"""from_predictions() entry point - audit from predictions only.

Used when you have predictions but not the model itself (e.g., external model,
model deleted, compliance verification).

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
    InvalidProtectedAttributesError,
    LengthMismatchError,
    NonBinaryClassificationError,
)


def _to_numpy(arr: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert pandas objects to numpy arrays safely."""
    import numpy as np
    import pandas as pd

    if isinstance(arr, (pd.Series, pd.DataFrame)):
        return arr.to_numpy()
    return np.asarray(arr)


def _validate_binary_classification(
    y: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray | None = None,
) -> None:
    """Validate that labels are binary (0/1 or two unique values).

    Args:
        y: Label array
        y_pred: Predicted labels (optional, for validation)

    Raises:
        GlassAlphaError (GAE1004): Non-binary classification

    """
    import numpy as np
    import pandas as pd

    # Convert to numpy
    y_arr = _to_numpy(y)

    # Check unique values
    unique_values = np.unique(y_arr[~pd.isna(y_arr)])
    n_classes = len(unique_values)

    if n_classes != 2:
        raise NonBinaryClassificationError(n_classes)

    # If y_pred provided, also validate it
    if y_pred is not None:
        y_pred_arr = _to_numpy(y_pred)
        unique_pred = np.unique(y_pred_arr[~pd.isna(y_pred_arr)])
        n_classes_pred = len(unique_pred)

        if n_classes_pred > 2:
            raise NonBinaryClassificationError(n_classes_pred)


def _normalize_protected_attributes(
    protected_attributes: Mapping[str, pd.Series | np.ndarray] | None,
    n_samples: int,
) -> dict[str, np.ndarray] | None:
    """Normalize protected attributes to dict of arrays with "Unknown" for NaN.

    Validates:
    - All arrays have length n_samples
    - Converts pandas Series to numpy arrays
    - Maps NaN → "Unknown" category (string)
    - Validates attribute names are strings

    Args:
        protected_attributes: Dict of attribute arrays or None
        n_samples: Expected number of samples

    Returns:
        Dict of numpy arrays with NaN → "Unknown", or None

    Raises:
        GlassAlphaError (GAE1001): Invalid format or names
        GlassAlphaError (GAE1003): Length mismatch

    """
    import numpy as np
    import pandas as pd

    if protected_attributes is None:
        return None

    # Validate it's a mapping
    if not isinstance(protected_attributes, Mapping):
        raise InvalidProtectedAttributesError(
            "protected_attributes must be a dictionary",
        )

    normalized = {}
    for name, arr in protected_attributes.items():
        # Validate attribute name is string
        if not isinstance(name, str):
            raise InvalidProtectedAttributesError(
                f"attribute names must be strings, got {type(name).__name__}",
            )

        # Convert to numpy array
        if isinstance(arr, pd.Series):
            arr_np = arr.values
        elif isinstance(arr, np.ndarray):
            arr_np = arr
        else:
            try:
                arr_np = np.asarray(arr)
            except Exception as e:
                raise InvalidProtectedAttributesError(
                    f"attribute '{name}' could not be converted to array: {e}",
                ) from e

        # Validate length
        if len(arr_np) != n_samples:
            raise LengthMismatchError(
                expected=n_samples,
                got=len(arr_np),
                name=f"protected_attributes['{name}']",
            )

        # Map NaN → "Unknown" (convert to object dtype to handle mixed types)
        arr_normalized = arr_np.astype(object)
        mask = pd.isna(arr_normalized)
        if mask.any():
            arr_normalized[mask] = "Unknown"

        normalized[name] = arr_normalized

    return normalized


def _compute_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    sample_weight: np.ndarray | None,
) -> dict[str, Any]:
    """Compute performance metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        sample_weight: Sample weights (optional)

    Returns:
        Dictionary with performance metrics

    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {}

    # Always compute label-based metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred, sample_weight=sample_weight))
    metrics["precision"] = float(
        precision_score(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            zero_division=0,
        ),
    )
    metrics["recall"] = float(
        recall_score(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            zero_division=0,
        ),
    )
    metrics["f1"] = float(
        f1_score(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            zero_division=0,
        ),
    )

    # Compute probability-based metrics if y_proba provided
    if y_proba is not None:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                y_proba,
                sample_weight=sample_weight,
            ),
        )

        # PR AUC
        precision, recall, _ = precision_recall_curve(y_true, y_proba, sample_weight=sample_weight)
        # Use trapezoidal integration for PR AUC
        metrics["pr_auc"] = float(np.trapezoid(precision, recall))

        # Brier score
        metrics["brier_score"] = float(
            brier_score_loss(
                y_true,
                y_proba,
                sample_weight=sample_weight,
            ),
        )

        # Log loss
        metrics["log_loss"] = float(
            log_loss(
                y_true,
                y_proba,
                sample_weight=sample_weight,
            ),
        )

    metrics["n_samples"] = len(y_true)

    return metrics


def _compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attributes: dict[str, np.ndarray],
    random_seed: int,
) -> dict[str, Any]:
    """Compute fairness metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        protected_attributes: Dict of protected attribute arrays
        random_seed: Random seed for bootstrap CIs

    Returns:
        Dictionary with fairness metrics

    """
    import numpy as np

    results = {}

    # For each protected attribute, compute demographic parity metrics
    for attr_name, attr_values in protected_attributes.items():
        # Convert to string to handle mixed types (e.g., "Unknown" mixed with numbers)
        attr_values_str = attr_values.astype(str)
        unique_groups = np.unique(attr_values_str)

        group_rates = []
        for group in unique_groups:
            mask = attr_values_str == group
            if np.sum(mask) > 0:
                rate = np.mean(y_pred[mask])
                results[f"{attr_name}_{group}"] = float(rate)
                group_rates.append(rate)

        # Compute max difference for this attribute
        if len(group_rates) >= 2:
            results[f"{attr_name}_max_diff"] = float(
                max(group_rates) - min(group_rates),
            )

    # Overall max diff across all attributes
    all_diffs = [v for k, v in results.items() if k.endswith("_max_diff")]
    if all_diffs:
        results["demographic_parity_max_diff"] = float(max(all_diffs))

    return results


def _compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    random_seed: int,
) -> dict[str, Any]:
    """Compute calibration metrics.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        random_seed: Random seed for bootstrap CIs

    Returns:
        Dictionary with calibration metrics

    """
    from glassalpha.metrics.calibration.quality import assess_calibration_quality

    # Compute calibration metrics (without CIs for now)
    calib_result = assess_calibration_quality(
        y_true=y_true,
        y_prob_pos=y_proba,
        n_bins=10,
        compute_confidence_intervals=False,
        seed=random_seed,
    )

    return calib_result


def from_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: pd.DataFrame | np.ndarray | None = None,
    *,
    protected_attributes: Mapping[str, pd.Series | np.ndarray] | None = None,
    sample_weight: pd.Series | np.ndarray | None = None,
    random_seed: int = 42,
    class_names: Sequence[str] | None = None,
    model_fingerprint: str | None = None,
    calibration: bool = True,
) -> AuditResult:
    """Generate audit from predictions (no model required).

    Use this when you have predictions but not the model itself
    (e.g., external model, model deleted, compliance verification).

    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        y_proba: Predicted probabilities (n_samples, n_classes) or (n_samples,) for binary.
            Required for AUC, Brier score, calibration metrics.
        protected_attributes: Dict mapping attribute names to arrays
        sample_weight: Sample weights for weighted metrics (optional)
        random_seed: Random seed for deterministic operations
        class_names: Class names (e.g., ["Denied", "Approved"])
        model_fingerprint: Optional model hash for tracking (default: "unknown")
        calibration: Compute calibration metrics (default: True, requires y_proba)

    Returns:
        AuditResult with performance and fairness metrics.
        No explanations or recourse (requires model).

    Raises:
        GlassAlphaError (GAE1001): Invalid protected_attributes format
        GlassAlphaError (GAE1003): Length mismatch
        GlassAlphaError (GAE1004): Non-binary classification

    Examples:
        Binary classification with probabilities:
        >>> result = ga.audit.from_predictions(
        ...     y_true=y_test,
        ...     y_pred=predictions,
        ...     y_proba=probabilities[:, 1],  # Positive class proba
        ...     protected_attributes={"gender": gender}
        ... )
        >>> result.performance.roc_auc
        0.891

        Without probabilities (no AUC/calibration):
        >>> result = ga.audit.from_predictions(
        ...     y_true=y_test,
        ...     y_pred=predictions,
        ...     protected_attributes={"gender": gender},
        ...     calibration=False
        ... )
        >>> result.performance.accuracy
        0.847
        >>> result.performance.get("roc_auc")  # None (no proba)
        None

    """

    import numpy as np

    import glassalpha
    from glassalpha.utils.canonicalization import compute_result_id, hash_data_for_manifest

    # Convert inputs to numpy arrays
    y_true_arr = _to_numpy(y_true)
    y_pred_arr = _to_numpy(y_pred)

    # Validate lengths match
    n_samples = len(y_true_arr)
    if len(y_pred_arr) != n_samples:
        raise LengthMismatchError(
            expected=n_samples,
            got=len(y_pred_arr),
            name="y_pred",
        )

    # Handle y_proba (extract positive class probabilities if 2D)
    y_proba_arr = None
    if y_proba is not None:
        y_proba_arr = _to_numpy(y_proba)

        # Validate length
        if len(y_proba_arr) != n_samples:
            raise LengthMismatchError(
                expected=n_samples,
                got=len(y_proba_arr),
                name="y_proba",
            )

        # If 2D (n_samples, n_classes), extract positive class
        if y_proba_arr.ndim == 2:
            y_proba_arr = y_proba_arr[:, 1]

    # Validate binary classification
    _validate_binary_classification(y_true_arr, y_pred_arr)

    # Normalize protected attributes
    protected_attrs_normalized = _normalize_protected_attributes(
        protected_attributes,
        n_samples,
    )

    # Compute performance metrics
    perf_metrics = _compute_performance_metrics(
        y_true_arr,
        y_pred_arr,
        y_proba_arr,
        sample_weight,
    )

    # Compute fairness metrics (if protected attributes provided)
    fairness_metrics = {}
    if protected_attrs_normalized is not None:
        fairness_metrics = _compute_fairness_metrics(
            y_true_arr,
            y_pred_arr,
            protected_attrs_normalized,
            random_seed,
        )

    # Compute calibration metrics (if y_proba provided and calibration=True)
    calibration_metrics = {}
    if calibration and y_proba_arr is not None:
        calibration_metrics = _compute_calibration_metrics(
            y_true_arr,
            y_proba_arr,
            random_seed,
        )

    # Build manifest
    manifest = {
        "glassalpha_version": glassalpha.__version__,
        "schema_version": "1.0.0",
        "random_seed": random_seed,
        "model_fingerprint": model_fingerprint or "unknown",
        "n_samples": len(y_true_arr),
        "data_hash_y": hash_data_for_manifest(y_true_arr),
        "protected_attributes_categories": {
            name: list(np.unique(arr.astype(str))) for name, arr in (protected_attrs_normalized or {}).items()
        },
    }

    # Compute result ID (deterministic hash of manifest)
    result_id = compute_result_id(manifest)

    # Create AuditResult
    return AuditResult(
        id=result_id,
        schema_version="1.0.0",
        manifest=manifest,
        performance=perf_metrics,
        fairness=fairness_metrics,
        calibration=calibration_metrics,
        stability={},
        explanations=None,
        recourse=None,
    )


__all__ = ["from_predictions"]
