from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


# ---------- Public API expected by tests ----------


def assess_calibration_quality(
    y_true: Iterable[int],
    y_prob_pos: Iterable[float],
    *,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute simple calibration quality diagnostics for binary classification.

    Returns a dict containing:
      - brier_score
      - expected_calibration_error (ECE, equal-width bins)
      - maximum_calibration_error (MCE)
    """
    y_true_arr = np.asarray(y_true).astype(int)
    prob = np.asarray(y_prob_pos, dtype=float)

    if y_true_arr.ndim != 1 or prob.ndim != 1 or y_true_arr.shape[0] != prob.shape[0]:
        raise ValueError("y_true and y_prob_pos must be 1D arrays of the same length")

    # Brier score
    brier = brier_score_loss(y_true_arr, prob)

    # ECE / MCE with equal-width bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(prob, bins) - 1
    ece = 0.0
    mce = 0.0
    n = len(prob)

    for b in range(n_bins):
        in_bin = bin_ids == b
        m = np.count_nonzero(in_bin)
        if m == 0:
            continue
        avg_conf = float(np.mean(prob[in_bin]))
        avg_acc = float(np.mean(y_true_arr[in_bin]))
        gap = abs(avg_acc - avg_conf)
        ece += (m / n) * gap
        mce = max(mce, gap)

    return {
        "brier_score": float(brier),
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
    }


def get_calibration_info(estimator: Any) -> dict[str, Any]:
    """Describe whether an estimator is a CalibratedClassifierCV and expose key fields."""
    info: dict[str, Any] = {
        "is_calibrated": False,
        "calibration_method": None,
        "cv_folds": None,
        "base_estimator_type": type(estimator).__name__,
    }

    if isinstance(estimator, CalibratedClassifierCV):
        info["is_calibrated"] = True
        # method_: attribute on CalibratedClassifierCV
        method = getattr(estimator, "method", None) or getattr(estimator, "method_", None)
        info["calibration_method"] = str(method) if method is not None else None
        info["cv_folds"] = getattr(estimator, "cv", None)
        base = getattr(estimator, "estimator", None)
        if base is not None:
            info["base_estimator_type"] = type(base).__name__

    return info


def validate_calibration_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize and validate a calibration configuration dict.

    Returns a dict with keys: method (str|None), cv (int), ensemble (bool)

    Raises ValueError for invalid configurations.
    """
    cfg = dict(cfg or {})
    method = cfg.get("method")
    cv = cfg.get("cv", 5)
    ensemble = cfg.get("ensemble", False)

    if method is not None:
        method_norm = str(method).lower()
        if method_norm not in {"isotonic", "sigmoid"}:
            raise ValueError("Unknown calibration method")
        method = method_norm

    # cv must be int >= 2
    if not isinstance(cv, int):
        raise ValueError("cv must be an integer")
    if cv < 2:
        raise ValueError("cv must be >= 2")

    if not isinstance(ensemble, bool):
        raise ValueError("ensemble must be a boolean")

    return {"method": method, "cv": cv, "ensemble": ensemble}


def recommend_calibration_method(model_name: str, n_samples: int, *, min_samples_isotonic: int = 1000) -> str:
    """Heuristic recommendation of calibration method based on model family and sample size.

    - Small datasets favor 'sigmoid'
    - Tree-based models favor 'isotonic' when sufficiently large
    - Unknown large datasets default to 'isotonic'
    """
    name = (model_name or "").lower()
    if n_samples < min_samples_isotonic:
        return "sigmoid"

    tree_families = {
        "randomforestclassifier",
        "xgbclassifier",
        "lgbmclassifier",
        "decisiontreeclassifier",
        "extratreesclassifier",
        "catboostclassifier",
        "gradientboostingclassifier",
    }
    if any(fam in name for fam in tree_families):
        return "isotonic"

    linear_families = {"logisticregression", "linearsvc", "sgdclassifier"}
    if any(fam in name for fam in linear_families):
        return "sigmoid"

    # Large unknown: prefer isotonic
    return "isotonic"


def maybe_calibrate(
    estimator: Any,
    method: str | None = None,
    cv: int | str | None = None,
    ensemble: bool = False,
) -> Any:
    """Return a CalibratedClassifierCV wrapper if a calibration method is requested;
    otherwise return the estimator unchanged.

    Note: This function does not call `.fit()`. It only constructs the wrapper so
    that callers can inspect configuration (e.g., for reporting) or fit later.
    """
    if method is None:
        return estimator

    method_norm = str(method).lower()
    if method_norm not in {"isotonic", "sigmoid"}:
        raise ValueError("Unknown calibration method")

    # default cv
    if cv is None:
        cv = 5
    if isinstance(cv, str) and cv.lower() in {"prefit"}:
        # We don't support prefit pathway here; keep interface simple for tests
        raise ValueError("cv='prefit' not supported in this context")

    if not isinstance(cv, int) or cv < 2:
        raise ValueError("cv must be an integer >= 2")

    try:
        wrapper = CalibratedClassifierCV(
            estimator=estimator,
            method=method_norm,
            cv=cv,
            ensemble=bool(ensemble),
        )
        return wrapper
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"Failed to create calibrated estimator: {exc}")
        logger.warning("Falling back to uncalibrated estimator")
        return estimator
