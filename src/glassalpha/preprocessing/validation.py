"""Validation functions for preprocessing artifacts."""

from pathlib import Path
from typing import Any

# Allowed sklearn transformer classes (tight allowlist for Phase 1)
ALLOWED_FQCN = {
    "sklearn.pipeline.Pipeline",
    "sklearn.compose._column_transformer.ColumnTransformer",
    "sklearn.impute._base.SimpleImputer",
    "sklearn.preprocessing._encoders.OneHotEncoder",
    "sklearn.preprocessing._encoders.OrdinalEncoder",
    "sklearn.preprocessing._data.StandardScaler",
    "sklearn.preprocessing._data.MinMaxScaler",
    "sklearn.preprocessing._data.RobustScaler",
    "sklearn.preprocessing._discretization.KBinsDiscretizer",
    "sklearn.preprocessing._polynomial.PolynomialFeatures",
}


def fqcn(obj: Any) -> str:
    """Get fully-qualified class name (module + class)."""
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__name__}"


def validate_classes(artifact: Any) -> None:
    """Validate all transformers in artifact are in allowlist.

    Recursively traverses Pipeline and ColumnTransformer.

    Args:
        artifact: Sklearn preprocessing artifact

    Raises:
        ValueError: If unsupported transformer found (with FQCN)

    """
    from sklearn.base import BaseEstimator
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    if isinstance(artifact, Pipeline):
        for _, step in artifact.steps:
            validate_classes(step)
    elif isinstance(artifact, ColumnTransformer):
        for _, transformer, _ in artifact.transformers:
            validate_classes(transformer)
    elif isinstance(artifact, BaseEstimator):
        class_name = fqcn(artifact)
        if class_name not in ALLOWED_FQCN:
            raise ValueError(
                f"Unsupported transformer {class_name} in preprocessing artifact. "
                f"Allowed classes: {', '.join(sorted(ALLOWED_FQCN))}",
            )


def validate_sparsity(actual_sparse: bool, expected_sparse: bool | None, artifact_path: Path) -> None:
    """Validate output sparsity matches expectations.

    Args:
        actual_sparse: Whether artifact produces sparse output
        expected_sparse: Expected sparsity (None = skip check)
        artifact_path: Path for error messages

    Raises:
        ValueError: If sparsity mismatch

    """
    if expected_sparse is None:
        return

    if actual_sparse != expected_sparse:
        raise ValueError(
            f"Sparsity mismatch in {artifact_path}: "
            f"expected sparse={expected_sparse}, got sparse={actual_sparse}. "
            f"Hint: Check OneHotEncoder sparse_output parameter.",
        )


def validate_output_shape(X_transformed: Any, model: Any, artifact: Any) -> None:
    """Validate transformed output matches model expectations.

    Checks both feature count and feature names (if available).

    Args:
        X_transformed: Transformed data
        model: Model with n_features_in_ and feature_names_in_
        artifact: Preprocessing artifact

    Raises:
        ValueError: If shape mismatch (with minimal diff)

    """
    # Get actual shape
    if hasattr(X_transformed, "shape"):
        actual_n_features = X_transformed.shape[1]
    else:
        actual_n_features = len(X_transformed[0]) if len(X_transformed) > 0 else 0

    # Check feature count
    expected_n_features = getattr(model, "n_features_in_", None)
    if expected_n_features is not None:
        if actual_n_features != expected_n_features:
            raise ValueError(
                f"Feature count mismatch: preprocessor outputs {actual_n_features} features, "
                f"but model expects {expected_n_features}. "
                f"Difference: {actual_n_features - expected_n_features:+d}",
            )

    # Check feature names if available
    expected_names = getattr(model, "feature_names_in_", None)
    actual_names = None

    if hasattr(artifact, "get_feature_names_out"):
        try:
            actual_names = list(artifact.get_feature_names_out())
        except Exception:
            pass

    if expected_names is not None and actual_names is not None:
        expected_set = set(expected_names)
        actual_set = set(actual_names)

        if expected_set != actual_set:
            missing = expected_set - actual_set
            extra = actual_set - expected_set

            error_parts = ["Feature name mismatch:"]
            if missing:
                error_parts.append(f"  Missing: {sorted(missing)[:5]}")
            if extra:
                error_parts.append(f"  Extra: {sorted(extra)[:5]}")

            raise ValueError("\n".join(error_parts))


def assert_runtime_versions(manifest: dict[str, Any], strict: bool, allow_minor: bool = False) -> None:
    """Validate sklearn/numpy/scipy versions match between artifact and audit runtime.

    Args:
        manifest: Manifest with artifact_runtime_versions and audit_runtime_versions
        strict: If True, raise on mismatch; if False, warn only
        allow_minor: If True, allow minor version drift (1.3.x -> 1.5.x)

    Raises:
        RuntimeError: If version mismatch in strict mode

    """
    import warnings

    artifact_versions = manifest.get("artifact_runtime_versions", {})
    audit_versions = manifest.get("audit_runtime_versions", {})

    mismatches = []
    for lib in ["sklearn", "numpy", "scipy"]:
        artifact_ver = artifact_versions.get(lib)
        audit_ver = audit_versions.get(lib)

        if artifact_ver and audit_ver and artifact_ver != audit_ver:
            # Always record mismatches, but check if they're acceptable
            is_minor_drift = False
            if allow_minor:
                try:
                    artifact_major = artifact_ver.split(".")[0]
                    audit_major = audit_ver.split(".")[0]
                    is_minor_drift = artifact_major == audit_major
                except Exception:
                    pass

            # In strict mode, minor drift is only acceptable if allow_minor=True
            # In non-strict mode, always warn but don't fail
            if (strict and not is_minor_drift) or not strict:
                mismatches.append(f"{lib}: {artifact_ver} -> {audit_ver}")

    if mismatches:
        message = f"Runtime version mismatch: {', '.join(mismatches)}"

        if strict:
            raise RuntimeError(message)
        warnings.warn(message, UserWarning, stacklevel=2)
