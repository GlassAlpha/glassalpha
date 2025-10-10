"""Introspect sklearn artifacts to extract learned parameters."""

from typing import Any

import numpy as np
import pandas as pd


def extract_sklearn_manifest(artifact: Any) -> dict[str, Any]:
    """Extract learned parameters from sklearn preprocessing artifact.

    Args:
        artifact: Sklearn Pipeline or ColumnTransformer

    Returns:
        Manifest dictionary with components, learned params, versions, etc.

    """
    import sklearn

    from glassalpha.preprocessing.manifest import MANIFEST_SCHEMA_VERSION

    manifest: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "components": [],
        "artifact_runtime_versions": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "audit_runtime_versions": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }

    # Add scipy if available
    try:
        import scipy

        manifest["artifact_runtime_versions"]["scipy"] = scipy.__version__
        manifest["audit_runtime_versions"]["scipy"] = scipy.__version__
    except ImportError:
        pass

    # Extract components

    # Use recursive extraction for all artifact types
    manifest["components"].extend(_extract_components_recursive("", artifact))

    # Detect output dtype and sparsity (requires sample transform)
    manifest["output_dtype"] = "unknown"  # Will be set during transform

    return manifest


def _extract_components_recursive(name: str, transformer: Any) -> list[dict[str, Any]]:
    """Recursively extract all leaf components from transformer (flattens nested structures)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    components = []

    if isinstance(transformer, Pipeline):
        # Recurse into pipeline steps
        # Use .steps which contains fitted transformers
        for step_name, step in transformer.steps:
            components.extend(_extract_components_recursive(f"{name}.{step_name}", step))
    elif isinstance(transformer, ColumnTransformer):
        # Recurse into column transformer transformers
        # IMPORTANT: Use transformers_ (with underscore) to get FITTED transformers
        transformers_to_use = (
            transformer.transformers_ if hasattr(transformer, "transformers_") else transformer.transformers
        )
        for trans_name, trans, columns in transformers_to_use:
            sub_components = _extract_components_recursive(f"{name}.{trans_name}", trans)
            # Attach column info
            for comp in sub_components:
                if "columns" not in comp:
                    comp["columns"] = list(columns) if columns is not None else None
            components.extend(sub_components)
    else:
        # Leaf transformer - extract its info
        comp = _extract_component(name, transformer)
        components.append(comp)

    return components


def _extract_component(name: str, transformer: Any) -> dict[str, Any]:
    """Extract parameters from a single transformer."""
    from glassalpha.preprocessing.validation import fqcn

    component: dict[str, Any] = {
        "name": name,
        "class": fqcn(transformer),
    }

    # Extract type-specific parameters
    class_name = transformer.__class__.__name__

    if class_name == "OneHotEncoder":
        component["handle_unknown"] = transformer.handle_unknown
        component["drop"] = str(transformer.drop)
        component["sparse_output"] = transformer.sparse_output
        # Categories (truncate if too long)
        if hasattr(transformer, "categories_"):
            n_categories = [len(cats) for cats in transformer.categories_]
            component["n_categories"] = n_categories
            # Only store first few categories per feature (full list goes to JSON)
            component["categories"] = [list(cats)[:50] for cats in transformer.categories_]

    elif class_name == "SimpleImputer":
        component["strategy"] = transformer.strategy
        if hasattr(transformer, "statistics_"):
            component["learned_stats"] = transformer.statistics_.tolist()

    elif class_name == "StandardScaler":
        if hasattr(transformer, "mean_"):
            component["mean"] = transformer.mean_.tolist()
        if hasattr(transformer, "scale_"):
            component["scale"] = transformer.scale_.tolist()

    elif class_name == "MinMaxScaler":
        if hasattr(transformer, "min_"):
            component["min"] = transformer.min_.tolist()
        if hasattr(transformer, "scale_"):
            component["scale"] = transformer.scale_.tolist()

    elif class_name == "RobustScaler":
        if hasattr(transformer, "center_"):
            component["center"] = transformer.center_.tolist()
        if hasattr(transformer, "scale_"):
            component["scale"] = transformer.scale_.tolist()

    return component


def compute_unknown_rates(artifact: Any, X: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute unknown category rates for eval data (pre-transform).

    Args:
        artifact: Fitted preprocessing artifact
        X: Raw evaluation DataFrame

    Returns:
        Dictionary mapping feature -> {rate, count, top_unknowns}

    """
    unknown_rates: dict[str, dict[str, Any]] = {}

    # Find all OneHotEncoders
    encoders = _find_encoders(artifact)

    for encoder_info in encoders:
        encoder = encoder_info["encoder"]
        columns = encoder_info["columns"]

        if not hasattr(encoder, "categories_"):
            continue

        for i, col in enumerate(columns):
            if col not in X.columns:
                continue

            training_cats = set(encoder.categories_[i])
            eval_cats = set(X[col].dropna().unique())
            unknown_cats = eval_cats - training_cats

            if unknown_cats:
                n_unknown = X[col].isin(unknown_cats).sum()
                n_total = len(X)
                rate = n_unknown / n_total if n_total > 0 else 0.0

                unknown_rates[col] = {
                    "rate": rate,
                    "count": n_unknown,
                    "top_unknowns": sorted(unknown_cats)[:10],
                }

    return unknown_rates


def _find_encoders(artifact: Any, columns: list[str] | None = None) -> list[dict[str, Any]]:
    """Find all OneHotEncoders in artifact with their columns."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    encoders = []

    if isinstance(artifact, OneHotEncoder):
        encoders.append({"encoder": artifact, "columns": columns or []})

    elif isinstance(artifact, Pipeline):
        # Propagate columns through pipeline
        for _, step in artifact.steps:
            encoders.extend(_find_encoders(step, columns))

    elif isinstance(artifact, ColumnTransformer):
        # Each transformer gets its own column mapping
        # IMPORTANT: Use transformers_ (with underscore) to get FITTED transformers
        transformers_to_use = artifact.transformers_ if hasattr(artifact, "transformers_") else artifact.transformers
        for _, transformer, trans_columns in transformers_to_use:
            encoders.extend(_find_encoders(transformer, trans_columns))

    return encoders
