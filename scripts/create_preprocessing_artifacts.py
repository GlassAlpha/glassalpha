#!/usr/bin/env python3
"""Create production preprocessing artifacts for GlassAlpha audits.

This script generates sklearn preprocessing pipelines that match the transformations
needed for each dataset. These artifacts are used for artifact-based preprocessing
verification in compliance audits.

Usage:
    python scripts/create_preprocessing_artifacts.py german_credit
    python scripts/create_preprocessing_artifacts.py --output-dir ./my_artifacts german_credit
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glassalpha.datasets import load_german_credit
from glassalpha.preprocessing.loader import compute_file_hash, compute_params_hash
from glassalpha.preprocessing.manifest import MANIFEST_SCHEMA_VERSION

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_german_credit_preprocessor() -> tuple[Pipeline, pd.DataFrame]:
    """Create production preprocessing pipeline for German Credit dataset.

    Returns:
        Tuple of (fitted_pipeline, training_data)

    """
    logger.info("Loading German Credit dataset...")
    train_data, test_data = load_german_credit(train_test_split=True, random_state=42)

    # Define feature types based on actual German Credit schema
    # These come from glassalpha/datasets/german_credit.py processing
    categorical_features = [
        "checking_account_status",
        "credit_history",
        "purpose",
        "savings_account",
        "employment_duration",
        "personal_status_sex",
        "other_debtors",
        "property",
        "other_installment_plans",
        "housing",
        "job",
        "telephone",
        "foreign_worker",
        "gender",  # Extracted demographic
        "age_group",  # Extracted demographic
    ]

    numeric_features = [
        "duration_months",
        "credit_amount",
        "installment_rate",
        "present_residence_since",
        "age_years",
        "existing_credits_count",
        "dependents_count",
    ]

    # Keep only features that exist in the data
    categorical_features = [f for f in categorical_features if f in train_data.columns]
    numeric_features = [f for f in numeric_features if f in train_data.columns]

    logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    logger.info(f"Numeric features ({len(numeric_features)}): {numeric_features}")

    # Create preprocessing pipeline matching production requirements
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ],
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "onehot",
                            OneHotEncoder(
                                drop="first",  # Drop first to avoid multicollinearity
                                handle_unknown="ignore",  # Production safety
                                sparse_output=False,  # Dense for most models
                            ),
                        ),
                    ],
                ),
                categorical_features,
            ),
        ],
        remainder="drop",  # Drop any other columns not specified
        verbose_feature_names_out=True,  # Useful for debugging
    )

    # Fit the preprocessor
    logger.info("Fitting preprocessor on training data...")
    X_train = train_data[categorical_features + numeric_features]
    preprocessor.fit(X_train)

    # Test transform to verify it works
    logger.info("Testing transform...")
    X_transformed = preprocessor.transform(X_train)
    logger.info(f"Transformed shape: {X_transformed.shape}")
    logger.info(f"Output type: {type(X_transformed)}")
    logger.info(f"Output dtype: {X_transformed.dtype}")

    return preprocessor, train_data


def save_artifact_with_manifest(
    preprocessor: Pipeline,
    train_data: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
) -> dict:
    """Save preprocessing artifact and generate manifest.

    Args:
        preprocessor: Fitted sklearn preprocessing pipeline
        train_data: Training data used to fit the preprocessor
        output_dir: Directory to save artifact and manifest
        dataset_name: Name of the dataset (e.g., "german_credit")

    Returns:
        Manifest dictionary with all metadata

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the artifact
    artifact_path = output_dir / f"{dataset_name}_preprocessor.joblib"
    logger.info(f"Saving artifact to {artifact_path}...")
    joblib.dump(preprocessor, artifact_path)

    # Compute hashes
    logger.info("Computing file hash...")
    file_hash = compute_file_hash(artifact_path)

    logger.info("Computing params hash...")
    # Extract manifest data
    from glassalpha.preprocessing.introspection import compute_unknown_rates, extract_sklearn_manifest

    manifest_data = extract_sklearn_manifest(preprocessor)

    # Add unknown category rates
    logger.info("Computing unknown category rates...")
    unknown_rates = compute_unknown_rates(preprocessor, train_data)
    manifest_data["unknown_categories"] = unknown_rates

    params_hash_value = compute_params_hash(manifest_data)

    # Create full manifest
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": dataset_name,
        "artifact_path": str(artifact_path),
        "created_at": pd.Timestamp.now().isoformat(),
        "file_hash": file_hash,
        "params_hash": params_hash_value,
        "sklearn_version": __import__("sklearn").__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "preprocessing": manifest_data,
    }

    # Save manifest
    manifest_path = output_dir / f"{dataset_name}_preprocessor_manifest.json"
    logger.info(f"Saving manifest to {manifest_path}...")
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("ARTIFACT CREATED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Artifact path: {artifact_path}")
    logger.info(f"Manifest path: {manifest_path}")
    logger.info(f"File hash:     {file_hash}")
    logger.info(f"Params hash:   {params_hash_value}")
    logger.info("=" * 80)

    # Print config snippet for easy copy-paste
    print("\n" + "=" * 80)
    print("CONFIG SNIPPET (copy to your audit config)")
    print("=" * 80)
    print("preprocessing:")
    print("  mode: artifact")
    print(f"  artifact_path: {artifact_path}")
    print(f"  expected_file_hash: '{file_hash}'")
    print(f"  expected_params_hash: '{params_hash_value}'")
    print("  expected_sparse: false")
    print("  fail_on_mismatch: true")
    print("=" * 80)

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Create preprocessing artifacts for GlassAlpha audits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        choices=["german_credit"],
        help="Dataset to create preprocessor for",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Output directory for artifacts (default: ./artifacts)",
    )

    args = parser.parse_args()

    logger.info(f"Creating preprocessing artifact for {args.dataset}")

    if args.dataset == "german_credit":
        preprocessor, train_data = create_german_credit_preprocessor()
        manifest = save_artifact_with_manifest(
            preprocessor,
            train_data,
            args.output_dir,
            args.dataset,
        )
    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

    logger.info("Done!")


if __name__ == "__main__":
    main()
