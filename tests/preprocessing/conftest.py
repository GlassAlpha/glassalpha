"""Fixtures for preprocessing tests."""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """Small DataFrame with 2 numeric, 2 categorical columns."""
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 30, 40],
            "credit_amount": [1000, 2000, 3000, 1500, 2500],
            "employment": ["skilled", "unskilled", "skilled", "management", "skilled"],
            "housing": ["own", "rent", "own", "own", "rent"],
        },
    )


@pytest.fixture
def sklearn_artifact(tmp_path: Path, toy_df: pd.DataFrame):
    """Creates and saves a fitted Pipeline with SimpleImputer+OneHotEncoder+StandardScaler."""
    numeric_features = ["age", "credit_amount"]
    categorical_features = ["employment", "housing"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ],
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(drop="first", sparse_output=True, handle_unknown="ignore")),
        ],
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
    )

    # Fit on toy data
    preprocessor.fit(toy_df)

    # Save to temp path
    artifact_path = tmp_path / "preprocessor.pkl"
    joblib.dump(preprocessor, artifact_path)

    return artifact_path


@pytest.fixture
def corrupted_artifact(tmp_path: Path) -> Path:
    """Writes junk bytes to simulate corrupted artifact."""
    artifact_path = tmp_path / "corrupted.pkl"
    artifact_path.write_bytes(b"JUNK DATA NOT A PICKLE")
    return artifact_path


@pytest.fixture
def mismatched_version_manifest() -> dict:
    """Fakes a version diff for strict tests."""
    return {
        "artifact_runtime_versions": {
            "sklearn": "1.3.2",
            "numpy": "1.24.0",
            "scipy": "1.10.0",
        },
        "audit_runtime_versions": {
            "sklearn": "1.5.0",  # Major mismatch
            "numpy": "1.26.0",  # Minor mismatch
            "scipy": "1.11.0",  # Minor mismatch
        },
    }


@pytest.fixture
def toy_df_with_unknowns(toy_df: pd.DataFrame) -> pd.DataFrame:
    """Add unseen categories to eval data."""
    eval_df = toy_df.copy()
    # Add unknown categories not in training
    eval_df.loc[0, "employment"] = "consultant"  # New category
    eval_df.loc[1, "housing"] = "parents"  # New category
    return eval_df


@pytest.fixture
def sparse_artifact(tmp_path: Path, toy_df: pd.DataFrame):
    """Creates artifact with explicit sparse_output=True."""
    # Use OneHotEncoder directly to ensure sparse output
    encoder = OneHotEncoder(sparse_output=True)
    encoder.fit(toy_df[["employment", "housing"]])

    artifact_path = tmp_path / "sparse_preprocessor.pkl"
    joblib.dump(encoder, artifact_path)
    return artifact_path


@pytest.fixture
def dense_artifact(tmp_path: Path, toy_df: pd.DataFrame):
    """Creates artifact with explicit sparse_output=False."""
    # Use OneHotEncoder directly to ensure dense output
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(toy_df[["employment", "housing"]])

    artifact_path = tmp_path / "dense_preprocessor.pkl"
    joblib.dump(encoder, artifact_path)
    return artifact_path
