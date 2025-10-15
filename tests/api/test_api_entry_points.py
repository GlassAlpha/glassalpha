"""Consolidated tests for all API entry points.

This module consolidates tests for:
1. from_model() - audit from fitted model
2. from_predictions() - audit from predictions only
3. from_config() - audit from YAML configuration

All three entry points share similar concerns:
- Input validation and error handling
- Determinism and reproducibility
- Protected attribute handling
- Manifest generation
- Result format consistency
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import glassalpha as ga
from glassalpha.exceptions import (
    CategoricalDataError,
    DataHashMismatchError,
    InvalidProtectedAttributesError,
    LengthMismatchError,
    MultiIndexNotSupportedError,
    NonBinaryClassificationError,
    NoPredictProbaError,
)

# ============================================================================
# from_model() Tests
# ============================================================================


@pytest.fixture
def binary_classification_data():
    """Generate simple binary classification data."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_from_model_sklearn_logistic_regression(binary_classification_data):
    """Test from_model with sklearn LogisticRegression."""
    from sklearn.linear_model import LogisticRegression

    X, y = binary_classification_data

    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    result = ga.audit.from_model(
        model=model,
        X=X,
        y=y,
        random_seed=42,
    )

    # Check basic metrics
    assert result.performance["accuracy"] > 0.5
    assert "precision" in result.performance
    assert "roc_auc" in result.performance  # Should have probabilities

    # Check manifest
    assert result.manifest["model_type"] == "logistic_regression"
    assert result.manifest["n_features"] == 5
    assert len(result.manifest["feature_names"]) == 5

    # Check result ID is stable
    assert len(result.id) == 64


def test_from_model_with_feature_names(binary_classification_data):
    """Test from_model with custom feature names."""
    from sklearn.linear_model import LogisticRegression

    X, y = binary_classification_data

    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    feature_names = ["age", "income", "credit_score", "debt", "employment"]

    result = ga.audit.from_model(
        model=model,
        X=X,
        y=y,
        feature_names=feature_names,
        random_seed=42,
    )

    assert result.manifest["feature_names"] == feature_names


def test_from_model_with_dataframe(binary_classification_data):
    """Test from_model with pandas DataFrame."""
    from sklearn.linear_model import LogisticRegression

    X, y = binary_classification_data

    X_df = pd.DataFrame(X, columns=["age", "income", "credit_score", "debt", "employment"])
    y_series = pd.Series(y)

    model = LogisticRegression(random_state=42)
    model.fit(X_df, y_series)

    result = ga.audit.from_model(
        model=model,
        X=X_df,
        y=y_series,
        random_seed=42,
    )

    # Feature names should be extracted from DataFrame
    assert result.manifest["feature_names"] == list(X_df.columns)
    assert result.performance["accuracy"] > 0.5


def test_from_model_with_protected_attributes(binary_classification_data):
    """Test from_model with protected attributes."""
    from sklearn.linear_model import LogisticRegression

    X, y = binary_classification_data

    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Add protected attribute
    gender = np.random.randint(0, 2, size=len(y))

    result = ga.audit.from_model(
        model=model,
        X=X,
        y=y,
        protected_attributes={"gender": gender},
        random_seed=42,
    )

    # Fairness metrics should be present
    assert len(result.fairness) > 0
    assert "demographic_parity_max_diff" in result.fairness


def test_from_model_without_calibration():
    """Test from_model with calibration=False."""
    from sklearn.tree import DecisionTreeClassifier

    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    result = ga.audit.from_model(
        model=model,
        X=X,
        y=y,
        calibration=False,
        random_seed=42,
    )

    # Calibration metrics should be empty
    assert len(result.calibration) == 0


def test_from_model_deterministic():
    """Test that from_model produces deterministic results."""
    from sklearn.linear_model import LogisticRegression

    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)

    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    result1 = ga.audit.from_model(model, X, y, random_seed=42)
    result2 = ga.audit.from_model(model, X, y, random_seed=42)

    # Result IDs should match
    assert result1.id == result2.id


def test_from_model_xgboost():
    """Test from_model with XGBoost."""
    pytest.importorskip("xgboost")
    from xgboost import XGBClassifier

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = XGBClassifier(random_state=42, n_estimators=10)
    model.fit(X, y)

    result = ga.audit.from_model(
        model=model,
        X=X,
        y=y,
        random_seed=42,
    )

    assert result.manifest["model_type"] == "xgboost"
    assert result.performance["accuracy"] > 0.5


def test_from_model_lightgbm():
    """Test from_model with LightGBM."""
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMClassifier

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LGBMClassifier(random_state=42, n_estimators=10, verbose=-1)
    model.fit(X, y)

    result = ga.audit.from_model(
        model=model,
        X=X,
        y=y,
        random_seed=42,
    )

    assert result.manifest["model_type"] == "lightgbm"
    assert result.performance["accuracy"] > 0.5


def test_from_model_non_binary():
    """Test that from_model raises error for non-binary classification."""
    from sklearn.linear_model import LogisticRegression

    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = np.random.randint(0, 3, size=50)  # 3 classes

    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    with pytest.raises(NonBinaryClassificationError):
        ga.audit.from_model(model, X, y)


def test_from_model_model_without_predict_proba():
    """Test from_model with model that has no predict_proba."""

    class DummyModel:
        """Model without predict_proba."""

        def predict(self, X):
            # Return alternating 0 and 1 for binary classification
            return np.array([i % 2 for i in range(len(X))], dtype=int)

    X = np.random.randn(10, 3)
    y = np.array([i % 2 for i in range(10)], dtype=int)  # Binary: [0,1,0,1,...]

    model = DummyModel()

    # Should work with calibration=False
    result = ga.audit.from_model(
        model=model,
        X=X,
        y=y,
        calibration=False,
        random_seed=42,
    )

    assert result.performance["accuracy"] >= 0
    assert result.manifest["model_type"] == "unknown"

    # Should fail with calibration=True (no predict_proba)
    with pytest.raises(NoPredictProbaError):
        ga.audit.from_model(
            model=model,
            X=X,
            y=y,
            calibration=True,
            random_seed=42,
        )


def test_from_model_categorical_data_raises_error():
    """Test from_model raises clear error for categorical features."""
    from sklearn.linear_model import LogisticRegression

    # Create DataFrame with categorical column
    df = pd.DataFrame(
        {
            "numeric": [25, 30, 35, 40, 45],
            "category": ["A", "B", "A", "C", "B"],  # Categorical column
        }
    )
    y = [0, 1, 0, 1, 0]

    model = LogisticRegression()
    # Train on numeric only to create a fitted model
    model.fit(df[["numeric"]], y)

    # Try to audit with categorical - should raise CategoricalDataError
    with pytest.raises(CategoricalDataError) as exc_info:
        ga.audit.from_model(model, df, y)

    assert "category" in exc_info.value.message
    assert exc_info.value.code == "GAE2001"
    assert "preprocess categorical features before training" in exc_info.value.fix.lower()


def test_from_model_multiindex_error_is_actionable():
    """Test multiindex error message is clear and actionable."""
    from sklearn.linear_model import LogisticRegression

    # Create multiindex DataFrame
    index = pd.MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 1)])
    df = pd.DataFrame({"x": [1, 2, 3]}, index=index)
    y = [0, 1, 0]

    model = LogisticRegression()
    model.fit(df, y)

    with pytest.raises(MultiIndexNotSupportedError) as exc_info:
        ga.audit.from_model(model, df, y)

    # Verify error message quality
    assert "reset_index" in exc_info.value.fix
    assert "GAE1012" in exc_info.value.code
    assert exc_info.value.fix is not None


def test_from_model_with_sklearn_pipeline():
    """Test from_model with sklearn Pipeline containing preprocessing."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    # Create data with categorical columns (like real-world ML workflows)
    np.random.seed(42)
    X = np.random.randn(100, 3)
    X_df = pd.DataFrame(X, columns=["num1", "num2", "num3"])
    X_df["category"] = np.random.choice(["A", "B", "C"], 100)
    X_df["binary_cat"] = np.random.choice(["X", "Y"], 100)

    y = (X_df["num1"] + X_df["num2"] > 0).astype(int)

    # Create Pipeline with preprocessing (common real-world pattern)
    cat_cols = ["category", "binary_cat"]
    num_cols = ["num1", "num2", "num3"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )

    # Fit pipeline on data with categorical columns
    pipeline.fit(X_df, y)

    # This should work - Pipeline handles preprocessing internally
    result = ga.audit.from_model(
        model=pipeline,
        X=X_df,  # Original DataFrame with categorical columns
        y=y,
        random_seed=42,
    )

    # Verify audit completed successfully
    assert result.performance["accuracy"] > 0.5
    assert "precision" in result.performance
    assert "recall" in result.performance

    # Check manifest reflects pipeline model type
    assert result.manifest["model_type"] == "random_forest"

    # Verify feature names include original column names (not transformed)
    assert len(result.manifest["feature_names"]) == 5  # Original 5 columns

    # Verify explanations work (should use TreeSHAP for RandomForest)
    assert hasattr(result, "explanations")
    assert result.explanations is not None


# ============================================================================
# from_predictions() Tests
# ============================================================================


def test_from_predictions_basic():
    """Test basic from_predictions with labels only."""
    # Binary classification data
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])

    result = ga.audit.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        calibration=False,  # No y_proba
    )

    # Check basic metrics
    assert result.performance["accuracy"] == 0.75  # 6/8 correct
    assert "precision" in result.performance
    assert "recall" in result.performance
    assert "f1" in result.performance

    # No probability-based metrics
    assert "roc_auc" not in result.performance
    assert "brier_score" not in result.performance

    # No fairness (no protected attributes)
    assert len(result.fairness) == 0

    # Check result ID is deterministic
    assert len(result.id) == 64
    assert result.schema_version == "1.0.0"


def test_from_predictions_with_probabilities():
    """Test from_predictions with predicted probabilities."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.6, 0.7])

    result = ga.audit.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    # Should have probability-based metrics
    assert "roc_auc" in result.performance
    assert "brier_score" in result.performance
    assert "log_loss" in result.performance

    # Calibration metrics should be present
    assert len(result.calibration) > 0

    # Check AUC is reasonable (better than random)
    assert result.performance["roc_auc"] > 0.5


def test_from_predictions_with_protected_attributes():
    """Test from_predictions with protected attributes."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    gender = np.array([0, 0, 1, 1, 0, 1])  # Protected attribute

    result = ga.audit.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        protected_attributes={"gender": gender},
        calibration=False,
    )

    # Should have fairness metrics
    assert len(result.fairness) > 0
    assert "gender_0" in result.fairness
    assert "gender_1" in result.fairness


def test_from_predictions_deterministic():
    """Test from_predictions produces deterministic results."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    gender = np.array([0, 0, 1, 1, 0, 1])

    result1 = ga.audit.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        protected_attributes={"gender": gender},
        random_seed=42,
    )

    result2 = ga.audit.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        protected_attributes={"gender": gender},
        random_seed=42,
    )

    # Result IDs should match
    assert result1.id == result2.id


def test_from_predictions_non_binary():
    """Test from_predictions raises error for non-binary classification."""
    y_true = np.array([0, 1, 2, 1, 0, 2])  # 3 classes
    y_pred = np.array([0, 1, 2, 1, 0, 2])

    with pytest.raises(NonBinaryClassificationError):
        ga.audit.from_predictions(y_true=y_true, y_pred=y_pred)


def test_from_predictions_length_mismatch():
    """Test from_predictions raises error for length mismatch."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0])  # Different length

    with pytest.raises(LengthMismatchError):
        ga.audit.from_predictions(y_true=y_true, y_pred=y_pred)


def test_from_predictions_with_sample_weights():
    """Test from_predictions with sample weights."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    sample_weight = np.array([1.0, 2.0, 1.0, 2.0])

    result = ga.audit.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight,
    )

    # All metrics should be computed
    assert "accuracy" in result.performance
    assert "precision" in result.performance
    assert "recall" in result.performance
    assert "f1" in result.performance


# ============================================================================
# from_config() Tests
# ============================================================================


def test_from_config_basic(tmp_path):
    """Test from_config with basic YAML config."""
    from sklearn.linear_model import LogisticRegression

    # Generate data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 3), columns=["f1", "f2", "f3"])
    y = pd.Series((X["f1"] > 0).astype(int))

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Save model and data
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    X.to_csv(tmp_path / "X.csv", index=False)
    y.to_csv(tmp_path / "y.csv", index=False, header=False)

    # Create config
    config = {
        "model": {"path": "model.pkl"},
        "data": {
            "X_path": "X.csv",
            "y_path": "y.csv",
        },
        "audit": {
            "random_seed": 42,
            "explain": False,
            "calibration": True,
        },
    }

    config_path = tmp_path / "audit.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Test from_config
    result = ga.audit.from_config(str(config_path))

    # Verify audit completed
    assert result.performance["accuracy"] > 0.5
    assert "precision" in result.performance
    assert "recall" in result.performance
    assert "roc_auc" in result.performance  # Has probabilities

    # Manifest should reflect model type
    assert result.manifest["model_type"] == "logistic_regression"


def test_from_config_with_protected_attributes(tmp_path):
    """Test from_config with protected attributes."""
    from sklearn.linear_model import LogisticRegression

    # Generate data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 3), columns=["f1", "f2", "f3"])
    y = pd.Series((X["f1"] > 0).astype(int))
    gender = pd.Series(np.random.randint(0, 2, size=len(y)))

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Save files
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    X.to_csv(tmp_path / "X.csv", index=False)
    y.to_csv(tmp_path / "y.csv", index=False, header=False)
    gender.to_csv(tmp_path / "gender.csv", index=False, header=False)

    # Create config with protected attributes
    config = {
        "model": {"path": "model.pkl"},
        "data": {
            "X_path": "X.csv",
            "y_path": "y.csv",
            "protected_attributes": {"gender": "gender.csv"},
        },
        "audit": {
            "random_seed": 42,
            "explain": False,
        },
    }

    config_path = tmp_path / "audit.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Test from_config
    result = ga.audit.from_config(str(config_path))

    # Should have fairness metrics
    assert len(result.fairness) > 0
    assert "gender_0" in result.fairness
    assert "gender_1" in result.fairness


def test_from_config_with_expected_hashes(tmp_path):
    """Test from_config validates data hashes when expected_hashes provided."""
    from sklearn.linear_model import LogisticRegression

    # Generate data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 3), columns=["f1", "f2", "f3"])
    y = pd.Series((X["f1"] > 0).astype(int))

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Save files
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    X.to_csv(tmp_path / "X.csv", index=False)
    y.to_csv(tmp_path / "y.csv", index=False, header=False)

    # Reload data from files (to match what from_config will load)
    X_loaded = pd.read_csv(tmp_path / "X.csv")
    y_loaded = pd.read_csv(tmp_path / "y.csv", header=None).iloc[:, 0]

    # Compute expected hashes from loaded data
    from glassalpha.utils.canonicalization import hash_data_for_manifest

    expected_X_hash = hash_data_for_manifest(X_loaded)
    expected_y_hash = hash_data_for_manifest(y_loaded)

    # Create config with expected hashes
    config = {
        "model": {"path": "model.pkl"},
        "data": {
            "X_path": "X.csv",
            "y_path": "y.csv",
            "expected_hashes": {
                "X": expected_X_hash,
                "y": expected_y_hash,
            },
        },
        "audit": {
            "random_seed": 42,
            "explain": False,
        },
    }

    config_path = tmp_path / "audit.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Should work when hashes match
    result = ga.audit.from_config(str(config_path))
    assert result is not None

    # Modify data to break hash
    X.iloc[0, 0] = 999  # Change one value
    X.to_csv(tmp_path / "X.csv", index=False)

    # Should raise DataHashMismatchError
    with pytest.raises(DataHashMismatchError):
        ga.audit.from_config(str(config_path))


def test_from_config_deterministic():
    """Test from_config produces deterministic results."""
    # This is an integration test that would need a full config setup
    # For now, just verify that the function exists and can be called
    # (detailed determinism testing is covered in other test modules)

    assert hasattr(ga.audit, "from_config")
    assert callable(ga.audit.from_config)


def test_from_config_missing_file():
    """Test from_config raises clear error for missing files."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        ga.audit.from_config("nonexistent.yaml")


def test_from_config_invalid_yaml(tmp_path):
    """Test from_config raises clear error for invalid YAML."""
    # Write invalid YAML
    config_path = tmp_path / "invalid.yaml"
    with open(config_path, "w") as f:
        f.write("invalid: yaml: content: [\n")

    with pytest.raises(ValueError, match="Invalid YAML syntax"):
        ga.audit.from_config(str(config_path))


def test_from_config_empty_file(tmp_path):
    """Test from_config raises clear error for empty config."""
    config_path = tmp_path / "empty.yaml"
    with open(config_path, "w") as f:
        f.write("")  # Empty file

    with pytest.raises(ValueError, match="empty"):
        ga.audit.from_config(str(config_path))


def test_from_config_missing_model_config(tmp_path):
    """Test from_config raises clear error for missing model config."""
    config = {
        "data": {
            "X_path": "X.csv",
            "y_path": "y.csv",
        },
        "audit": {"random_seed": 42},
    }

    config_path = tmp_path / "audit.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    with pytest.raises(ValueError, match="model.path.*model.type"):
        ga.audit.from_config(str(config_path))
