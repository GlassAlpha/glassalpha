"""Tests for from_model() entry point."""

import numpy as np
import pandas as pd
import pytest

import glassalpha as ga
from glassalpha.exceptions import NonBinaryClassificationError, NoPredictProbaError


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
