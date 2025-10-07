"""Fixtures for stability metrics tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def simple_df():
    """Small DataFrame with 4 features (2 protected, 2 regular)."""
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 30],
            "income": [30000, 50000, 70000, 90000, 40000],
            "gender": [0, 1, 0, 1, 0],  # protected
            "race": [0, 0, 1, 1, 0],  # protected
        },
    )


@pytest.fixture
def simple_model():
    """Trained LogisticRegression on simple synthetic data."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def simple_model_dataframe():
    """Trained model with DataFrame input (for feature name handling)."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 100),
            "income": np.random.randint(20000, 100000, 100),
            "gender": np.random.randint(0, 2, 100),
            "race": np.random.randint(0, 3, 100),
        },
    )
    y = (df["age"] + df["income"] / 1000 > 50).astype(int)

    model = LogisticRegression(random_state=42)
    model.fit(df, y)
    return model, df


@pytest.fixture
def single_feature_df():
    """DataFrame with only one non-protected feature."""
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 30],
            "gender": [0, 1, 0, 1, 0],  # protected
        },
    )


@pytest.fixture
def single_feature_model():
    """Trained model for single feature test case."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 100),
            "gender": np.random.randint(0, 2, 100),
        },
    )
    y = (df["age"] > 40).astype(int)

    model = LogisticRegression(random_state=42)
    model.fit(df, y)
    return model


@pytest.fixture
def all_protected_df():
    """DataFrame where all features are protected (should raise error)."""
    return pd.DataFrame(
        {
            "gender": [0, 1, 0, 1, 0],
            "race": [0, 0, 1, 1, 0],
        },
    )


@pytest.fixture
def empty_df():
    """Empty DataFrame (edge case)."""
    return pd.DataFrame(
        {
            "age": [],
            "income": [],
        },
    )
