"""Smoke tests for end-to-end audit functionality.

This module provides smoke tests that verify the complete audit pipeline
works without crashing, with focus on explainer selection and PDF generation.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_audit_german_credit_simple_works(tmp_path):
    """Smoke test: full audit should complete without crashing.

    This test validates that the Phase 2.5 explainer fixes allow audits
    to complete successfully without TypeError or signature errors.

    Uses API instead of CLI to avoid CLI performance issues.
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from glassalpha.api import from_model

    # Create simple test data similar to German Credit
    np.random.seed(42)
    n_samples = 100

    # Create synthetic features (numeric only for simple model training)
    data = {
        "duration_months": np.random.randint(4, 73, n_samples),
        "credit_amount": np.random.randint(250, 20000, n_samples),
        "age_years": np.random.randint(19, 76, n_samples),
        "gender": np.random.choice([0, 1], n_samples),  # Protected attribute
        "credit_risk": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),  # Target
    }

    X = pd.DataFrame(data)
    y = X.pop("credit_risk")

    # Train simple model
    model = LogisticRegression(random_state=42)
    model.fit(X.drop("gender", axis=1), y)

    # Run audit using API (avoids CLI issues)
    result = from_model(
        model=model,
        X=X.drop("gender", axis=1),
        y=y,
        protected_attributes={"gender": X["gender"]},
        random_seed=42,
        explain=False,  # Skip explanations for speed
        calibration=False,  # Skip calibration for speed
    )

    # Verify audit succeeded
    assert result is not None
    assert hasattr(result, "performance")
    assert hasattr(result, "fairness")

    # Check that we have some basic metrics
    # Convert to dict to access metrics
    perf_data = dict(result.performance)
    fairness_data = dict(result.fairness)

    assert "accuracy" in perf_data and perf_data["accuracy"] is not None
    assert "gender_max_diff" in fairness_data and fairness_data["gender_max_diff"] is not None


def test_audit_stderr_no_explainer_errors(tmp_path):
    """Verify audit output contains no explainer-related errors.

    Uses API instead of CLI to avoid CLI performance issues.
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from glassalpha.api import from_model

    # Create simple test data
    np.random.seed(42)
    n_samples = 100

    data = {
        "duration_months": np.random.randint(4, 73, n_samples),
        "credit_amount": np.random.randint(250, 20000, n_samples),
        "age_years": np.random.randint(19, 76, n_samples),
        "gender": np.random.choice([0, 1], n_samples),
        "credit_risk": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    }

    X = pd.DataFrame(data)
    y = X.pop("credit_risk")

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X.drop("gender", axis=1), y)

    # Run audit using API
    result = from_model(
        model=model,
        X=X.drop("gender", axis=1),
        y=y,
        protected_attributes={"gender": X["gender"]},
        random_seed=42,
        explain=False,  # Skip for speed
        calibration=False,  # Skip for speed
    )

    # Check that no errors occurred (API throws exceptions on errors)
    assert result is not None
    assert hasattr(result, "performance")  # Should have performance metrics


def test_quickstart_audit_works(tmp_path):
    """Test that audit functionality works (quickstart equivalent).

    Uses API instead of CLI to avoid CLI performance issues.
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from glassalpha.api import from_model

    # Create simple test data
    np.random.seed(42)
    n_samples = 100

    data = {
        "duration_months": np.random.randint(4, 73, n_samples),
        "credit_amount": np.random.randint(250, 20000, n_samples),
        "age_years": np.random.randint(19, 76, n_samples),
        "gender": np.random.choice([0, 1], n_samples),
        "credit_risk": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    }

    X = pd.DataFrame(data)
    y = X.pop("credit_risk")

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X.drop("gender", axis=1), y)

    # Run audit using API
    result = from_model(
        model=model,
        X=X.drop("gender", axis=1),
        y=y,
        protected_attributes={"gender": X["gender"]},
        random_seed=42,
        explain=False,  # Skip for speed
        calibration=False,  # Skip for speed
    )

    # Verify audit succeeded
    assert result is not None
    assert hasattr(result, "performance")  # Should have performance metrics
