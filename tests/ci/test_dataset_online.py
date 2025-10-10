"""Test online dataset fetching."""

import sys

import pytest

sys.path.insert(0, "src")
from glassalpha.config.loader import load_config_from_file
from glassalpha.pipeline.audit import AuditPipeline


@pytest.mark.ci
def test_online_mode_can_fetch_datasets():
    """Online mode must successfully fetch and load datasets."""
    # Load config and override offline mode for online test
    config = load_config_from_file("configs/german_credit_simple.yaml")

    # Force online mode for this test
    config.data.offline = False
    config.data.fetch = "if_missing"

    # Create pipeline (this should trigger dataset fetching if needed)
    pipeline = AuditPipeline(config)

    # Test path resolution
    resolved_path = pipeline._resolve_dataset_path()
    assert resolved_path.exists(), f"Dataset not found at {resolved_path}"

    # Verify we can load the data
    import pandas as pd

    df = pd.read_csv(resolved_path)
    assert len(df) > 0, "Dataset is empty"
    assert df.shape[1] > 0, "Dataset has no columns"

    # Check that target column exists
    assert "credit_risk" in df.columns, "Target column 'credit_risk' not found"

    # Verify data quality
    assert df["credit_risk"].isin([0, 1]).all(), "Target column has invalid values"
