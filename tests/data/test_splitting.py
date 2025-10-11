"""Data splitting tests.

Tests for train/test splitting functionality in TabularDataLoader.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from glassalpha.data.tabular import TabularDataLoader, TabularDataSchema


def _parquet_engine_available() -> bool:
    """Check if a parquet engine (pyarrow or fastparquet) is available."""
    return bool(importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"))


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feature3": ["A", "B", "A", "B", "A"],
            "sensitive_attr": ["M", "F", "M", "F", "M"],
            "target": [0, 1, 0, 1, 0],
        },
    )


@pytest.fixture
def sample_csv_file(sample_csv_data, tmp_path):
    """Create a temporary CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    sample_csv_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def larger_dataset():
    """Create a larger dataset for more comprehensive testing."""
    np.random.seed(42)
    n_samples = 1000

    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n_samples),
            "income": np.random.normal(50000, 15000, n_samples),
            "education": np.random.choice(["HS", "College", "Graduate"], n_samples),
            "gender": np.random.choice(["M", "F", "Other"], n_samples),
            "ethnicity": np.random.choice(["A", "B", "C", "D"], n_samples),
            "employment_status": np.random.choice(["Employed", "Unemployed", "Retired"], n_samples),
            "credit_score": np.random.randint(300, 850, n_samples),
            "approved": np.random.choice([0, 1], n_samples),
        },
    )


@pytest.fixture
def larger_csv_file(larger_dataset, tmp_path):
    """Create a temporary CSV file with larger dataset."""
    csv_path = tmp_path / "large_data.csv"
    larger_dataset.to_csv(csv_path, index=False)
    return csv_path


class TestTabularDataLoaderIntegration:
    """Test integration scenarios."""

    def test_full_workflow(self, sample_csv_file):
        """Test complete workflow from loading to processing."""
        from glassalpha.data.tabular import create_schema_from_data

        loader = TabularDataLoader()

        # 1. Load data
        data = loader.load(sample_csv_file)
        assert isinstance(data, pd.DataFrame)

        # 2. Infer schema
        schema = create_schema_from_data(data, target_column="target")
        assert isinstance(schema, TabularDataSchema)

        # 3. Validate schema
        validation_result = loader.validate_schema(data, schema)
        assert validation_result is None  # Success returns None

        # 4. Extract protected attributes (if any sensitive features)
        if schema.sensitive_features:
            protected_attrs = data[schema.sensitive_features]
            assert isinstance(protected_attrs, pd.DataFrame)

        # 5. Compute hash
        data_hash = loader.hash_data(data)
        assert isinstance(data_hash, str)

    def test_reproducible_processing(self, sample_csv_file):
        """Test that processing is reproducible."""
        loader = TabularDataLoader()

        # Process same file twice
        data1 = loader.load(sample_csv_file)
        hash1 = loader.hash_data(data1)

        data2 = loader.load(sample_csv_file)
        hash2 = loader.hash_data(data2)

        # Results should be identical
        pd.testing.assert_frame_equal(data1, data2)
        assert hash1 == hash2

        # Compute hashes for different data (use data1 that we already loaded)
        hash1_again = loader.hash_data(data1)
        modified_data = data1.copy()
        modified_data.iloc[0, 0] = 999  # Change first value
        hash2_modified = loader.hash_data(modified_data)
