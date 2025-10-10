"""Register built-in datasets with the registry.

This module registers the built-in datasets that ship with GlassAlpha,
making them available for automatic fetching and caching.
"""

from pathlib import Path

from .adult_income import AdultIncomeDataset, load_adult_income
from .german_credit import GermanCreditDataset, load_german_credit  # Import the existing loader and class
from .registry import REGISTRY, DatasetSpec


def _fetch_german_credit() -> Path:
    """Fetch German Credit dataset and return path to processed CSV."""
    # Load the dataset (this handles download, processing, and caching)
    data = load_german_credit()

    # The loader should have saved the processed data to the expected location
    dataset = GermanCreditDataset()
    processed_file = dataset.processed_file

    # Create integrity metadata for Phase 3 verification
    try:
        from ..utils.integrity import create_dataset_metadata, save_metadata

        columns = list(data.columns)
        metadata = create_dataset_metadata(
            dataset_key="german_credit",
            schema_version="v1",
            source_url="https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
            file_path=processed_file,
            columns=columns,
            processing_notes="Downloaded from UCI, processed with categorical encoding and demographic feature extraction",
        )
        save_metadata(metadata, processed_file)
    except Exception as e:
        # Don't fail the fetch if metadata creation fails (for Phase 3)
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to create dataset metadata: {e}")

    return processed_file


def _fetch_adult_income() -> Path:
    """Fetch Adult Income dataset and return path to processed CSV."""
    # Load the dataset (this handles download, processing, and caching)
    data = load_adult_income()

    # The loader should have saved the processed data to the expected location
    dataset = AdultIncomeDataset()
    processed_file = dataset.processed_file

    # Create integrity metadata for Phase 3 verification
    try:
        from ..utils.integrity import create_dataset_metadata, save_metadata

        columns = list(data.columns)
        metadata = create_dataset_metadata(
            dataset_key="adult_income",
            schema_version="v1",
            source_url="https://archive.ics.uci.edu/ml/datasets/Adult",
            file_path=processed_file,
            columns=columns,
            processing_notes="Downloaded from UCI, combined train/test sets, processed with education grouping and age binning",
        )
        save_metadata(metadata, processed_file)
    except Exception as e:
        # Don't fail the fetch if metadata creation fails (for Phase 3)
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Could not create dataset metadata: {e}")

    return processed_file


def register_builtin_datasets() -> None:
    """Register all built-in datasets with the registry.

    This function is called during package import to make datasets
    available for automatic fetching.
    """
    # Register German Credit dataset
    REGISTRY["german_credit"] = DatasetSpec(
        key="german_credit",
        default_relpath="german_credit_processed.csv",
        fetch_fn=_fetch_german_credit,  # Returns path to processed file
        schema_version="v1",
        # checksum will be added in Phase 3
    )

    # Register Adult Income dataset
    REGISTRY["adult_income"] = DatasetSpec(
        key="adult_income",
        default_relpath="adult_income_processed.csv",
        fetch_fn=_fetch_adult_income,  # Returns path to processed file
        schema_version="v1",
        # checksum will be added in Phase 3
    )
