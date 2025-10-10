"""Adult Income (Census Income) dataset loader.

This module provides loading and preprocessing for the Adult Income dataset,
a canonical benchmark for income prediction and fairness analysis.

Source: UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/ml/datasets/Adult
Records: 48,842 (32,561 training + 16,281 test)
Features: 14 demographic and employment attributes
Target: Binary income classification (>50K / <=50K)
Protected Attributes: Race, sex, age
"""

import logging
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# UCI Adult dataset URLs
ADULT_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

# Column names (dataset has no header)
COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

TARGET_NAME = "income"


class AdultIncomeDataset:
    """Comprehensive Adult Income dataset loader and preprocessor."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize Adult Income dataset loader.

        Args:
            cache_dir: Directory to cache downloaded files (default: ~/.glassalpha/data)

        """
        self.cache_dir = cache_dir or Path.home() / ".glassalpha" / "data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.train_file = self.cache_dir / "adult.data"
        self.test_file = self.cache_dir / "adult.test"
        self.processed_file = self.cache_dir / "adult_income_processed.csv"

        logger.debug(f"Adult Income dataset cache directory: {self.cache_dir}")

    def download_raw_data(self, force_redownload: bool = False) -> tuple[Path, Path]:
        """Download raw Adult Income data from UCI repository.

        Args:
            force_redownload: Force redownload even if files exist

        Returns:
            Tuple of (train_file, test_file) paths

        """
        # Download training data
        if self.train_file.exists() and not force_redownload:
            logger.debug("Using cached Adult Income training data")
        else:
            logger.info("Downloading Adult Income training data from UCI repository...")
            try:
                urllib.request.urlretrieve(ADULT_TRAIN_URL, self.train_file)
                logger.info(f"Downloaded training data to {self.train_file}")
            except Exception as e:
                raise RuntimeError(f"Failed to download Adult Income training data: {e}") from e

        # Download test data
        if self.test_file.exists() and not force_redownload:
            logger.debug("Using cached Adult Income test data")
        else:
            logger.info("Downloading Adult Income test data from UCI repository...")
            try:
                urllib.request.urlretrieve(ADULT_TEST_URL, self.test_file)
                logger.info(f"Downloaded test data to {self.test_file}")
            except Exception as e:
                raise RuntimeError(f"Failed to download Adult Income test data: {e}") from e

        return self.train_file, self.test_file

    def load_raw_data(self) -> pd.DataFrame:
        """Load and combine raw train/test data.

        Returns:
            Combined DataFrame with all records

        """
        # Download if needed
        self.download_raw_data()

        # Load training data
        train_df = pd.read_csv(
            self.train_file,
            names=COLUMN_NAMES,
            sep=", ",  # Note: UCI Adult uses ", " as separator
            engine="python",
            na_values=" ?",  # Missing values are marked as " ?"
        )

        # Load test data (has extra line at top that needs to be skipped)
        test_df = pd.read_csv(
            self.test_file,
            names=COLUMN_NAMES,
            sep=", ",
            engine="python",
            na_values=" ?",
            skiprows=1,  # Skip the header line in test file
        )

        # Clean target values (test file has periods: " >50K." vs " >50K")
        test_df["income"] = test_df["income"].str.rstrip(".")

        # Combine train and test
        df = pd.concat([train_df, test_df], ignore_index=True)

        logger.info(f"Loaded {len(df)} records from Adult Income dataset")

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Adult Income data for ML auditing.

        Args:
            df: Raw DataFrame

        Returns:
            Processed DataFrame with cleaned and encoded features

        """
        df = df.copy()

        # Remove rows with missing values (about 7% of data)
        df = df.dropna()

        # Binary target: 1 if >50K, 0 if <=50K
        df["income_over_50k"] = (df["income"] == ">50K").astype(int)

        # Create age groups for fairness analysis
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 100],
            labels=["<25", "25-34", "35-44", "45-54", "55+"],
        )

        # Simplify education to major categories
        education_map = {
            "Preschool": "Less than HS",
            "1st-4th": "Less than HS",
            "5th-6th": "Less than HS",
            "7th-8th": "Less than HS",
            "9th": "Less than HS",
            "10th": "Less than HS",
            "11th": "Less than HS",
            "12th": "Less than HS",
            "HS-grad": "High School",
            "Some-college": "Some College",
            "Assoc-voc": "Associates",
            "Assoc-acdm": "Associates",
            "Bachelors": "Bachelors",
            "Masters": "Advanced",
            "Prof-school": "Advanced",
            "Doctorate": "Advanced",
        }
        df["education_level"] = df["education"].map(education_map)

        # Select relevant features for modeling
        selected_cols = [
            "age",
            "education_num",
            "hours_per_week",
            "capital_gain",
            "capital_loss",
            "race",
            "sex",
            "age_group",
            "education_level",
            "workclass",
            "marital_status",
            "occupation",
            "relationship",
            "native_country",
            "income_over_50k",  # Target
        ]

        df = df[selected_cols]

        logger.info(f"Preprocessed Adult Income data: {len(df)} records, {len(selected_cols)} features")

        return df

    def load_processed(self, force_reprocess: bool = False) -> pd.DataFrame:
        """Load preprocessed Adult Income data (download and process if needed).

        Args:
            force_reprocess: Force reprocessing even if processed file exists

        Returns:
            Processed DataFrame ready for ML auditing

        """
        # Use cached processed file if available
        if self.processed_file.exists() and not force_reprocess:
            logger.debug(f"Loading cached processed Adult Income data from {self.processed_file}")
            return pd.read_csv(self.processed_file)

        # Download and process
        logger.info("Processing Adult Income dataset...")
        raw_df = self.load_raw_data()
        processed_df = self.preprocess(raw_df)

        # Save processed data
        processed_df.to_csv(self.processed_file, index=False)
        logger.info(f"Saved processed Adult Income data to {self.processed_file}")

        return processed_df


def load_adult_income(cache_dir: Path | None = None, force_reprocess: bool = False) -> pd.DataFrame:
    """Load preprocessed Adult Income dataset.

    This is the main entry point for loading the Adult Income dataset.
    It handles downloading, preprocessing, and caching automatically.

    Args:
        cache_dir: Directory to cache files (default: ~/.glassalpha/data)
        force_reprocess: Force reprocessing even if cached file exists

    Returns:
        Processed DataFrame ready for ML auditing

    Example:
        >>> df = load_adult_income()
        >>> print(df.columns)
        >>> # Train model on df

    """
    dataset = AdultIncomeDataset(cache_dir=cache_dir)
    return dataset.load_processed(force_reprocess=force_reprocess)


def get_adult_income_schema() -> dict[str, list[str]]:
    """Get Adult Income dataset schema for configuration.

    Returns:
        Dictionary with feature names and protected attributes

    Example:
        >>> schema = get_adult_income_schema()
        >>> print(schema["protected_attributes"])
        ['race', 'sex', 'age_group']

    """
    return {
        "target_column": "income_over_50k",
        "feature_columns": [
            "age",
            "education_num",
            "hours_per_week",
            "capital_gain",
            "capital_loss",
            "race",
            "sex",
            "age_group",
            "education_level",
            "workclass",
            "marital_status",
            "occupation",
            "relationship",
            "native_country",
        ],
        "protected_attributes": ["race", "sex", "age_group"],
        "categorical_features": [
            "race",
            "sex",
            "age_group",
            "education_level",
            "workclass",
            "marital_status",
            "occupation",
            "relationship",
            "native_country",
        ],
        "numerical_features": [
            "age",
            "education_num",
            "hours_per_week",
            "capital_gain",
            "capital_loss",
        ],
    }
