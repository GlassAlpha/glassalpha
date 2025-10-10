"""German Credit Dataset loader and preprocessing.

This module provides comprehensive loading and preprocessing for the
German Credit dataset, a canonical benchmark for credit risk assessment
and fairness analysis in machine learning.

The dataset contains information about 1000 loan applicants with features
including demographics, financial history, and loan details, with a binary
classification target indicating credit risk (good/bad).
"""

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import TabularDataSchema

logger = logging.getLogger(__name__)

# German Credit dataset URLs and metadata
GERMAN_CREDIT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
GERMAN_CREDIT_INFO_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc"

# Original attribute names and descriptions
FEATURE_NAMES = [
    "checking_account_status",  # Status of existing checking account
    "duration_months",  # Duration in months
    "credit_history",  # Credit history
    "purpose",  # Purpose of the loan
    "credit_amount",  # Credit amount
    "savings_account",  # Savings account/bonds
    "employment_duration",  # Present employment since
    "installment_rate",  # Installment rate in percentage of disposable income
    "personal_status_sex",  # Personal status and sex
    "other_debtors",  # Other debtors/guarantors
    "present_residence_since",  # Present residence since
    "property",  # Property
    "age_years",  # Age in years
    "other_installment_plans",  # Other installment plans
    "housing",  # Housing
    "existing_credits_count",  # Number of existing credits at this bank
    "job",  # Job
    "dependents_count",  # Number of people being liable to provide maintenance for
    "telephone",  # Telephone
    "foreign_worker",  # Foreign worker
]

TARGET_NAME = "credit_risk"


class GermanCreditDataset:
    """Comprehensive German Credit dataset loader and preprocessor."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize German Credit dataset loader.

        Args:
            cache_dir: Directory to cache downloaded files (default: ~/.glassalpha/data)

        """
        self.cache_dir = cache_dir or Path.home() / ".glassalpha" / "data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.data_file = self.cache_dir / "german_credit.data"
        self.processed_file = self.cache_dir / "german_credit_processed.csv"

        logger.debug(f"German Credit dataset cache directory: {self.cache_dir}")

    def download_raw_data(self, force_redownload: bool = False) -> Path:
        """Download raw German Credit data from UCI repository.

        Args:
            force_redownload: Force redownload even if file exists

        Returns:
            Path to downloaded data file

        """
        if self.data_file.exists() and not force_redownload:
            logger.debug("Using cached German Credit data")
            return self.data_file

        logger.info("Downloading German Credit dataset from UCI repository...")

        try:
            urllib.request.urlretrieve(GERMAN_CREDIT_URL, self.data_file)
            logger.info(f"Downloaded German Credit dataset to {self.data_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to download German Credit dataset: {e}") from e

        return self.data_file

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw German Credit data as DataFrame.

        Returns:
            Raw DataFrame with original encodings

        """
        # Download if not cached
        data_file = self.download_raw_data()

        # Load data (space-separated, no header)
        try:
            data = pd.read_csv(data_file, sep=" ", header=None, names=FEATURE_NAMES + [TARGET_NAME])

            logger.info(f"Loaded raw German Credit data: {data.shape}")
            return data

        except Exception as e:
            raise ValueError(f"Failed to load German Credit data from {data_file}: {e}") from e

    def preprocess_data(self, raw_data: pd.DataFrame | None = None) -> pd.DataFrame:
        """Preprocess German Credit data with proper feature engineering.

        Args:
            raw_data: Raw data DataFrame (loads if None)

        Returns:
            Preprocessed DataFrame with interpretable features

        """
        if raw_data is None:
            raw_data = self.load_raw_data()

        logger.info("Preprocessing German Credit dataset...")

        # Create a copy for processing
        data = raw_data.copy()

        # Process target variable (1=good, 2=bad -> 1=good, 0=bad)
        data[TARGET_NAME] = (data[TARGET_NAME] == 1).astype(int)

        # Process categorical features with meaningful labels
        data = self._process_categorical_features(data)

        # Process numerical features
        data = self._process_numerical_features(data)

        # Extract demographic information for fairness analysis
        data = self._extract_demographics(data)

        # Validate processed data
        self._validate_processed_data(data)

        logger.info(f"Preprocessed German Credit data: {data.shape}")

        return data

    def _process_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process categorical features with meaningful labels."""
        # Checking account status
        checking_map = {"A11": "< 0 DM", "A12": "0 <= ... < 200 DM", "A13": ">= 200 DM", "A14": "no checking account"}
        data["checking_account_status"] = data["checking_account_status"].map(checking_map)

        # Credit history
        history_map = {
            "A30": "no credits taken",
            "A31": "all credits paid back duly",
            "A32": "existing credits paid back duly till now",
            "A33": "delay in paying back in the past",
            "A34": "critical account",
        }
        data["credit_history"] = data["credit_history"].map(history_map)

        # Purpose
        purpose_map = {
            "A40": "new car",
            "A41": "used car",
            "A42": "furniture/equipment",
            "A43": "radio/television",
            "A44": "domestic appliances",
            "A45": "repairs",
            "A46": "education",
            "A47": "vacation",
            "A48": "retraining",
            "A49": "business",
            "A410": "others",
        }
        data["purpose"] = data["purpose"].map(purpose_map)

        # Savings account
        savings_map = {
            "A61": "< 100 DM",
            "A62": "100 <= ... < 500 DM",
            "A63": "500 <= ... < 1000 DM",
            "A64": ">= 1000 DM",
            "A65": "unknown/no savings account",
        }
        data["savings_account"] = data["savings_account"].map(savings_map)

        # Employment duration
        employment_map = {
            "A71": "unemployed",
            "A72": "< 1 year",
            "A73": "1 <= ... < 4 years",
            "A74": "4 <= ... < 7 years",
            "A75": ">= 7 years",
        }
        data["employment_duration"] = data["employment_duration"].map(employment_map)

        # Other debtors/guarantors
        debtors_map = {"A101": "none", "A102": "co-applicant", "A103": "guarantor"}
        data["other_debtors"] = data["other_debtors"].map(debtors_map)

        # Property
        property_map = {
            "A121": "real estate",
            "A122": "building society savings agreement/life insurance",
            "A123": "car or other",
            "A124": "unknown/no property",
        }
        data["property"] = data["property"].map(property_map)

        # Other installment plans
        installment_map = {"A141": "bank", "A142": "stores", "A143": "none"}
        data["other_installment_plans"] = data["other_installment_plans"].map(installment_map)

        # Housing
        housing_map = {"A151": "rent", "A152": "own", "A153": "for free"}
        data["housing"] = data["housing"].map(housing_map)

        # Job
        job_map = {
            "A171": "unemployed/unskilled - non-resident",
            "A172": "unskilled - resident",
            "A173": "skilled employee/official",
            "A174": "management/self-employed/highly qualified employee",
        }
        data["job"] = data["job"].map(job_map)

        # Telephone
        telephone_map = {"A191": "none", "A192": "yes"}
        data["telephone"] = data["telephone"].map(telephone_map)

        # Foreign worker
        foreign_map = {"A201": "yes", "A202": "no"}
        data["foreign_worker"] = data["foreign_worker"].map(foreign_map)

        return data

    def _process_numerical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and validate numerical features."""
        # Ensure numerical columns are proper numeric types
        numeric_columns = [
            "duration_months",
            "credit_amount",
            "installment_rate",
            "present_residence_since",
            "age_years",
            "existing_credits_count",
            "dependents_count",
        ]

        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        # Handle any missing values (should be rare in this dataset)
        numeric_missing = data[numeric_columns].isnull().sum()
        if numeric_missing.any():
            logger.warning(f"Missing values in numeric columns: {numeric_missing[numeric_missing > 0]}")
            # Fill with median values
            for col in numeric_columns:
                if data[col].isnull().any():
                    data[col] = data[col].fillna(data[col].median())

        return data

    def _extract_demographics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract demographic information for fairness analysis."""
        # Extract gender from personal_status_sex
        personal_status_map = {
            "A91": "male : divorced/separated",
            "A92": "female : divorced/separated/married",
            "A93": "male : single",
            "A94": "male : married/widowed",
            "A95": "female : single",
        }

        data["personal_status_sex"] = data["personal_status_sex"].map(personal_status_map)

        # Create separate gender column for fairness analysis
        def extract_gender(status):
            if pd.isna(status):
                return "unknown"
            return "female" if "female" in status.lower() else "male"

        data["gender"] = data["personal_status_sex"].apply(extract_gender)

        # Create age groups for analysis
        data["age_group"] = pd.cut(
            data["age_years"],
            bins=[0, 25, 35, 50, 100],
            labels=["young", "middle_young", "middle_aged", "senior"],
            right=False,
        )

        return data

    def _validate_processed_data(self, data: pd.DataFrame) -> None:
        """Validate processed data quality and completeness."""
        # Check target distribution
        target_dist = data[TARGET_NAME].value_counts()
        logger.info(f"Target distribution: {target_dist.to_dict()}")

        if len(target_dist) != 2:
            raise ValueError(f"Expected binary target, got {len(target_dist)} classes")

        # Check for missing values
        missing_counts = data.isnull().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            logger.warning(f"Dataset has {total_missing} missing values: {missing_counts[missing_counts > 0]}")

        # Check data types
        categorical_cols = data.select_dtypes(include=["object"]).columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns

        logger.info(f"Processed data: {len(categorical_cols)} categorical, {len(numerical_cols)} numerical features")

    def load_processed_data(self, force_reprocess: bool = False) -> pd.DataFrame:
        """Load processed German Credit dataset.

        Args:
            force_reprocess: Force reprocessing even if cached file exists

        Returns:
            Processed DataFrame ready for ML

        """
        if self.processed_file.exists() and not force_reprocess:
            logger.debug("Loading cached processed German Credit data")
            try:
                return pd.read_csv(self.processed_file)
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}. Reprocessing...")

        # Process and cache
        processed_data = self.preprocess_data()

        try:
            processed_data.to_csv(self.processed_file, index=False)
            logger.info(f"Cached processed data to {self.processed_file}")
        except Exception as e:
            logger.warning(f"Failed to cache processed data: {e}")

        return processed_data

    def get_train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get train/test split of German Credit data.

        Args:
            test_size: Fraction of data for test set
            random_state: Random seed for reproducibility
            stratify: Whether to stratify split by target

        Returns:
            Tuple of (train_data, test_data)

        """
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise ImportError("sklearn not available - install scikit-learn or fix CI environment") from None

        data = self.load_processed_data()

        stratify_by = data[TARGET_NAME] if stratify else None

        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_by,
        )

        logger.info(f"Train/test split: {len(train_data)}/{len(test_data)} samples")

        return train_data, test_data


def get_german_credit_schema() -> TabularDataSchema:
    """Get predefined schema for German Credit dataset.

    Returns:
        TabularDataSchema with feature definitions and sensitive attributes

    """
    # Define all features (including derived ones for fairness analysis)
    features = [
        "checking_account_status",
        "duration_months",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings_account",
        "employment_duration",
        "installment_rate",
        "personal_status_sex",
        "other_debtors",
        "present_residence_since",
        "property",
        "age_years",
        "other_installment_plans",
        "housing",
        "existing_credits_count",
        "job",
        "dependents_count",
        "telephone",
        "foreign_worker",
        "gender",  # Derived from personal_status_sex
        "age_group",  # Derived from age_years
    ]

    # Sensitive attributes for fairness analysis
    sensitive_features = ["gender", "age_group", "foreign_worker"]

    # Categorical features
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
        "gender",
        "age_group",
    ]

    # Numerical features (note: correct attribute name is 'numeric_features')
    numeric_features = [
        "duration_months",
        "credit_amount",
        "installment_rate",
        "present_residence_since",
        "age_years",
        "existing_credits_count",
        "dependents_count",
    ]

    return TabularDataSchema(
        target=TARGET_NAME,
        features=features,
        sensitive_features=sensitive_features,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )


def load_german_credit(
    cache_dir: Path | None = None,
    train_test_split: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function to load German Credit dataset.

    Args:
        cache_dir: Directory to cache files
        train_test_split: Whether to return train/test split
        test_size: Fraction for test set if splitting
        random_state: Random seed for splitting

    Returns:
        Full dataset or (train_data, test_data) if train_test_split=True

    """
    dataset = GermanCreditDataset(cache_dir)

    if train_test_split:
        return dataset.get_train_test_split(test_size, random_state)
    return dataset.load_processed_data()


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    print("Loading German Credit dataset...")
    data = load_german_credit()

    print(f"Dataset shape: {data.shape}")
    print(f"Target distribution:\n{data[TARGET_NAME].value_counts()}")

    schema = get_german_credit_schema()
    print(f"Schema: {len(schema.features)} features, {len(schema.sensitive_features)} sensitive")

    # Test train/test split
    train_data, test_data = load_german_credit(train_test_split=True)
    print(f"Train: {train_data.shape}, Test: {test_data.shape}")
