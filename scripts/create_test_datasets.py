#!/usr/bin/env python3
"""Create test datasets for CI testing.

This script creates minimal test datasets that can be used in CI environments
where network downloads are not available or desired. These datasets are
smaller versions of the real datasets but maintain the same structure for testing.

IMPORTANT: This script reads the schema from the actual configuration files
to ensure test datasets match the expected schema. This prevents the recurring
issue where test datasets have incomplete schemas that don't match what tests expect.

Environment Variables:
    GLASSALPHA_DATA_DIR: Override default cache directory (useful for CI)
    GLASSALPHA_CI: Set to '1' to enable CI-specific behavior
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from glassalpha.config.loader import load_config_from_file
from glassalpha.utils.cache_dirs import resolve_data_root


def create_german_credit_test():
    """Create a test version of the German Credit dataset.

    This creates a structurally complete dataset that matches the schema defined
    in the German Credit configuration file. This ensures tests work with
    realistic schema expectations and prevents the recurring issue where test
    datasets have incomplete schemas.
    """
    np.random.seed(42)

    # Create smaller dataset (100 samples vs 1000)
    n_samples = 100

    # Load schema from the actual German Credit config file
    # This ensures test data matches what tests and configs expect
    config_path = Path(__file__).parent.parent / "src" / "glassalpha" / "data" / "configs" / "german_credit.yaml"
    config = load_config_from_file(config_path)

    # Extract expected columns from config
    expected_features = config.data.feature_columns
    expected_target = config.data.target_column

    print("📋 Creating German Credit test dataset from config schema:")
    print(f"   Features: {len(expected_features)} columns")
    print(f"   Target: {expected_target}")

    # Generate data for each expected feature column
    data = {}

    for feature in expected_features:
        # Determine appropriate values based on feature name patterns
        if "age" in feature.lower():
            data[feature] = np.random.randint(19, 76, n_samples)
        elif "amount" in feature.lower() or "credit" in feature.lower():
            data[feature] = np.random.randint(250, 20000, n_samples)
        elif "duration" in feature.lower() or "months" in feature.lower():
            data[feature] = np.random.randint(4, 73, n_samples)
        elif "rate" in feature.lower() or "count" in feature.lower():
            if "installment" in feature.lower() or "existing" in feature.lower():
                data[feature] = np.random.randint(1, 5, n_samples)
            elif "dependents" in feature.lower():
                data[feature] = np.random.randint(1, 3, n_samples)
            else:
                data[feature] = np.random.randint(1, 10, n_samples)
        elif "residence" in feature.lower():
            data[feature] = np.random.randint(1, 5, n_samples)
        elif "gender" in feature.lower():
            # For gender, use categorical values to match expected data types
            data[feature] = np.random.choice(["male", "female"], n_samples)
        elif "age_group" in feature.lower():
            data[feature] = np.random.choice(["young", "middle", "senior"], n_samples)
        # For categorical features, use common values that match UCI dataset patterns
        elif "account" in feature.lower() or "checking" in feature.lower():
            data[feature] = np.random.choice(
                ["no_checking", "negative_balance", "1_to_200", "above_200"],
                n_samples,
            )
        elif "history" in feature.lower() or "credit_history" in feature.lower():
            data[feature] = np.random.choice(
                ["critical", "delay", "existing_paid", "all_paid", "none"],
                n_samples,
            )
        elif "purpose" in feature.lower():
            data[feature] = np.random.choice(
                [
                    "new_car",
                    "used_car",
                    "furniture",
                    "radio_tv",
                    "appliances",
                    "repairs",
                    "education",
                    "vacation",
                    "retraining",
                    "business",
                    "other",
                ],
                n_samples,
            )
        elif "savings" in feature.lower():
            data[feature] = np.random.choice(
                ["unknown", "none", "below_100", "100_to_500", "500_to_1000", "above_1000"],
                n_samples,
            )
        elif "employment" in feature.lower():
            data[feature] = np.random.choice(
                ["unemployed", "below_1_year", "1_to_4_years", "4_to_7_years", "above_7_years"],
                n_samples,
            )
        elif "status" in feature.lower() or "sex" in feature.lower():
            data[feature] = np.random.choice(
                ["male_divorced", "female", "male_single", "male_married"],
                n_samples,
            )
        elif "debtors" in feature.lower():
            data[feature] = np.random.choice(
                ["none", "co_applicant", "guarantor"],
                n_samples,
            )
        elif "property" in feature.lower():
            data[feature] = np.random.choice(
                ["real_estate", "savings_agreement", "car", "none"],
                n_samples,
            )
        elif "plans" in feature.lower() or "installment" in feature.lower():
            data[feature] = np.random.choice(
                ["bank", "stores", "none"],
                n_samples,
            )
        elif "housing" in feature.lower():
            data[feature] = np.random.choice(
                ["rent", "own", "free"],
                n_samples,
            )
        elif "job" in feature.lower():
            data[feature] = np.random.choice(
                ["unemployed", "unskilled", "skilled", "highly_skilled"],
                n_samples,
            )
        elif "telephone" in feature.lower():
            data[feature] = np.random.choice(
                ["none", "yes"],
                n_samples,
            )
        elif "worker" in feature.lower() or "foreign" in feature.lower():
            data[feature] = np.random.choice(
                ["yes", "no"],
                n_samples,
            )
        else:
            # Default categorical values for any remaining features
            print(f"⚠️  Warning: Unknown feature pattern '{feature}', using default categorical values")
            data[feature] = np.random.choice(["A", "B", "C"], n_samples)

    # Add target column
    data[expected_target] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 70% good risk

    df = pd.DataFrame(data)

    # Save to cache directory
    cache_dir = resolve_data_root()
    output_path = cache_dir / "german_credit_processed.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"✅ Created test German Credit dataset: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {', '.join(df.columns.tolist())}")

    return output_path


def create_adult_income_test():
    """Create a test version of the Adult Income dataset."""
    np.random.seed(42)

    # Create smaller dataset (200 samples vs 32k)
    n_samples = 200

    data = {
        "age": np.random.randint(17, 91, n_samples),
        "workclass": np.random.choice(
            [
                "Private",
                "Self-emp-not-inc",
                "Self-emp-inc",
                "Federal-gov",
                "Local-gov",
                "State-gov",
                "Without-pay",
                "Never-worked",
            ],
            n_samples,
        ),
        "education": np.random.choice(
            [
                "Bachelors",
                "Some-college",
                "11th",
                "HS-grad",
                "Prof-school",
                "Assoc-acdm",
                "Assoc-voc",
                "9th",
                "7th-8th",
                "12th",
                "Masters",
                "1st-4th",
                "10th",
                "Doctorate",
                "5th-6th",
                "Preschool",
            ],
            n_samples,
        ),
        "marital_status": np.random.choice(
            [
                "Married-civ-spouse",
                "Divorced",
                "Never-married",
                "Separated",
                "Widowed",
                "Married-spouse-absent",
                "Married-AF-spouse",
            ],
            n_samples,
        ),
        "occupation": np.random.choice(
            [
                "Tech-support",
                "Craft-repair",
                "Other-service",
                "Sales",
                "Exec-managerial",
                "Prof-specialty",
                "Handlers-cleaners",
                "Machine-op-inspct",
                "Adm-clerical",
                "Farming-fishing",
                "Transport-moving",
                "Priv-house-serv",
                "Protective-serv",
                "Armed-Forces",
            ],
            n_samples,
        ),
        "relationship": np.random.choice(
            [
                "Wife",
                "Own-child",
                "Husband",
                "Not-in-family",
                "Other-relative",
                "Unmarried",
            ],
            n_samples,
        ),
        "race": np.random.choice(["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"], n_samples),
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "hours_per_week": np.random.randint(1, 99, n_samples),
        "native_country": np.random.choice(
            [
                "United-States",
                "Cambodia",
                "England",
                "Puerto-Rico",
                "Canada",
                "Germany",
                "Outlying-US(Guam-USVI-etc)",
                "India",
                "Japan",
                "Greece",
                "South",
                "China",
                "Cuba",
                "Iran",
                "Honduras",
                "Philippines",
                "Italy",
                "Poland",
                "Jamaica",
                "Vietnam",
                "Mexico",
                "Portugal",
                "Ireland",
                "France",
                "Dominican-Republic",
                "Laos",
                "Ecuador",
                "Taiwan",
                "Haiti",
                "Columbia",
                "Hungary",
                "Guatemala",
                "Nicaragua",
                "Scotland",
                "Thailand",
                "Yugoslavia",
                "El-Salvador",
                "Trinadad&Tobago",
                "Peru",
                "Hong",
                "Holand-Netherlands",
            ],
            n_samples,
        ),
        "income": np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),  # 75% <=50K
    }

    df = pd.DataFrame(data)

    # Save to cache directory
    cache_dir = resolve_data_root()
    output_path = cache_dir / "adult_income_processed.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"✅ Created test Adult Income dataset: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {', '.join(df.columns.tolist())}")

    return output_path


def main():
    """Create all test datasets."""
    is_ci = os.getenv("GLASSALPHA_CI") == "1"

    if is_ci:
        print("🚀 Creating test datasets for CI environment...")
    else:
        print("🚀 Creating test datasets...")

    cache_dir = resolve_data_root()
    print(f"   Cache directory: {cache_dir}")

    # In CI, show more detailed information
    if is_ci:
        print(f"   Python version: {os.sys.version}")
        print(f"   Working directory: {os.getcwd()}")
        print(f"   Environment: GLASSALPHA_DATA_DIR={os.getenv('GLASSALPHA_DATA_DIR', 'not set')}")

    # Ensure cache directory exists and is writable
    try:
        from glassalpha.utils.cache_dirs import ensure_dir_writable

        ensure_dir_writable(cache_dir)
        print(f"   Cache directory verified writable: {cache_dir}")
    except Exception as e:
        print(f"❌ Failed to create/verify cache directory: {e}")
        return 1

    try:
        german_credit_path = create_german_credit_test()
        adult_income_path = create_adult_income_test()

        print("\n✅ All test datasets created successfully!")

        print("\nTo verify datasets are available:")
        print(f"   ls -la {german_credit_path.parent}")
        print(f"   head -5 {german_credit_path}")

        # Final verification that datasets can be loaded
        import pandas as pd

        german_df = pd.read_csv(german_credit_path)
        adult_df = pd.read_csv(adult_income_path)
        print(f"   Verification: German Credit ({len(german_df)} rows), Adult Income ({len(adult_df)} rows)")

        return 0

    except Exception as e:
        print(f"❌ Failed to create datasets: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
