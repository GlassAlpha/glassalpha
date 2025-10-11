"""Schema validation tests for test datasets.

This module ensures that test datasets match the expected schema from configuration files.
This prevents the recurring issue where test datasets have incomplete schemas that don't
match the real dataset schemas expected by tests and configs.

Tests in this module run automatically to catch schema mismatches early.
"""

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from glassalpha.config import load_config_from_file
from glassalpha.utils.cache_dirs import resolve_data_root


@pytest.fixture(scope="session", autouse=True)
def ensure_test_datasets() -> None:
    """Ensure test datasets exist and are valid before running any tests.

    This fixture runs once per test session and ensures that both German Credit
    and Adult Income test datasets exist in the cache directory. If they don't
    exist or are corrupted, it creates them using the dataset creation script.
    """
    cache_dir = resolve_data_root()

    german_path = cache_dir / "german_credit_processed.csv"
    adult_path = cache_dir / "adult_income_processed.csv"

    # Check if datasets exist and are valid
    datasets_valid = False
    if german_path.exists() and adult_path.exists():
        try:
            german_df = pd.read_csv(german_path)
            adult_df = pd.read_csv(adult_path)

            if len(german_df) > 0 and len(adult_df) > 0:
                datasets_valid = True
                print(
                    f"✅ Test datasets verified: German Credit ({len(german_df)} rows), Adult Income ({len(adult_df)} rows)",
                )
            else:
                print("⚠️  Existing test datasets are empty, recreating...")
        except Exception:
            print("⚠️  Existing test datasets are corrupted, recreating...")
    else:
        print("🔧 Test datasets not found, creating them...")

    # Create datasets if they don't exist or are invalid
    if not datasets_valid:
        script_path = Path(__file__).parent.parent / "scripts" / "create_test_datasets.py"

        # Run the dataset creation script with retry logic for CI environments
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                result = subprocess.run(
                    ["python3", str(script_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    cwd=Path(__file__).parent.parent,
                    timeout=60,  # Prevent hanging in CI
                )
                break
            except subprocess.TimeoutExpired:
                if attempt < max_attempts - 1:
                    print(f"⚠️  Dataset creation timed out (attempt {attempt + 1}/{max_attempts}), retrying...")
                    continue
                pytest.fail(f"Dataset creation timed out after {max_attempts} attempts")
            except subprocess.CalledProcessError as e:
                if attempt < max_attempts - 1:
                    print(f"⚠️  Dataset creation failed (attempt {attempt + 1}/{max_attempts}), retrying...")
                    continue
                pytest.fail(
                    f"Dataset creation failed after {max_attempts} attempts.\n"
                    f"Script output: {result.stdout}\n"
                    f"Script errors: {result.stderr}"
                    if "result" in locals()
                    else f"Script errors: {e}",
                )

        # Verify datasets were created successfully
        if not (german_path.exists() and adult_path.exists()):
            pytest.fail(
                "Dataset creation script ran but datasets not found.\n"
                f"Script output: {result.stdout}\n"
                f"Script errors: {result.stderr}",
            )

        print("✅ Test datasets created successfully")

    # Final verification that datasets can be loaded
    try:
        german_df = pd.read_csv(german_path)
        adult_df = pd.read_csv(adult_path)

        if len(german_df) == 0 or len(adult_df) == 0:
            pytest.fail("Test datasets exist but are empty")

        print(f"✅ Test datasets ready: German Credit ({len(german_df)} rows), Adult Income ({len(adult_df)} rows)")

    except Exception as e:
        pytest.fail(f"Test datasets exist but cannot be loaded: {e}")


class TestTestDatasetSchemas:
    """Test that test datasets match expected schemas from configs."""

    def test_german_credit_test_dataset_matches_real_schema(self):
        """Verify test German Credit dataset has same schema as real config expects.

        This prevents the recurring issue where test datasets have incomplete schemas
        that cause "Missing feature columns" errors in integration tests.

        Regression guard for: https://github.com/glassalpha/glassalpha/issues/X
        """
        # Load test dataset (guaranteed to exist by ensure_test_datasets fixture)
        test_dataset_path = resolve_data_root() / "german_credit_processed.csv"
        test_df = pd.read_csv(test_dataset_path)

        # Load expected schema from German Credit config
        config_path = Path(__file__).parent.parent / "src" / "glassalpha" / "data" / "configs" / "german_credit.yaml"
        config = load_config_from_file(config_path)

        # Extract expected columns from config
        expected_features = config.data.feature_columns
        expected_target = config.data.target_column
        expected_columns = expected_features + [expected_target]

        # Verify test dataset has all expected columns
        missing_columns = set(expected_columns) - set(test_df.columns)
        assert not missing_columns, (
            f"Test dataset missing columns: {missing_columns}\n"
            f"Expected columns: {expected_columns}\n"
            f"Test dataset columns: {list(test_df.columns)}"
        )

        # Verify test dataset doesn't have unexpected extra columns
        # (Allow extra columns like 'gender', 'age_group' that might be in preprocessing)
        expected_minimal = expected_features + [expected_target]
        extra_columns = set(test_df.columns) - set(expected_minimal)
        allowed_extra = {"gender", "age_group"}  # These are added by preprocessing
        unexpected_extra = extra_columns - allowed_extra

        if unexpected_extra:
            pytest.fail(
                f"Test dataset has unexpected extra columns: {unexpected_extra}\n"
                f"This might indicate schema drift between test data and expected schema.",
            )

        # Verify dataset is not empty
        assert len(test_df) > 0, "Test dataset should not be empty"

        # Verify target column exists and is binary
        assert expected_target in test_df.columns, f"Target column '{expected_target}' not found"
        unique_targets = test_df[expected_target].unique()
        assert len(unique_targets) >= 2, f"Target should have at least 2 classes, found: {unique_targets}"

        print("✅ German Credit test dataset schema validation passed")
        print(f"   Test dataset: {len(test_df)} rows × {len(test_df.columns)} columns")
        print(f"   Expected: {len(expected_columns)} columns")
        print(f"   Target column: {expected_target} ({len(unique_targets)} classes)")

    def test_adult_income_test_dataset_matches_real_schema(self):
        """Verify test Adult Income dataset has correct structure."""
        # Load test dataset (guaranteed to exist by ensure_test_datasets fixture)
        test_dataset_path = resolve_data_root() / "adult_income_processed.csv"
        test_df = pd.read_csv(test_dataset_path)

        # Basic structure checks for Adult Income
        expected_columns = [
            "age",
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "gender",
            "hours_per_week",
            "native_country",
            "income",
        ]

        missing_columns = set(expected_columns) - set(test_df.columns)
        assert not missing_columns, (
            f"Adult Income test dataset missing columns: {missing_columns}\n"
            f"Expected: {expected_columns}\n"
            f"Found: {list(test_df.columns)}"
        )

        # Verify target column is binary
        assert "income" in test_df.columns, "Target column 'income' not found"
        unique_incomes = test_df["income"].unique()
        assert len(unique_incomes) == 2, f"Income should be binary, found: {unique_incomes}"

        # Verify reasonable data size
        assert len(test_df) > 0, "Test dataset should not be empty"
        assert len(test_df.columns) >= len(expected_columns), "Should have at least expected columns"

        print("✅ Adult Income test dataset schema validation passed")
        print(f"   Test dataset: {len(test_df)} rows × {len(test_df.columns)} columns")

    def test_test_datasets_are_deterministic(self):
        """Verify test datasets are deterministic across runs.

        This ensures tests are reproducible and CI runs are consistent.
        """
        from glassalpha.utils.cache_dirs import resolve_data_root

        # Generate datasets twice and compare
        german_credit_1 = resolve_data_root() / "german_credit_processed.csv"
        adult_income_1 = resolve_data_root() / "adult_income_processed.csv"

        # Remove existing datasets
        if german_credit_1.exists():
            german_credit_1.unlink()
        if adult_income_1.exists():
            adult_income_1.unlink()

        # Generate first time
        import subprocess

        result1 = subprocess.run(
            ["python3", "scripts/create_test_datasets.py"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=Path(__file__).parent.parent,
        )
        assert result1.returncode == 0, f"First dataset creation failed: {result1.stderr}"

        # Read first datasets
        df1_german = pd.read_csv(german_credit_1)
        df1_adult = pd.read_csv(adult_income_1)

        # Remove and regenerate
        german_credit_1.unlink()
        adult_income_1.unlink()

        result2 = subprocess.run(
            ["python3", "scripts/create_test_datasets.py"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=Path(__file__).parent.parent,
        )
        assert result2.returncode == 0, f"Second dataset creation failed: {result2.stderr}"

        # Read second datasets
        df2_german = pd.read_csv(german_credit_1)
        df2_adult = pd.read_csv(adult_income_1)

        # Compare - should be identical due to fixed seed
        pd.testing.assert_frame_equal(df1_german, df2_german, check_dtype=False)
        pd.testing.assert_frame_equal(df1_adult, df2_adult, check_dtype=False)

        print("✅ Test datasets are deterministic across runs")

    def test_test_datasets_have_reasonable_data_types(self):
        """Verify test datasets have appropriate data types for their columns."""
        # Check German Credit (guaranteed to exist by ensure_test_datasets fixture)
        german_path = resolve_data_root() / "german_credit_processed.csv"
        german_df = pd.read_csv(german_path)

        # Basic sanity checks - ensure columns exist and have reasonable types
        # We're more concerned with schema matching than exact pandas dtypes

        # Numeric columns should exist and not be completely wrong types
        numeric_cols = [
            "duration_months",
            "credit_amount",
            "installment_rate",
            "present_residence_since",
            "age_years",
            "existing_credits_count",
            "dependents_count",
        ]

        for col in numeric_cols:
            assert col in german_df.columns, f"Numeric column {col} missing"
            # Should not be object (string) - that's the main issue to catch
            assert german_df[col].dtype != "object", f"Column {col} should not be object (expected numeric)"

        # Target should be numeric (0/1)
        assert "credit_risk" in german_df.columns, "Target column missing"
        assert german_df["credit_risk"].dtype != "object", "Target should not be object"

        # Categorical columns should exist
        categorical_cols = [
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

        for col in categorical_cols:
            assert col in german_df.columns, f"Categorical column {col} missing"

        # The main requirement is that columns exist and data can be loaded
        # Exact pandas dtypes are less critical as long as the schema is correct
        print("✅ Test datasets have appropriate column structure and types")

    def test_test_datasets_have_reasonable_value_ranges(self):
        """Verify test datasets have realistic value ranges."""
        # German Credit dataset (guaranteed to exist by ensure_test_datasets fixture)
        german_path = resolve_data_root() / "german_credit_processed.csv"
        german_df = pd.read_csv(german_path)

        # Check age range (should be reasonable for adults)
        assert german_df["age_years"].min() >= 18, "Age should be at least 18"
        assert german_df["age_years"].max() <= 90, "Age should be reasonable"

        # Check credit amount range (should be positive and reasonable)
        assert german_df["credit_amount"].min() > 0, "Credit amount should be positive"
        assert german_df["credit_amount"].max() < 1000000, "Credit amount should be reasonable"

        # Check duration range
        assert german_df["duration_months"].min() >= 1, "Duration should be at least 1 month"
        assert german_df["duration_months"].max() <= 120, "Duration should be reasonable"

        # Check target distribution (should not be all one class)
        target_dist = german_df["credit_risk"].value_counts()
        assert len(target_dist) == 2, "Target should have 2 classes"
        assert target_dist.min() > 0, "Both classes should have samples"

        print("✅ Test datasets have reasonable value ranges")

    def test_ci_datasets_available(self):
        """Verify that CI test datasets are available when expected.

        This test runs in CI to ensure test datasets are properly created.
        Note: Dataset creation is handled by the ensure_test_datasets fixture.
        """
        # Datasets are guaranteed to exist by ensure_test_datasets fixture
        german_path = resolve_data_root() / "german_credit_processed.csv"
        adult_path = resolve_data_root() / "adult_income_processed.csv"

        # Verify they can be loaded
        german_df = pd.read_csv(german_path)
        adult_df = pd.read_csv(adult_path)

        assert len(german_df) > 0, "German Credit dataset empty"
        assert len(adult_df) > 0, "Adult Income dataset empty"

        print("✅ CI test datasets available and loadable")


# Run schema validation automatically when this module is imported
# This catches issues early in the test run
if __name__ == "__main__":
    # Run basic validation
    test_instance = TestTestDatasetSchemas()

    print("🔍 Running test dataset schema validation...")
    try:
        test_instance.test_german_credit_test_dataset_matches_real_schema()
        test_instance.test_adult_income_test_dataset_matches_real_schema()
        test_instance.test_test_datasets_have_reasonable_data_types()
        test_instance.test_test_datasets_have_reasonable_value_ranges()
        print("✅ All schema validations passed!")
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
        raise
