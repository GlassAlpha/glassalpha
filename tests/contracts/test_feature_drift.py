"""Regression tests for feature-name drift contract compliance.

Prevents sklearn feature name mismatch errors during prediction.
"""

import numpy as np
import pandas as pd
import pytest

from glassalpha.models._features import align_features


class TestFeatureDriftContract:
    """Test feature alignment contract compliance."""

    def test_renamed_same_width_positional_mapping(self) -> None:
        """Test renamed-same-width → positional mapping contract.

        When columns have same width but different names, should accept positionally.
        """
        # Training feature names
        original_features = ["age", "income", "credit_score"]

        # Test data with renamed but same-order columns
        test_data = pd.DataFrame(
            {
                "customer_age": [25, 35, 45],
                "annual_income": [50000, 75000, 100000],
                "credit_rating": [700, 750, 800],
            },
        )

        aligned = align_features(test_data, original_features)

        # Contract: Should return DataFrame with correct column names
        assert isinstance(aligned, pd.DataFrame)  # noqa: S101
        assert list(aligned.columns) == original_features  # noqa: S101
        assert aligned.shape == test_data.shape  # noqa: S101

        # Values should be preserved (positional mapping)
        np.testing.assert_array_equal(aligned.values, test_data.values)
        assert aligned.index.equals(test_data.index)  # noqa: S101

    def test_missing_columns_filled_with_zero(self) -> None:
        """Test missing columns → filled with 0 contract."""
        expected_features = ["age", "income", "credit_score", "employment_years"]

        # Test data missing some columns
        test_data = pd.DataFrame(
            {
                "age": [25, 35],
                "income": [50000, 75000],
                # Missing: credit_score, employment_years
            },
        )

        aligned = align_features(test_data, expected_features)

        # Contract: Should reindex with fill_value=0
        assert isinstance(aligned, pd.DataFrame)  # noqa: S101
        assert list(aligned.columns) == expected_features  # noqa: S101
        assert aligned.shape == (2, 4)  # noqa: S101

        # Existing values preserved
        assert aligned["age"].tolist() == [25, 35]  # noqa: S101
        assert aligned["income"].tolist() == [50000, 75000]  # noqa: S101

        # Missing columns filled with 0
        assert aligned["credit_score"].tolist() == [0, 0]  # noqa: S101
        assert aligned["employment_years"].tolist() == [0, 0]  # noqa: S101

    def test_extra_columns_dropped(self) -> None:
        """Test extra columns → dropped contract."""
        expected_features = ["age", "income"]

        # Test data with extra columns
        test_data = pd.DataFrame(
            {
                "age": [25, 35],
                "income": [50000, 75000],
                "extra_col1": [1, 2],  # Should be dropped
                "extra_col2": [3, 4],  # Should be dropped
            },
        )

        aligned = align_features(test_data, expected_features)

        # Contract: Should drop extra columns
        assert isinstance(aligned, pd.DataFrame)  # noqa: S101
        assert list(aligned.columns) == expected_features  # noqa: S101
        assert aligned.shape == (2, 2)  # noqa: S101

        # Expected values preserved
        assert aligned["age"].tolist() == [25, 35]  # noqa: S101
        assert aligned["income"].tolist() == [50000, 75000]  # noqa: S101

    def test_exact_match_unchanged(self) -> None:
        """Test exact feature match → unchanged."""
        features = ["age", "income", "credit_score"]

        test_data = pd.DataFrame(
            {
                "age": [25, 35],
                "income": [50000, 75000],
                "credit_score": [700, 750],
            },
        )

        aligned = align_features(test_data, features)

        # Should be unchanged
        pd.testing.assert_frame_equal(aligned, test_data)

    def test_non_dataframe_pass_through(self) -> None:
        """Test non-DataFrame inputs pass through unchanged."""
        test_cases = [
            np.array([[1, 2, 3], [4, 5, 6]]),
            [[1, 2, 3], [4, 5, 6]],
            "not_a_dataframe",
            None,
        ]

        for test_input in test_cases:
            result = align_features(test_input, ["a", "b", "c"])
            assert result is test_input  # Should be identical object  # noqa: S101

    def test_no_feature_names_pass_through(self) -> None:
        """Test no feature_names → pass through."""
        test_data = pd.DataFrame(
            {
                "col1": [1, 2],
                "col2": [3, 4],
            },
        )

        result = align_features(test_data, None)
        pd.testing.assert_frame_equal(result, test_data)

        result = align_features(test_data, [])
        pd.testing.assert_frame_equal(result, test_data)

    def test_wrapper_predict_no_sklearn_error(self) -> None:
        """Test that wrappers don't raise sklearn feature name errors.

        This is the main regression test for the original problem:
        "ValueError: The feature names should match those that were passed during fit"
        """
        from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper  # noqa: PLC0415

        # Skip if sklearn not available
        try:
            import sklearn  # noqa: PLC0415
        except ImportError:
            pytest.skip("sklearn not available")

        wrapper = LogisticRegressionWrapper()

        # Training data
        X_train = pd.DataFrame(  # noqa: N806
            {
                "feature_a": [1, 2, 3, 4],
                "feature_b": [5, 6, 7, 8],
            },
        )
        y_train = [0, 1, 0, 1]

        # Fit the wrapper
        wrapper.fit(X_train, y_train)

        # Test data with renamed but same-width columns
        X_test_renamed = pd.DataFrame(  # noqa: N806
            {
                "renamed_a": [1.5, 2.5],
                "renamed_b": [5.5, 6.5],
            },
        )

        # This should NOT raise sklearn feature name errors
        try:
            predictions = wrapper.predict(X_test_renamed)
            assert len(predictions) == 2  # noqa: PLR2004, S101
            assert isinstance(predictions, np.ndarray)  # noqa: S101

            probabilities = wrapper.predict_proba(X_test_renamed)
            assert probabilities.shape == (2, 2)  # 2 samples, 2 classes  # noqa: S101

        except ValueError as e:
            if "feature names should match" in str(e):
                pytest.fail(f"Feature drift handling failed: {e}")
            else:
                raise

    @pytest.mark.parametrize(
        "wrapper_class_name",
        [
            "LogisticRegressionWrapper",
            # Could add XGBoostWrapper, LightGBMWrapper tests here
        ],
    )
    def test_all_wrappers_handle_feature_drift(self, wrapper_class_name: str) -> None:
        """Test that all wrappers properly handle feature drift."""
        # Import wrapper class by name
        if wrapper_class_name == "LogisticRegressionWrapper":
            from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper as WrapperClass  # noqa: PLC0415
        else:
            pytest.skip(f"Wrapper {wrapper_class_name} not implemented in test")

        # Skip if dependencies not available
        try:
            wrapper = WrapperClass()
        except ImportError:
            pytest.skip(f"Dependencies for {wrapper_class_name} not available")

        # Simple training data
        X_train = pd.DataFrame({"x1": [1, 2], "x2": [3, 4]})  # noqa: N806
        y_train = [0, 1]

        wrapper.fit(X_train, y_train)

        # Test various drift scenarios
        test_cases = [
            # Same width, different names
            pd.DataFrame({"renamed1": [1.5], "renamed2": [3.5]}),
            # Extra columns
            pd.DataFrame({"x1": [1.5], "x2": [3.5], "x3": [999]}),
            # Missing columns (will be filled with 0)
            pd.DataFrame({"x1": [1.5]}),
        ]

        for i, X_test in enumerate(test_cases):  # noqa: N806
            try:
                predictions = wrapper.predict(X_test)
                assert len(predictions) == 1, f"Test case {i} failed shape check"  # noqa: S101
            except Exception as e:  # noqa: BLE001
                pytest.fail(f"Test case {i} failed: {e}")
