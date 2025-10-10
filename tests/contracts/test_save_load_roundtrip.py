"""Regression tests for save/load round-trip contract compliance.

Prevents wrapper serialization regressions that cause state loss
or incompatible model loading.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestSaveLoadRoundtrip:
    """Test save/load round-trip contract compliance."""

    def test_xgboost_wrapper_roundtrip(self) -> None:
        """Test XGBoost wrapper save/load preserves state and functionality."""
        try:
            import xgboost as xgb  # noqa: PLC0415

            from glassalpha.models.tabular.xgboost import XGBoostWrapper  # noqa: PLC0415
        except ImportError:
            pytest.skip("XGBoost not available")

        # Create and fit wrapper
        wrapper1 = XGBoostWrapper()

        X_train = pd.DataFrame(  # noqa: N806
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
            },
        )
        y_train = [0, 1, 0, 1, 0]

        wrapper1.fit(X_train, y_train, random_state=42)

        # Get predictions from original
        X_test = pd.DataFrame(  # noqa: N806
            {
                "feature1": [2.5, 3.5],
                "feature2": [25, 35],
            },
        )
        original_predictions = wrapper1.predict(X_test)
        original_probabilities = wrapper1.predict_proba(X_test)

        # Save wrapper
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            wrapper1.save(tmp.name)
            save_path = tmp.name

        try:
            # Load into new wrapper
            wrapper2 = XGBoostWrapper()
            wrapper2.load(save_path)

            # Contract: After load, state should be preserved
            assert wrapper2.model is not None, "Model should be loaded"  # noqa: S101
            assert wrapper2.n_classes == wrapper1.n_classes, "n_classes should match"  # noqa: S101
            assert wrapper2.feature_names_ == wrapper1.feature_names_, "Feature names should match"  # noqa: S101

            # Contract: Predictions should be numerically close
            loaded_predictions = wrapper2.predict(X_test)
            loaded_probabilities = wrapper2.predict_proba(X_test)

            np.testing.assert_array_almost_equal(
                original_predictions,
                loaded_predictions,
                decimal=5,
                err_msg="Predictions should be nearly identical after load",
            )

            np.testing.assert_array_almost_equal(
                original_probabilities,
                loaded_probabilities,
                decimal=5,
                err_msg="Probabilities should be nearly identical after load",
            )

        finally:
            # Cleanup
            Path(save_path).unlink(missing_ok=True)

    def test_sklearn_wrapper_roundtrip(self) -> None:
        """Test sklearn wrapper save/load preserves state and functionality."""
        try:
            import sklearn  # noqa: PLC0415

            from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper  # noqa: PLC0415
        except ImportError:
            pytest.skip("sklearn not available")

        # Create and fit wrapper
        wrapper1 = LogisticRegressionWrapper()

        X_train = pd.DataFrame(  # noqa: N806
            {
                "x1": [1, 2, 3, 4, 5],
                "x2": [5, 4, 3, 2, 1],
            },
        )
        y_train = [0, 0, 1, 1, 1]

        wrapper1.fit(X_train, y_train, random_state=42)

        # Get predictions from original
        X_test = pd.DataFrame(  # noqa: N806
            {
                "x1": [2.5, 3.5],
                "x2": [3.5, 2.5],
            },
        )
        original_predictions = wrapper1.predict(X_test)
        original_probabilities = wrapper1.predict_proba(X_test)

        # Save wrapper
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            wrapper1.save(tmp.name)
            save_path = tmp.name

        try:
            # Load into new wrapper
            wrapper2 = LogisticRegressionWrapper()
            wrapper2.load(save_path)

            # Contract: After load, state should be preserved
            assert wrapper2.model is not None, "Model should be loaded"  # noqa: S101
            assert wrapper2.n_classes == wrapper1.n_classes, "n_classes should match"  # noqa: S101
            assert wrapper2.feature_names_ == wrapper1.feature_names_, "Feature names should match"  # noqa: S101
            assert wrapper2._is_fitted == True, "_is_fitted should be True after load"  # noqa: E712, SLF001, S101

            # Contract: Predictions should be identical for sklearn
            loaded_predictions = wrapper2.predict(X_test)
            loaded_probabilities = wrapper2.predict_proba(X_test)

            np.testing.assert_array_equal(
                original_predictions,
                loaded_predictions,
                err_msg="sklearn predictions should be identical after load",
            )

            np.testing.assert_array_almost_equal(
                original_probabilities,
                loaded_probabilities,
                decimal=10,
                err_msg="sklearn probabilities should be nearly identical after load",
            )

        finally:
            # Cleanup
            Path(save_path).unlink(missing_ok=True)

    def test_lightgbm_wrapper_roundtrip(self) -> None:
        """Test LightGBM wrapper save/load preserves state and functionality."""
        try:
            import lightgbm  # noqa: PLC0415

            from glassalpha.models.tabular.lightgbm import LightGBMWrapper  # noqa: PLC0415
        except ImportError:
            pytest.skip("LightGBM not available")

        # Create and fit wrapper
        wrapper1 = LightGBMWrapper()

        X_train = pd.DataFrame(  # noqa: N806
            {
                "feat_a": [1, 2, 3, 4, 5, 6],
                "feat_b": [10, 20, 30, 40, 50, 60],
            },
        )
        y_train = [0, 1, 0, 1, 0, 1]

        wrapper1.fit(X_train, y_train, random_state=42)

        # Get predictions from original
        X_test = pd.DataFrame(  # noqa: N806
            {
                "feat_a": [2.5, 4.5],
                "feat_b": [25, 45],
            },
        )
        original_predictions = wrapper1.predict(X_test)
        original_probabilities = wrapper1.predict_proba(X_test)

        # Save wrapper
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            wrapper1.save(tmp.name)
            save_path = tmp.name

        try:
            # Load into new wrapper
            wrapper2 = LightGBMWrapper()
            wrapper2.load(save_path)

            # Contract: After load, state should be preserved
            assert wrapper2.model is not None, "Model should be loaded"  # noqa: S101
            assert wrapper2.n_classes == wrapper1.n_classes, "n_classes should match"  # noqa: S101
            assert wrapper2.feature_names_ == wrapper1.feature_names_, "Feature names should match"  # noqa: S101

            # Contract: Predictions should be numerically close
            loaded_predictions = wrapper2.predict(X_test)
            loaded_probabilities = wrapper2.predict_proba(X_test)

            np.testing.assert_array_almost_equal(
                original_predictions,
                loaded_predictions,
                decimal=5,
                err_msg="LightGBM predictions should be nearly identical after load",
            )

            np.testing.assert_array_almost_equal(
                original_probabilities,
                loaded_probabilities,
                decimal=5,
                err_msg="LightGBM probabilities should be nearly identical after load",
            )

        finally:
            # Cleanup
            Path(save_path).unlink(missing_ok=True)

    def test_wrapper_load_returns_self(self) -> None:
        """Test that load() methods return self for method chaining."""
        try:
            from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper  # noqa: PLC0415
        except ImportError:
            pytest.skip("sklearn not available")

        # Create, fit, and save wrapper
        wrapper1 = LogisticRegressionWrapper()
        X_train = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})  # noqa: N806
        y_train = [0, 1, 0]
        wrapper1.fit(X_train, y_train)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            wrapper1.save(tmp.name)
            save_path = tmp.name

        try:
            # Test that load returns self
            wrapper2 = LogisticRegressionWrapper()
            result = wrapper2.load(save_path)

            assert result is wrapper2, "load() should return self"  # noqa: S101

            # Should be able to chain methods
            X_test = pd.DataFrame({"x1": [1.5], "x2": [4.5]})  # noqa: N806
            predictions = wrapper2.load(save_path).predict(X_test)  # Chain load->predict
            assert len(predictions) == 1  # noqa: S101

        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_save_creates_parent_directories(self) -> None:
        """Test that save() creates parent directories if needed."""
        try:
            from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper  # noqa: PLC0415
        except ImportError:
            pytest.skip("sklearn not available")

        wrapper = LogisticRegressionWrapper()
        X_train = pd.DataFrame({"x": [1, 2], "y": [3, 4]})  # noqa: N806
        y_train = [0, 1]
        wrapper.fit(X_train, y_train)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Path with nested directories that don't exist
            nested_path = Path(temp_dir) / "deep" / "nested" / "path" / "model.pkl"

            # Should not raise FileNotFoundError
            wrapper.save(nested_path)

            # File should exist
            assert nested_path.exists()  # noqa: S101
            assert nested_path.is_file()  # noqa: S101

    def test_load_nonexistent_file_raises_error(self) -> None:
        """Test that loading nonexistent file raises proper error."""
        try:
            from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper  # noqa: PLC0415
        except ImportError:
            pytest.skip("sklearn not available")

        wrapper = LogisticRegressionWrapper()
        nonexistent_path = "/tmp/nonexistent_model_file.pkl"  # noqa: S108

        with pytest.raises(FileNotFoundError):
            wrapper.load(nonexistent_path)

    @pytest.mark.parametrize(
        "wrapper_name",
        [
            "LogisticRegressionWrapper",
            # Could add others as they become available
        ],
    )
    def test_empty_wrapper_states(self, wrapper_name: str) -> None:
        """Test that empty/default wrapper states are handled properly."""
        if wrapper_name == "LogisticRegressionWrapper":
            try:
                from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper as WrapperClass  # noqa: PLC0415
            except ImportError:
                pytest.skip("sklearn not available")
        else:
            pytest.skip(f"Wrapper {wrapper_name} not implemented in test")

        wrapper = WrapperClass()

        # Should raise error when trying to save unfitted model
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
            from glassalpha.constants import NO_MODEL_MSG  # noqa: PLC0415

            with pytest.raises(ValueError, match=NO_MODEL_MSG):
                wrapper.save(tmp.name)
