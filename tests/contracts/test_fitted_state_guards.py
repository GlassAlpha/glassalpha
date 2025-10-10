"""Regression tests for fitted state guard contract compliance.

Prevents silent misuse of wrappers by ensuring proper error messages
when methods are called before fit/load.
"""

from pathlib import Path
from typing import Any

import pytest

from glassalpha.constants import ERR_NOT_FITTED


class TestFittedStateGuards:
    """Test fitted state guard contract compliance."""

    def test_xgboost_wrapper_unfitted_guards(self) -> None:
        """Test XGBoost wrapper raises correct errors when unfitted."""
        from glassalpha.models.tabular.xgboost import XGBoostWrapper  # noqa: PLC0415

        wrapper = XGBoostWrapper()

        # Predict before fit/load should raise exact error
        with pytest.raises(ValueError, match=ERR_NOT_FITTED):
            wrapper.predict([[1, 2, 3]])

        with pytest.raises(ValueError, match=ERR_NOT_FITTED):
            wrapper.predict_proba([[1, 2, 3]])

        with pytest.raises(ValueError, match=ERR_NOT_FITTED):
            wrapper.save("dummy_path.json")

    def test_lightgbm_wrapper_unfitted_guards(self) -> None:
        """Test LightGBM wrapper raises correct errors when unfitted."""
        try:
            from glassalpha.models.tabular.lightgbm import LightGBMWrapper  # noqa: PLC0415
        except ImportError:
            pytest.skip("LightGBM not available")

        wrapper = LightGBMWrapper()

        with pytest.raises(ValueError, match=ERR_NOT_FITTED):
            wrapper.predict([[1, 2, 3]])

        with pytest.raises(ValueError, match=ERR_NOT_FITTED):
            wrapper.predict_proba([[1, 2, 3]])

    def test_sklearn_wrapper_unfitted_guards(self) -> None:
        """Test sklearn wrappers raise correct errors when unfitted."""
        try:
            from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper  # noqa: PLC0415
        except ImportError:
            pytest.skip("sklearn not available")

        wrapper = LogisticRegressionWrapper()

        with pytest.raises(ValueError, match=ERR_NOT_FITTED):
            wrapper.predict([[1, 2, 3]])

        with pytest.raises(ValueError, match=ERR_NOT_FITTED):
            wrapper.predict_proba([[1, 2, 3]])

        with pytest.raises(ValueError, match=ERR_NOT_FITTED):
            wrapper.save("dummy_path.pkl")

    def test_fitted_state_after_fit(self) -> None:
        """Test that fitted state is properly tracked after fit."""
        try:
            import pandas as pd  # noqa: PLC0415

            from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper  # noqa: PLC0415
        except ImportError:
            pytest.skip("Dependencies not available")

        wrapper = LogisticRegressionWrapper()

        # Should raise error before fit
        with pytest.raises(ValueError, match=ERR_NOT_FITTED):
            wrapper.predict([[1, 2]])

        # Fit the model
        X_train = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})  # noqa: N806
        y_train = [0, 1, 0]
        wrapper.fit(X_train, y_train)

        # Should work after fit
        X_test = pd.DataFrame({"x1": [1.5], "x2": [4.5]})  # noqa: N806
        predictions = wrapper.predict(X_test)
        assert len(predictions) == 1  # noqa: S101

        probabilities = wrapper.predict_proba(X_test)
        assert probabilities.shape[0] == 1  # noqa: S101

    def test_fitted_state_after_load(self) -> None:
        """Test that fitted state is properly tracked after load."""
        try:
            import tempfile  # noqa: PLC0415

            import pandas as pd  # noqa: PLC0415

            from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper  # noqa: PLC0415
        except ImportError:
            pytest.skip("Dependencies not available")

        # Create and fit a wrapper
        wrapper1 = LogisticRegressionWrapper()
        X_train = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})  # noqa: N806
        y_train = [0, 1, 0]
        wrapper1.fit(X_train, y_train)

        # Save it
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            wrapper1.save(tmp.name)
            save_path = tmp.name

        # Create new wrapper and verify it's unfitted
        wrapper2 = LogisticRegressionWrapper()
        with pytest.raises(ValueError, match=ERR_NOT_FITTED):
            wrapper2.predict([[1, 2]])

        # Load the saved model
        wrapper2.load(save_path)

        # Should work after load
        X_test = pd.DataFrame({"x1": [1.5], "x2": [4.5]})  # noqa: N806
        predictions = wrapper2.predict(X_test)
        assert len(predictions) == 1  # noqa: S101

        # Cleanup

        Path(save_path).unlink()

    def test_exact_error_message_consistency(self) -> None:
        """Test that all wrappers use the exact same error message."""
        wrappers_to_test = []

        # Collect available wrappers
        try:
            from glassalpha.models.tabular.xgboost import XGBoostWrapper  # noqa: PLC0415

            wrappers_to_test.append(("XGBoost", XGBoostWrapper))
        except ImportError:
            pass

        try:
            from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper  # noqa: PLC0415

            wrappers_to_test.append(("LogisticRegression", LogisticRegressionWrapper))
        except ImportError:
            pass

        try:
            from glassalpha.models.tabular.lightgbm import LightGBMWrapper  # noqa: PLC0415

            wrappers_to_test.append(("LightGBM", LightGBMWrapper))
        except ImportError:
            pass

        if not wrappers_to_test:
            pytest.skip("No wrappers available to test")

        # Test that all use exact same error message
        for wrapper_name, wrapper_class in wrappers_to_test:
            wrapper = wrapper_class()

            try:
                wrapper.predict([[1, 2, 3]])
                pytest.fail(f"{wrapper_name} wrapper should raise error when unfitted")
            except ValueError as e:
                assert str(e) == ERR_NOT_FITTED, (  # noqa: PT017, S101
                    f"{wrapper_name} wrapper error message mismatch. Got: '{e}', Expected: '{ERR_NOT_FITTED}'"
                )

    def test_guards_decorator_functionality(self) -> None:
        """Test the requires_fitted decorator works correctly."""
        from glassalpha.models._guards import requires_fitted  # noqa: PLC0415

        class MockWrapper:
            def __init__(self) -> None:
                self.model = None

            @requires_fitted
            def predict(self, X: Any) -> str:  # noqa: N803, ARG002, ANN401
                return "prediction"

            def fit(self) -> Any:  # noqa: ANN401
                self.model = "fitted_model"
                return self

        wrapper = MockWrapper()

        # Should raise error when model is None
        with pytest.raises(ValueError, match=ERR_NOT_FITTED):
            wrapper.predict([[1, 2, 3]])

        # Should work after fitting
        wrapper.fit()
        result = wrapper.predict([[1, 2, 3]])
        assert result == "prediction"  # noqa: S101

    def test_method_signatures_preserved_by_decorator(self) -> None:
        """Test that @requires_fitted preserves method signatures and metadata."""
        import inspect  # noqa: PLC0415

        from glassalpha.models._guards import requires_fitted  # noqa: PLC0415

        class MockWrapper:
            def __init__(self) -> None:
                self.model = "fitted"

            @requires_fitted
            def predict(self, X: Any, *, verbose: bool = False) -> str:  # noqa: N803, ARG002, ANN401
                """Predict method with docstring."""
                return f"prediction_verbose={verbose}"

        wrapper = MockWrapper()

        # Check that method signature is preserved
        sig = inspect.signature(wrapper.predict)
        params = list(sig.parameters.keys())
        assert "X" in params  # noqa: S101
        assert "verbose" in params  # noqa: S101

        # Check that docstring is preserved
        assert "Predict method with docstring" in wrapper.predict.__doc__  # noqa: S101

        # Check that method works correctly
        result = wrapper.predict([[1, 2]], verbose=True)
        assert result == "prediction_verbose=True"  # noqa: S101
