"""Contract tests for unfitted model guards.

Validates that model wrappers raise exact error messages when
predict/save methods are called before fit/load operations.
"""

import pytest


def test_sklearn_wrapper_unfitted_guard() -> None:
    """Test that sklearn wrapper raises exact error message when unfitted."""
    try:
        from glassalpha.constants import NO_MODEL_MSG  # noqa: PLC0415
        from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper  # noqa: PLC0415
    except ImportError:
        pytest.skip("sklearn wrapper or constants not available")

    wrapper = LogisticRegressionWrapper()

    # Test predict before fit
    with pytest.raises(ValueError, match=NO_MODEL_MSG) as exc_info:
        wrapper.predict([[1, 2, 3]])
    assert str(exc_info.value) == NO_MODEL_MSG  # noqa: S101

    # Test save before fit
    with pytest.raises(ValueError, match=NO_MODEL_MSG) as exc_info:
        wrapper.save("dummy_path.pkl")
    assert str(exc_info.value) == NO_MODEL_MSG  # noqa: S101


def test_xgboost_wrapper_unfitted_guard() -> None:
    """Test that XGBoost wrapper raises exact error message when unfitted."""
    try:
        from glassalpha.constants import NO_MODEL_MSG  # noqa: PLC0415
        from glassalpha.models.tabular.xgboost import XGBoostWrapper  # noqa: PLC0415
    except ImportError:
        pytest.skip("XGBoost wrapper or constants not available")

    wrapper = XGBoostWrapper()

    # Test predict before fit
    with pytest.raises(ValueError, match=NO_MODEL_MSG) as exc_info:
        wrapper.predict([[1, 2, 3]])
    assert str(exc_info.value) == NO_MODEL_MSG  # noqa: S101

    # Test predict_proba before fit
    with pytest.raises(ValueError, match=NO_MODEL_MSG) as exc_info:
        wrapper.predict_proba([[1, 2, 3]])
    assert str(exc_info.value) == NO_MODEL_MSG  # noqa: S101

    # Test save before fit
    with pytest.raises(ValueError, match=NO_MODEL_MSG) as exc_info:
        wrapper.save("dummy_path.json")
    assert str(exc_info.value) == NO_MODEL_MSG  # noqa: S101


def test_lightgbm_wrapper_unfitted_guard() -> None:
    """Test that LightGBM wrapper raises exact error message when unfitted."""
    try:
        from glassalpha.constants import NO_MODEL_MSG  # noqa: PLC0415
        from glassalpha.models.tabular.lightgbm import LightGBMWrapper  # noqa: PLC0415
    except ImportError:
        pytest.skip("LightGBM wrapper or constants not available")

    wrapper = LightGBMWrapper()

    # Test predict before fit
    with pytest.raises(ValueError, match=NO_MODEL_MSG) as exc_info:
        wrapper.predict([[1, 2, 3]])
    assert str(exc_info.value) == NO_MODEL_MSG  # noqa: S101

    # Test predict_proba before fit
    with pytest.raises(ValueError, match=NO_MODEL_MSG) as exc_info:
        wrapper.predict_proba([[1, 2, 3]])
    assert str(exc_info.value) == NO_MODEL_MSG  # noqa: S101


def test_base_wrapper_guard_consistency() -> None:
    """Test that all available wrappers use the same error message."""
    from glassalpha.constants import NO_MODEL_MSG  # noqa: PLC0415

    # Collect all available wrapper classes
    wrapper_classes = []

    try:
        from glassalpha.models.tabular.sklearn import LogisticRegressionWrapper  # noqa: PLC0415

        wrapper_classes.append(("LogisticRegression", LogisticRegressionWrapper))
    except ImportError:
        pass

    try:
        from glassalpha.models.tabular.xgboost import XGBoostWrapper  # noqa: PLC0415

        wrapper_classes.append(("XGBoost", XGBoostWrapper))
    except ImportError:
        pass

    try:
        from glassalpha.models.tabular.lightgbm import LightGBMWrapper  # noqa: PLC0415

        wrapper_classes.append(("LightGBM", LightGBMWrapper))
    except ImportError:
        pass

    if not wrapper_classes:
        pytest.skip("No wrapper classes available to test")

    # Test that all use exact same error message
    for wrapper_name, wrapper_class in wrapper_classes:
        wrapper = wrapper_class()

        with pytest.raises(ValueError, match=NO_MODEL_MSG) as exc_info:
            wrapper.predict([[1, 2, 3]])

        error_msg = str(exc_info.value)
        assert error_msg == NO_MODEL_MSG, (  # noqa: S101
            f"{wrapper_name} wrapper error message mismatch. Got: '{error_msg}', Expected: '{NO_MODEL_MSG}'"
        )


def test_constant_import_from_base_module() -> None:
    """Test that base module can import the constant without errors."""
    # This is the specific regression test for the import-time failure
    try:
        from glassalpha.models.tabular.base import _ensure_fitted  # noqa: PLC0415

        # If this import succeeds, the constant import is working
        assert True  # noqa: S101
    except ImportError as e:
        pytest.fail(f"Base module import failed: {e}")


def test_no_model_msg_constant_exists() -> None:
    """Test that NO_MODEL_MSG constant exists and can be imported."""
    from glassalpha.constants import NO_MODEL_MSG  # noqa: PLC0415

    # Validate it's the exact string expected by tests
    assert NO_MODEL_MSG == "Model not loaded. Load a model first."  # noqa: S101
    assert isinstance(NO_MODEL_MSG, str)  # noqa: S101
    assert len(NO_MODEL_MSG) > 0  # noqa: S101
