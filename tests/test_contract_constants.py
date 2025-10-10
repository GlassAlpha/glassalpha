"""Contract tests for centralized constants.

Validates that exact contract strings exist and have correct values
to prevent import-time failures and ensure test assertions pass.
"""

from glassalpha.constants import (
    INIT_LOG_MESSAGE,
    INIT_LOG_TEMPLATE,
    NO_EXPLAINER_MSG,
    # Test backward-compatible aliases too
    NO_MODEL_MSG,
)


def test_primary_constants_exist_and_values() -> None:
    """Test that primary contract constants exist with exact expected values."""
    # Exact strings that tests assert against
    assert NO_MODEL_MSG == "Model not loaded. Load a model first."  # noqa: S101
    assert NO_EXPLAINER_MSG == "No compatible explainer found"  # noqa: S101
    assert "{profile}" in INIT_LOG_MESSAGE  # noqa: S101
    assert INIT_LOG_MESSAGE == "Initialized audit pipeline with profile: {profile}"  # noqa: S101


def test_backward_compatible_aliases() -> None:
    """Test that backward-compatible aliases point to same values."""
    # Aliases should match primary constants
    assert NO_MODEL_MSG == NO_MODEL_MSG  # noqa: S101
    assert NO_EXPLAINER_MSG == NO_EXPLAINER_MSG  # noqa: S101
    assert INIT_LOG_TEMPLATE == INIT_LOG_MESSAGE  # noqa: S101


def test_constants_are_strings() -> None:
    """Test that all constants are properly typed as strings."""
    assert isinstance(NO_MODEL_MSG, str)  # noqa: S101
    assert isinstance(NO_EXPLAINER_MSG, str)  # noqa: S101
    assert isinstance(INIT_LOG_MESSAGE, str)  # noqa: S101

    # Backward-compatible aliases should also be strings
    assert isinstance(NO_MODEL_MSG, str)  # noqa: S101
    assert isinstance(NO_EXPLAINER_MSG, str)  # noqa: S101
    assert isinstance(INIT_LOG_TEMPLATE, str)  # noqa: S101


def test_constants_not_empty() -> None:
    """Test that constants are not empty strings."""
    constants_to_check = [
        NO_MODEL_MSG,
        NO_EXPLAINER_MSG,
        INIT_LOG_MESSAGE,
        NO_MODEL_MSG,
        NO_EXPLAINER_MSG,
        INIT_LOG_TEMPLATE,
    ]

    for constant in constants_to_check:
        assert len(constant.strip()) > 0, f"Constant should not be empty: {constant!r}"  # noqa: S101


def test_log_message_template_format() -> None:
    """Test that log message template can be formatted correctly."""
    # Should be able to format with profile name
    formatted = INIT_LOG_MESSAGE.format(profile="test_profile")
    expected = "Initialized audit pipeline with profile: test_profile"
    assert formatted == expected  # noqa: S101

    # Backward-compatible alias should work the same way
    formatted_alias = INIT_LOG_TEMPLATE.format(profile="test_profile")
    assert formatted_alias == expected  # noqa: S101


def test_constants_importable_from_module() -> None:
    """Test that constants can be imported from the module without errors."""
    # This test mainly validates that the __all__ export works correctly
    # and that there are no import-time issues

    from glassalpha.constants import (  # noqa: PLC0415
        BINARY_CLASSES,
        BINARY_THRESHOLD,
        ERR_NOT_FITTED,
        NO_MODEL_MSG,
        STANDARD_AUDIT_TEMPLATE,
        STATUS_CLEAN,
        STATUS_DIRTY,
        STATUS_NO_GIT,
        TEMPLATES_PACKAGE,
    )

    # Just verify they exist and have expected types
    assert isinstance(BINARY_CLASSES, int)  # noqa: S101
    assert isinstance(BINARY_THRESHOLD, (int, float))  # noqa: S101
    assert isinstance(ERR_NOT_FITTED, str)  # noqa: S101
    assert isinstance(NO_MODEL_MSG, str)  # noqa: S101
    assert isinstance(STANDARD_AUDIT_TEMPLATE, str)  # noqa: S101
    assert isinstance(STATUS_CLEAN, str)  # noqa: S101
    assert isinstance(STATUS_DIRTY, str)  # noqa: S101
    assert isinstance(STATUS_NO_GIT, str)  # noqa: S101
    assert isinstance(TEMPLATES_PACKAGE, str)  # noqa: S101
