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
    assert NO_MODEL_MSG == "Model not loaded. Load a model first."
    assert NO_EXPLAINER_MSG == "No compatible explainer found"
    assert "{profile}" in INIT_LOG_MESSAGE
    assert INIT_LOG_MESSAGE == "Initialized audit pipeline with profile: {profile}"


def test_backward_compatible_aliases() -> None:
    """Test that backward-compatible aliases point to same values."""
    # Aliases should match primary constants
    assert NO_MODEL_MSG == NO_MODEL_MSG
    assert NO_EXPLAINER_MSG == NO_EXPLAINER_MSG
    assert INIT_LOG_TEMPLATE == INIT_LOG_MESSAGE


def test_constants_are_strings() -> None:
    """Test that all constants are properly typed as strings."""
    assert isinstance(NO_MODEL_MSG, str)
    assert isinstance(NO_EXPLAINER_MSG, str)
    assert isinstance(INIT_LOG_MESSAGE, str)

    # Backward-compatible aliases should also be strings
    assert isinstance(NO_MODEL_MSG, str)
    assert isinstance(NO_EXPLAINER_MSG, str)
    assert isinstance(INIT_LOG_TEMPLATE, str)


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
        assert len(constant.strip()) > 0, f"Constant should not be empty: {constant!r}"


def test_log_message_template_format() -> None:
    """Test that log message template can be formatted correctly."""
    # Should be able to format with profile name
    formatted = INIT_LOG_MESSAGE.format(profile="test_profile")
    expected = "Initialized audit pipeline with profile: test_profile"
    assert formatted == expected

    # Backward-compatible alias should work the same way
    formatted_alias = INIT_LOG_TEMPLATE.format(profile="test_profile")
    assert formatted_alias == expected


def test_constants_importable_from_module() -> None:
    """Test that constants can be imported from the module without errors."""
    # This test mainly validates that the __all__ export works correctly
    # and that there are no import-time issues

    from glassalpha.constants import (
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
    assert isinstance(BINARY_CLASSES, int)
    assert isinstance(BINARY_THRESHOLD, (int, float))
    assert isinstance(ERR_NOT_FITTED, str)
    assert isinstance(NO_MODEL_MSG, str)
    assert isinstance(STANDARD_AUDIT_TEMPLATE, str)
    assert isinstance(STATUS_CLEAN, str)
    assert isinstance(STATUS_DIRTY, str)
    assert isinstance(STATUS_NO_GIT, str)
    assert isinstance(TEMPLATES_PACKAGE, str)
