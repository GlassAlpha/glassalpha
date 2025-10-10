# SPDX-License-Identifier: Apache-2.0
"""Exact logger contract test - prevents regression of the chronic CI failure.

This test validates the EXACT logger call that the CI test spies.
It uses the same constructor path and assertion that caused us to thrash.
"""

from types import SimpleNamespace
from unittest.mock import patch


def test_pipeline_init_logger_exact_contract() -> None:
    """Test the exact logger call that CI expects.

    This is a tripwire test - it prevents regression of the specific issue
    where logger.info("%s", arg) vs logger.info(f"{arg}") caused CI thrashing.

    The CI test does: mock_logger.info.assert_called_with("Initialized...")
    That requires a single string argument, not printf-style formatting.
    """
    # Import after potential wheel installation
    import glassalpha.pipeline.audit as audit_module  # noqa: PLC0415
    from glassalpha.constants import INIT_LOG_TEMPLATE  # noqa: PLC0415
    from glassalpha.pipeline.audit import AuditPipeline  # noqa: PLC0415

    # Spy the exact logger the CI test spies
    with patch.object(audit_module, "logger") as logger_spy:
        # Use the exact constructor pattern CI uses
        config = SimpleNamespace(audit_profile="tabular_compliance")
        AuditPipeline(config)

        # The EXACT assertion that caused us to thrash
        expected_message = INIT_LOG_TEMPLATE.format(profile="tabular_compliance")
        logger_spy.info.assert_called_with(expected_message)

        # Verify it was called with a SINGLE argument (not printf style)
        call_args = logger_spy.info.call_args
        assert len(call_args[0]) == 1, "Logger must be called with single argument, not printf-style"  # noqa: S101
        assert isinstance(call_args[0][0], str), "Logger argument must be a string"  # noqa: S101

        # Verify no keyword arguments (would indicate printf style)
        assert not call_args[1], "Logger must not use keyword arguments (indicates printf style)"  # noqa: S101


def test_logger_contract_with_different_profiles() -> None:
    """Test that the logger works with different audit profiles.

    This ensures our f-string approach works for any profile name,
    not just the hardcoded test case.
    """
    from unittest.mock import patch  # noqa: PLC0415

    import glassalpha.pipeline.audit as audit_module  # noqa: PLC0415
    from glassalpha.pipeline.audit import AuditPipeline  # noqa: PLC0415

    profiles_to_test = [
        "tabular_compliance",
        "llm_safety",
        "custom_profile",
        "profile-with-dashes",
        "profile_with_underscores",
    ]

    for profile in profiles_to_test:
        with patch.object(audit_module, "logger") as logger_spy:
            config = SimpleNamespace(audit_profile=profile)
            AuditPipeline(config)

            # Verify the message contains the profile name
            call_args = logger_spy.info.call_args[0][0]
            assert profile in call_args, f"Profile {profile} not in log message: {call_args}"  # noqa: S101
            assert call_args.startswith("Initialized audit pipeline with profile:"), (  # noqa: S101
                f"Log message format incorrect for profile {profile}: {call_args}"
            )
