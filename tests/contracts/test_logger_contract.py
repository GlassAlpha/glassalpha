"""Regression tests for logging contract compliance.

Prevents drift back to printf-style logging that causes contract test failures.
"""

from types import SimpleNamespace
from unittest.mock import patch

from glassalpha.constants import INIT_LOG_TEMPLATE


class TestLoggerContract:
    """Test exact logging contract compliance."""

    def test_pipeline_init_single_arg_logging(self) -> None:
        """Ensure pipeline initialization uses single-arg logging.

        Prevents regression to printf-style logging that fails contract tests.
        """
        import glassalpha.pipeline.audit as audit_mod
        from glassalpha.pipeline.audit import AuditPipeline

        config = SimpleNamespace(audit_profile="tabular_compliance")

        with patch.object(audit_mod, "logger") as logger_spy:
            AuditPipeline(config)

            # Contract: Exactly one positional argument with complete message
            expected_message = INIT_LOG_TEMPLATE.format(profile="tabular_compliance")
            logger_spy.info.assert_called_with(expected_message)

            # Ensure it's a single argument call (not printf-style)
            call_args = logger_spy.info.call_args
            assert len(call_args[0]) == 1, "Should be single-arg call, not printf-style"
            assert call_args[0][0] == expected_message

    def test_no_printf_style_logging_in_logging_utils(self) -> None:
        """Ensure logging utilities don't use printf-style internally."""
        import logging
        from unittest.mock import Mock

        from glassalpha.logging_utils import log_pipeline_init

        mock_logger = Mock(spec=logging.Logger)

        log_pipeline_init(mock_logger, "test_profile")

        # Should be single argument call
        call_args = mock_logger.info.call_args
        assert len(call_args[0]) == 1
        assert "test_profile" in call_args[0][0]
        assert "%s" not in call_args[0][0], "Should not contain printf formatters"
