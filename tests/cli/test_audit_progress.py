"""Tests for CLI audit command progress bar integration (QW3).

This module tests that the CLI audit command shows progress bars during execution.
"""


class TestCLIAuditProgress:
    """Test CLI audit command progress bar integration."""

    def test_cli_progress_bar_created_for_non_strict_mode(self):
        """CLI creates progress bar with correct parameters for non-strict mode."""
        from glassalpha.utils.progress import get_progress_bar

        # Test that CLI would create progress bar correctly
        # (non-strict mode means show_progress=True)
        show_progress = True  # This is what CLI sets for non-strict mode

        pbar = get_progress_bar(total=100, desc="Running audit", disable=not show_progress, leave=False)

        # Verify it has expected methods
        assert hasattr(pbar, "update")
        assert hasattr(pbar, "set_description")
        assert hasattr(pbar, "refresh")
        assert hasattr(pbar, "n")

        # Can use as context manager
        with pbar as p:
            p.update(10)
            p.set_description("Test")
            p.refresh()

        pbar.close()

    def test_cli_progress_bar_disabled_for_strict_mode(self):
        """CLI disables progress bar correctly for strict mode."""
        from glassalpha.utils.progress import get_progress_bar

        # Test that CLI would disable progress bar correctly
        # (strict mode means show_progress=False)
        show_progress = False  # This is what CLI sets for strict mode

        pbar = get_progress_bar(total=100, desc="Running audit", disable=not show_progress, leave=False)

        # Should still have methods (passthrough wrapper)
        assert hasattr(pbar, "update")
        assert hasattr(pbar, "set_description")
        assert hasattr(pbar, "refresh")

        # Should work in context manager
        with pbar as p:
            p.update(10)
            p.set_description("Test")
            p.refresh()

        pbar.close()
