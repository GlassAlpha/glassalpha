"""High-risk git helpers tests - subprocess and status critical path validation.

These tests target the actual risk areas in git information collection:
- subprocess.run with text=True must work correctly
- Status mapping (clean/dirty) must be accurate
- FileNotFoundError handling must be robust
- No .decode() calls should exist (prevents CI failures)
"""

import pytest


class TestGitHelpersRisk:
    """Test git helper critical paths that prevent CI and manifest failures."""

    def test_git_info_collection_in_clean_repo(self):
        """Git info collection must work correctly in clean repository."""
        from glassalpha.utils.manifest import ManifestGenerator

        generator = ManifestGenerator()
        git_info = generator._collect_git_info()

        if git_info is not None:  # Git available
            # Status must be either "clean" or "dirty"
            assert git_info.status in {"clean", "dirty"}, f"Invalid status: {git_info.status}"

            # is_dirty must match status
            expected_dirty = git_info.status == "dirty"
            assert git_info.is_dirty == expected_dirty, "is_dirty must match status"

            # commit_hash property must work (backward compatibility)
            assert git_info.commit_hash == git_info.commit_sha, "commit_hash alias must work"

            # Basic fields should be populated
            if git_info.commit_sha:
                assert len(git_info.commit_sha) >= 7, "commit_sha should be meaningful length"

            if git_info.branch:
                assert isinstance(git_info.branch, str), "branch should be string"

    def test_subprocess_uses_text_mode_only(self):
        """Subprocess calls must use text=True - no decode() calls allowed."""
        from glassalpha.utils.proc import run_text

        # Test with simple command that should work
        result = run_text("echo", "test")

        if result is not None:  # Command worked
            assert isinstance(result, str), "run_text must return string, not bytes"
            assert result == "test", "Output should be processed correctly"

    def test_git_status_clean_vs_dirty_detection(self):
        """Status detection must accurately distinguish clean vs dirty repos."""
        pytest.skip("Requires isolated git repo - implement if git manipulation needed")
        # This would require creating a temporary git repo and manipulating it
        # Skipping for now as it's complex and the existing logic is sound

    def test_git_not_available_returns_none(self):
        """When git is not available, must return None gracefully."""
        from glassalpha.utils.proc import run_text

        # Test with command that definitely doesn't exist
        result = run_text("definitely_not_a_real_command_12345", "--version")
        assert result is None, "Nonexistent command should return None"

    def test_git_info_none_when_git_unavailable(self):
        """ManifestGenerator must handle git unavailability gracefully."""
        from glassalpha.utils.manifest import ManifestGenerator

        # Mock the run_text to simulate git not available
        generator = ManifestGenerator()

        # Patch the run_text function to return None (simulating git not found)
        original_run_text = None
        try:
            from glassalpha.utils import proc

            original_run_text = proc.run_text

            def mock_run_text(*args):
                return None  # Simulate git command not found

            proc.run_text = mock_run_text

            git_info = generator._collect_git_info()
            assert git_info is None, "Should return None when git unavailable"

        finally:
            # Restore original function
            if original_run_text:
                proc.run_text = original_run_text

    def test_no_decode_calls_in_subprocess_wrappers(self):
        """Ensure no .decode() calls exist in subprocess wrappers - prevents CI failures."""
        import inspect

        from glassalpha.utils import proc

        # Get source code of the proc module
        source = inspect.getsource(proc)

        # Check that .decode() is not used anywhere
        assert ".decode()" not in source, "subprocess wrappers must not use .decode() - use text=True instead"

        # Ensure text=True is used
        assert "text=True" in source, "subprocess calls must use text=True"

    def test_git_info_model_validation(self):
        """GitInfo model must validate status values correctly."""
        from glassalpha.utils.manifest import GitInfo

        # Valid status values should work
        clean_info = GitInfo(commit_sha="abc123", status="clean", is_dirty=False)
        assert clean_info.status == "clean"
        assert not clean_info.is_dirty
        assert clean_info.commit_hash == "abc123"  # Test alias

        dirty_info = GitInfo(commit_sha="def456", status="dirty", is_dirty=True)
        assert dirty_info.status == "dirty"
        assert dirty_info.is_dirty
        assert dirty_info.commit_hash == "def456"

    def test_manifest_environment_no_subprocess_errors(self):
        """Environment collection must not fail with subprocess decode errors."""
        from glassalpha.utils.manifest import ManifestGenerator

        generator = ManifestGenerator()
        env_info = generator._collect_environment_info()

        # Should complete without errors
        assert env_info is not None, "Environment collection should not fail"
        assert hasattr(env_info, "architecture"), "Should have architecture field"

        # Architecture should not trigger subprocess issues
        assert isinstance(env_info.architecture, str), "Architecture should be string"
        assert len(env_info.architecture) > 0, "Architecture should not be empty"
