"""Tests for OS-aware cache directory resolution."""

from pathlib import Path

import pytest

from glassalpha.utils.cache_dirs import ensure_dir_writable, get_cache_path, resolve_data_root


class TestGetDataRoot:
    """Test OS-aware cache directory resolution."""

    def test_env_override_uses_realpath(self, tmp_path, monkeypatch):
        """Test that env override uses canonical realpath."""
        target = tmp_path / "ga-cache"
        monkeypatch.setenv("GLASSALPHA_DATA_DIR", str(target))
        p = resolve_data_root()
        # Compare canonical paths (handles /tmp vs /private/tmp on macOS)
        assert p == Path(str(target)).expanduser().resolve()
        ensure_dir_writable(p)
        assert p.exists() and p.is_dir()

    def test_env_override_canonicalize(self, tmp_path, monkeypatch):
        """Test that env override canonicalizes paths."""
        custom_path = tmp_path / "cache" / "dir"
        monkeypatch.setenv("GLASSALPHA_DATA_DIR", str(custom_path))

        cache_root = resolve_data_root()
        assert cache_root == custom_path.resolve()

        # Test that we can make it writable
        writable_root = ensure_dir_writable(cache_root)
        assert writable_root.exists()
        assert writable_root.is_dir()

    def test_cache_directory_creation_canonical(self, tmp_path, monkeypatch):
        """Test that cache directory creation uses canonical paths."""
        custom_cache = tmp_path / "custom_cache"
        monkeypatch.setenv("GLASSALPHA_DATA_DIR", str(custom_cache))

        # Should create directory and test writability
        cache_root = ensure_dir_writable(resolve_data_root())
        assert cache_root.exists()
        assert cache_root.is_dir()

        # Test file should be writable
        test_file = cache_root / "test.txt"
        test_file.write_text("test")
        assert test_file.read_text() == "test"
        test_file.unlink()

    def test_cache_directory_permission_error_canonical(self, monkeypatch, tmp_path):
        """Test handling of permission errors with canonical paths."""
        # Create a read-only directory to simulate permission error
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o555)  # Read-only, no write permission

        # Point to a subdirectory within readonly_dir that can't be created
        target_dir = readonly_dir / "subdir"
        monkeypatch.setenv("GLASSALPHA_DATA_DIR", str(target_dir))

        with pytest.raises(RuntimeError, match="Cannot create or write to cache directory"):
            ensure_dir_writable(resolve_data_root())


class TestGetCachePath:
    """Test cache path construction for datasets."""

    def test_cache_path_construction_canonical(self, monkeypatch):
        """Test construction of cache paths uses canonical realpaths."""
        monkeypatch.setenv("GLASSALPHA_DATA_DIR", "/cache/root")

        cache_path = get_cache_path("german_credit", "processed.csv")

        expected = Path("/cache/root/german_credit/processed.csv")
        assert cache_path == expected.resolve()

    def test_cache_path_different_dataset_canonical(self, monkeypatch):
        """Test cache path with different dataset uses canonical realpaths."""
        monkeypatch.setenv("GLASSALPHA_DATA_DIR", "/cache/root")

        cache_path = get_cache_path("test_dataset", "test_file.json")

        expected = Path("/cache/root/test_dataset/test_file.json")
        assert cache_path == expected.resolve()


class TestEnsureDirWritable:
    """Test ensure_dir_writable function."""

    def test_ensure_dir_writable_creates_directory(self, tmp_path):
        """Test that ensure_dir_writable creates the directory."""
        test_dir = tmp_path / "test_cache"

        result = ensure_dir_writable(test_dir)

        assert result == test_dir
        assert test_dir.exists()
        assert test_dir.is_dir()

    def test_ensure_dir_writable_with_existing_directory(self, tmp_path):
        """Test that ensure_dir_writable works with existing directory."""
        test_dir = tmp_path / "existing_cache"
        test_dir.mkdir()

        result = ensure_dir_writable(test_dir)

        assert result == test_dir
        assert test_dir.exists()
        assert test_dir.is_dir()

    def test_ensure_dir_writable_creates_parent_directories(self, tmp_path):
        """Test that ensure_dir_writable creates parent directories."""
        test_dir = tmp_path / "deep" / "nested" / "cache"

        result = ensure_dir_writable(test_dir)

        assert result == test_dir
        assert test_dir.exists()
        assert test_dir.is_dir()
        assert test_dir.parent.exists()
        assert test_dir.parent.parent.exists()

    def test_ensure_dir_writable_tests_writability(self, tmp_path):
        """Test that ensure_dir_writable verifies directory is writable."""
        test_dir = tmp_path / "writable_cache"

        result = ensure_dir_writable(test_dir)

        assert result == test_dir

        # Should be able to write a file
        test_file = test_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.read_text() == "test content"
        test_file.unlink()

    def test_ensure_dir_writable_permission_error(self, tmp_path):
        """Test that ensure_dir_writable raises error for unwritable directory."""
        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o555)  # Read-only, no write permission

        # Try to create a subdirectory that should fail
        target_dir = readonly_dir / "subdir"

        with pytest.raises(RuntimeError, match="Cannot create or write to cache directory"):
            ensure_dir_writable(target_dir)
