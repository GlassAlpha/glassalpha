"""Tests for concurrent dataset fetching with file locking."""

import threading
import time
from pathlib import Path

import pytest

from glassalpha.config.schema import DataConfig
from glassalpha.datasets.registry import REGISTRY, DatasetSpec
from glassalpha.pipeline.audit import AuditPipeline
from glassalpha.utils.locks import file_lock


class TestConcurrencyFetch:
    """Test concurrent dataset fetching behavior."""

    def setup_method(self):
        """Set up test environment."""
        # Clear registry for clean tests
        REGISTRY.clear()

        # Track fetch calls
        self.fetch_calls = []

        def mock_fetch():
            # Simulate some processing time
            time.sleep(0.1)
            self.fetch_calls.append(time.time())

            # Actually create the file so fetcher behaves realistically
            output = Path("/tmp/test_dataset_temp.csv")
            output.write_text("test,data\n1,2\n3,4\n")
            return output

        REGISTRY["test_dataset"] = DatasetSpec(
            key="test_dataset",
            default_relpath="test_dataset.csv",
            fetch_fn=mock_fetch,
            schema_version="v1",
        )

    def teardown_method(self):
        """Clean up after tests."""
        REGISTRY.clear()
        self.fetch_calls.clear()

        # Clean up test files
        test_files = [
            Path("/tmp/test_dataset_temp.csv"),
            Path("/tmp/concurrent_test.csv"),
        ]
        for f in test_files:
            if f.exists():
                f.unlink()

    def test_concurrent_fetch_same_dataset(self, tmp_path, monkeypatch):
        """Test that concurrent fetches of the same dataset work correctly."""
        # Use tmp_path for cache to avoid permission issues
        monkeypatch.setenv("GLASSALPHA_DATA_DIR", str(tmp_path / "cache"))

        # Use registry dataset without path (clean policy)
        config = DataConfig(
            dataset="test_dataset",
            fetch="if_missing",
            offline=False,
        )

        pipeline = AuditPipeline.__new__(AuditPipeline)
        pipeline.config = type("Config", (), {"data": config})()

        # Get requested path via pipeline's resolver
        requested_path = pipeline._resolve_dataset_path()
        results = []

        def fetch_worker():
            try:
                result = pipeline._ensure_dataset_availability(requested_path)
                results.append(result)
            except Exception as e:
                results.append(e)

        # Start multiple threads trying to fetch the same dataset
        threads = []
        for i in range(3):
            thread = threading.Thread(target=fetch_worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete with timeout to prevent hangs
        for thread in threads:
            thread.join(timeout=10.0)
            if thread.is_alive():
                raise TimeoutError(f"Thread {thread.name} did not complete within timeout")

        # All should succeed
        assert len(results) == 3
        assert all(isinstance(r, Path) for r in results), (
            f"Non-Path results: {[r for r in results if not isinstance(r, Path)]}"
        )

        # Only one fetch should have actually happened (due to locking)
        assert len(self.fetch_calls) == 1, f"Expected 1 fetch call, got {len(self.fetch_calls)}"

        # All results should point to the same file
        assert all(r == results[0] for r in results)

    def test_file_lock_timeout(self, tmp_path):
        """Test that file lock times out appropriately."""
        lock_path = tmp_path / "test_lock"
        locked = threading.Event()

        def hold_lock():
            with file_lock(lock_path, timeout_s=1.0):
                locked.set()  # Signal we've acquired the lock
                time.sleep(2)  # Hold past the other thread's timeout

        # Start thread that will hold the lock
        t = threading.Thread(target=hold_lock, daemon=True)
        t.start()

        # Wait until the lock is definitely held (fail fast if it never is)
        assert locked.wait(1.0), "holder thread never acquired the lock"

        # Now this acquisition should time out quickly
        with pytest.raises(TimeoutError), file_lock(lock_path, timeout_s=0.5, retry_ms=50):
            pass

        # Clean up
        t.join()

    def test_file_lock_context_manager(self, tmp_path):
        """Test that file lock properly releases on exception."""
        lock_path = tmp_path / "test_lock2"

        with pytest.raises(ValueError), file_lock(lock_path, timeout_s=1.0):
            # Simulate an exception during the critical section
            raise ValueError("Test exception")

        # Lock file should be cleaned up even after exception
        assert not lock_path.exists()

    def test_file_lock_multiple_acquisitions(self, tmp_path):
        """Test that file lock can be acquired multiple times sequentially."""
        lock_path = tmp_path / "test_lock3"

        # First acquisition
        with file_lock(lock_path, timeout_s=1.0):
            assert lock_path.exists()
            time.sleep(0.1)

        # Lock should be released
        assert not lock_path.exists()

        # Second acquisition
        with file_lock(lock_path, timeout_s=1.0):
            assert lock_path.exists()
            time.sleep(0.1)

        # Lock should be released again
        assert not lock_path.exists()

    def test_lock_path_generation(self):
        """Test that lock paths are generated correctly."""
        from glassalpha.utils.locks import get_lock_path

        target = Path("/tmp/test/file.csv")
        lock_path = get_lock_path(target)

        expected = Path("/tmp/test/file.csv.lock")
        assert lock_path == expected
