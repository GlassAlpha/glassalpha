"""Test concurrent dataset fetching."""

import sys
import threading

import pytest

sys.path.insert(0, "src")

from glassalpha.config.schema import DataConfig
from glassalpha.pipeline.audit import AuditPipeline


@pytest.mark.ci
def test_concurrent_dataset_fetching_is_safe():
    """Multiple threads fetching same dataset must not cause race conditions."""
    # Setup test configuration
    config = DataConfig(
        dataset="german_credit",
        fetch="if_missing",
        offline=False,
    )

    pipeline = AuditPipeline.__new__(AuditPipeline)
    pipeline.config = type("Config", (), {"data": config})()

    results = []
    errors = []

    def fetch_worker():
        try:
            # Resolve path first (needs to happen per-thread to be safe)
            resolved_path = pipeline._resolve_dataset_path()
            results.append(resolved_path)
        except Exception as e:
            import traceback

            errors.append((e, traceback.format_exc()))

    # Start multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=fetch_worker)
        threads.append(thread)
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    # Verify results
    assert len(errors) == 0, f"Errors occurred: {[str(e) for e, _ in errors]}"
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    # All should point to the same file
    assert all(r == results[0] for r in results), "Not all results point to same file"

    # File should exist and be readable
    import pandas as pd

    df = pd.read_csv(results[0])
    assert len(df) > 0, "Dataset is empty"
