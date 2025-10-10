"""Test offline mode behavior."""

import sys

import pytest

sys.path.insert(0, "src")
from glassalpha.config.schema import DataConfig
from glassalpha.pipeline.audit import AuditPipeline
from glassalpha.utils.cache_dirs import resolve_data_root


@pytest.mark.ci
def test_offline_mode_prevents_fetching():
    """Offline mode must raise FileNotFoundError when dataset not cached."""
    # Clear cache to ensure dataset doesn't exist
    cache_root = resolve_data_root()
    german_credit_path = cache_root / "german_credit_processed.csv"
    if german_credit_path.exists():
        german_credit_path.unlink()

    # Test offline mode with missing file
    config = DataConfig(
        dataset="german_credit",
        fetch="if_missing",
        offline=True,
    )

    pipeline = AuditPipeline.__new__(AuditPipeline)
    pipeline.config = type("Config", (), {"data": config})()

    # Should raise FileNotFoundError mentioning offline mode
    with pytest.raises(FileNotFoundError) as exc_info:
        pipeline._resolve_dataset_path()

    assert "offline" in str(exc_info.value).lower()
