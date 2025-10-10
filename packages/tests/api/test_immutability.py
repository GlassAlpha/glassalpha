"""Phase 2: Immutability and Result Object Tests

Tests for deep immutability, frozen dataclasses, and read-only arrays.
"""

import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from glassalpha.api.metrics import (
    CalibrationMetrics,
    FairnessMetrics,
    PerformanceMetrics,
    ReadonlyMetrics,
    StabilityMetrics,
)
from glassalpha.api.result import AuditResult, _freeze_array

# Mark as contract test (must pass before release)
pytestmark = pytest.mark.contract


@pytest.fixture
def sample_metrics():
    """Sample metrics for testing."""
    return {
        "accuracy": 0.847,
        "precision": 0.823,
        "recall": 0.891,
        "f1": 0.856,
        "confusion_matrix": np.array([[50, 10], [5, 35]]),
    }


@pytest.fixture
def sample_result():
    """Sample AuditResult for testing."""
    return AuditResult(
        id="abc123" * 10 + "abcdef",  # 64 hex chars
        schema_version="0.2.0",
        manifest={
            "model_fingerprint": "model_hash_123",
            "model_type": "xgboost.XGBClassifier",
            "data_hash_X": "sha256:X_hash_123",
            "data_hash_y": "sha256:y_hash_123",
            "n_samples": 100,
            "n_features": 10,
            "protected_attributes_categories": {"gender": ["F", "M", "Unknown"]},
            "random_seed": 42,
            "execution": {
                "status": "completed",
                "start_time": "2025-01-01T10:00:00Z",
                "end_time": "2025-01-01T10:00:30Z",
                "duration_seconds": 30.0,
            },
        },
        performance={"accuracy": 0.847, "f1": 0.856},
        fairness={"demographic_parity_max_diff": 0.023},
        calibration={"ece": 0.045},
        stability={"monotonicity_violations": 0},
    )


class TestFreezeArray:
    """Tests for _freeze_array helper."""

    def test_makes_array_read_only(self):
        """Array becomes read-only."""
        arr = np.array([1, 2, 3])
        frozen = _freeze_array(arr)

        assert not frozen.flags.writeable

    def test_write_raises_error(self):
        """Writing to frozen array raises ValueError."""
        arr = np.array([1, 2, 3])
        frozen = _freeze_array(arr)

        with pytest.raises(ValueError, match="read-only"):
            frozen[0] = 999

    def test_c_contiguous(self):
        """Frozen array is C-contiguous."""
        arr = np.array([[1, 2], [3, 4]]).T  # Fortran order
        frozen = _freeze_array(arr)

        assert frozen.flags.c_contiguous

    def test_original_unchanged(self):
        """Original array not modified (copy is made)."""
        arr = np.array([1, 2, 3])
        frozen = _freeze_array(arr)

        # Original is still writeable
        assert arr.flags.writeable
        # Can still write to original
        arr[0] = 999
        # Frozen copy is different
        assert frozen[0] == 1


class TestReadonlyMetrics:
    """Tests for ReadonlyMetrics base class."""

    def test_dict_style_access(self, sample_metrics):
        """Dict-style access works."""
        metrics = ReadonlyMetrics(sample_metrics)

        assert metrics["accuracy"] == 0.847
        assert metrics["precision"] == 0.823

    def test_attribute_style_access(self, sample_metrics):
        """Attribute-style access works."""
        metrics = ReadonlyMetrics(sample_metrics)

        assert metrics.accuracy == 0.847
        assert metrics.precision == 0.823

    def test_missing_key_raises_key_error(self, sample_metrics):
        """Dict access to missing key raises KeyError."""
        metrics = ReadonlyMetrics(sample_metrics)

        with pytest.raises(KeyError):
            _ = metrics["nonexistent"]

    def test_missing_attr_raises_attribute_error(self, sample_metrics):
        """Attribute access to missing metric raises AttributeError."""
        metrics = ReadonlyMetrics(sample_metrics)

        with pytest.raises(AttributeError, match="not available in this result"):
            _ = metrics.nonexistent

    def test_keys_method(self, sample_metrics):
        """keys() returns metric names."""
        metrics = ReadonlyMetrics(sample_metrics)

        keys = list(metrics.keys())
        assert "accuracy" in keys
        assert "precision" in keys

    def test_values_method(self, sample_metrics):
        """values() returns metric values."""
        metrics = ReadonlyMetrics(sample_metrics)

        values = list(metrics.values())
        assert 0.847 in values
        assert 0.823 in values

    def test_items_method(self, sample_metrics):
        """items() returns (name, value) pairs."""
        metrics = ReadonlyMetrics(sample_metrics)

        items = dict(metrics.items())
        assert items["accuracy"] == 0.847

    def test_get_method(self, sample_metrics):
        """get() with default works."""
        metrics = ReadonlyMetrics(sample_metrics)

        assert metrics.get("accuracy") == 0.847
        assert metrics.get("nonexistent", None) is None
        assert metrics.get("nonexistent", 0.0) == 0.0

    def test_len(self, sample_metrics):
        """len() returns number of metrics."""
        metrics = ReadonlyMetrics(sample_metrics)

        assert len(metrics) == len(sample_metrics)

    def test_iteration(self, sample_metrics):
        """Can iterate over metric names."""
        metrics = ReadonlyMetrics(sample_metrics)

        names = list(metrics)
        assert "accuracy" in names
        assert len(names) == len(sample_metrics)

    def test_nested_dict_immutable(self):
        """Nested dicts are frozen."""
        data = {"outer": {"inner": 42}}
        metrics = ReadonlyMetrics(data)

        # Nested dict is MappingProxyType
        import types

        assert isinstance(metrics["outer"], types.MappingProxyType)

        # Cannot mutate nested dict
        with pytest.raises(TypeError):
            metrics["outer"]["inner"] = 999

    def test_nested_array_immutable(self):
        """Nested arrays are read-only."""
        data = {"matrix": np.array([[1, 2], [3, 4]])}
        metrics = ReadonlyMetrics(data)

        matrix = metrics["matrix"]
        with pytest.raises(ValueError, match="read-only"):
            matrix[0, 0] = 999

    def test_list_becomes_tuple(self):
        """Lists are converted to tuples."""
        data = {"items": [1, 2, 3]}
        metrics = ReadonlyMetrics(data)

        assert isinstance(metrics["items"], tuple)
        assert metrics["items"] == (1, 2, 3)

    def test_cannot_set_attribute(self, sample_metrics):
        """Cannot set new attributes."""
        metrics = ReadonlyMetrics(sample_metrics)

        with pytest.raises(AttributeError, match="immutable"):
            metrics.new_metric = 0.5


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics subclass."""

    def test_inherits_readonly_behavior(self, sample_metrics):
        """Inherits ReadonlyMetrics behavior."""
        metrics = PerformanceMetrics(sample_metrics)

        assert metrics["accuracy"] == 0.847
        assert metrics.accuracy == 0.847

    def test_has_proba_required_set(self):
        """Has _PROBA_REQUIRED class attribute."""
        assert hasattr(PerformanceMetrics, "_PROBA_REQUIRED")
        assert "roc_auc" in PerformanceMetrics._PROBA_REQUIRED

    def test_plot_methods_exist(self, sample_metrics):
        """Plot methods exist (even if not implemented yet)."""
        metrics = PerformanceMetrics(sample_metrics)

        assert hasattr(metrics, "plot_confusion_matrix")
        assert hasattr(metrics, "plot_roc_curve")


class TestMetricSubclasses:
    """Tests for other metric subclasses."""

    def test_fairness_metrics(self):
        """FairnessMetrics works."""
        data = {"demographic_parity_diff": 0.023}
        metrics = FairnessMetrics(data)

        assert metrics["demographic_parity_diff"] == 0.023

    def test_calibration_metrics(self):
        """CalibrationMetrics works."""
        data = {"ece": 0.045}
        metrics = CalibrationMetrics(data)

        assert metrics["ece"] == 0.045

    def test_stability_metrics(self):
        """StabilityMetrics works."""
        data = {"monotonicity_violations": 0}
        metrics = StabilityMetrics(data)

        assert metrics["monotonicity_violations"] == 0


class TestAuditResultImmutability:
    """Tests for AuditResult immutability."""

    def test_dataclass_frozen(self, sample_result):
        """Cannot mutate dataclass fields."""
        with pytest.raises(FrozenInstanceError):
            sample_result.id = "new_id"

        with pytest.raises(FrozenInstanceError):
            sample_result.schema_version = "0.3.0"

    def test_manifest_immutable(self, sample_result):
        """Manifest is wrapped in MappingProxyType."""
        import types

        assert isinstance(sample_result.manifest, types.MappingProxyType)

        with pytest.raises(TypeError):
            sample_result.manifest["new_key"] = "value"

    def test_performance_metrics_immutable(self, sample_result):
        """Performance metrics immutable."""
        with pytest.raises(TypeError):
            sample_result.performance["accuracy"] = 0.999

    def test_fairness_metrics_immutable(self, sample_result):
        """Fairness metrics immutable."""
        with pytest.raises(TypeError):
            sample_result.fairness["new_metric"] = 0.1


class TestAuditResultEquality:
    """Tests for AuditResult equality."""

    def test_eq_via_id(self, sample_result):
        """Equality uses result.id."""
        result2 = AuditResult(
            id=sample_result.id,  # Same ID
            schema_version="0.2.0",
            manifest={"different": "manifest"},
            performance={"different": 0.5},
            fairness={},
            calibration={},
            stability={},
        )

        assert sample_result == result2

    def test_not_eq_different_id(self, sample_result):
        """Different IDs not equal."""
        result2 = AuditResult(
            id="different" * 10 + "123456",
            schema_version=sample_result.schema_version,
            manifest=dict(sample_result.manifest),
            performance=dict(sample_result.performance),
            fairness=dict(sample_result.fairness),
            calibration=dict(sample_result.calibration),
            stability=dict(sample_result.stability),
        )

        assert sample_result != result2

    def test_eq_not_implemented_for_other_types(self, sample_result):
        """Returns NotImplemented for non-AuditResult."""
        assert (sample_result == "string") is False
        assert (sample_result == 42) is False
        assert (sample_result == None) is False


class TestAuditResultHash:
    """Tests for AuditResult hashing."""

    def test_hashable(self, sample_result):
        """Result is hashable."""
        hash_val = hash(sample_result)
        assert isinstance(hash_val, int)

    def test_hash_stable(self, sample_result):
        """Hash is stable across multiple calls."""
        hash1 = hash(sample_result)
        hash2 = hash(sample_result)

        assert hash1 == hash2

    @pytest.mark.xfail(reason="Pickle support requires API exposure (Phase 3)")
    def test_hash_stable_across_pickle(self, sample_result):
        """Hash stable after pickle round-trip."""
        hash_before = hash(sample_result)

        restored = pickle.loads(pickle.dumps(sample_result))
        hash_after = hash(restored)

        assert hash_before == hash_after

    def test_can_use_in_set(self, sample_result):
        """Can be used in sets."""
        result_set = {sample_result}
        assert sample_result in result_set

    def test_can_use_as_dict_key(self, sample_result):
        """Can be used as dict key."""
        result_dict = {sample_result: "value"}
        assert result_dict[sample_result] == "value"


class TestAuditResultEquals:
    """Tests for tolerance-based equality."""

    def test_equals_same_id(self, sample_result):
        """Returns True for same ID."""
        assert sample_result.equals(sample_result)

    def test_equals_with_tiny_difference(self):
        """Tolerance-based comparison allows small differences."""
        result1 = AuditResult(
            id="id1" * 16,
            schema_version="0.2.0",
            manifest={},
            performance={"accuracy": 0.847},
            fairness={},
            calibration={},
            stability={},
        )

        result2 = AuditResult(
            id="id2" * 16,
            schema_version="0.2.0",
            manifest={},
            performance={"accuracy": 0.847 + 1e-9},  # Tiny difference
            fairness={},
            calibration={},
            stability={},
        )

        # Strict equality fails
        assert result1 != result2

        # Tolerance-based passes
        assert result1.equals(result2, rtol=1e-5, atol=1e-8)

    def test_equals_different_keys(self):
        """Returns False if different metric keys."""
        result1 = AuditResult(
            id="id1" * 16,
            schema_version="0.2.0",
            manifest={},
            performance={"accuracy": 0.847},
            fairness={},
            calibration={},
            stability={},
        )

        result2 = AuditResult(
            id="id2" * 16,
            schema_version="0.2.0",
            manifest={},
            performance={"precision": 0.823},  # Different key
            fairness={},
            calibration={},
            stability={},
        )

        assert not result1.equals(result2)

    def test_equals_handles_none(self):
        """Gracefully handles None values."""
        result1 = AuditResult(
            id="id1" * 16,
            schema_version="0.2.0",
            manifest={},
            performance={"metric": None},
            fairness={},
            calibration={},
            stability={},
        )

        result2 = AuditResult(
            id="id2" * 16,
            schema_version="0.2.0",
            manifest={},
            performance={"metric": None},
            fairness={},
            calibration={},
            stability={},
        )

        assert result1.equals(result2)


class TestAuditResultRepr:
    """Tests for result representations."""

    def test_repr_compact(self, sample_result):
        """__repr__ is compact for logging."""
        repr_str = repr(sample_result)

        assert "AuditResult" in repr_str
        assert sample_result.id[:12] in repr_str
        assert sample_result.schema_version in repr_str

    def test_repr_html_exists(self, sample_result):
        """_repr_html_ method exists for Jupyter."""
        html = sample_result._repr_html_()

        assert isinstance(html, str)
        assert "GlassAlpha Audit Result" in html
        assert sample_result.id[:16] in html

    def test_repr_html_fast(self, sample_result):
        """_repr_html_ completes quickly (O(1))."""
        import time

        start = time.time()
        _ = sample_result._repr_html_()
        duration = time.time() - start

        assert duration < 0.01  # <10ms

    def test_summary_returns_dict(self, sample_result):
        """summary() returns dictionary."""
        summary = sample_result.summary()

        assert isinstance(summary, dict)
        assert "id" in summary
        assert "schema_version" in summary
        assert "performance" in summary

    def test_summary_fast(self, sample_result):
        """summary() completes quickly (O(1))."""
        import time

        start = time.time()
        _ = sample_result.summary()
        duration = time.time() - start

        assert duration < 0.001  # <1ms


class TestAuditResultMethods:
    """Tests for result methods."""

    def test_to_config_returns_dict(self, sample_result):
        """to_config() returns dictionary."""
        config = sample_result.to_config()

        assert isinstance(config, dict)
        assert "model" in config
        assert "data" in config
        assert "protected_attributes" in config
        assert "random_seed" in config
        assert "expected_result_id" in config

    def test_to_config_includes_hashes(self, sample_result):
        """to_config() includes data hashes."""
        config = sample_result.to_config()

        assert config["data"]["X_hash"] == "sha256:X_hash_123"
        assert config["data"]["y_hash"] == "sha256:y_hash_123"

    def test_to_json_not_implemented_yet(self, sample_result):
        """to_json() raises NotImplementedError (Phase 3)."""
        with pytest.raises(NotImplementedError):
            sample_result.to_json("output.json")

    def test_to_pdf_requires_pdf_deps(self, sample_result, tmp_path):
        """to_pdf() works when PDF deps available, raises ImportError otherwise."""
        output = tmp_path / "audit.pdf"

        # Will either succeed or raise ImportError (not NotImplementedError)
        try:
            sample_result.to_pdf(output)
            assert output.exists(), "PDF should be created"
        except ImportError as e:
            # Expected if PDF dependencies not installed
            assert "glassalpha[pdf]" in str(e)

    def test_save_pdf_requires_pdf_deps(self, sample_result, tmp_path):
        """save() with .pdf extension works when PDF deps available."""
        output = tmp_path / "audit.pdf"

        # Will either succeed or raise ImportError (not NotImplementedError)
        try:
            sample_result.save(output)
            assert output.exists(), "PDF should be created"
        except ImportError as e:
            # Expected if PDF dependencies not installed
            assert "glassalpha[pdf]" in str(e)
