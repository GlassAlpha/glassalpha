"""Phase 6: Coverage Edge Cases

Additional tests to achieve 95%+ coverage for implemented code.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from glassalpha.api.metrics import ReadonlyMetrics, _freeze_nested
from glassalpha.api.result import AuditResult
from glassalpha.core.canonicalization import canonicalize

# Mark as contract test (must pass before release)
pytestmark = pytest.mark.contract


class TestFreezeNestedEdgeCases:
    """Tests for _freeze_nested edge cases."""

    def test_already_mappingproxy(self):
        """Already frozen dict passes through"""
        import types

        frozen = types.MappingProxyType({"a": 1})
        result = _freeze_nested(frozen)

        # Should return a new MappingProxyType
        assert isinstance(result, types.MappingProxyType)

    def test_nested_list_in_dict(self):
        """Nested lists in dict become tuples"""
        data = {"outer": [1, [2, 3], 4]}
        result = _freeze_nested(data)

        assert result["outer"] == (1, (2, 3), 4)
        assert isinstance(result["outer"], tuple)
        assert isinstance(result["outer"][1], tuple)


class TestReadonlyMetricsEdgeCases:
    """Tests for ReadonlyMetrics edge cases."""

    def test_already_frozen_input(self):
        """Can initialize with already frozen data"""
        import types

        frozen_data = types.MappingProxyType({"a": 1, "b": 2})
        metrics = ReadonlyMetrics(frozen_data)

        assert metrics["a"] == 1
        assert metrics["b"] == 2


class TestAuditResultPickle:
    """Tests for AuditResult pickle support."""

    @pytest.mark.skip(reason="Pickle/unpickle contaminates sys.modules, breaks exception identity in subsequent tests")
    def test_pickle_round_trip(self):
        """Result can be pickled and unpickled"""
        result = AuditResult(
            id="abc123" * 10 + "abcdef",
            schema_version="0.2.0",
            manifest={"seed": 42},
            performance={"accuracy": 0.847},
            fairness={},
            calibration={},
            stability={},
        )

        # Pickle and unpickle
        pickled = pickle.dumps(result)
        restored = pickle.loads(pickled)

        assert restored.id == result.id
        assert restored.schema_version == result.schema_version
        assert dict(restored.manifest) == dict(result.manifest)
        assert dict(restored.performance) == dict(result.performance)

    @pytest.mark.skip(reason="Pickle/unpickle contaminates sys.modules, breaks exception identity in subsequent tests")
    def test_pickle_with_optional_sections(self):
        """Pickle works with optional sections"""
        result = AuditResult(
            id="abc123" * 10 + "abcdef",
            schema_version="0.2.0",
            manifest={},
            performance={},
            fairness={},
            calibration={},
            stability={},
            explanations={"shap_values": [0.1, 0.2]},
            recourse={"feasible": True},
        )

        pickled = pickle.dumps(result)
        restored = pickle.loads(pickled)

        assert dict(restored.explanations) == {"shap_values": [0.1, 0.2]}
        assert dict(restored.recourse) == {"feasible": True}

    @pytest.mark.skip(reason="Pickle/unpickle contaminates sys.modules, breaks exception identity in subsequent tests")
    def test_pickle_preserves_immutability(self):
        """Unpickled result is still immutable"""
        result = AuditResult(
            id="abc123" * 10 + "abcdef",
            schema_version="0.2.0",
            manifest={},
            performance={"accuracy": 0.847},
            fairness={},
            calibration={},
            stability={},
        )

        restored = pickle.loads(pickle.dumps(result))

        # Still immutable
        with pytest.raises(TypeError):
            restored.performance["accuracy"] = 0.999


class TestCanonicalizeEdgeCases:
    """Tests for canonicalize edge cases."""

    def test_empty_dict(self):
        """Empty dict canonicalizes"""
        result = canonicalize({})
        assert result == {}

    def test_empty_list(self):
        """Empty list canonicalizes"""
        result = canonicalize([])
        assert result == []

    def test_empty_string(self):
        """Empty string passes through"""
        result = canonicalize("")
        assert result == ""

    def test_zero_int(self):
        """Zero integer passes through"""
        result = canonicalize(0)
        assert result == 0

    def test_zero_float(self):
        """Zero float passes through"""
        result = canonicalize(0.0)
        assert result == 0.0

    def test_pd_series(self):
        """Pandas Series canonicalizes to string"""
        series = pd.Series([1, 2, 3])
        result = canonicalize(series)
        # Series gets converted to string (fallback)
        assert isinstance(result, str)
        assert "1" in result and "2" in result and "3" in result

    def test_custom_object_fallback(self):
        """Unknown objects fallback to str()"""

        class CustomObj:
            def __str__(self):
                return "custom_object"

        result = canonicalize(CustomObj())
        assert result == "custom_object"


class TestResultEqualsEdgeCases:
    """Tests for result.equals() edge cases."""

    def test_equals_with_nan_metrics(self):
        """equals() handles NaN in metrics"""
        result1 = AuditResult(
            id="id1" * 16,
            schema_version="0.2.0",
            manifest={},
            performance={"metric": float("nan")},
            fairness={},
            calibration={},
            stability={},
        )

        result2 = AuditResult(
            id="id2" * 16,
            schema_version="0.2.0",
            manifest={},
            performance={"metric": float("nan")},
            fairness={},
            calibration={},
            stability={},
        )

        # NaN == NaN with equal_nan=True
        assert result1.equals(result2)

    def test_equals_with_inf_metrics(self):
        """equals() handles Inf in metrics"""
        result1 = AuditResult(
            id="id1" * 16,
            schema_version="0.2.0",
            manifest={},
            performance={"metric": float("inf")},
            fairness={},
            calibration={},
            stability={},
        )

        result2 = AuditResult(
            id="id2" * 16,
            schema_version="0.2.0",
            manifest={},
            performance={"metric": float("inf")},
            fairness={},
            calibration={},
            stability={},
        )

        assert result1.equals(result2)

    def test_equals_with_none_value(self):
        """equals() handles None vs number"""
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
            performance={"metric": 0.5},
            fairness={},
            calibration={},
            stability={},
        )

        assert not result1.equals(result2)

    def test_equals_with_string_metrics(self):
        """equals() handles string metrics"""
        result1 = AuditResult(
            id="id1" * 16,
            schema_version="0.2.0",
            manifest={},
            performance={"model_type": "xgboost"},
            fairness={},
            calibration={},
            stability={},
        )

        result2 = AuditResult(
            id="id2" * 16,
            schema_version="0.2.0",
            manifest={},
            performance={"model_type": "xgboost"},
            fairness={},
            calibration={},
            stability={},
        )

        assert result1.equals(result2)


class TestResultSummaryEdgeCases:
    """Tests for result.summary() edge cases."""

    def test_summary_with_missing_metrics(self):
        """summary() handles missing standard metrics"""
        result = AuditResult(
            id="abc123" * 10 + "abcdef",
            schema_version="0.2.0",
            manifest={},
            performance={"custom_metric": 0.9},  # No accuracy/precision/recall/f1
            fairness={"custom_fairness": 0.1},  # No standard fairness metrics
            calibration={},
            stability={},
        )

        summary = result.summary()

        # Should not crash
        assert "id" in summary
        assert "schema_version" in summary
        assert summary["performance"] == {}  # No standard metrics
        assert summary["fairness"] == {}  # No standard metrics

    def test_summary_with_all_standard_metrics(self):
        """summary() includes all standard metrics when present"""
        result = AuditResult(
            id="abc123" * 10 + "abcdef",
            schema_version="0.2.0",
            manifest={},
            performance={
                "accuracy": 0.847,
                "precision": 0.823,
                "recall": 0.891,
                "f1": 0.856,
            },
            fairness={
                "demographic_parity_max_diff": 0.023,
                "equalized_odds_max_diff": 0.031,
            },
            calibration={},
            stability={},
        )

        summary = result.summary()

        assert len(summary["performance"]) == 4
        assert len(summary["fairness"]) == 2
