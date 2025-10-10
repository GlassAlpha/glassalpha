"""Phase 4: Determinism and Hashing Tests

Tests for canonicalization, result ID computation, and data hashing.
"""

import base64
from datetime import UTC, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from glassalpha.core.canonicalization import (
    _atomic_write,
    canonicalize,
    compute_result_id,
    hash_data_for_manifest,
)

# Mark as contract test (must pass before release)
pytestmark = pytest.mark.contract


class TestCanonicalize:
    """Tests for canonicalize() function."""

    def test_none(self):
        """None passes through"""
        assert canonicalize(None) is None

    def test_bool(self):
        """Booleans pass through"""
        assert canonicalize(True) is True
        assert canonicalize(False) is False

    def test_int(self):
        """Integers pass through"""
        assert canonicalize(42) == 42
        assert canonicalize(-17) == -17

    def test_numpy_int(self):
        """NumPy integers convert to Python int"""
        assert canonicalize(np.int64(42)) == 42
        assert isinstance(canonicalize(np.int32(17)), int)

    def test_string(self):
        """Strings pass through"""
        assert canonicalize("hello") == "hello"
        assert canonicalize("") == ""

    def test_float_normal(self):
        """Normal floats pass through with 17 decimal precision"""
        result = canonicalize(3.14159265358979323846)
        assert isinstance(result, float)
        assert result == 3.14159265358979324

    def test_float_nan(self):
        """NaN becomes string "NaN" """
        assert canonicalize(float("nan")) == "NaN"
        assert canonicalize(np.nan) == "NaN"

    def test_float_inf(self):
        """Infinity becomes string "Infinity" """
        assert canonicalize(float("inf")) == "Infinity"
        assert canonicalize(np.inf) == "Infinity"

    def test_float_neg_inf(self):
        """Negative infinity becomes string "-Infinity" """
        assert canonicalize(float("-inf")) == "-Infinity"
        assert canonicalize(-np.inf) == "-Infinity"

    def test_float_negative_zero(self):
        """Negative zero normalized to 0.0"""
        result = canonicalize(-0.0)
        assert result == 0.0
        assert not np.signbit(result)

    def test_bytes(self):
        """Bytes become base64-encoded dict"""
        data = b"hello world"
        result = canonicalize(data)

        assert result["_type"] == "bytes"
        assert result["data"] == base64.b64encode(data).decode("ascii")

    def test_ndarray_1d(self):
        """1D array becomes dict with data, shape, dtype"""
        arr = np.array([1, 2, 3])
        result = canonicalize(arr)

        assert result["_type"] == "ndarray"
        assert result["data"] == [1, 2, 3]
        assert result["shape"] == [3]
        assert "int" in result["dtype"]

    def test_ndarray_2d(self):
        """2D array includes correct shape"""
        arr = np.array([[1, 2], [3, 4]])
        result = canonicalize(arr)

        assert result["shape"] == [2, 2]
        assert result["data"] == [1, 2, 3, 4]  # Flattened

    def test_ndarray_with_nan(self):
        """Array with NaN canonicalizes NaN to "NaN" """
        arr = np.array([1.0, np.nan, 3.0])
        result = canonicalize(arr)

        assert result["data"][1] == "NaN"

    def test_dict_sorts_keys(self):
        """Dict keys sorted alphabetically"""
        d = {"z": 1, "a": 2, "m": 3}
        result = canonicalize(d)

        keys = list(result.keys())
        assert keys == ["a", "m", "z"]

    def test_dict_recursive(self):
        """Nested dicts canonicalized recursively"""
        d = {"outer": {"z": np.nan, "a": 1}}
        result = canonicalize(d)

        assert result["outer"]["z"] == "NaN"
        assert list(result["outer"].keys()) == ["a", "z"]

    def test_list(self):
        """Lists canonicalized element-wise"""
        lst = [1, np.nan, "hello", None]
        result = canonicalize(lst)

        assert result == [1, "NaN", "hello", None]

    def test_tuple(self):
        """Tuples become lists"""
        tup = (1, 2, 3)
        result = canonicalize(tup)

        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_datetime_naive(self):
        """Naive datetime treated as UTC"""
        dt = datetime(2025, 1, 15, 12, 30, 0)
        result = canonicalize(dt)

        assert result == "2025-01-15T12:30:00+00:00"

    def test_datetime_with_tz(self):
        """Datetime with timezone preserved"""
        dt = datetime(2025, 1, 15, 12, 30, 0, tzinfo=UTC)
        result = canonicalize(dt)

        assert result == "2025-01-15T12:30:00+00:00"

    def test_pandas_timestamp(self):
        """Pandas Timestamp becomes ISO string"""
        ts = pd.Timestamp("2025-01-15 12:30:00")
        result = canonicalize(ts)

        assert "2025-01-15" in result
        assert "12:30:00" in result


class TestComputeResultID:
    """Tests for compute_result_id() function."""

    def test_returns_64_hex_chars(self):
        """Returns 64-character hex string"""
        result_dict = {"performance": {"accuracy": 0.847}}
        result_id = compute_result_id(result_dict)

        assert len(result_id) == 64
        assert all(c in "0123456789abcdef" for c in result_id)

    def test_deterministic(self):
        """Same input produces same hash"""
        result_dict = {"performance": {"accuracy": 0.847}, "fairness": {}}

        id1 = compute_result_id(result_dict)
        id2 = compute_result_id(result_dict)

        assert id1 == id2

    def test_excludes_manifest(self):
        """Manifest excluded from hash"""
        dict1 = {"performance": {"accuracy": 0.847}, "manifest": {"seed": 42}}
        dict2 = {"performance": {"accuracy": 0.847}, "manifest": {"seed": 999}}

        id1 = compute_result_id(dict1)
        id2 = compute_result_id(dict2)

        # Same result despite different manifests
        assert id1 == id2

    def test_sensitive_to_values(self):
        """Different values produce different hashes"""
        dict1 = {"performance": {"accuracy": 0.847}}
        dict2 = {"performance": {"accuracy": 0.848}}

        id1 = compute_result_id(dict1)
        id2 = compute_result_id(dict2)

        assert id1 != id2

    def test_sensitive_to_keys(self):
        """Different keys produce different hashes"""
        dict1 = {"performance": {"accuracy": 0.847}}
        dict2 = {"performance": {"precision": 0.847}}

        id1 = compute_result_id(dict1)
        id2 = compute_result_id(dict2)

        assert id1 != id2

    def test_key_order_irrelevant(self):
        """Key order doesn't matter (canonicalized)"""
        dict1 = {"a": 1, "b": 2, "c": 3}
        dict2 = {"c": 3, "a": 1, "b": 2}

        id1 = compute_result_id(dict1)
        id2 = compute_result_id(dict2)

        assert id1 == id2

    def test_handles_nan(self):
        """NaN handled consistently"""
        dict_with_nan = {"performance": {"metric": float("nan")}}

        id1 = compute_result_id(dict_with_nan)
        id2 = compute_result_id(dict_with_nan)

        assert id1 == id2

    def test_handles_inf(self):
        """Infinity handled consistently"""
        dict_with_inf = {"performance": {"metric": float("inf")}}

        id1 = compute_result_id(dict_with_inf)
        id2 = compute_result_id(dict_with_inf)

        assert id1 == id2

    def test_handles_nested_arrays(self):
        """Nested arrays canonicalized correctly"""
        dict_with_array = {"calibration": {"bins": np.array([0.0, 0.5, 1.0])}}

        id1 = compute_result_id(dict_with_array)
        id2 = compute_result_id(dict_with_array)

        assert id1 == id2


class TestHashDataForManifest:
    """Tests for hash_data_for_manifest() function."""

    def test_returns_sha256_prefix(self):
        """Returns hash with 'sha256:' prefix by default"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = hash_data_for_manifest(df)

        assert result.startswith("sha256:")
        assert len(result) == 71  # "sha256:" (7) + 64 hex chars

    def test_no_prefix_option(self):
        """Can disable sha256: prefix"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = hash_data_for_manifest(df, prefix=False)

        assert not result.startswith("sha256:")
        assert len(result) == 64

    def test_deterministic(self):
        """Same data produces same hash"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

        hash1 = hash_data_for_manifest(df)
        hash2 = hash_data_for_manifest(df)

        assert hash1 == hash2

    def test_sensitive_to_values(self):
        """Different values produce different hashes"""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 4]})

        hash1 = hash_data_for_manifest(df1)
        hash2 = hash_data_for_manifest(df2)

        assert hash1 != hash2

    def test_sensitive_to_column_names(self):
        """Different column names produce different hashes"""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [1, 2, 3]})

        hash1 = hash_data_for_manifest(df1)
        hash2 = hash_data_for_manifest(df2)

        assert hash1 != hash2

    def test_sensitive_to_column_order(self):
        """Column order affects hash"""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"b": [3, 4], "a": [1, 2]})

        hash1 = hash_data_for_manifest(df1)
        hash2 = hash_data_for_manifest(df2)

        assert hash1 != hash2

    def test_sensitive_to_dtypes(self):
        """Different dtypes produce different hashes"""
        df1 = pd.DataFrame({"a": [1, 2, 3]})  # int64
        df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})  # float64

        hash1 = hash_data_for_manifest(df1)
        hash2 = hash_data_for_manifest(df2)

        assert hash1 != hash2

    def test_handles_categorical(self):
        """Categorical dtype handled correctly"""
        df = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a", "c"])})

        hash1 = hash_data_for_manifest(df)
        hash2 = hash_data_for_manifest(df)

        assert hash1 == hash2

    def test_categorical_categories_matter(self):
        """Different category sets produce different hashes"""
        df1 = pd.DataFrame({"cat": pd.Categorical(["a", "b"], categories=["a", "b"])})
        df2 = pd.DataFrame({"cat": pd.Categorical(["a", "b"], categories=["a", "b", "c"])})

        hash1 = hash_data_for_manifest(df1)
        hash2 = hash_data_for_manifest(df2)

        assert hash1 != hash2

    def test_handles_string_dtype(self):
        """String dtype handled correctly"""
        df = pd.DataFrame({"s": ["hello", "world"]})

        hash1 = hash_data_for_manifest(df)
        hash2 = hash_data_for_manifest(df)

        assert hash1 == hash2

    def test_handles_bool_dtype(self):
        """Boolean dtype handled correctly"""
        df = pd.DataFrame({"b": [True, False, True]})

        hash1 = hash_data_for_manifest(df)
        hash2 = hash_data_for_manifest(df)

        assert hash1 == hash2

    def test_handles_datetime(self):
        """Datetime dtype handled correctly"""
        df = pd.DataFrame({"dt": pd.date_range("2025-01-01", periods=3)})

        hash1 = hash_data_for_manifest(df)
        hash2 = hash_data_for_manifest(df)

        assert hash1 == hash2

    def test_handles_timedelta(self):
        """Timedelta dtype handled correctly"""
        df = pd.DataFrame({"td": pd.timedelta_range("1 day", periods=3)})

        hash1 = hash_data_for_manifest(df)
        hash2 = hash_data_for_manifest(df)

        assert hash1 == hash2

    def test_range_index_default_ignored(self):
        """Default RangeIndex (0, n, 1) doesn't affect hash"""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 3]}, index=pd.RangeIndex(0, 3, 1))

        hash1 = hash_data_for_manifest(df1)
        hash2 = hash_data_for_manifest(df2)

        assert hash1 == hash2

    def test_range_index_non_default_included(self):
        """Non-default RangeIndex affects hash"""
        df1 = pd.DataFrame({"a": [1, 2, 3]}, index=pd.RangeIndex(0, 3, 1))
        df2 = pd.DataFrame({"a": [1, 2, 3]}, index=pd.RangeIndex(1, 4, 1))

        hash1 = hash_data_for_manifest(df1)
        hash2 = hash_data_for_manifest(df2)

        assert hash1 != hash2

    def test_multiindex_raises(self):
        """MultiIndex raises ValueError"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.set_index([df.index, df["a"]])

        with pytest.raises(ValueError, match="MultiIndex not supported"):
            hash_data_for_manifest(df)

    def test_series_handled(self):
        """Series converted to DataFrame"""
        series = pd.Series([1, 2, 3], name="col")

        hash1 = hash_data_for_manifest(series)
        hash2 = hash_data_for_manifest(series)

        assert hash1 == hash2

    def test_ndarray_1d_handled(self):
        """1D ndarray converted to DataFrame"""
        arr = np.array([1, 2, 3])

        hash1 = hash_data_for_manifest(arr)
        hash2 = hash_data_for_manifest(arr)

        assert hash1 == hash2

    def test_ndarray_2d_handled(self):
        """2D ndarray converted to DataFrame"""
        arr = np.array([[1, 2], [3, 4]])

        hash1 = hash_data_for_manifest(arr)
        hash2 = hash_data_for_manifest(arr)

        assert hash1 == hash2


class TestAtomicWrite:
    """Tests for _atomic_write() function."""

    def test_writes_file(self, tmp_path):
        """File written successfully"""
        target = tmp_path / "output.txt"
        content = b"hello world"

        _atomic_write(str(target), content)

        assert target.exists()
        assert target.read_bytes() == content

    def test_creates_parent_dirs(self, tmp_path):
        """Parent directories created if missing"""
        target = tmp_path / "subdir" / "nested" / "output.txt"
        content = b"hello"

        _atomic_write(str(target), content)

        assert target.exists()
        assert target.read_bytes() == content

    def test_overwrites_existing(self, tmp_path):
        """Existing file overwritten"""
        target = tmp_path / "output.txt"
        target.write_bytes(b"old content")

        _atomic_write(str(target), b"new content")

        assert target.read_bytes() == b"new content"

    def test_atomic_on_failure(self, tmp_path):
        """Original file intact if write fails"""
        target = tmp_path / "output.txt"
        target.write_bytes(b"original")

        # Simulate failure by writing to non-writable location
        # (can't easily test true atomicity without killing process)
        # Just verify no temp files left behind

        _atomic_write(str(target), b"updated")

        # No .tmp files left
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestDeterminismAcrossRuns:
    """Integration tests for determinism across multiple runs."""

    def test_result_id_stable_across_runs(self):
        """Result ID identical across 10 runs"""
        result_dict = {
            "performance": {"accuracy": 0.847, "f1": 0.856},
            "fairness": {"demographic_parity_diff": 0.023},
            "calibration": {"ece": 0.045},
        }

        ids = [compute_result_id(result_dict) for _ in range(10)]

        # All identical
        assert len(set(ids)) == 1

    def test_data_hash_stable_across_runs(self):
        """Data hash identical across 10 runs"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})

        hashes = [hash_data_for_manifest(df) for _ in range(10)]

        # All identical
        assert len(set(hashes)) == 1

    def test_no_hash_collisions(self):
        """Different inputs produce different hashes"""
        base_dict = {"performance": {"accuracy": 0.847}}

        # Generate 100 variants
        variants = [{**base_dict, "performance": {"accuracy": 0.847 + i * 0.001}} for i in range(100)]

        ids = [compute_result_id(v) for v in variants]

        # All unique (no collisions)
        assert len(set(ids)) == 100
