"""Phase 5: Error Handling Tests

Tests for GlassAlphaError and specific error subclasses.
"""

import pytest

from glassalpha.exceptions import (
    AUCWithoutProbaError,
    DataHashMismatchError,
    FileExistsError,
    GlassAlphaError,
    InvalidProtectedAttributesError,
    LengthMismatchError,
    MultiIndexNotSupportedError,
    NonBinaryClassificationError,
    NoPredictProbaError,
    ResultIDMismatchError,
    UnsupportedMissingPolicyError,
)

# Mark as contract test (must pass before release)
pytestmark = pytest.mark.contract


class TestGlassAlphaError:
    """Tests for base GlassAlphaError class."""

    def test_has_code(self):
        """Error has code attribute"""
        error = GlassAlphaError(
            code="GAE9999",
            message="Test error",
            fix="Fix it",
        )

        assert error.code == "GAE9999"

    def test_has_message(self):
        """Error has message attribute"""
        error = GlassAlphaError(
            code="GAE9999",
            message="Test error",
            fix="Fix it",
        )

        assert error.message == "Test error"

    def test_has_fix(self):
        """Error has fix attribute"""
        error = GlassAlphaError(
            code="GAE9999",
            message="Test error",
            fix="Fix it like this",
        )

        assert error.fix == "Fix it like this"

    def test_has_docs(self):
        """Error has docs URL"""
        error = GlassAlphaError(
            code="GAE9999",
            message="Test error",
            fix="Fix it",
        )

        assert error.docs == "https://glassalpha.com/errors/GAE9999"

    def test_custom_docs_url(self):
        """Can provide custom docs URL"""
        error = GlassAlphaError(
            code="GAE9999",
            message="Test error",
            fix="Fix it",
            docs="https://custom.com/error",
        )

        assert error.docs == "https://custom.com/error"

    def test_str_includes_all_fields(self):
        """String representation includes code, message, fix, docs"""
        error = GlassAlphaError(
            code="GAE9999",
            message="Test error",
            fix="Fix it",
        )

        error_str = str(error)
        assert "GAE9999" in error_str
        assert "Test error" in error_str
        assert "Fix it" in error_str
        assert "https://glassalpha.com/errors/GAE9999" in error_str

    def test_repr_includes_code_and_message(self):
        """Repr includes code and message"""
        error = GlassAlphaError(
            code="GAE9999",
            message="Test error",
            fix="Fix it",
        )

        repr_str = repr(error)
        assert "GAE9999" in repr_str
        assert "Test error" in repr_str

    def test_is_exception(self):
        """GlassAlphaError is an Exception"""
        error = GlassAlphaError(
            code="GAE9999",
            message="Test error",
            fix="Fix it",
        )

        assert isinstance(error, Exception)

    def test_can_be_raised(self):
        """Error can be raised and caught"""
        with pytest.raises(GlassAlphaError) as exc_info:
            raise GlassAlphaError(
                code="GAE9999",
                message="Test error",
                fix="Fix it",
            )

        assert exc_info.value.code == "GAE9999"


class TestInvalidProtectedAttributesError:
    """Tests for GAE1001 error."""

    def test_has_correct_code(self):
        """Error has code GAE1001"""
        error = InvalidProtectedAttributesError("keys must be strings")

        assert error.code == "GAE1001"

    def test_includes_reason(self):
        """Error message includes reason"""
        error = InvalidProtectedAttributesError("keys must be strings")

        assert "keys must be strings" in error.message

    def test_has_fix(self):
        """Error has actionable fix"""
        error = InvalidProtectedAttributesError("keys must be strings")

        assert "dict with string keys" in error.fix
        assert "Example:" in error.fix or "example" in error.fix.lower()

    def test_is_glassalpha_error(self):
        """Is subclass of GlassAlphaError"""
        error = InvalidProtectedAttributesError("test")

        assert isinstance(error, GlassAlphaError)


class TestLengthMismatchError:
    """Tests for GAE1003 error."""

    def test_has_correct_code(self):
        """Error has code GAE1003"""
        error = LengthMismatchError(expected=100, got=50, name="y")

        assert error.code == "GAE1003"

    def test_includes_lengths(self):
        """Error message includes expected and actual lengths"""
        error = LengthMismatchError(expected=100, got=50, name="y")

        assert "100" in error.message
        assert "50" in error.message

    def test_includes_name(self):
        """Error message includes array name"""
        error = LengthMismatchError(expected=100, got=50, name="y")

        assert "y" in error.message

    def test_has_fix(self):
        """Error has actionable fix"""
        error = LengthMismatchError(expected=100, got=50, name="y")

        assert "same length" in error.fix.lower()
        assert "100" in error.fix


class TestNonBinaryClassificationError:
    """Tests for GAE1004 error."""

    def test_has_correct_code(self):
        """Error has code GAE1004"""
        error = NonBinaryClassificationError(n_classes=3)

        assert error.code == "GAE1004"

    def test_includes_n_classes(self):
        """Error message includes number of classes"""
        error = NonBinaryClassificationError(n_classes=5)

        assert "5" in error.message

    def test_has_fix(self):
        """Error has actionable fix"""
        error = NonBinaryClassificationError(n_classes=3)

        assert "binary" in error.fix.lower()
        assert "one-vs-rest" in error.fix.lower() or "filter" in error.fix.lower()


class TestUnsupportedMissingPolicyError:
    """Tests for GAE1005 error."""

    def test_has_correct_code(self):
        """Error has code GAE1005"""
        error = UnsupportedMissingPolicyError("invalid_policy")

        assert error.code == "GAE1005"

    def test_includes_policy(self):
        """Error message includes invalid policy"""
        error = UnsupportedMissingPolicyError("invalid_policy")

        assert "invalid_policy" in error.message

    def test_has_fix(self):
        """Error has actionable fix with valid options"""
        error = UnsupportedMissingPolicyError("invalid_policy")

        assert "unknown" in error.fix.lower()
        assert "drop" in error.fix.lower()


class TestNoPredictProbaError:
    """Tests for GAE1008 error."""

    def test_has_correct_code(self):
        """Error has code GAE1008"""
        error = NoPredictProbaError("SVC")

        assert error.code == "GAE1008"

    def test_includes_model_type(self):
        """Error message includes model type"""
        error = NoPredictProbaError("SVC")

        assert "SVC" in error.message

    def test_has_fix(self):
        """Error has actionable fix"""
        error = NoPredictProbaError("SVC")

        assert "probability=True" in error.fix or "predict_proba" in error.fix


class TestAUCWithoutProbaError:
    """Tests for GAE1009 error."""

    def test_has_correct_code(self):
        """Error has code GAE1009"""
        error = AUCWithoutProbaError("roc_auc")

        assert error.code == "GAE1009"

    def test_includes_metric(self):
        """Error message includes metric name"""
        error = AUCWithoutProbaError("roc_auc")

        assert "roc_auc" in error.message

    def test_has_fix(self):
        """Error has actionable fix with alternatives"""
        error = AUCWithoutProbaError("roc_auc")

        assert "accuracy" in error.fix or "precision" in error.fix
        assert "predict_proba" in error.fix


class TestMultiIndexNotSupportedError:
    """Tests for GAE1012 error."""

    def test_has_correct_code(self):
        """Error has code GAE1012"""
        error = MultiIndexNotSupportedError("X")

        assert error.code == "GAE1012"

    def test_includes_data_name(self):
        """Error message includes data name"""
        error = MultiIndexNotSupportedError("X")

        assert "X" in error.message

    def test_has_fix(self):
        """Error has actionable fix"""
        error = MultiIndexNotSupportedError("X")

        assert "reset_index" in error.fix


class TestResultIDMismatchError:
    """Tests for GAE2002 error."""

    def test_has_correct_code(self):
        """Error has code GAE2002"""
        error = ResultIDMismatchError(expected="abc123" * 11, got="def456" * 11)

        assert error.code == "GAE2002"

    def test_includes_hashes(self):
        """Error message includes (truncated) hashes"""
        expected = "abc123" * 11
        got = "def456" * 11
        error = ResultIDMismatchError(expected=expected, got=got)

        # Should include first 16 chars
        assert expected[:16] in error.message
        assert got[:16] in error.message

    def test_has_fix(self):
        """Error has actionable fix with checklist"""
        error = ResultIDMismatchError(expected="abc" * 22, got="def" * 22)

        assert "seed" in error.fix.lower()
        assert "data" in error.fix.lower() or "hash" in error.fix.lower()


class TestDataHashMismatchError:
    """Tests for GAE2003 error."""

    def test_has_correct_code(self):
        """Error has code GAE2003"""
        error = DataHashMismatchError("X", expected="sha256:abc123" * 5, got="sha256:def456" * 5)

        assert error.code == "GAE2003"

    def test_includes_data_name(self):
        """Error message includes data name"""
        error = DataHashMismatchError("y", expected="sha256:abc" * 10, got="sha256:def" * 10)

        assert "y" in error.message

    def test_includes_hashes(self):
        """Error message includes (truncated) hashes"""
        expected = "sha256:abc123" * 5
        got = "sha256:def456" * 5
        error = DataHashMismatchError("X", expected=expected, got=got)

        # Should include first 20 chars
        assert expected[:20] in error.message
        assert got[:20] in error.message

    def test_has_fix(self):
        """Error has actionable fix"""
        error = DataHashMismatchError("X", expected="sha256:abc" * 10, got="sha256:def" * 10)

        assert "data has changed" in error.fix.lower()
        assert "X" in error.fix


class TestFileExistsError:
    """Tests for GAE4001 error."""

    def test_has_correct_code(self):
        """Error has code GAE4001"""
        error = FileExistsError("/path/to/file.pdf")

        assert error.code == "GAE4001"

    def test_includes_path(self):
        """Error message includes file path"""
        error = FileExistsError("/path/to/file.pdf")

        assert "/path/to/file.pdf" in error.message

    def test_has_fix(self):
        """Error has actionable fix"""
        error = FileExistsError("/path/to/file.pdf")

        assert "overwrite=True" in error.fix


class TestErrorCodeCoverage:
    """Tests for error code documentation coverage."""

    def test_all_documented_codes_have_classes(self):
        """All error codes mentioned in docstrings have corresponding classes"""
        # Error codes mentioned in API (from Phase 3)
        documented_codes = [
            "GAE1001",  # InvalidProtectedAttributesError
            "GAE1003",  # LengthMismatchError
            "GAE1004",  # NonBinaryClassificationError
            "GAE1005",  # UnsupportedMissingPolicyError
            "GAE1008",  # NoPredictProbaError
            "GAE1009",  # AUCWithoutProbaError
            "GAE1012",  # MultiIndexNotSupportedError
            "GAE2002",  # ResultIDMismatchError
            "GAE2003",  # DataHashMismatchError
            "GAE4001",  # FileExistsError
        ]

        # All should have corresponding error classes
        error_classes = [
            InvalidProtectedAttributesError("test"),
            LengthMismatchError(100, 50, "y"),
            NonBinaryClassificationError(3),
            UnsupportedMissingPolicyError("test"),
            NoPredictProbaError("SVC"),
            AUCWithoutProbaError("roc_auc"),
            MultiIndexNotSupportedError("X"),
            ResultIDMismatchError("a" * 64, "b" * 64),
            DataHashMismatchError("X", "sha256:a" * 10, "sha256:b" * 10),
            FileExistsError("/tmp/file"),
        ]

        error_codes = [e.code for e in error_classes]

        for code in documented_codes:
            assert code in error_codes, f"Documented code {code} has no corresponding error class"

    def test_all_errors_inherit_from_base(self):
        """All error classes inherit from GlassAlphaError"""
        error_classes = [
            InvalidProtectedAttributesError("test"),
            LengthMismatchError(100, 50, "y"),
            NonBinaryClassificationError(3),
            UnsupportedMissingPolicyError("test"),
            NoPredictProbaError("SVC"),
            AUCWithoutProbaError("roc_auc"),
            MultiIndexNotSupportedError("X"),
            ResultIDMismatchError("a" * 64, "b" * 64),
            DataHashMismatchError("X", "sha256:a" * 10, "sha256:b" * 10),
            FileExistsError("/tmp/file"),
        ]

        for error in error_classes:
            assert isinstance(error, GlassAlphaError)

    def test_all_errors_have_fix(self):
        """All errors have non-empty fix message"""
        error_classes = [
            InvalidProtectedAttributesError("test"),
            LengthMismatchError(100, 50, "y"),
            NonBinaryClassificationError(3),
            UnsupportedMissingPolicyError("test"),
            NoPredictProbaError("SVC"),
            AUCWithoutProbaError("roc_auc"),
            MultiIndexNotSupportedError("X"),
            ResultIDMismatchError("a" * 64, "b" * 64),
            DataHashMismatchError("X", "sha256:a" * 10, "sha256:b" * 10),
            FileExistsError("/tmp/file"),
        ]

        for error in error_classes:
            assert error.fix
            assert len(error.fix) > 10  # Substantive fix message

    def test_all_errors_have_docs(self):
        """All errors have docs URL"""
        error_classes = [
            InvalidProtectedAttributesError("test"),
            LengthMismatchError(100, 50, "y"),
            NonBinaryClassificationError(3),
            UnsupportedMissingPolicyError("test"),
            NoPredictProbaError("SVC"),
            AUCWithoutProbaError("roc_auc"),
            MultiIndexNotSupportedError("X"),
            ResultIDMismatchError("a" * 64, "b" * 64),
            DataHashMismatchError("X", "sha256:a" * 10, "sha256:b" * 10),
            FileExistsError("/tmp/file"),
        ]

        for error in error_classes:
            assert error.docs
            assert error.docs.startswith("https://")
            assert error.code in error.docs
