"""Contract tests for reason code extraction.

These tests verify the core contracts of the reason_codes module:
1. Determinism: Same seed → same output
2. Protected attribute exclusion: Never leak protected features
3. Top-N ranking: Correct ordering by SHAP magnitude
4. Error handling: Clear errors for invalid inputs
"""

import numpy as np
import pandas as pd
import pytest

from glassalpha.explain.reason_codes import (
    DEFAULT_PROTECTED_ATTRIBUTES,
    ReasonCode,
    ReasonCodeResult,
    extract_reason_codes,
    format_adverse_action_notice,
)


class TestReasonCodeExtraction:
    """Test reason code extraction from SHAP values."""

    def test_extract_returns_top_n_negative_contributions(self):
        """Verify only top-N negative contributions are returned."""
        shap_values = np.array([-0.5, 0.3, -0.2, 0.1, -0.4, 0.05])
        feature_names = ["debt", "income", "duration", "savings", "credit_history", "employment"]
        feature_values = pd.Series([5000, 30000, 24, 1000, 2, 3])

        result = extract_reason_codes(
            shap_values=shap_values,
            feature_names=feature_names,
            feature_values=feature_values,
            instance_id=1,
            prediction=0.35,
            top_n=2,
            seed=42,
        )

        # Should return exactly 2 reason codes
        assert len(result.reason_codes) == 2

        # All should be negative contributions
        for code in result.reason_codes:
            assert code.contribution < 0

        # Should be top 2 most negative: debt (-0.5), credit_history (-0.4)
        assert result.reason_codes[0].feature == "debt"
        assert result.reason_codes[0].contribution == -0.5
        assert result.reason_codes[1].feature == "credit_history"
        assert result.reason_codes[1].contribution == -0.4

    def test_extract_is_deterministic_with_seed(self):
        """Same seed → same ranking even with ties."""
        shap_values = np.array([-0.5, -0.5, -0.3, 0.2])  # Two ties at -0.5
        feature_names = ["debt", "loans", "duration", "income"]
        feature_values = pd.Series([5000, 5000, 24, 30000])

        # Run twice with same seed
        result1 = extract_reason_codes(
            shap_values=shap_values,
            feature_names=feature_names,
            feature_values=feature_values,
            instance_id=1,
            prediction=0.35,
            top_n=3,
            seed=42,
        )

        result2 = extract_reason_codes(
            shap_values=shap_values,
            feature_names=feature_names,
            feature_values=feature_values,
            instance_id=1,
            prediction=0.35,
            top_n=3,
            seed=42,
        )

        # Should be identical
        assert len(result1.reason_codes) == len(result2.reason_codes)
        for code1, code2 in zip(result1.reason_codes, result2.reason_codes, strict=False):
            assert code1.feature == code2.feature
            assert code1.contribution == code2.contribution
            assert code1.rank == code2.rank

    def test_extract_excludes_protected_attributes(self):
        """Protected attributes never appear in reason codes."""
        shap_values = np.array([-0.8, -0.5, -0.3, -0.2])
        feature_names = ["age", "debt", "gender", "duration"]  # age, gender protected
        feature_values = pd.Series([25, 5000, 1, 24])

        result = extract_reason_codes(
            shap_values=shap_values,
            feature_names=feature_names,
            feature_values=feature_values,
            instance_id=1,
            prediction=0.35,
            top_n=4,
            seed=42,
        )

        # Should exclude age and gender despite being most negative
        reason_features = [code.feature for code in result.reason_codes]
        assert "age" not in reason_features
        assert "gender" not in reason_features

        # Should only include debt and duration
        assert set(reason_features) == {"debt", "duration"}

        # Should track excluded features
        assert "age" in result.excluded_features
        assert "gender" in result.excluded_features

    def test_extract_ranks_by_magnitude(self):
        """Most negative SHAP value gets rank 1."""
        shap_values = np.array([-0.2, -0.5, -0.3, 0.1])
        feature_names = ["duration", "debt", "credit_history", "income"]
        feature_values = pd.Series([24, 5000, 2, 30000])

        result = extract_reason_codes(
            shap_values=shap_values,
            feature_names=feature_names,
            feature_values=feature_values,
            instance_id=1,
            prediction=0.35,
            top_n=3,
            seed=42,
        )

        # Ranks should be sequential
        ranks = [code.rank for code in result.reason_codes]
        assert ranks == [1, 2, 3]

        # Rank 1 should be most negative
        assert result.reason_codes[0].feature == "debt"
        assert result.reason_codes[0].contribution == -0.5

        # Rank 2 should be second most negative
        assert result.reason_codes[1].feature == "credit_history"
        assert result.reason_codes[1].contribution == -0.3

    def test_extract_handles_all_positive_shap(self):
        """Raise clear error if no negative contributions."""
        shap_values = np.array([0.3, 0.5, 0.2, 0.1])  # All positive
        feature_names = ["income", "savings", "employment", "duration"]
        feature_values = pd.Series([50000, 10000, 5, 36])

        with pytest.raises(ValueError, match="No negative contributions"):
            extract_reason_codes(
                shap_values=shap_values,
                feature_names=feature_names,
                feature_values=feature_values,
                instance_id=1,
                prediction=0.85,
                top_n=2,
                seed=42,
            )

    def test_extract_handles_all_protected_features(self):
        """Raise clear error if all features are protected."""
        shap_values = np.array([-0.5, -0.3, -0.2])
        feature_names = ["age", "gender", "race"]  # All protected
        feature_values = pd.Series([25, 1, 0])

        with pytest.raises(ValueError, match="All features are protected"):
            extract_reason_codes(
                shap_values=shap_values,
                feature_names=feature_names,
                feature_values=feature_values,
                instance_id=1,
                prediction=0.35,
                top_n=2,
                seed=42,
            )

    def test_extract_validates_shap_shape(self):
        """Reject non-1D SHAP values."""
        shap_values = np.array([[-0.5, 0.3], [-0.2, 0.1]])  # 2D
        feature_names = ["debt", "income"]
        feature_values = pd.Series([5000, 30000])

        with pytest.raises(ValueError, match="Expected 1D SHAP values"):
            extract_reason_codes(
                shap_values=shap_values,
                feature_names=feature_names,
                feature_values=feature_values,
                instance_id=1,
                prediction=0.35,
                seed=42,
            )

    def test_extract_validates_feature_name_mismatch(self):
        """Raise clear error if SHAP values don't match feature names."""
        shap_values = np.array([-0.5, 0.3])
        feature_names = ["debt", "income", "duration"]  # Mismatch
        feature_values = pd.Series([5000, 30000, 24])

        with pytest.raises(ValueError, match="don't match feature names"):
            extract_reason_codes(
                shap_values=shap_values,
                feature_names=feature_names,
                feature_values=feature_values,
                instance_id=1,
                prediction=0.35,
                seed=42,
            )

    def test_extract_validates_feature_value_mismatch(self):
        """Raise clear error if feature values don't match feature names."""
        shap_values = np.array([-0.5, 0.3, -0.2])
        feature_names = ["debt", "income", "duration"]
        feature_values = pd.Series([5000, 30000])  # Mismatch

        with pytest.raises(ValueError, match="don't match feature names"):
            extract_reason_codes(
                shap_values=shap_values,
                feature_names=feature_names,
                feature_values=feature_values,
                instance_id=1,
                prediction=0.35,
                seed=42,
            )

    def test_extract_handles_custom_protected_list(self):
        """Allow custom protected attribute list."""
        shap_values = np.array([-0.5, -0.3, -0.2])
        feature_names = ["debt", "custom_protected", "duration"]
        feature_values = pd.Series([5000, 100, 24])

        result = extract_reason_codes(
            shap_values=shap_values,
            feature_names=feature_names,
            feature_values=feature_values,
            instance_id=1,
            prediction=0.35,
            protected_attributes=["custom_protected"],
            top_n=3,
            seed=42,
        )

        # Should exclude custom_protected
        reason_features = [code.feature for code in result.reason_codes]
        assert "custom_protected" not in reason_features
        assert "custom_protected" in result.excluded_features

    def test_extract_decision_matches_threshold(self):
        """Decision should match prediction vs threshold."""
        shap_values = np.array([-0.5, 0.3])
        feature_names = ["debt", "income"]
        feature_values = pd.Series([5000, 30000])

        # Below threshold → denied
        result_denied = extract_reason_codes(
            shap_values=shap_values,
            feature_names=feature_names,
            feature_values=feature_values,
            instance_id=1,
            prediction=0.35,
            threshold=0.5,
            seed=42,
        )
        assert result_denied.decision == "denied"

        # Above threshold → approved
        result_approved = extract_reason_codes(
            shap_values=shap_values,
            feature_names=feature_names,
            feature_values=feature_values,
            instance_id=2,
            prediction=0.65,
            threshold=0.5,
            seed=42,
        )
        assert result_approved.decision == "approved"

    def test_extract_includes_audit_trail(self):
        """Result should include complete audit trail."""
        shap_values = np.array([-0.5, 0.3])
        feature_names = ["debt", "income"]
        feature_values = pd.Series([5000, 30000])

        result = extract_reason_codes(
            shap_values=shap_values,
            feature_names=feature_names,
            feature_values=feature_values,
            instance_id=42,
            prediction=0.35,
            model_hash="abc123",
            seed=123,
        )

        # Should include all audit trail fields
        assert result.instance_id == 42
        assert result.prediction == 0.35
        assert result.model_hash == "abc123"
        assert result.seed == 123
        assert result.timestamp is not None
        assert "T" in result.timestamp  # ISO 8601 format


class TestAdverseActionNotice:
    """Test adverse action notice formatting."""

    def test_format_includes_required_fields(self):
        """AAN template has all ECOA-required fields."""
        # Create sample result
        reason_codes = [
            ReasonCode(feature="debt", contribution=-0.5, feature_value=5000, rank=1),
            ReasonCode(feature="duration", contribution=-0.3, feature_value=24, rank=2),
        ]
        result = ReasonCodeResult(
            instance_id=42,
            prediction=0.35,
            decision="denied",
            reason_codes=reason_codes,
            excluded_features=["age"],
            timestamp="2025-01-01T00:00:00+00:00",
            model_hash="abc123",
            seed=42,
        )

        notice = format_adverse_action_notice(result)

        # ECOA required fields
        assert "ADVERSE ACTION NOTICE" in notice
        assert "Equal Credit Opportunity Act" in notice
        assert "Application ID: 42" in notice
        assert "DECISION: DENIED" in notice
        assert "Predicted Score: 35.0%" in notice

        # Reason codes
        assert "1. Debt:" in notice
        assert "2. Duration:" in notice

        # Rights notice
        assert "IMPORTANT RIGHTS UNDER FEDERAL LAW" in notice
        assert "right to a statement of specific reasons" in notice

        # Audit trail
        assert "Model Hash: abc123" in notice

    def test_format_accepts_custom_organization(self):
        """Allow custom organization name."""
        reason_codes = [
            ReasonCode(feature="debt", contribution=-0.5, feature_value=5000, rank=1),
        ]
        result = ReasonCodeResult(
            instance_id=42,
            prediction=0.35,
            decision="denied",
            reason_codes=reason_codes,
            excluded_features=[],
            timestamp="2025-01-01T00:00:00+00:00",
            model_hash="abc123",
            seed=42,
        )

        notice = format_adverse_action_notice(
            result,
            organization="Example Bank",
            contact_info="1-800-555-0199",
        )

        assert "Example Bank" in notice
        assert "1-800-555-0199" in notice

    def test_format_handles_custom_template(self, tmp_path):
        """Allow custom AAN template."""
        # Create custom template
        template_path = tmp_path / "custom_aan.txt"
        template_path.write_text(
            "Custom Notice\nID: {instance_id}\nDecision: {decision}",
        )

        reason_codes = [
            ReasonCode(feature="debt", contribution=-0.5, feature_value=5000, rank=1),
        ]
        result = ReasonCodeResult(
            instance_id=42,
            prediction=0.35,
            decision="denied",
            reason_codes=reason_codes,
            excluded_features=[],
            timestamp="2025-01-01T00:00:00+00:00",
            model_hash="abc123",
            seed=42,
        )

        notice = format_adverse_action_notice(result, template_path=template_path)

        assert "Custom Notice" in notice
        assert "ID: 42" in notice
        assert "Decision: DENIED" in notice  # Decision is uppercased in default template

    def test_format_raises_on_missing_template(self, tmp_path):
        """Raise clear error if template doesn't exist."""
        template_path = tmp_path / "nonexistent.txt"

        reason_codes = [
            ReasonCode(feature="debt", contribution=-0.5, feature_value=5000, rank=1),
        ]
        result = ReasonCodeResult(
            instance_id=42,
            prediction=0.35,
            decision="denied",
            reason_codes=reason_codes,
            excluded_features=[],
            timestamp="2025-01-01T00:00:00+00:00",
            model_hash="abc123",
            seed=42,
        )

        with pytest.raises(FileNotFoundError, match="Template not found"):
            format_adverse_action_notice(result, template_path=template_path)


class TestProtectedAttributes:
    """Test protected attribute handling."""

    def test_default_protected_list_includes_ecoa_attributes(self):
        """Default list covers ECOA protected classes."""
        # ECOA prohibited bases
        ecoa_attributes = [
            "age",
            "gender",
            "race",
            "religion",
            "national_origin",
            "marital_status",
        ]

        for attr in ecoa_attributes:
            assert attr in DEFAULT_PROTECTED_ATTRIBUTES

    def test_protected_matching_is_case_insensitive(self):
        """Protected attribute matching should be case-insensitive."""
        shap_values = np.array([-0.5, -0.3])
        feature_names = ["AGE", "debt"]  # Uppercase AGE
        feature_values = pd.Series([25, 5000])

        result = extract_reason_codes(
            shap_values=shap_values,
            feature_names=feature_names,
            feature_values=feature_values,
            instance_id=1,
            prediction=0.35,
            seed=42,
        )

        # Should still exclude AGE (case-insensitive match)
        reason_features = [code.feature for code in result.reason_codes]
        assert "AGE" not in reason_features
        assert "AGE" in result.excluded_features


class TestDeterminism:
    """Test deterministic behavior across platforms."""

    def test_byte_identical_output_same_seed(self):
        """Same inputs + seed → byte-identical output."""
        shap_values = np.array([-0.5, -0.5, -0.3, 0.2])  # Ties
        feature_names = ["debt", "loans", "duration", "income"]
        feature_values = pd.Series([5000, 5000, 24, 30000])

        results = []
        for _ in range(5):
            result = extract_reason_codes(
                shap_values=shap_values,
                feature_names=feature_names,
                feature_values=feature_values,
                instance_id=1,
                prediction=0.35,
                top_n=3,
                seed=42,  # Same seed
            )
            results.append(result)

        # All results should be identical (ignoring timestamp)
        for i in range(1, len(results)):
            assert len(results[i].reason_codes) == len(results[0].reason_codes)
            for code_i, code_0 in zip(results[i].reason_codes, results[0].reason_codes, strict=False):
                assert code_i.feature == code_0.feature
                assert code_i.contribution == code_0.contribution
                assert code_i.rank == code_0.rank

    def test_different_seeds_may_differ_with_ties(self):
        """Different seeds may break ties differently."""
        shap_values = np.array([-0.5, -0.5, 0.2])  # Perfect tie
        feature_names = ["debt", "loans", "income"]
        feature_values = pd.Series([5000, 5000, 30000])

        result1 = extract_reason_codes(
            shap_values=shap_values,
            feature_names=feature_names,
            feature_values=feature_values,
            instance_id=1,
            prediction=0.35,
            top_n=1,
            seed=42,
        )

        result2 = extract_reason_codes(
            shap_values=shap_values,
            feature_names=feature_names,
            feature_values=feature_values,
            instance_id=1,
            prediction=0.35,
            top_n=1,
            seed=99,
        )

        # Both should have 1 reason code with same contribution
        assert len(result1.reason_codes) == 1
        assert len(result2.reason_codes) == 1
        assert result1.reason_codes[0].contribution == result2.reason_codes[0].contribution

        # But may select different tied feature (this is acceptable)
        # Both "debt" and "loans" are valid choices
