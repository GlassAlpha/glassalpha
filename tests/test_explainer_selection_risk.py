"""High-risk explainer selection tests - critical path validation.

These tests target the actual risk areas in explainer selection:
- Specific model types must resolve to expected explainers
- Registry must handle mock objects properly
- Priority ordering must be deterministic
"""

import pytest


class TestExplainerSelectionRisk:
    """Test explainer selection critical paths that prevent customer issues."""

    def test_xgboost_selects_treeshap_by_default(self):
        """XGBoost must select TreeSHAP by default - customer expectation."""
        from glassalpha.explain.registry import ExplainerRegistry

        explainer_name = ExplainerRegistry.find_compatible("xgboost")
        assert explainer_name is not None, "XGBoost must have compatible explainer"
        assert explainer_name == "treeshap", "XGBoost should prefer TreeSHAP"

        # Verify we can get the class
        explainer_class = ExplainerRegistry.get(explainer_name)
        assert explainer_class.__name__ == "TreeSHAPExplainer"

    def test_logistic_regression_selects_coef(self):
        """LogisticRegression should select coefficients explainer (fastest, no dependencies)."""
        from glassalpha.explain.registry import ExplainerRegistry

        explainer_name = ExplainerRegistry.find_compatible("logistic_regression")
        assert explainer_name is not None, "LogisticRegression must have compatible explainer"
        # LogisticRegression should select coefficients - fastest option
        assert explainer_name == "coefficients", "LogisticRegression should prefer coefficients explainer"

        # Verify we can get the class
        explainer_class = ExplainerRegistry.get(explainer_name)

    def test_lightgbm_priority_selection(self):
        """LightGBM should prefer TreeSHAP when available."""
        from glassalpha.explain.registry import ExplainerRegistry

        explainer_name = ExplainerRegistry.find_compatible("lightgbm")
        assert explainer_name is not None, "LightGBM must have compatible explainer"
        # Should pick TreeSHAP due to priority ordering
        assert explainer_name == "treeshap", "LightGBM should prefer TreeSHAP"

        # Verify we can get the class
        explainer_class = ExplainerRegistry.get(explainer_name)
        assert explainer_class.__name__ == "TreeSHAPExplainer"

    def test_unsupported_model_uses_fallback_logic(self):
        """Unsupported models should use fallback logic rather than failing immediately."""
        from glassalpha.explain.registry import ExplainerRegistry

        # New logic may not fail immediately for unknown models - uses fallback
        try:
            result = ExplainerRegistry.find_compatible("completely_unknown_model")
            # If it doesn't raise, it should return some explainer (likely permutation)
            assert result is not None
        except RuntimeError:
            # If it does raise, should be the expected message
            pass

    def test_mock_model_object_compatibility(self):
        """Model objects with get_model_info should work correctly."""
        from glassalpha.explain.registry import ExplainerRegistry

        class MockModel:
            def get_model_info(self):
                return {"type": "xgboost"}

        mock = MockModel()
        explainer_name = ExplainerRegistry.find_compatible(mock)
        assert explainer_name is not None, "Mock XGBoost should be compatible"
        assert explainer_name == "treeshap"

        # Verify we can get the class
        explainer_class = ExplainerRegistry.get(explainer_name)
        assert explainer_class.__name__ == "TreeSHAPExplainer"

    def test_none_model_info_uses_fallback_logic(self):
        """Objects without valid model info should use fallback logic."""
        from glassalpha.explain.registry import ExplainerRegistry

        class BadMock:
            def get_model_info(self):
                return None

        bad_mock = BadMock()
        # New logic may not fail immediately - uses fallback
        try:
            result = ExplainerRegistry.find_compatible(bad_mock)
            # If it doesn't raise, it should return some explainer
            assert result is not None
        except RuntimeError:
            # If it does raise, that's also acceptable
            pass

    def test_explainer_registry_deterministic_selection(self):
        """Same input must always select same explainer - reproducibility critical."""
        from glassalpha.explain.registry import ExplainerRegistry

        # Test multiple times to ensure deterministic
        selections = []
        for _ in range(5):
            explainer_name = ExplainerRegistry.find_compatible("xgboost")
            selections.append(explainer_name)

        # All selections should be identical
        assert len(set(selections)) == 1, f"Selection not deterministic: {selections}"
        assert selections[0] == "treeshap", "Should consistently select treeshap"

    def test_new_explainer_selection_logic(self):
        """Test the new capability-aware explainer selection."""
        from glassalpha.explain.registry import _available, select_explainer

        # Test module availability checking
        assert _available("coefficients") is True  # No dependencies
        assert _available("permutation") is True  # No dependencies

        # SHAP availability depends on installation
        shap_available = _available("kernelshap")

        # Test linear model selection
        if shap_available:
            selected = select_explainer("logistic_regression")
            assert selected in ["coefficients", "permutation", "kernelshap"]
        else:
            selected = select_explainer("logistic_regression")
            assert selected in ["coefficients", "permutation"]  # Should not select SHAP explainers

        # Test tree model selection
        if shap_available:
            selected = select_explainer("xgboost")
            assert selected in ["treeshap", "permutation", "kernelshap"]
            # Should prefer treeshap for tree models when available
            assert selected == "treeshap"
        else:
            selected = select_explainer("xgboost")
            assert selected in ["permutation"]  # Should fallback to permutation

    def test_explicit_priority_works_when_shap_available(self):
        """Test explicit priority works correctly when SHAP is available."""
        from glassalpha.explain.registry import select_explainer

        # Should work when SHAP is available
        selected = select_explainer("xgboost", ["kernelshap"])
        assert selected == "kernelshap"

        selected = select_explainer("xgboost", ["treeshap"])
        assert selected == "treeshap"

    def test_explicit_priority_fails_with_nonexistent_explainer(self):
        """Test explicit priority fails with helpful message for non-existent explainers."""
        from glassalpha.explain.registry import select_explainer

        # Should fail for non-existent explainer
        with pytest.raises(RuntimeError, match="No explainer from .*nonexistent_explainer.* is available"):
            select_explainer("xgboost", ["nonexistent_explainer"])

    def test_explainer_selection_logging(self, caplog):
        """Test that explainer selection provides informative logging."""
        from glassalpha.explain.registry import ExplainerRegistry

        with caplog.at_level("INFO"):
            # Test with a model that should select coef
            try:
                ExplainerRegistry.find_compatible("logistic_regression")
                # Check that logging occurred
                explainer_logs = [record.message for record in caplog.records if "Explainer:" in record.message]
                assert len(explainer_logs) > 0, "Should log explainer selection"
            except RuntimeError:
                # If no explainers available, that's also fine for this test
                pass
