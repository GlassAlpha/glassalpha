"""Regression tests for metrics normalization contract compliance.

Prevents Jinja template failures when metrics are float values instead of dicts.
"""

import pytest

from glassalpha.report.context import normalize_metrics


class TestMetricsNormalization:
    """Test metrics normalization for template compatibility."""

    def test_float_metrics_normalized_for_templates(self) -> None:
        """Ensure float metrics are wrapped for template access.

        Prevents "TypeError: 'float' object has no attribute 'accuracy'" in Jinja2.
        """
        # Raw metrics with float values (common case)
        raw_metrics = {
            "accuracy": 0.87,
            "precision": 0.82,
            "recall": 0.91,
            "f1": 0.86,
        }

        normalized = normalize_metrics(raw_metrics)

        # Contract: Each float should be wrapped with "value" key
        for metric_name, metric_value in raw_metrics.items():
            assert metric_name in normalized  # noqa: S101
            assert isinstance(normalized[metric_name], dict)  # noqa: S101
            assert "value" in normalized[metric_name]  # noqa: S101
            assert normalized[metric_name]["value"] == float(metric_value)  # noqa: S101

    def test_dict_metrics_preserved(self) -> None:
        """Ensure existing dict metrics are preserved unchanged."""
        raw_metrics = {
            "accuracy": {"value": 0.87, "confidence_interval": [0.85, 0.89]},
            "precision": 0.82,  # Mix of dict and float
            "confusion_matrix": {"tn": 100, "fp": 20, "fn": 15, "tp": 85},
        }

        normalized = normalize_metrics(raw_metrics)

        # Dict values should be preserved
        assert normalized["accuracy"] == raw_metrics["accuracy"]  # noqa: S101
        assert normalized["confusion_matrix"] == raw_metrics["confusion_matrix"]  # noqa: S101

        # Float should be wrapped
        assert normalized["precision"] == {"value": 0.82}  # noqa: S101

    def test_empty_metrics_handled(self) -> None:
        """Ensure empty/None metrics don't cause errors."""
        test_cases = [None, {}, []]

        for empty_case in test_cases:
            result = normalize_metrics(empty_case)
            assert isinstance(result, dict)  # noqa: S101
            assert len(result) == 0  # noqa: S101

    def test_html_renderer_uses_normalized_metrics(self) -> None:
        """Test that HTML renderer properly normalizes metrics.

        This prevents the specific error: "PDF generation failed: 'float' has no attribute 'accuracy'".
        """
        from glassalpha.pipeline.audit import AuditResults  # noqa: PLC0415
        from glassalpha.report.renderer import AuditReportRenderer  # noqa: PLC0415

        # Create mock audit results with float metrics
        mock_results = AuditResults()
        mock_results.model_performance = {
            "accuracy": 0.87,
            "f1": 0.82,
        }
        mock_results.success = True
        mock_results.execution_info = {"audit_profile": "test"}

        renderer = AuditReportRenderer()

        # This should not raise "float has no attribute" errors
        try:
            context = renderer._prepare_template_context(mock_results, embed_plots=False)  # noqa: SLF001

            # Verify metrics are properly normalized
            model_performance = context["model_performance"]
            assert isinstance(model_performance["accuracy"], dict)  # noqa: S101
            assert "value" in model_performance["accuracy"]  # noqa: S101
            assert model_performance["accuracy"]["value"] == 0.87  # noqa: PLR2004, S101

        except (TypeError, AttributeError) as e:
            pytest.fail(f"Metrics normalization failed: {e}")

    def test_template_rendering_with_float_metrics(self) -> None:
        """Test complete template rendering with float metrics.

        Ensures the full rendering pipeline handles normalized metrics correctly.
        """
        from glassalpha.pipeline.audit import AuditResults  # noqa: PLC0415
        from glassalpha.report.renderer import AuditReportRenderer  # noqa: PLC0415

        # Mock audit results
        mock_results = AuditResults()
        mock_results.model_performance = {"accuracy": 0.87}
        mock_results.success = True
        mock_results.execution_info = {}

        renderer = AuditReportRenderer()

        try:
            # This should render without template errors
            html_content = renderer.render_audit_report(
                audit_results=mock_results,
                embed_plots=False,
            )

            # Verify the metric value appears in rendered HTML
            assert "0.87" in html_content or "87" in html_content  # noqa: S101
            assert "<html" in html_content.lower()  # noqa: S101

        except Exception as e:  # noqa: BLE001
            pytest.fail(f"Template rendering failed with normalized metrics: {e}")

    @pytest.mark.parametrize(
        "metric_value",
        [
            0.0,
            1.0,
            0.5,
            42,
            -0.1,
            3.14159,
        ],
    )
    def test_various_numeric_types_normalized(self, metric_value: float) -> None:
        """Test normalization handles various numeric values correctly."""
        metrics = {"test_metric": metric_value}
        normalized = normalize_metrics(metrics)

        assert "test_metric" in normalized  # noqa: S101
        assert normalized["test_metric"]["value"] == float(metric_value)  # noqa: S101
