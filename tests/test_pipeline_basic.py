"""Basic pipeline functionality tests.

Tests core AuditPipeline functionality including initialization,
component loading, and basic execution flow. These tests focus
on covering the main pipeline logic without requiring real data/models.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from glassalpha.config.schema import (
    AuditConfig,
    DataConfig,
    ExplainerConfig,
    MetricsConfig,
    ModelConfig,
    ReproducibilityConfig,
)
from glassalpha.pipeline.audit import AuditPipeline, AuditResults, run_audit_pipeline


@pytest.fixture
def minimal_config():
    """Create minimal valid audit configuration for testing."""
    return AuditConfig(
        audit_profile="tabular_compliance",
        model=ModelConfig(type="xgboost"),
        data=DataConfig(dataset="custom", path="test_data.csv"),
        explainers=ExplainerConfig(strategy="first_compatible", priority=["treeshap", "kernelshap"]),
        metrics=MetricsConfig(performance=["accuracy"]),
        reproducibility=ReproducibilityConfig(random_seed=42),
    )


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [0.1, 0.2, 0.3, 0.4, 0.5], "target": [0, 1, 0, 1, 0]})


class TestAuditResults:
    """Test AuditResults dataclass functionality."""

    def test_audit_results_initialization(self):
        """Test AuditResults creates with proper defaults."""
        results = AuditResults()

        assert results.success is False
        assert results.error_message is None
        assert isinstance(results.model_performance, dict)
        assert isinstance(results.fairness_analysis, dict)
        assert isinstance(results.drift_analysis, dict)
        assert isinstance(results.explanations, dict)
        assert isinstance(results.data_summary, dict)
        assert isinstance(results.schema_info, dict)
        assert isinstance(results.model_info, dict)
        assert isinstance(results.selected_components, dict)
        assert isinstance(results.execution_info, dict)
        assert isinstance(results.manifest, dict)

    def test_audit_results_success_state(self):
        """Test setting AuditResults to success state."""
        results = AuditResults()
        results.success = True
        results.model_performance = {"accuracy": 0.85}

        assert results.success is True
        assert results.model_performance["accuracy"] == 0.85
        assert results.error_message is None

    def test_audit_results_error_state(self):
        """Test setting AuditResults to error state."""
        results = AuditResults()
        results.success = False
        results.error_message = "Model loading failed"

        assert results.success is False
        assert results.error_message == "Model loading failed"


class TestAuditPipelineInitialization:
    """Test AuditPipeline initialization and setup."""

    def test_pipeline_initialization(self, minimal_config):
        """Test basic pipeline initialization."""
        pipeline = AuditPipeline(minimal_config)

        assert pipeline.config == minimal_config
        assert isinstance(pipeline.results, AuditResults)
        assert pipeline.data_loader is not None
        assert pipeline.model is None
        assert pipeline.explainer is None
        assert isinstance(pipeline.selected_metrics, dict)
        assert pipeline.manifest_generator is not None

    def test_pipeline_config_storage(self, minimal_config):
        """Test that pipeline stores config correctly."""
        pipeline = AuditPipeline(minimal_config)

        assert pipeline.config.audit_profile == "tabular_compliance"
        assert pipeline.config.model.type == "xgboost"
        assert pipeline.config.reproducibility.random_seed == 42


class TestAuditPipelineReproducibility:
    """Test reproducibility setup functionality."""

    @patch("glassalpha.pipeline.audit.set_global_seed")
    def test_setup_reproducibility(self, mock_set_seed, minimal_config):
        """Test reproducibility setup calls seed functions."""
        pipeline = AuditPipeline(minimal_config)

        # Call the private method directly for testing
        pipeline._setup_reproducibility()

        mock_set_seed.assert_called_once_with(42)

    @patch("glassalpha.pipeline.audit.set_global_seed")
    def test_setup_reproducibility_with_different_seed(self, mock_set_seed, minimal_config):
        """Test reproducibility setup with different seed."""
        minimal_config.reproducibility.random_seed = 123
        pipeline = AuditPipeline(minimal_config)

        pipeline._setup_reproducibility()

        mock_set_seed.assert_called_once_with(123)


class TestAuditPipelineDataLoading:
    """Test data loading functionality."""

    def test_load_data_file_path_validation(self, minimal_config):
        """Test data loading validates file path."""
        pipeline = AuditPipeline(minimal_config)

        # File doesn't exist, should get error
        with pytest.raises(FileNotFoundError):
            pipeline._load_data()

    def test_load_data_checks_path_exists(self, minimal_config):
        """Test data loading checks path existence."""
        pipeline = AuditPipeline(minimal_config)

        # Should check if file exists and fail appropriately
        with pytest.raises(FileNotFoundError, match="Custom data file not found"):
            pipeline._load_data()


class TestAuditPipelineComponentSelection:
    """Test component selection logic."""

    def test_select_explainer_basic(self, minimal_config):
        """Test explainer selection error handling when explainer unavailable."""
        pipeline = AuditPipeline(minimal_config)
        pipeline.model = Mock()

        # Request an explainer that doesn't exist in the registry
        # This tests the error path when user specifies unavailable explainer
        pipeline.config.explainers = ExplainerConfig(
            strategy="first_compatible",
            priority=["nonexistent_explainer_xyz_123"],  # Guaranteed to not exist
        )

        # Should raise RuntimeError when explainer is not available
        # Note: Error message may be either from direct selection or legacy fallback
        with pytest.raises(RuntimeError, match="No (explainer from.*is available|compatible explainer found)"):
            pipeline._select_explainer()


class TestAuditPipelineExecution:
    """Test main pipeline execution logic."""

    def test_progress_callback(self, minimal_config):
        """Test progress callback functionality."""
        pipeline = AuditPipeline(minimal_config)

        callback_mock = Mock()
        pipeline._update_progress(callback_mock, "Test message", 50)

        # Check that callback was called (signature may vary)
        assert callback_mock.called

    def test_progress_callback_none(self, minimal_config):
        """Test progress callback when None is provided."""
        pipeline = AuditPipeline(minimal_config)

        # Should not raise exception
        pipeline._update_progress(None, "Test message", 50)

    @patch("glassalpha.pipeline.audit.AuditPipeline._setup_reproducibility")
    @patch("glassalpha.pipeline.audit.AuditPipeline._load_data")
    @patch("glassalpha.pipeline.audit.AuditPipeline._load_model")
    def test_run_pipeline_basic_flow(
        self,
        mock_load_model,
        mock_load_data,
        mock_setup,
        minimal_config,
        sample_dataframe,
    ):
        """Test basic pipeline execution flow."""
        # Setup mocks
        mock_schema = Mock()
        mock_schema.features = ["feature1", "feature2"]
        mock_schema.target = "target"
        mock_load_data.return_value = (sample_dataframe, mock_schema)
        mock_load_model.return_value = Mock()

        pipeline = AuditPipeline(minimal_config)

        # Mock other dependencies to avoid full execution
        with (
            patch.object(pipeline, "_select_explainer", return_value=Mock()),
            patch.object(pipeline, "_generate_explanations", return_value={}),
            patch.object(pipeline, "_compute_metrics"),
            patch.object(pipeline, "_finalize_results"),
        ):
            results = pipeline.run()

        assert isinstance(results, AuditResults)
        mock_setup.assert_called_once()
        mock_load_data.assert_called_once()
        mock_load_model.assert_called_once()

    @patch("glassalpha.pipeline.audit.AuditPipeline._load_data")
    def test_run_pipeline_data_loading_error(self, mock_load_data, minimal_config):
        """Test pipeline handles data loading errors gracefully."""
        mock_load_data.side_effect = Exception("Data loading failed")

        pipeline = AuditPipeline(minimal_config)
        results = pipeline.run()

        assert results.success is False
        assert "Data loading failed" in results.error_message

    def test_run_pipeline_with_progress_callback(self, minimal_config):
        """Test pipeline execution with progress callback."""
        callback_mock = Mock()

        pipeline = AuditPipeline(minimal_config)

        # Mock all major methods to avoid dependencies
        mock_schema = Mock()
        mock_schema.features = ["feature1", "feature2"]
        mock_schema.target = "target"

        # Create a mock DataFrame that behaves like pandas DataFrame
        mock_data = Mock()
        mock_data.columns = ["feature1", "feature2", "target"]
        # Configure shape to return an iterable tuple
        mock_data.configure_mock(shape=(100, 3))

        # Mock the specific operations used in the pipeline
        mock_target_column = Mock()
        mock_target_column.nunique.return_value = 2  # 2 classes
        mock_data.__getitem__ = Mock(return_value=mock_target_column)
        mock_data.__contains__ = Mock(return_value=True)  # contains target column

        with (
            patch.object(pipeline, "_setup_reproducibility"),
            patch.object(pipeline, "_load_data", return_value=(mock_data, mock_schema)),
            patch.object(pipeline, "_load_model", return_value=Mock()),
            patch.object(pipeline, "_select_explainer", return_value=Mock()),
            patch.object(pipeline, "_generate_explanations", return_value={}),
            patch.object(pipeline, "_compute_metrics"),
            patch("glassalpha.provenance.run_manifest._get_dataset_provenance", return_value={"shape": [100, 3]}),
            # Don't mock _finalize_results so the final progress callback can be called
        ):
            results = pipeline.run(progress_callback=callback_mock)

        assert isinstance(results, AuditResults)
        # Should have been called multiple times with progress updates
        # The pipeline calls progress callback at 10%, 20%, 35%, 45%, 60%, 80%, 100%
        assert callback_mock.call_count >= 7


class TestAuditPipelineResults:
    """Test results finalization and stats computation."""

    def test_compute_explanation_stats_empty(self, minimal_config):
        """Test explanation stats computation with empty explanations."""
        pipeline = AuditPipeline(minimal_config)

        stats = pipeline._compute_explanation_stats({})

        # Should return a dict with some stats
        assert isinstance(stats, dict)

    def test_compute_explanation_stats_with_data(self, minimal_config):
        """Test explanation stats computation with explanations."""
        pipeline = AuditPipeline(minimal_config)

        explanations = {
            "global": {"shap_values": np.array([[1, 2, 3], [4, 5, 6]])},
            "local": {"shap_values": np.array([[7, 8, 9]])},
        }

        stats = pipeline._compute_explanation_stats(explanations)

        # Should return a dict with stats about explanations
        assert isinstance(stats, dict)

    def test_finalize_results(self, minimal_config):
        """Test results finalization."""
        pipeline = AuditPipeline(minimal_config)
        explanations = {"test": "explanation"}

        pipeline._finalize_results(explanations)

        # Should update results object with execution info
        assert pipeline.results.error_message is None
        assert isinstance(pipeline.results.execution_info, dict)


class TestAuditPipelineUtils:
    """Test utility methods."""

    def test_preprocess_for_training(self, minimal_config, sample_dataframe):
        """Test data preprocessing for training."""
        pipeline = AuditPipeline(minimal_config)

        processed = pipeline._preprocess_for_training(sample_dataframe)

        assert isinstance(processed, pd.DataFrame)
        # Should return a processed version (exact behavior depends on implementation)
        assert len(processed) == len(sample_dataframe)

    def test_get_seeded_context(self, minimal_config):
        """Test seeded context generation."""
        pipeline = AuditPipeline(minimal_config)

        with pipeline._get_seeded_context("test_component"):
            # Context manager should work without error
            assert True

    def test_ensure_components_loaded(self, minimal_config):
        """Test component loading."""
        pipeline = AuditPipeline(minimal_config)

        # Should not raise exception
        pipeline._ensure_components_loaded()


class TestRunAuditPipelineFunction:
    """Test standalone run_audit_pipeline function."""

    @patch("glassalpha.pipeline.audit.AuditPipeline")
    def test_run_audit_pipeline_function(self, mock_pipeline_class, minimal_config):
        """Test run_audit_pipeline function creates pipeline and runs it."""
        mock_pipeline = Mock()
        mock_results = AuditResults()
        mock_results.success = True
        mock_pipeline.run.return_value = mock_results
        mock_pipeline_class.return_value = mock_pipeline

        callback_mock = Mock()
        results = run_audit_pipeline(minimal_config, callback_mock)

        mock_pipeline_class.assert_called_once_with(minimal_config)
        mock_pipeline.run.assert_called_once_with(callback_mock)
        assert results == mock_results

    def test_run_audit_pipeline_no_callback(self, minimal_config):
        """Test run_audit_pipeline function without callback."""
        with patch("glassalpha.pipeline.audit.AuditPipeline") as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_results = AuditResults()
            mock_pipeline.run.return_value = mock_results
            mock_pipeline_class.return_value = mock_pipeline

            results = run_audit_pipeline(minimal_config)

            mock_pipeline.run.assert_called_once_with(None)
            assert results == mock_results


class TestAuditPipelineEdgeCases:
    """Test edge cases and error conditions."""

    def test_pipeline_with_invalid_model_type(self, minimal_config):
        """Test pipeline handles invalid model type."""
        minimal_config.model.type = "invalid_model"

        pipeline = AuditPipeline(minimal_config)

        # Pipeline should initialize but may fail during execution
        assert pipeline.config.model.type == "invalid_model"

    def test_pipeline_with_empty_metrics_list(self, minimal_config):
        """Test pipeline with empty metrics configuration."""
        minimal_config.metrics.performance = []

        pipeline = AuditPipeline(minimal_config)

        assert pipeline.config.metrics.performance == []

    @patch("glassalpha.pipeline.audit.logger")
    def test_pipeline_logs_initialization(self, mock_logger, minimal_config):
        """Test that pipeline logs initialization."""
        AuditPipeline(minimal_config)

        mock_logger.info.assert_called_with("Initialized audit pipeline with profile: tabular_compliance")
