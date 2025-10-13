"""Main audit pipeline orchestrator.

This module provides the central AuditPipeline class that coordinates
all components from data loading through model analysis to generate
comprehensive audit results with full reproducibility tracking.
"""

import logging
import os
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from glassalpha.config import AuditConfig

# Constants import removed - using f-string directly for logger format
from glassalpha.data import TabularDataLoader, TabularDataSchema
from glassalpha.utils import ManifestGenerator, get_component_seed, set_global_seed
from glassalpha.utils.preprocessing import preprocess_auto

# Built-in dataset loaders (lazy imported when needed)
BUILT_IN_DATASETS = {
    "german_credit": ("glassalpha.datasets", "load_german_credit"),
    "adult_income": ("glassalpha.datasets", "load_adult_income"),
}

logger = logging.getLogger(__name__)

# Suppress sklearn feature name warnings during audit (they're noisy but not errors)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*", category=UserWarning)

# Suppress sklearn numerical underflow warnings during optimization (harmless)
# These occur during logistic regression gradient computation and don't affect results
warnings.filterwarnings("ignore", message=".*underflow encountered in divide.*", category=RuntimeWarning)


@dataclass
class AuditResults:
    """Container for comprehensive audit results."""

    # Core results
    model_performance: dict[str, Any] = field(default_factory=dict)
    fairness_analysis: dict[str, Any] = field(default_factory=dict)
    drift_analysis: dict[str, Any] = field(default_factory=dict)
    stability_analysis: dict[str, Any] = field(default_factory=dict)
    explanations: dict[str, Any] = field(default_factory=dict)

    # Data information
    data_summary: dict[str, Any] = field(default_factory=dict)
    schema_info: dict[str, Any] = field(default_factory=dict)

    # Model information
    model_info: dict[str, Any] = field(default_factory=dict)
    selected_components: dict[str, Any] = field(default_factory=dict)
    trained_model: Any = None  # The actual trained model object for export

    # Audit metadata
    execution_info: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    # Success indicators
    success: bool = False
    error_message: str | None = None

    def _repr_html_(self) -> str:
        """Jupyter/IPython HTML representation for inline display.

        Returns:
            HTML string with audit summary (performance, fairness, features, lineage)

        Note:
            Returns visible error card if rendering fails (no silent failures).
            Reuses existing Jinja2 template engine (no new dependencies).

        """
        try:
            # Lazy import to avoid circular dependency
            from glassalpha.report.renderer import AuditReportRenderer

            renderer = AuditReportRenderer()
            return renderer.render_audit_report(
                audit_results=self,
                template_name="inline_summary.html",
                embed_plots=False,  # No plots in inline view (keep it fast)
            )
        except Exception as e:
            # Return visible error card (not silent fallback)
            error_type = type(e).__name__
            error_msg = str(e)
            trace = traceback.format_exc()

            # HTML error card with GitHub-style red/pink theme
            return f"""
            <div style="border: 2px solid #e74c3c; padding: 16px; margin: 10px 0; background-color: #fadbd8; border-radius: 6px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                <h3 style="color: #c0392b; margin: 0 0 8px 0; font-size: 16px;">WARNING: Inline Display Failed</h3>
                <p style="margin: 4px 0; font-size: 14px;">
                    <strong>Error:</strong> {error_type}: {error_msg}
                </p>
                <p style="margin: 8px 0 4px 0; font-size: 13px; color: #6a737d;">
                    Use <code style="background: #f6f8fa; padding: 2px 6px; border-radius: 3px; font-family: 'SF Mono', Monaco, monospace;">result.to_pdf('output.pdf')</code> to generate full report.
                </p>
                <details style="margin-top: 12px;">
                    <summary style="cursor: pointer; font-size: 13px; color: #0366d6;">Show full error trace</summary>
                    <pre style="font-size: 11px; margin-top: 8px; padding: 8px; background: #f6f8fa; border-radius: 3px; overflow-x: auto;">{trace}</pre>
                </details>
            </div>
            """


class AuditPipeline:
    """Main pipeline for conducting comprehensive ML model audits."""

    def __init__(
        self,
        config: AuditConfig,
        selected_explainer: str | None = None,
        requested_model: str | None = None,
    ) -> None:
        """Initialize audit pipeline with configuration.

        Args:
            config: Validated audit configuration
            selected_explainer: Pre-selected explainer name to use (avoids re-selection)
            requested_model: Originally requested model type (for fallback tracking)

        """
        self.config = config
        self.selected_explainer = selected_explainer
        self.requested_model = requested_model
        self.results = AuditResults()

        # Ensure all components are imported and registered
        self._ensure_components_loaded()

        # Get seed from config for deterministic manifest generation
        seed = None
        if hasattr(config, "reproducibility") and hasattr(config.reproducibility, "random_seed"):
            seed = config.reproducibility.random_seed
        elif isinstance(config, dict):
            seed = config.get("reproducibility", {}).get("random_seed")

        # Initialize manifest generator with seed for deterministic audit IDs
        self.manifest_generator = ManifestGenerator(seed=seed)

        # Handle both pydantic and plain object configs (contract compliance)
        if hasattr(config, "model_dump"):
            cfg_dict = config.model_dump()
        elif isinstance(config, dict):
            cfg_dict = config
        else:
            cfg_dict = dict(vars(config))
        self.manifest_generator.add_config(cfg_dict)

        # Component instances (will be populated during execution)
        self.data_loader = TabularDataLoader()
        self.model = None
        self._preprocessing_artifact_path = None  # Track preprocessing artifact path for model metadata
        self.explainer = None
        self.selected_metrics = {}

        # Runtime context for from_model() API (optional)
        self._runtime_model = None
        self._runtime_X_test = None
        self._runtime_y_test = None
        self._runtime_data = None

        # Contract compliance: Exact f-string for wheel contract test
        logger.info(f"Initialized audit pipeline with profile: {cfg_dict.get('audit_profile', 'default')}")

    @classmethod
    def from_model(
        cls,
        model: Any,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        protected_attributes: list[str],
        *,
        random_seed: int = 42,
        audit_profile: str = "tabular_compliance",
        explainer: str | None = None,
        fairness_threshold: float | None = None,
        recourse_config: dict | None = None,
        feature_names: list[str] | None = None,
        target_name: str | None = None,
        **config_overrides: Any,
    ) -> AuditResults:
        """Create and run audit from in-memory model and data.

        Enables 3-line audits without YAML config files:

        >>> result = AuditPipeline.from_model(
        ...     model=xgb_model,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ...     protected_attributes=["gender", "age"]
        ... )

        Args:
            model: Fitted model (sklearn, xgboost, lightgbm, etc.)
            X_test: Test features (DataFrame or array)
            y_test: Test labels (Series or array)
            protected_attributes: Protected attribute column names
            random_seed: Random seed for reproducibility (default: 42)
            audit_profile: Audit profile name (default: "tabular_compliance")
            explainer: Explainer to use (default: auto-select based on model)
            fairness_threshold: Classification threshold (default: 0.5)
            recourse_config: Recourse configuration dict (default: None)
            feature_names: Feature names if X_test is array (default: auto-detect)
            target_name: Target column name (default: "target")
            **config_overrides: Additional config overrides (advanced)

        Returns:
            AuditResults object with inline HTML display (auto-renders in Jupyter)

        Raises:
            ValueError: If model type cannot be detected
            ValueError: If protected attributes not in X_test columns
            TypeError: If X_test/y_test have incompatible types

        Examples:
            Basic usage with DataFrame:

            >>> from sklearn.linear_model import LogisticRegression
            >>> model = LogisticRegression()
            >>> model.fit(X_train, y_train)
            >>> result = AuditPipeline.from_model(
            ...     model=model,
            ...     X_test=X_test,
            ...     y_test=y_test,
            ...     protected_attributes=["gender", "age"]
            ... )

            With numpy arrays (requires feature_names):

            >>> result = AuditPipeline.from_model(
            ...     model=model,
            ...     X_test=X_array,
            ...     y_test=y_array,
            ...     protected_attributes=["feature_0"],
            ...     feature_names=["feature_0", "feature_1", "feature_2"]
            ... )

            With custom threshold and recourse:

            >>> result = AuditPipeline.from_model(
            ...     model=model,
            ...     X_test=X_test,
            ...     y_test=y_test,
            ...     protected_attributes=["gender"],
            ...     fairness_threshold=0.6,
            ...     recourse_config={"enabled": True, "immutable_features": ["age"]}
            ... )

        """
        from ..config.builder import build_config_from_model

        logger.info("Building audit configuration from in-memory model and data")

        # Build config from inputs
        audit_config, runtime_context = build_config_from_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            protected_attributes=protected_attributes,
            random_seed=random_seed,
            audit_profile=audit_profile,
            explainer=explainer,
            fairness_threshold=fairness_threshold,
            recourse_config=recourse_config,
            feature_names=feature_names,
            target_name=target_name,
            **config_overrides,
        )

        # Create pipeline instance
        pipeline = cls(audit_config)

        # Store in-memory model and data for runtime use
        pipeline._runtime_model = runtime_context["model"]
        pipeline._runtime_X_test = runtime_context["X_test"]
        pipeline._runtime_y_test = runtime_context["y_test"]

        # Build DataFrame for pipeline (if needed)
        if isinstance(X_test, pd.DataFrame):
            pipeline._runtime_data = X_test.copy()
        else:
            # Convert array to DataFrame
            pipeline._runtime_data = pd.DataFrame(
                X_test,
                columns=runtime_context["feature_names"],
            )

        # Add target column
        pipeline._runtime_data[runtime_context["target_name"]] = y_test

        logger.info("Running audit pipeline with in-memory model and data")

        # Create progress bar (auto-detects notebook, respects strict mode)
        from glassalpha.utils.progress import get_progress_bar

        show_progress = not getattr(getattr(audit_config, "runtime", None), "strict_mode", False)

        with get_progress_bar(total=100, desc="Audit", disable=not show_progress, leave=False) as pbar:

            def update_progress(message: str, percent: int):
                """Update progress bar with current step."""
                pbar.set_description(f"Audit: {message}")
                pbar.n = percent
                pbar.refresh()

            # Run pipeline with progress callback
            results = pipeline.run(progress_callback=update_progress)

        return results

    def run(self, progress_callback: Callable | None = None) -> AuditResults:
        """Execute the complete audit pipeline.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Comprehensive audit results

        """
        from datetime import datetime

        # Store progress callback for use in nested methods
        self._progress_callback = progress_callback

        # Friend's spec: Before doing work - capture start time
        start = datetime.now(UTC).isoformat()

        try:
            logger.info("Starting audit pipeline execution")

            # Step 1: Setup reproducibility (~5% of time)
            self._setup_reproducibility()
            self._update_progress(progress_callback, "Setting seeds for reproducibility", 5)

            # Step 2: Load and validate data (~10% of time)
            data, schema = self._load_data()
            # Store dataset for provenance tracking
            self._dataset_df = data.copy()  # Store copy for provenance
            self._feature_count = len(schema.features) if schema.features else data.shape[1] - 1
            self._class_count = data[schema.target].nunique() if schema.target in data.columns else None
            self._update_progress(progress_callback, "Loading and validating dataset", 15)

            # Step 3: Load/initialize model (~10% of time)
            self.model = self._load_model(data, schema)
            self._update_progress(progress_callback, "Training/loading model", 25)

            # Step 4: Select and initialize explainer (~1% of time)
            self.explainer = self._select_explainer()
            self._update_progress(progress_callback, "Selecting explainer (SHAP, coefficients, etc.)", 26)

            # Step 5: Generate explanations (~20% of time for coefficients, ~50% for permutation)
            explanations = self._generate_explanations(data, schema)
            self._update_progress(progress_callback, "Generating feature explanations", 45)

            # Step 6: Compute metrics (~50% of time - fairness CIs are expensive)
            self._compute_metrics(data, schema)
            self._update_progress(progress_callback, "All metrics computed", 95)

            # Step 7: Finalize results and manifest (~5% of time)
            self._finalize_results(explanations)
            self._update_progress(progress_callback, "Finalizing audit results", 100)

            # Friend's spec: On success - set end time and call set execution info
            end = datetime.now(UTC).isoformat()
            logger.debug(f"Pipeline execution: start={start}, end={end}")

            # Update manifest generator with execution info
            self.manifest_generator.mark_completed("completed", None)

            # Store the trained model for export
            self.results.trained_model = self.model

            self.results.success = True
            logger.info("Audit pipeline completed successfully")

        except Exception as e:
            # Friend's spec: On exception - set error_message and still write start/end times
            end = datetime.now(UTC).isoformat()
            logger.debug(f"Pipeline execution failed: start={start}, end={end}")
            error_msg = f"Audit pipeline failed: {e!s}"

            logger.exception(error_msg)
            logger.debug(f"Full traceback: {traceback.format_exc()}")

            self.results.success = False
            self.results.error_message = error_msg

            # Mark manifest as failed with timing info
            self.manifest_generator.mark_completed("failed", error_msg)

        return self.results

    def _setup_reproducibility(self) -> None:
        """Set up reproducible execution environment."""
        logger.info("Setting up reproducible execution environment")

        # Set global seed from config
        master_seed = self.config.reproducibility.random_seed if self.config.reproducibility else 42
        set_global_seed(master_seed)

        # Store seed in execution_info for report generation
        if not hasattr(self.results, "execution_info") or self.results.execution_info is None:
            self.results.execution_info = {}
        self.results.execution_info["random_seed"] = master_seed

        # Apply advanced reproduction controls if configured
        if (
            self.config.reproducibility
            and hasattr(self.config.reproducibility, "strict")
            and self.config.reproducibility.strict
        ):
            from ..runtime import set_repro

            logger.info("Applying advanced deterministic reproduction controls")
            repro_status = set_repro(
                seed=master_seed,
                strict=self.config.reproducibility.strict,
                thread_control=getattr(self.config.reproducibility, "thread_control", False),
                warn_on_failure=getattr(self.config.reproducibility, "warn_on_failure", True),
            )

            # Store repro status in results for provenance
            if not hasattr(self.results, "execution_info") or self.results.execution_info is None:
                self.results.execution_info = {}
            self.results.execution_info["reproduction_status"] = repro_status

            successful = sum(1 for control in repro_status["controls"].values() if control.get("success", False))
            total = len(repro_status["controls"])
            logger.info(f"Advanced reproduction controls: {successful}/{total} successful")

        # Add seed information to manifest
        self.manifest_generator.add_seeds()

        logger.debug(f"Global seed set to {master_seed}")

    def _load_data(self) -> tuple[pd.DataFrame, TabularDataSchema]:
        """Load and validate dataset.

        Returns:
            Tuple of (data, schema)

        """
        logger.info("Loading and validating dataset")

        # Check for in-memory data (from from_model())
        if self._runtime_data is not None:
            logger.info("Using in-memory data from from_model() API")
            data = self._runtime_data

            # Create schema from config
            target_col = self.config.data.target_column or "target"
            feature_cols = self.config.data.feature_columns or []

            if not feature_cols:
                # Use all columns except target
                feature_cols = [col for col in data.columns if col != target_col]

            schema = TabularDataSchema(
                target=target_col,
                features=feature_cols,
                sensitive_features=self.config.data.protected_attributes,
            )

            # Validate schema
            self.data_loader.validate_schema(data, schema)

            # Store data information in results
            self.results.data_summary = self.data_loader.get_data_summary(data)
            self.results.schema_info = schema.model_dump()

            # Add dataset to manifest (mark as in-memory)
            self.manifest_generator.add_dataset(
                "primary_dataset",
                data=data,
                file_path=":memory:",
                target_column=schema.target,
                sensitive_features=schema.sensitive_features,
            )

            logger.info(f"Loaded in-memory data: {data.shape[0]} rows, {data.shape[1]} columns")

            return data, schema

        # Resolve dataset path with offline and fetch policy enforcement
        data_path = self._resolve_dataset_path()

        # Load schema if specified
        schema = None
        if self.config.data.schema_path:
            # For now, create schema from config
            # TODO(dev): Implement schema loading from file
            pass

        # Create schema from data config
        if not schema:
            # Extract schema information from config
            target_col = self.config.data.target_column or "target"
            feature_cols = self.config.data.feature_columns or []

            # If no feature columns specified, use all except target
            if not feature_cols:
                # We'll need to load data first to get column names
                temp_data = pd.read_csv(data_path)
                feature_cols = [col for col in temp_data.columns if col != target_col]

            schema = TabularDataSchema(
                target=target_col,
                features=feature_cols,
                sensitive_features=self.config.data.protected_attributes,
            )

        # Load data
        data = self.data_loader.load(data_path, schema)

        # First-class schema validation before proceeding
        from ..data.schema import get_schema_summary, validate_config_schema, validate_data_quality

        try:
            # Convert config to dict for schema validation
            config_dict = {"data": self.config.data.model_dump()}
            validated_schema = validate_config_schema(data, config_dict)

            # Run data quality checks
            validate_data_quality(data, validated_schema)

            # Log schema summary
            schema_summary = get_schema_summary(data, validated_schema)
            logger.info(
                f"Schema validation passed: {schema_summary['n_features']} features, "
                f"{schema_summary['n_classes']} classes, "
                f"{schema_summary['n_protected_attributes']} protected attributes",
            )

        except ValueError as e:
            logger.error(f"Schema validation failed: {e}")
            raise

        # Validate schema (legacy validation)
        self.data_loader.validate_schema(data, schema)

        # Store data information in results
        self.results.data_summary = self.data_loader.get_data_summary(data)
        self.results.schema_info = schema.model_dump()

        # Add dataset to manifest
        self.manifest_generator.add_dataset(
            "primary_dataset",
            data=data,
            file_path=data_path,
            target_column=schema.target,
            sensitive_features=schema.sensitive_features,
        )

        logger.info(f"Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")

        # Warn about performance for large datasets
        n_rows = len(data)
        n_protected = len(schema.sensitive_features) if schema.sensitive_features else 0

        if n_rows > 50000:
            # Very large dataset - will be slow
            estimated_time = n_rows / 1000  # Rough estimate: ~1 second per 1000 rows
            logger.warning(
                f"Large dataset detected ({n_rows:,} rows). "
                f"Audit may take {estimated_time:.0f}-{estimated_time * 2:.0f} seconds. "
                "Consider using --sample for faster iteration during development.",
            )
        elif n_rows > 10000:
            # Moderately large dataset
            estimated_time = n_rows / 1500
            logger.info(
                f"Dataset has {n_rows:,} rows. Estimated audit time: {estimated_time:.0f}-{estimated_time * 1.5:.0f} seconds. "
                "Use --sample for faster iteration.",
            )

        # Warn about fairness computation complexity
        if n_protected > 1 and n_rows > 5000:
            logger.info(
                f"Computing fairness metrics for {n_protected} protected attributes on {n_rows:,} samples. "
                "Bootstrap confidence intervals may take additional time.",
            )

        return data, schema

    def _load_model(self, data: pd.DataFrame, schema: TabularDataSchema) -> Any:
        """Load or train model.

        Args:
            data: Dataset for training
            schema: Data schema

        Returns:
            Trained/loaded model instance

        """
        logger.info("Loading/training model")

        # Check for in-memory model (from from_model())
        if self._runtime_model is not None:
            model = self._runtime_model
            logger.info("Using in-memory model from from_model() API")

            # Store model info - use proper model type from config
            model_type = self.config.model.type

            # Try to get feature importance if available
            feature_importance = {}
            if hasattr(model, "get_feature_importance"):
                try:
                    importance = model.get_feature_importance()
                    if hasattr(importance, "to_dict"):
                        feature_importance = importance.to_dict()
                    elif isinstance(importance, dict):
                        feature_importance = importance
                    else:
                        feature_importance = dict(importance) if importance is not None else {}
                except Exception as e:
                    logger.warning(f"Could not extract feature importance: {e}")

            self.results.model_info = {
                "type": model_type,
                "capabilities": model.get_capabilities() if hasattr(model, "get_capabilities") else {},
                "feature_importance": feature_importance,
            }

            # Track in manifest
            model_info = {"name": model_type, "type": "model"}
            if self.requested_model and self.requested_model != model_type:
                model_info["requested"] = self.requested_model
            self.results.selected_components["model"] = model_info
            self.manifest_generator.add_component(
                "model",
                model_type,
                model,
                details={"source": "in_memory"},
            )

            return model

        # Default to trainable model if no config provided (E2E compliance)
        if not hasattr(self.config, "model") or self.config.model is None:
            logger.info("Using default trainable model: LogisticRegressionWrapper")

            # Create default config for LogisticRegression
            from types import SimpleNamespace

            default_model_config = SimpleNamespace()
            default_model_config.type = "logistic_regression"
            default_model_config.params = {"random_state": get_component_seed("model")}

            # Create temporary config with default model
            temp_config = SimpleNamespace()
            temp_config.model = default_model_config
            temp_config.reproducibility = getattr(self.config, "reproducibility", None)

            # Extract features and target for training
            X, y, _ = self.data_loader.extract_features_target(data, schema)
            X_processed = self._preprocess_for_training(X)

            # Use train_from_config for consistency
            from .train import train_from_config

            model = train_from_config(temp_config, X_processed, y)
            logger.info("Default model training completed using configuration")

            # Store model info and tracking
            self.results.model_info = {
                "type": "logistic_regression",
                "capabilities": model.get_capabilities() if hasattr(model, "get_capabilities") else {},
                "feature_importance": {},
            }
            model_info = {"name": "logistic_regression", "type": "model"}
            if self.requested_model and self.requested_model != "logistic_regression":
                model_info["requested"] = self.requested_model
            self.results.selected_components["model"] = model_info

            # Add to manifest
            self.manifest_generator.add_component(
                "model",
                "logistic_regression",
                model,
                details={"default": True, "fitted": True},
            )
            return model

        # Get model configuration
        model_type = self.config.model.type
        model_path = getattr(self.config.model, "path", None)

        # Load model using explicit dispatch
        from glassalpha.models import load_model

        try:
            model_class = load_model(model_type)
        except ValueError:
            msg = f"Unknown model type: {model_type}"
            raise ValueError(msg)

        if model_path and Path(model_path).exists():
            # Load existing model
            model = model_class.from_file(Path(model_path))
            logger.info(f"Loaded model from {model_path}")
        else:
            # Train new model using configuration-driven approach
            logger.info("Training new model from configuration")

            # Extract features and target
            X, y, _ = self.data_loader.extract_features_target(data, schema)
            X_processed = self._preprocess_for_training(X)

            # Use the new train_from_config function
            from .train import train_from_config

            model = train_from_config(self.config, X_processed, y)
            logger.info("Model training completed using configuration")

            # Auto-save model if save_path is configured
            model_save_path = getattr(self.config.model, "save_path", None)
            if model_save_path:
                try:
                    import json

                    import joblib

                    save_path = Path(model_save_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, save_path)
                    logger.info(f"Model auto-saved to: {save_path}")

                    # Save metadata for compatibility checks
                    meta_path = save_path.with_suffix(".meta.json")
                    metadata = {
                        "model_type": model_type,
                        "feature_names": list(X_processed.columns),
                        "target_column": self.config.data.target_column,
                        "protected_attributes": self.config.data.protected_attributes or [],
                        "preprocessing_applied": self._preprocessing_artifact_path is not None,
                        "n_features": len(X_processed.columns),
                    }
                    if self._preprocessing_artifact_path:
                        metadata["preprocessing_artifact"] = self._preprocessing_artifact_path
                    meta_path.write_text(json.dumps(metadata, indent=2))
                    logger.info(f"Model metadata saved to: {meta_path}")
                except Exception as e:
                    logger.warning(f"Failed to auto-save model to {model_save_path}: {e}")

        # Store model information
        feature_importance = {}
        if hasattr(model, "get_feature_importance"):
            try:
                importance = model.get_feature_importance()
                if hasattr(importance, "to_dict"):
                    feature_importance = importance.to_dict()
                elif isinstance(importance, dict):
                    feature_importance = importance
                else:
                    feature_importance = dict(importance) if importance is not None else {}
            except Exception as e:
                logger.warning(f"Could not extract feature importance: {e}")
                feature_importance = {}

        self.results.model_info = {
            "type": model_type,
            "capabilities": model.get_capabilities() if hasattr(model, "get_capabilities") else {},
            "feature_importance": feature_importance,
        }

        # Friend's spec: Track the model in both results and manifest
        # Track in results.selected_components with exact structure
        model_info = {"name": model_type, "type": "model"}
        if self.requested_model and self.requested_model != model_type:
            model_info["requested"] = self.requested_model
        self.results.selected_components["model"] = model_info

        # Add model to manifest using new signature with details
        model_config = self.config.model.model_dump() if self.config.model else {}
        self.manifest_generator.add_component(
            "model",
            model_type,
            model,
            details={
                "config": model_config,
                "priority": getattr(model, "priority", None),
            },
        )

        return model

    def _select_explainer(self) -> Any:
        """Select best explainer based on model capabilities and configuration.

        Returns:
            Selected explainer instance

        """
        logger.info("Selecting explainer based on model capabilities")

        # Select explainer using explicit dispatch
        from glassalpha.explain import select_explainer

        # Use pre-selected explainer if provided, otherwise select based on config
        explainer_class = None
        if self.selected_explainer:
            # selected_explainer is a string (explainer name), not an instance
            selected_name = self.selected_explainer
            explainer_class = self._get_explainer_class(selected_name)
            explainer_instance = explainer_class()
        else:
            explainer_name = select_explainer(
                model_type=self.config.model.type,
                config=self.config.model_dump(),
            )
            selected_name = explainer_name

            # Map explainer name to class
            explainer_class = self._get_explainer_class(explainer_name)
            explainer_instance = explainer_class()

        logger.info(f"Selected explainer: {selected_name}")

        # If no explainer found, raise error (required by tests)
        # Only check explainer_class when it was actually assigned
        if explainer_class is not None and not explainer_class:
            msg = "No compatible explainer found"
            raise RuntimeError(msg)

        # Store selection info
        self.results.selected_components["explainer"] = {
            "name": selected_name,
            "capabilities": getattr(explainer_instance, "capabilities", {}),
        }

        # Add to manifest with new signature
        self.manifest_generator.add_component(
            "explainer",
            selected_name,
            explainer_instance,
            details={
                "implementation": selected_name,
                "priority": getattr(explainer_instance, "priority", None),
            },
        )

        return explainer_instance

    def _generate_explanations(self, data: pd.DataFrame, schema: TabularDataSchema) -> dict[str, Any]:
        """Generate model explanations.

        Args:
            data: Dataset
            schema: Data schema

        Returns:
            Dictionary with explanation results

        """
        logger.info("Generating model explanations")

        # Extract features for explanation
        X, y_target, _ = self.data_loader.extract_features_target(data, schema)

        # For in-memory models from from_model(), check if we need to exclude protected attributes
        # Only exclude if model was trained on fewer features than provided
        if self._runtime_model is not None and schema.sensitive_features:
            model_n_features = getattr(self._runtime_model, "n_features_in_", None)
            if model_n_features is not None and model_n_features < len(X.columns):
                # Model trained on subset - likely excluding protected attributes
                protected_cols = [col for col in schema.sensitive_features if col in X.columns]
                if protected_cols:
                    logger.info(f"Excluding protected attributes from model input: {protected_cols}")
                    X = X.drop(columns=protected_cols)

        # Preprocess features same way as training
        X_processed = self._preprocess_for_training(X)

        # Generate explanations with explainer seed
        with self._get_seeded_context("explainer"):
            # Fit explainer with model and background data (use sample of data as background)
            background_sample = X_processed.sample(n=min(100, len(X_processed)), random_state=42)
            self.explainer.fit(self.model, background_sample, feature_names=list(X_processed.columns))

            # Get strict mode from config for progress bar control
            strict_mode = False
            if hasattr(self.config, "runtime") and hasattr(self.config.runtime, "strict_mode"):
                strict_mode = self.config.runtime.strict_mode
            elif isinstance(self.config, dict):
                strict_mode = self.config.get("runtime", {}).get("strict_mode", False)

            # Generate explanations with progress settings
            # Note: Different explainers have different signatures:
            # - CoefficientsExplainer: explain(model, X, y, **kwargs)
            # - PermutationExplainer: explain(X, y, **kwargs) [model stored in fit]
            # - TreeSHAP: explain(X, **kwargs) [model stored in fit]

            # Check explainer signature to determine if it needs model parameter
            import inspect

            sig = inspect.signature(self.explainer.explain)
            params = list(sig.parameters.keys())

            # If first param is 'model', pass it; otherwise explainer has model from fit()
            if params and params[0] == "model":
                explanations = self.explainer.explain(
                    self.model,  # Some explainers need explicit model reference
                    X_processed,
                    y=y_target,  # Required for PermutationExplainer
                    show_progress=True,  # Enable progress by default
                    strict_mode=strict_mode,  # Respect strict mode setting
                )
            else:
                explanations = self.explainer.explain(
                    X_processed,
                    y=y_target,  # Required for PermutationExplainer
                    show_progress=True,  # Enable progress by default
                    strict_mode=strict_mode,  # Respect strict mode setting
                )

        # Normalize explanations to canonical format
        normalized_explanations = self._normalize_explanations(
            explanations,
            X_processed,
            feature_names=list(X_processed.columns),
        )

        explanation_results = {
            "global_importance": normalized_explanations["global_importance"],
            "local_explanations_sample": normalized_explanations["local_explanations_sample"],
            "summary_statistics": self._compute_explanation_stats(normalized_explanations),
        }

        # Store in results
        self.results.explanations = explanation_results

        logger.info("Explanation generation completed")

        return explanation_results

    def _normalize_explanations(self, explanations: Any, X: pd.DataFrame, feature_names: list[str]) -> dict[str, Any]:
        """Normalize explainer outputs to canonical format.

        Args:
            explanations: Raw output from explainer
            X: Input data for shape reference
            feature_names: Feature names

        Returns:
            Dictionary with canonical explanation format

        """
        logger.info("Normalizing explainer outputs to canonical format")

        # Handle different explainer output formats
        if isinstance(explanations, dict):
            # Already structured format
            return self._normalize_dict_explanations(explanations, X, feature_names)
        if isinstance(explanations, np.ndarray):
            # Raw SHAP values array
            return self._normalize_array_explanations(explanations, X, feature_names)
        # Unknown format - create empty structure
        logger.warning(f"Unknown explainer output format: {type(explanations)}")
        return {
            "global_importance": {},
            "local_explanations_sample": [],
            "ranking": [],
        }

    def _normalize_dict_explanations(
        self,
        explanations: dict,
        X: pd.DataFrame,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """Normalize dictionary-format explanations."""
        n_samples = len(X)

        # Handle global_importance
        global_importance = explanations.get("global_importance", {})
        if isinstance(global_importance, dict):
            # Already a dict - ensure all values are scalars
            normalized_global = {}
            for feature, value in global_importance.items():
                if isinstance(value, (list, np.ndarray)):
                    # Take mean if it's an array
                    normalized_global[feature] = float(np.mean(value))
                else:
                    normalized_global[feature] = float(value)
            global_importance = normalized_global
        elif isinstance(global_importance, (list, np.ndarray)):
            # Convert array to dict
            global_importance = {
                feature_names[i]: float(global_importance[i])
                for i in range(min(len(feature_names), len(global_importance)))
            }

        # Handle local_explanations
        local_explanations = explanations.get("local_explanations", [])
        if isinstance(local_explanations, np.ndarray):
            # Check dimensionality and reduce if needed
            if local_explanations.ndim == 3:
                # Multi-class: (n_samples, n_features, n_classes) or similar
                # Reduce to 2D by taking mean across class dimension
                # Try to identify class axis (usually last or second)
                if local_explanations.shape[-1] <= local_explanations.shape[1]:
                    # Class axis is likely last: (n_samples, n_features, n_classes)
                    local_explanations = np.mean(np.abs(local_explanations), axis=-1)
                else:
                    # Class axis is likely middle: (n_samples, n_classes, n_features)
                    local_explanations = np.mean(np.abs(local_explanations), axis=1)

            # Now convert 2D array to list of dicts
            local_explanations_sample = []
            n_samples_to_show = min(5, local_explanations.shape[0])
            n_features_local = min(local_explanations.shape[1], len(feature_names))

            for i in range(n_samples_to_show):
                sample_dict = {}
                for j in range(n_features_local):
                    value = local_explanations[i, j]
                    sample_dict[feature_names[j]] = float(value)
                local_explanations_sample.append(sample_dict)
        else:
            # Already in expected format or empty
            local_explanations_sample = local_explanations[:5] if local_explanations else []

        # Create ranking from global importance
        ranking = sorted(global_importance.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            "global_importance": global_importance,
            "local_explanations_sample": local_explanations_sample,
            "ranking": ranking,
        }

    def _normalize_array_explanations(
        self,
        explanations: np.ndarray,
        X: pd.DataFrame,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """Normalize array-format explanations."""
        n_samples = len(X)

        # Handle different array shapes
        if explanations.ndim == 2:
            # (n_samples, n_features) - per-sample attributions
            n_samples, n_features = explanations.shape
            if n_features != len(feature_names):
                logger.warning(
                    f"Feature count mismatch: explanations has {n_features} features, expected {len(feature_names)}",
                )
                n_features = min(n_features, len(feature_names))

            # Global importance = mean absolute per-sample
            global_importance = np.mean(np.abs(explanations), axis=0)
            global_importance_dict = {}
            for i in range(n_features):
                value = global_importance[i]
                # Handle arrays by taking mean
                if isinstance(value, (list, np.ndarray)):
                    global_importance_dict[feature_names[i]] = float(np.mean(value))
                else:
                    global_importance_dict[feature_names[i]] = float(value)
            global_importance = global_importance_dict

            # Local explanations sample
            local_explanations_sample = []
            for i in range(min(5, n_samples)):
                sample_dict = {}
                for j in range(n_features):
                    value = explanations[i, j]
                    # Handle arrays by taking mean or first element
                    if isinstance(value, (list, np.ndarray)):
                        sample_dict[feature_names[j]] = float(np.mean(value))
                    else:
                        sample_dict[feature_names[j]] = float(value)
                local_explanations_sample.append(sample_dict)

        elif explanations.ndim == 3:
            # (n_samples, n_classes, n_features) or (n_samples, n_features, n_classes)
            # Need to reduce class dimension
            shape = explanations.shape

            # Try to identify class axis by looking at sizes
            # Class axis typically has smallest dimension (often 2 for binary)
            if shape[-1] <= shape[1]:
                # Class axis is likely last: (n_samples, n_features, n_classes)
                class_axis = -1
                n_samples, n_features = shape[0], shape[1]
            else:
                # Class axis is likely middle: (n_samples, n_classes, n_features)
                class_axis = 1
                n_samples, n_features = shape[0], shape[2]

            # Reduce class dimension - use mean absolute across classes
            reduced_explanations = np.mean(np.abs(explanations), axis=class_axis)

            # Now treat as (n_samples, n_features)
            return self._normalize_array_explanations(reduced_explanations, X, feature_names)

        else:
            logger.warning(f"Unsupported explanation array shape: {explanations.shape}")
            return {
                "global_importance": {},
                "local_explanations_sample": [],
                "ranking": [],
            }

        # Create ranking
        ranking = sorted(global_importance.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            "global_importance": global_importance,
            "local_explanations_sample": local_explanations_sample,
            "ranking": ranking,
        }

    def _compute_metrics(self, data: pd.DataFrame, schema: TabularDataSchema) -> None:
        """Compute all configured metrics.

        Args:
            data: Dataset
            schema: Data schema

        """
        logger.info("Computing audit metrics")

        # Get progress callback from instance if available
        progress_callback = getattr(self, "_progress_callback", None)

        # Extract data components
        X, y_true, sensitive_features = self.data_loader.extract_features_target(data, schema)

        # For in-memory models from from_model(), check if we need to exclude protected attributes
        # Only exclude if model was trained on fewer features than provided
        X_for_model = X
        if self._runtime_model is not None and schema.sensitive_features:
            model_n_features = getattr(self._runtime_model, "n_features_in_", None)
            if model_n_features is not None and model_n_features < len(X.columns):
                # Model trained on subset - likely excluding protected attributes
                protected_cols = [col for col in schema.sensitive_features if col in X.columns]
                if protected_cols:
                    logger.info(f"Excluding protected attributes from model input: {protected_cols}")
                    X_for_model = X.drop(columns=protected_cols)

        # Use processed features for predictions (same as training)
        X_processed = self._preprocess_for_training(X_for_model)

        # Friend's spec: Fit-or-fail approach - never skip metrics
        if self.model is None:
            msg = "Model is None during metrics computation - this should not happen"
            raise RuntimeError(msg)

        # Check if model needs fitting and fit it (don't skip metrics)
        model_needs_fitting = False

        # For in-memory models from from_model(), they're already fitted
        if self._runtime_model is not None:
            logger.debug("Using pre-fitted in-memory model from from_model()")
            model_needs_fitting = False
        # Contract: simplified training logic guard (string match required by tests)
        elif getattr(self.model, "model", None) is None:
            logger.debug("No underlying model instance set; proceeding with wrapper defaults")
            model_needs_fitting = True
        elif hasattr(self.model, "model") and getattr(self.model, "model", None) is None:
            # Model wrapper exists but internal model is None - needs fitting
            model_needs_fitting = True
        elif hasattr(self.model, "_is_fitted") and not getattr(self.model, "_is_fitted", True):
            # Model wrapper tracks fitted state explicitly
            model_needs_fitting = True

        if model_needs_fitting:
            logger.warning("Model not fitted - this should not happen with proper wrapper-based training")
            logger.warning("All models should be trained via train_from_config() in _load_model()")

            # This is a fallback that should rarely be used
            if not hasattr(self.model, "fit"):
                msg = f"Model type {type(self.model).__name__} needs fitting but doesn't support fit method"
                raise RuntimeError(msg)

            logger.info("Fallback: fitting model with available data")
            model_seed = (
                self.manifest_generator.manifest.seeds.get("model", 42)
                if hasattr(self.manifest_generator, "manifest")
                else 42
            )

            # Get random state from config if available
            if hasattr(self.config, "reproducibility") and self.config.reproducibility:
                model_seed = getattr(self.config.reproducibility, "random_state", model_seed)

            # Use wrapper fit method with proper parameters
            if hasattr(self.config, "model") and hasattr(self.config.model, "params"):
                model_params = dict(self.config.model.params)
                model_params["random_state"] = model_seed
                self.model.fit(X_processed, y_true, **model_params)
            else:
                self.model.fit(X_processed, y_true, random_state=model_seed)
            logger.info("Fallback model fitting completed")

        # Generate probability predictions first
        y_proba = None
        if hasattr(self.model, "predict_proba"):
            try:
                y_proba = self.model.predict_proba(X_processed)
                # Keep full probability matrix for multiclass, don't extract single column
                logger.debug(f"Generated probability predictions with shape: {y_proba.shape}")
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")
                # Try fallback preprocessing if feature name mismatch
                if "feature names" in str(e).lower():
                    logger.info("Attempting fallback preprocessing due to feature name mismatch")
                    X_processed = self._preprocess_for_training_fallback(X_for_model)
                    try:
                        y_proba = self.model.predict_proba(X_processed)
                        logger.debug(f"Fallback preprocessing succeeded, shape: {y_proba.shape}")
                    except Exception as fallback_e:
                        logger.warning(f"Fallback preprocessing also failed: {fallback_e}")

        # Generate predictions using threshold policy (if binary classification with probabilities)
        if y_proba is not None and y_proba.shape[1] == 2:  # Binary classification
            # Use threshold policy for binary classification
            y_pred, threshold_info = self._apply_threshold_policy(y_true, y_proba[:, 1])
            # Store threshold information in results
            self.results.model_performance["threshold_selection"] = threshold_info
        else:
            # Fallback to model's default predict method (multiclass or no probabilities)
            try:
                y_pred = self.model.predict(X_processed)
                logger.info("Using model's default predictions (multiclass or no probabilities available)")
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                # Try fallback preprocessing if feature name mismatch
                if "feature names" in str(e).lower():
                    logger.info("Attempting fallback preprocessing for prediction")
                    X_processed = self._preprocess_for_training_fallback(X_for_model)
                    try:
                        y_pred = self.model.predict(X_processed)
                        logger.info("Fallback preprocessing succeeded for prediction")
                    except Exception as fallback_e:
                        logger.error(f"Fallback preprocessing also failed: {fallback_e}")
                        raise ValueError(
                            f"Model prediction failed with both preprocessing methods. "
                            f"Original error: {e}. Fallback error: {fallback_e}. "
                            f"This typically indicates a mismatch between model training preprocessing "
                            f"and audit pipeline preprocessing. Consider using consistent preprocessing.",
                        ) from fallback_e
                else:
                    raise

        # Compute performance metrics (~5% of time)
        self._update_progress(progress_callback, "Computing performance (accuracy, precision, recall)", 50)
        self._compute_performance_metrics(y_true, y_pred, y_proba)
        self._update_progress(progress_callback, "Performance metrics complete", 55)

        # Friend's spec: Ensure accuracy is always computed for each model type
        try:
            from sklearn.metrics import accuracy_score

            acc = float(accuracy_score(y_true, y_pred))
            if not hasattr(self.results, "model_performance") or self.results.model_performance is None:
                self.results.model_performance = {}
            # Only set if not already set by _compute_performance_metrics
            if "accuracy" not in self.results.model_performance:
                self.results.model_performance["accuracy"] = {
                    "accuracy": acc,
                    "n_samples": len(y_true),
                }
            logger.debug(f"Computed explicit accuracy: {acc:.4f}")
        except Exception:
            logger.exception("Failed to compute explicit accuracy:")

        # Compute fairness metrics if sensitive features available (~33% of time - bootstrap CIs are expensive)
        if sensitive_features is not None:
            self._update_progress(progress_callback, "Computing fairness (demographic parity, equal opportunity)", 60)
            self._compute_fairness_metrics(y_true, y_pred, y_proba, sensitive_features, X)
            self._update_progress(progress_callback, "Fairness metrics complete", 88)

        # Compute stability metrics (~2% of time)
        self._update_progress(progress_callback, "Computing stability (monotonicity, consistency)", 90)
        self._compute_stability_metrics(X, sensitive_features)
        self._update_progress(progress_callback, "Stability metrics complete", 92)

        # Compute drift metrics (placeholder for now)
        self._compute_drift_metrics(X, y_true)

        logger.info("Metrics computation completed")

    def _apply_threshold_policy(self, y_true: np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply threshold selection policy for binary classification.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class

        Returns:
            Tuple of (predictions, threshold_info)

        """
        from ..metrics.thresholds import pick_threshold, validate_threshold_config

        # Get threshold configuration from report config
        threshold_config = {}
        if hasattr(self.config, "report") and hasattr(self.config.report, "threshold") and self.config.report.threshold:
            threshold_config = self.config.report.threshold.model_dump()

        # Validate and normalize config
        validated_config = validate_threshold_config(threshold_config)

        logger.info(f"Applying threshold policy: {validated_config['policy']}")

        # Select threshold using policy
        threshold_result = pick_threshold(y_true, y_proba, **validated_config)

        # Generate predictions using selected threshold
        selected_threshold = threshold_result["threshold"]
        y_pred = (y_proba >= selected_threshold).astype(int)

        logger.info(f"Selected threshold: {selected_threshold:.3f} using {threshold_result['policy']} policy")

        return y_pred, threshold_result

    def _generate_provenance_manifest(self) -> None:
        """Generate comprehensive provenance manifest for audit reproducibility."""
        from ..provenance import generate_run_manifest

        logger.info("Generating comprehensive provenance manifest")

        # Gather all provenance information
        config_dict = self.config.model_dump() if hasattr(self.config, "model_dump") else dict(vars(self.config))

        # Get dataset information
        dataset_path = getattr(self.config.data, "path", None) if hasattr(self.config, "data") else None
        dataset_df = getattr(self, "_dataset_df", None)  # Store dataset if available

        # Get model information
        model_info = {
            "type": self.model.__class__.__name__ if self.model else None,
            "parameters": getattr(self.model, "get_params", dict)() if self.model else {},
            "calibration": self._get_calibration_info(),
            "feature_count": getattr(self, "_feature_count", None),
            "class_count": getattr(self, "_class_count", None),
        }

        # Get selected components
        selected_components = {
            "explainer": self.explainer.__class__.__name__ if self.explainer else None,
            "metrics": list(self.selected_metrics.keys()) if self.selected_metrics else [],
            "threshold_policy": self.results.model_performance.get("threshold_selection", {}).get("policy")
            if hasattr(self.results, "model_performance")
            else None,
        }

        # Get execution information
        execution_info = {
            "start_time": getattr(self, "_start_time", None),
            "end_time": getattr(self, "_end_time", None),
            "success": True,  # If we're here, it succeeded
            "random_seed": getattr(self.config.reproducibility, "random_seed", None)
            if hasattr(self.config, "reproducibility")
            else None,
        }

        # Generate the manifest
        manifest = generate_run_manifest(
            config=config_dict,
            dataset_path=dataset_path,
            dataset_df=dataset_df,
            model_info=model_info,
            selected_components=selected_components,
            execution_info=execution_info,
            seed=getattr(self.config.reproducibility, "random_seed", None),
        )

        # Store in results for PDF embedding
        self.results.execution_info["provenance_manifest"] = manifest

        logger.info(f"Provenance manifest generated with {len(manifest)} sections")

    def _get_calibration_info(self) -> dict[str, Any] | None:
        """Get calibration information from model if available."""
        if not self.model:
            return None

        try:
            from ..models.calibration import get_calibration_info

            base_estimator = getattr(self.model, "model", self.model)
            return get_calibration_info(base_estimator)
        except Exception:
            return None

    def _compute_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> None:
        """Compute performance metrics using auto-detecting engine.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities

        """
        from ..metrics.core import compute_classification_metrics

        logger.debug("Computing performance metrics with auto-detection engine")

        # Get averaging strategy from config if specified
        performance_config = self.config.metrics.performance
        if hasattr(performance_config, "config"):
            averaging_override = performance_config.config.get("average")
        else:
            averaging_override = None

        # Use the new auto-detecting metrics engine
        try:
            metrics_result = compute_classification_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                average=averaging_override,
            )

            # Convert to the expected format for compatibility
            results = {
                "accuracy": {
                    "accuracy": metrics_result["accuracy"],
                    "n_samples": len(y_true),
                },
                "precision": {
                    "precision": metrics_result["precision"],
                    "n_samples": len(y_true),
                    "problem_type": metrics_result["problem_type"],
                    "averaging_strategy": metrics_result["averaging_strategy"],
                },
                "recall": {
                    "recall": metrics_result["recall"],
                    "n_samples": len(y_true),
                    "problem_type": metrics_result["problem_type"],
                    "averaging_strategy": metrics_result["averaging_strategy"],
                },
                "f1": {
                    "f1": metrics_result["f1_score"],
                    "n_samples": len(y_true),
                    "problem_type": metrics_result["problem_type"],
                    "averaging_strategy": metrics_result["averaging_strategy"],
                },
                "classification_report": {
                    "accuracy": metrics_result["accuracy"],
                    "n_samples": len(y_true),
                    "n_classes": metrics_result["n_classes"],
                    "problem_type": metrics_result["problem_type"],
                    "averaging_strategy": metrics_result["averaging_strategy"],
                },
            }

            # Add probability-based metrics if available
            if metrics_result["roc_auc"] is not None:
                results["auc_roc"] = {
                    "roc_auc": metrics_result["roc_auc"],
                    "n_samples": len(y_true),
                    "problem_type": metrics_result["problem_type"],
                }

            if metrics_result["log_loss"] is not None:
                results["log_loss"] = {
                    "log_loss": metrics_result["log_loss"],
                    "n_samples": len(y_true),
                    "problem_type": metrics_result["problem_type"],
                }

            # Log any warnings or errors from the metrics engine
            if metrics_result["warnings"]:
                for warning in metrics_result["warnings"]:
                    logger.warning(f"Metrics engine warning: {warning}")

            if metrics_result["errors"]:
                for error in metrics_result["errors"]:
                    logger.error(f"Metrics engine error: {error}")

            # Filter out None values to maintain compatibility
            results = {
                k: v
                for k, v in results.items()
                if v and all(val is not None for val in v.values() if isinstance(val, (int, float)))
            }

            logger.info(f"Successfully computed {len(results)} performance metrics using auto-detection engine")

        except Exception as e:
            logger.error(f"Failed to compute performance metrics with auto-detection engine: {e}")
            # Fallback to empty results with error
            results = {"error": f"Auto-detection engine failed: {e}"}

        self.results.model_performance = results

        # E10+: Compute calibration with confidence intervals
        if y_proba is not None and y_proba.shape[1] == 2:  # Binary classification with probabilities
            try:
                from ..metrics.calibration.quality import assess_calibration_quality
                from ..utils.seeds import get_component_seed

                logger.debug("Computing E10+: Calibration with confidence intervals")

                # Get seed for deterministic bootstrap
                seed = get_component_seed("calibration_ci")

                # Get bootstrap count from config
                n_bootstrap = 500  # Default (reduced for performance - was 1000, now 500 for balance)
                if hasattr(self.config, "metrics") and hasattr(self.config.metrics, "n_bootstrap"):
                    n_bootstrap = self.config.metrics.n_bootstrap
                elif isinstance(self.config, dict):
                    n_bootstrap = self.config.get("metrics", {}).get("n_bootstrap", 1000)

                # Compute calibration with CIs
                calibration_result = assess_calibration_quality(
                    y_true=y_true,
                    y_prob_pos=y_proba[:, 1],  # Positive class probabilities
                    n_bins=10,
                    compute_confidence_intervals=self.config.metrics.compute_confidence_intervals,
                    n_bootstrap=n_bootstrap,
                    confidence_level=0.95,
                    seed=seed,
                )

                # Add to model_performance
                if isinstance(calibration_result, dict):
                    self.results.model_performance["calibration_ci"] = calibration_result
                else:
                    self.results.model_performance["calibration_ci"] = calibration_result.to_dict()

                logger.info("E10+: Calibration metrics with CIs computed successfully")

            except Exception as e:
                logger.warning(f"E10+: Failed to compute calibration with CIs: {e}")
                # Don't fail entire performance analysis if calibration CIs fail
                self.results.model_performance["calibration_ci"] = {
                    "error": str(e),
                    "status": "failed",
                }

    def _compute_fairness_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        sensitive_features: pd.DataFrame,
        X: pd.DataFrame,
    ) -> None:
        """Compute fairness metrics including E11 individual fairness.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            sensitive_features: Sensitive attributes
            X: Full feature DataFrame (for individual fairness)

        """
        logger.debug("Computing fairness metrics")

        # Preprocess features for individual fairness (needs same feature space as model)
        X_processed = self._preprocess_for_training(X)

        # E12: Compute dataset-level bias metrics first (foundational check)
        logger.debug("Computing E12: Dataset-level bias metrics")
        try:
            from ..metrics.fairness.dataset import compute_dataset_bias_metrics

            # Prepare full dataset with protected attributes
            data = X.copy()
            for col in sensitive_features.columns:
                if col not in data.columns:
                    data[col] = sensitive_features[col]

            # Get feature columns (non-protected)
            feature_cols = [c for c in X.columns if c not in sensitive_features.columns]
            protected_cols = list(sensitive_features.columns)

            # Get seed for reproducibility
            seed = get_component_seed("dataset_bias")

            # Compute dataset bias metrics
            dataset_bias = compute_dataset_bias_metrics(
                data=data,
                feature_cols=feature_cols,
                protected_attrs=protected_cols,
                seed=seed,
                compute_proxy=True,
                compute_drift=True,
                compute_power=True,
                compute_imbalance=False,  # Would need train/test split indicator column
            )

            # Store in results (will be added to fairness_analysis later)
            self._dataset_bias_results = dataset_bias.to_dict()
            logger.info("E12: Dataset bias metrics computed successfully")

        except Exception as e:
            logger.warning(f"E12: Failed to compute dataset bias metrics: {e}")
            # Don't fail entire fairness analysis if dataset bias fails
            self._dataset_bias_results = {
                "error": str(e),
                "status": "failed",
            }

        # Get available fairness metrics
        all_metric_names = self._get_available_fairness_metrics()
        fairness_metrics = []

        for name in all_metric_names:
            try:
                metric_class = self._get_fairness_metric_class(name)
                # Check if it's a fairness metric
                if (hasattr(metric_class, "metric_type") and metric_class.metric_type == "fairness") or name in [
                    "demographic_parity",
                    "equal_opportunity",
                    "equalized_odds",
                    "predictive_parity",
                ]:
                    fairness_metrics.append(metric_class)
            except (KeyError, ImportError):
                # Skip unavailable metrics
                continue

        if not fairness_metrics:
            # Provide detailed error message based on config structure
            if hasattr(self.config.data, "protected_attributes") and not self.config.data.protected_attributes:
                error_msg = "No protected attributes configured in data.protected_attributes"
                logger.warning(f"Fairness metrics skipped: {error_msg}")
                self.results.fairness_analysis = {"skipped": error_msg}
            else:
                # Config structure issue - provide clear guidance
                error_msg = (
                    "No fairness metrics found. Check config structure:\n"
                    "  Expected: metrics.fairness.metrics = ['demographic_parity', ...]\n"
                    "  Or configure: metrics.fairness with nested 'metrics' key"
                )
                logger.warning(error_msg)
                self.results.fairness_analysis = {"error": "No fairness metrics configured", "hint": error_msg}
            return

        # Use the new fairness runner with E10 confidence intervals
        from ..metrics.fairness.runner import run_fairness_metrics

        # Get seed for deterministic bootstrap CIs (global seed already set in _setup_reproducibility)
        seed = get_component_seed("fairness_ci")

        # Get intersections from config (E5.1 - Intersectional Fairness)
        intersections = []
        if hasattr(self.config, "data") and hasattr(self.config.data, "intersections"):
            intersections = self.config.data.intersections
        elif isinstance(self.config, dict):
            intersections = self.config.get("data", {}).get("intersections", [])

        # Get bootstrap count from config
        n_bootstrap = 1000  # Default
        compute_confidence_intervals = True  # Default
        performance_mode = False  # Default

        if hasattr(self.config, "metrics") and hasattr(self.config.metrics, "n_bootstrap"):
            n_bootstrap = self.config.metrics.n_bootstrap
            compute_confidence_intervals = self.config.metrics.compute_confidence_intervals
            performance_mode = self.config.metrics.performance_mode

        if isinstance(self.config, dict):
            n_bootstrap = self.config.get("metrics", {}).get("n_bootstrap", 1000)

        try:
            # Apply performance mode: reduce fairness metrics for faster computation
            metrics_to_compute = fairness_metrics
            intersections_to_compute = intersections

            # Apply performance mode optimizations
            if performance_mode:
                # In performance mode, reduce fairness metrics and intersections for faster computation
                if len(fairness_metrics) > 1:
                    # Only compute the first fairness metric
                    metrics_to_compute = fairness_metrics[:1]
                    logger.info(
                        f"Performance mode: computing only {len(metrics_to_compute)} fairness metrics instead of {len(fairness_metrics)}",
                    )

                if intersections and len(intersections) > 1:
                    # Only compute the first intersection
                    intersections_to_compute = intersections[:1]
                    logger.info(
                        f"Performance mode: computing only {len(intersections_to_compute)} intersections instead of {len(intersections)}",
                    )
            else:
                metrics_to_compute = fairness_metrics
                intersections_to_compute = intersections

            fairness_results = run_fairness_metrics(
                y_true,
                y_pred,
                sensitive_features,
                metrics_to_compute,
                compute_confidence_intervals=compute_confidence_intervals,
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                seed=seed,
                intersections=intersections_to_compute,  # E5.1: Pass intersections from config
            )

            # E11: Compute individual fairness metrics
            # Check if individual fairness is enabled in config
            individual_fairness_enabled = True
            if hasattr(self.config, "metrics") and hasattr(self.config.metrics, "individual_fairness"):
                individual_fairness_config = self.config.metrics.individual_fairness
                if individual_fairness_config is not None:
                    # Handle both dict and object config formats
                    if isinstance(individual_fairness_config, dict):
                        individual_fairness_enabled = individual_fairness_config.get("enabled", True)
                    elif hasattr(individual_fairness_config, "enabled"):
                        individual_fairness_enabled = individual_fairness_config.enabled

            if individual_fairness_enabled:
                logger.debug("Computing E11: Individual fairness metrics")
                try:
                    from ..metrics.fairness.individual import IndividualFairnessMetrics

                    # Get protected attribute names
                    protected_attrs = list(sensitive_features.columns)

                    # Use preprocessed features (X_processed) that match model's expected input
                    # Individual fairness needs the same feature space the model was trained on
                    X_with_protected = X_processed.copy()
                    for col in protected_attrs:
                        if col not in X_with_protected.columns:
                            X_with_protected[col] = sensitive_features[col]

                    # Extract predictions for individual fairness (binary class probability)
                    if y_proba is not None and len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                        predictions = y_proba[:, 1]  # Positive class probability
                    else:
                        # Fallback to predicted labels (convert to 0-1 scale)
                        predictions = y_pred.astype(float)

                    # Get configuration from metrics.individual_fairness
                    distance_metric = "euclidean"
                    similarity_percentile = 90
                    prediction_diff_threshold = 0.1
                    threshold = 0.5

                    if hasattr(self.config, "metrics") and hasattr(self.config.metrics, "individual_fairness"):
                        ifc = self.config.metrics.individual_fairness
                        if ifc is not None:
                            # Handle both dict and object config formats
                            if isinstance(ifc, dict):
                                distance_metric = ifc.get("distance_metric", distance_metric)
                                similarity_percentile = ifc.get("similarity_percentile", similarity_percentile)
                                prediction_diff_threshold = ifc.get(
                                    "prediction_diff_threshold", prediction_diff_threshold
                                )
                                threshold = ifc.get("threshold", threshold)
                            else:
                                if hasattr(ifc, "distance_metric"):
                                    distance_metric = ifc.distance_metric
                                if hasattr(ifc, "similarity_percentile"):
                                    similarity_percentile = ifc.similarity_percentile
                                if hasattr(ifc, "prediction_diff_threshold"):
                                    prediction_diff_threshold = ifc.prediction_diff_threshold
                                if hasattr(ifc, "threshold"):
                                    threshold = ifc.threshold

                    # Initialize individual fairness metrics
                    individual_metrics = IndividualFairnessMetrics(
                        model=self.model,
                        features=X_with_protected,  # Now using preprocessed features
                        predictions=predictions,
                        protected_attributes=protected_attrs,
                        distance_metric=distance_metric,
                        similarity_percentile=similarity_percentile,
                        prediction_diff_threshold=prediction_diff_threshold,
                        threshold=threshold,
                        seed=seed,
                    )

                    # Compute metrics
                    individual_results = individual_metrics.compute()

                    # Add to fairness results
                    fairness_results["individual_fairness"] = individual_results
                    logger.info("E11: Individual fairness metrics computed successfully")

                except Exception as e:
                    logger.warning(f"E11: Failed to compute individual fairness metrics: {e}")
                    # Don't fail entire fairness analysis if individual fairness fails
                    fairness_results["individual_fairness"] = {
                        "error": str(e),
                        "status": "failed",
                    }
            else:
                logger.debug("E11: Individual fairness metrics disabled by configuration")

            # E12: Add dataset bias results to fairness analysis
            if hasattr(self, "_dataset_bias_results"):
                fairness_results["dataset_bias"] = self._dataset_bias_results

            self.results.fairness_analysis = fairness_results
            logger.info(f"Computed fairness metrics with CIs: {list(fairness_results.keys())}")

        except Exception as e:
            logger.error(f"Failed to compute fairness metrics: {e}")
            # Store error in a way that won't break template rendering
            self.results.fairness_analysis = {}
            self.results.error_message = f"Fairness analysis failed: {e}"

    def _compute_stability_metrics(
        self,
        X: pd.DataFrame,
        sensitive_features: pd.DataFrame | None,
    ) -> None:
        """Compute stability metrics (E6+ perturbation sweeps).

        Args:
            X: Full feature DataFrame
            sensitive_features: Sensitive attributes to exclude from perturbation

        """
        # Check if stability metrics are enabled
        if not hasattr(self.config, "metrics") or not hasattr(self.config.metrics, "stability"):
            logger.debug("Stability metrics not configured, skipping")
            self.results.stability_analysis = {}
            return

        stability_config = self.config.metrics.stability
        if stability_config is None or not stability_config.enabled:
            logger.debug("Stability metrics disabled, skipping")
            self.results.stability_analysis = {}
            return

        logger.debug("Computing stability metrics (E6+ perturbation sweeps)")

        try:
            from ..metrics.stability import run_perturbation_sweep

            # Get protected feature names
            protected_features = []
            if sensitive_features is not None:
                protected_features = list(sensitive_features.columns)

            # Get perturbation config
            epsilon_values = stability_config.epsilon_values
            threshold = stability_config.threshold
            seed = get_component_seed("stability_perturbation")

            # Preprocess features to handle categorical columns (prevent "could not convert string to float" errors)
            # This ensures stability metrics work on datasets with categorical features like German Credit
            X_processed = self._preprocess_for_training(X)

            # Run perturbation sweep
            logger.info(
                f"Running perturbation sweep: epsilon={epsilon_values}, threshold={threshold}, seed={seed}",
            )

            result = run_perturbation_sweep(
                model=self.model,
                X_test=X_processed,  # Use preprocessed features
                protected_features=protected_features,
                epsilon_values=epsilon_values,
                threshold=threshold,
                seed=seed,
            )

            # Store results
            self.results.stability_analysis = result.to_dict()
            logger.info(
                f"Stability metrics computed: robustness_score={result.robustness_score:.6f}, "
                f"gate={result.gate_status}",
            )

        except Exception as e:
            logger.error(f"Failed to compute stability metrics: {e}")
            self.results.stability_analysis = {
                "error": str(e),
                "status": "failed",
            }

    def _compute_drift_metrics(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Compute drift metrics (placeholder implementation).

        Args:
            X: Features
            y: Target values

        """
        logger.debug("Computing drift metrics (placeholder)")

        # For now, just record that we would compute drift metrics
        # This requires reference data which we don't have in this context
        self.results.drift_analysis = {
            "status": "not_computed",
            "reason": "no_reference_data",
            "available_methods": ["psi", "kl_divergence", "ks_test"],
        }

    def _finalize_results(self, explanations: dict[str, Any]) -> None:
        """Finalize audit results and manifest.

        Args:
            explanations: Generated explanations

        """
        logger.info("Finalizing audit results and manifest")

        # Store execution information (update existing dict to preserve preprocessing info)
        # Handle both dict and Pydantic configs
        if hasattr(self.config, "model_dump"):
            cfg_dict = self.config.model_dump()
        elif isinstance(self.config, dict):
            cfg_dict = self.config
        else:
            cfg_dict = dict(vars(self.config))

        self.results.execution_info.update(
            {
                "config_hash": self.manifest_generator.manifest.config_hash,
                "audit_profile": cfg_dict.get("audit_profile", "default"),
                "strict_mode": cfg_dict.get("strict_mode", False),
                "components_used": len(self.manifest_generator.manifest.selected_components),
            },
        )

        # Add result hashes to manifest
        from glassalpha.utils import hash_object

        if self.results.model_performance:
            self.manifest_generator.add_result_hash("performance_metrics", hash_object(self.results.model_performance))

        if self.results.fairness_analysis:
            self.manifest_generator.add_result_hash("fairness_metrics", hash_object(self.results.fairness_analysis))

        if explanations:
            self.manifest_generator.add_result_hash("explanations", hash_object(explanations))

        # Populate selected_components in results from manifest (friend's spec)
        self.results.selected_components = self.manifest_generator.manifest.selected_components

        # Mark manifest as completed successfully before finalizing (friend's spec)
        self.manifest_generator.mark_completed("success")

        # Record end time for provenance
        from glassalpha.utils.determinism import get_deterministic_timestamp

        # Use deterministic timestamp for reproducibility
        seed = self.results.execution_info.get("random_seed") if self.results.execution_info else None
        self._end_time = get_deterministic_timestamp(seed=seed).isoformat()

        # Generate comprehensive provenance manifest
        self._generate_provenance_manifest()

        # Finalize manifest
        final_manifest = self.manifest_generator.finalize()
        # Use mode='json' to properly serialize datetime objects
        self.results.manifest = (
            final_manifest.model_dump(mode="json") if hasattr(final_manifest, "model_dump") else final_manifest
        )

        logger.info("Audit results finalized")

    def _compute_explanation_stats(self, explanations: dict[str, Any]) -> dict[str, Any]:
        """Compute summary statistics for explanations.

        Args:
            explanations: Raw explanation results

        Returns:
            Summary statistics

        """

        def _to_scalar(v: Any) -> float:
            """Convert value to scalar, handling lists/arrays as specified by friend."""
            if isinstance(v, (list, tuple, np.ndarray)):
                return float(np.mean(np.abs(v)))
            return float(abs(v))

        stats = {}

        if "global_importance" in explanations:
            importance = explanations["global_importance"]
            if isinstance(importance, dict):
                values = list(importance.values())
                if values:
                    # Convert all values to scalars before computing stats
                    scalar_values = [_to_scalar(v) for v in values]
                    stats["mean_importance"] = float(np.mean(scalar_values))
                    stats["std_importance"] = float(np.std(scalar_values))
                    stats["top_features"] = sorted(importance.items(), key=lambda x: _to_scalar(x[1]), reverse=True)[:5]

        return stats

    def _get_seeded_context(self, component_name: str) -> Any:
        """Get seeded context manager for component.

        Args:
            component_name: Name of component for seed generation

        Returns:
            Context manager with component seed

        """
        from glassalpha.utils import with_component_seed

        return with_component_seed(component_name)

    def _update_progress(self, callback: Callable, message: str, progress: int) -> None:
        """Update progress if callback provided.

        Args:
            callback: Optional progress callback
            message: Progress message
            progress: Progress percentage (0-100)

        """
        if callback:
            callback(message, progress)

        logger.debug(f"Progress: {progress}% - {message}")

    def _preprocess_for_training(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for model training.

        Uses artifact-based preprocessing if configured, otherwise falls back to
        automatic OneHotEncoder-based preprocessing.

        Args:
            X: Raw features DataFrame

        Returns:
            Processed features DataFrame suitable for training

        """
        # Check if preprocessing artifact is configured
        if hasattr(self.config, "preprocessing") and self.config.preprocessing.mode == "artifact":
            return self._preprocess_with_artifact(X)
        return self._preprocess_auto(X)

    def _preprocess_for_training_fallback(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fallback preprocessing using label encoding.

        Used when one-hot encoding causes feature name mismatches with pre-trained models.
        This uses label encoding to maintain original column names while converting
        categorical values to numeric.

        Args:
            X: Raw features DataFrame

        Returns:
            Processed features DataFrame with label encoding

        """
        from sklearn.preprocessing import LabelEncoder

        logger.info("Using fallback label encoding preprocessing for feature name compatibility")

        X_processed = X.copy()

        # Identify categorical columns
        categorical_cols = list(X.select_dtypes(include=["object"]).columns)

        if not categorical_cols:
            logger.debug("No categorical columns found for fallback preprocessing")
            return X_processed

        # Apply label encoding to each categorical column
        for col in categorical_cols:
            if col in X_processed.columns and X_processed[col].dtype == "object":
                # Use LabelEncoder for each column separately (maintains column names)
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col])

        logger.info(f"Applied label encoding to {len(categorical_cols)} categorical columns")
        return X_processed

    def _preprocess_with_artifact(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess using production artifact (artifact mode).

        Args:
            X: Raw features DataFrame

        Returns:
            Transformed features using artifact preprocessing

        Raises:
            ValueError: If artifact validation fails

        """
        from glassalpha.preprocessing import (
            assert_runtime_versions,
            compute_file_hash,
            compute_params_hash,
            compute_unknown_rates,
            extract_sklearn_manifest,
            load_artifact,
            validate_classes,
            validate_sparsity,
        )

        config = self.config.preprocessing
        logger.info(f"Loading preprocessing artifact from {config.artifact_path}")

        # Track preprocessing artifact path for model metadata
        self._preprocessing_artifact_path = str(config.artifact_path)

        # Load artifact
        artifact = load_artifact(config.artifact_path)

        # Compute actual hashes
        actual_file_hash = compute_file_hash(config.artifact_path)
        manifest = extract_sklearn_manifest(artifact)
        actual_params_hash = compute_params_hash(manifest)

        # Verify file hash
        if config.expected_file_hash:
            if actual_file_hash != config.expected_file_hash:
                msg = (
                    f"Preprocessing artifact file hash mismatch!\n"
                    f"  Expected: {config.expected_file_hash}\n"
                    f"  Actual:   {actual_file_hash}\n"
                    f"  File:     {config.artifact_path}\n"
                    f"This indicates the artifact file has been modified or corrupted."
                )
                if config.fail_on_mismatch:
                    raise ValueError(msg)
                logger.warning(msg)

        # Verify params hash
        if config.expected_params_hash:
            if actual_params_hash != config.expected_params_hash:
                msg = (
                    f"Preprocessing artifact params hash mismatch!\n"
                    f"  Expected: {config.expected_params_hash}\n"
                    f"  Actual:   {actual_params_hash}\n"
                    f"This indicates the learned parameters (encoders, scalers) have changed."
                )
                if config.fail_on_mismatch:
                    raise ValueError(msg)
                logger.warning(msg)

        # Validate artifact classes
        try:
            validate_classes(artifact)
            logger.debug("Artifact class allowlist validation passed")
        except ValueError as e:
            logger.error(f"Artifact contains unsupported transformer class: {e}")
            raise

        # Validate runtime versions
        try:
            # Handle both dict and Pydantic configs
            if hasattr(self.config, "runtime") and hasattr(self.config.runtime, "strict_mode"):
                strict_mode = self.config.runtime.strict_mode
            elif isinstance(self.config, dict):
                strict_mode = self.config.get("runtime", {}).get("strict_mode", False)
            else:
                strict_mode = False

            assert_runtime_versions(
                manifest,
                strict=strict_mode,
                allow_minor=config.version_policy.allow_minor_in_strict,
            )
            logger.debug("Runtime version validation passed")
        except (ValueError, RuntimeError) as e:
            logger.error(f"Runtime version mismatch: {e}")
            raise

        # Compute unknown category rates
        try:
            unknown_rates = compute_unknown_rates(artifact, X)
            if unknown_rates:
                # Check thresholds
                for col, rate in unknown_rates.items():
                    if rate > config.thresholds.fail_unknown_rate:
                        msg = (
                            f"Column '{col}' has {rate:.1%} unknown categories "
                            f"(threshold: {config.thresholds.fail_unknown_rate:.1%})"
                        )
                        logger.error(msg)
                        raise ValueError(msg)
                    if rate > config.thresholds.warn_unknown_rate:
                        logger.warning(
                            f"Column '{col}' has {rate:.1%} unknown categories "
                            f"(threshold: {config.thresholds.warn_unknown_rate:.1%})",
                        )
                # Store for manifest
                self.results.execution_info["preprocessing_unknown_rates"] = unknown_rates
        except Exception as e:
            logger.warning(f"Failed to compute unknown rates: {e}")

        # Transform the data
        logger.info("Transforming data with artifact preprocessor...")
        X_transformed = artifact.transform(X)

        # Note: Output shape validation against model is deferred until after model loading
        # (validate_output_shape requires the model's n_features_in_ and feature_names_in_)

        # Validate sparsity if expected
        if config.expected_sparse is not None:
            try:
                # Check if output is sparse
                from scipy.sparse import issparse

                actual_sparse = issparse(X_transformed)
                validate_sparsity(actual_sparse, config.expected_sparse, config.artifact_path)
                logger.debug(f"Sparsity validation passed (expected_sparse={config.expected_sparse})")
            except ValueError as e:
                logger.error(f"Sparsity validation failed: {e}")
                raise

        # Capture feature names before and after preprocessing for reasons/recourse
        feature_names_before = list(X.columns)

        # Convert to DataFrame if needed
        if not isinstance(X_transformed, pd.DataFrame):
            # Get feature names from artifact
            feature_names = None
            if hasattr(artifact, "get_feature_names_out"):
                try:
                    feature_names = artifact.get_feature_names_out()
                except Exception:
                    pass

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

            X_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

        feature_names_after = list(X_transformed.columns)

        # Store preprocessing info in results
        self.results.execution_info["preprocessing"] = {
            "mode": "artifact",
            "artifact_path": str(config.artifact_path),
            "file_hash": actual_file_hash,
            "params_hash": actual_params_hash,
            "manifest": manifest,
            "feature_names_before": feature_names_before,
            "feature_names_after": feature_names_after,
            "feature_dtypes": {col: str(X[col].dtype) for col in X.columns},
        }

        # Add to manifest
        self.manifest_generator.add_component(
            "preprocessing",
            "artifact",
            artifact,
            details={
                "artifact_path": str(config.artifact_path),
                "file_hash": actual_file_hash,
                "params_hash": actual_params_hash,
            },
        )

        # Convert to DataFrame if needed
        if not isinstance(X_transformed, pd.DataFrame):
            # Get feature names from artifact
            feature_names = None
            if hasattr(artifact, "get_feature_names_out"):
                try:
                    feature_names = artifact.get_feature_names_out()
                except Exception:
                    pass

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

            X_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

        logger.info(f"Artifact preprocessing complete: {X_transformed.shape}")
        return X_transformed

    def _preprocess_auto(self, X: pd.DataFrame) -> pd.DataFrame:
        """Auto preprocessing (fallback mode - NOT compliant for production).

        Handles categorical features (like German Credit strings "< 0 DM") with OneHotEncoder
        to prevent ValueError: could not convert string to float during training.

        Args:
            X: Raw features DataFrame

        Returns:
            Processed features DataFrame suitable for training

        """
        # Use shared preprocessing utility
        X_processed = preprocess_auto(X)

        # Store preprocessing info for audit results
        categorical_cols = list(X.select_dtypes(include=["object"]).columns)
        numeric_cols = list(X.select_dtypes(exclude=["object"]).columns)

        # Capture feature names after preprocessing for reasons/recourse compatibility
        feature_names_before = list(X.columns)
        feature_names_after = list(X_processed.columns)

        self.results.execution_info["preprocessing"] = {
            "mode": "auto",
            "warning": "not_compliant",
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
            "feature_names_before": feature_names_before,
            "feature_names_after": feature_names_after,
            "feature_dtypes": {col: str(X[col].dtype) for col in X.columns},
        }

        return X_processed

    def _ensure_components_loaded(self) -> None:
        """Ensure all required components are imported and registered."""
        try:
            # Import explainer modules to trigger registration
            from glassalpha.explain import coefficients  # noqa: F401

            # Import SHAP explainers (may not be available)
            try:
                from glassalpha.explain.shap import kernel, tree  # noqa: F401
            except ImportError:
                pass  # SHAP is optional

            # Import metrics modules
            from glassalpha.metrics.fairness import bias_detection  # noqa: F401
            from glassalpha.metrics.performance import classification  # noqa: F401

            logger.debug("All component modules imported and registered")
        except ImportError as e:
            logger.warning(f"Some components could not be imported: {e}")

    def _get_explainer_class(self, explainer_name: str) -> type:
        """Get explainer class from explainer name.

        Args:
            explainer_name: Name of explainer (treeshap, kernelshap, coefficients)

        Returns:
            Explainer class

        Raises:
            RuntimeError: If explainer name is not recognized

        """
        explainer_map = {
            "treeshap": "glassalpha.explain.shap.tree.TreeSHAPExplainer",
            "kernelshap": "glassalpha.explain.shap.kernel.KernelSHAPExplainer",
            "coefficients": "glassalpha.explain.coefficients.CoefficientsExplainer",
        }

        if explainer_name not in explainer_map:
            raise RuntimeError(f"Unknown explainer: {explainer_name}")

        module_path, class_name = explainer_map[explainer_name].rsplit(".", 1)

        try:
            # Import the module dynamically
            import importlib

            module = importlib.import_module(module_path)
            explainer_class = getattr(module, class_name)
            return explainer_class
        except (ImportError, AttributeError) as e:
            raise RuntimeError(f"Failed to load explainer {explainer_name}: {e}") from e

    def _get_available_fairness_metrics(self) -> list[str]:
        """Get list of available fairness metric names from config.

        Returns:
            List of fairness metric names

        """
        # Try to get metrics from config
        if hasattr(self.config, "metrics"):
            metrics_config = self.config.metrics

            # Handle both nested structure (metrics.fairness.metrics) and legacy flat structure
            if hasattr(metrics_config, "fairness"):
                fairness_config = metrics_config.fairness

                logger.debug(f"Fairness config type: {type(fairness_config)}")
                logger.debug(f"Fairness config has metrics attr: {hasattr(fairness_config, 'metrics')}")

                # New structure: metrics.fairness.metrics = [...]
                if hasattr(fairness_config, "metrics") and fairness_config.metrics:
                    logger.debug(f"Returning metrics from config: {fairness_config.metrics}")
                    return fairness_config.metrics

                # Legacy structure: metrics.fairness = [...] (list directly)
                if isinstance(fairness_config, list):
                    logger.debug(f"Returning metrics as list: {fairness_config}")
                    return fairness_config

        # Fallback to default metrics if not configured
        logger.debug("Using default fairness metrics")
        return [
            "demographic_parity",
            "equal_opportunity",
            "equalized_odds",
            "predictive_parity",
        ]

    def _get_fairness_metric_class(self, metric_name: str) -> type:
        """Get fairness metric class from metric name.

        Args:
            metric_name: Name of the fairness metric

        Returns:
            Metric class

        Raises:
            RuntimeError: If metric name is not recognized

        """
        # Import metrics dynamically to avoid circular imports
        if metric_name == "demographic_parity":
            from glassalpha.metrics.fairness.bias_detection import DemographicParityMetric

            return DemographicParityMetric
        if metric_name == "equal_opportunity":
            from glassalpha.metrics.fairness.bias_detection import EqualOpportunityMetric

            return EqualOpportunityMetric
        if metric_name == "equalized_odds":
            from glassalpha.metrics.fairness.bias_detection import EqualizedOddsMetric

            return EqualizedOddsMetric
        if metric_name == "predictive_parity":
            from glassalpha.metrics.fairness.bias_detection import PredictiveParityMetric

            return PredictiveParityMetric
        raise RuntimeError(f"Unknown fairness metric: {metric_name}")

    def _resolve_dataset_path(self) -> Path:
        """Resolve dataset path with offline and fetch policy enforcement.

        This is the single source of truth for dataset access:
        - Honors offline mode (raises if file missing)
        - Respects fetch policy
        - Handles both custom and built-in datasets

        Returns:
            Path to the available dataset file

        Raises:
            FileNotFoundError: If dataset is not available and cannot be fetched
            ValueError: If configuration is invalid

        """
        from ..utils.cache_dirs import resolve_data_root

        cfg = self.config.data

        # Custom dataset: user provides explicit path
        if cfg.dataset == "custom":
            path = Path(cfg.path).expanduser().resolve()

            if path.exists():
                return path

            # File doesn't exist - check policy
            if cfg.offline:
                raise FileNotFoundError(
                    f"Offline mode: dataset file not found at {path} with fetch='{cfg.fetch}'. "
                    "Provide a local file or disable offline.",
                )

            if cfg.fetch in {None, "never"}:
                raise FileNotFoundError(
                    f"Dataset file not found at {path} and fetch='{cfg.fetch}'.",
                )

            # For custom paths, we don't have a fetch function
            raise FileNotFoundError(
                f"Custom data file not found: {path}\nPlease provide the file at this location.",
            )

        # Built-in dataset
        if cfg.dataset not in BUILT_IN_DATASETS:
            available = list(BUILT_IN_DATASETS.keys())
            raise ValueError(
                f"Unknown dataset key: {cfg.dataset}\n\n"
                f"Available datasets:\n  " + ", ".join(available) + "\n\n"
                "Use 'glassalpha list' to see all available components.",
            )

        cache_root = resolve_data_root()
        # Default cache path based on dataset name
        cache_path = (cache_root / f"{cfg.dataset}.csv").resolve()

        # Check if cached
        if cache_path.exists():
            return cache_path

        # Not cached - check if we can fetch
        if cfg.offline:
            raise FileNotFoundError(
                f"Offline mode: dataset '{cfg.dataset}' not cached at {cache_path}.",
            )

        if cfg.fetch == "never":
            raise FileNotFoundError(
                f"Dataset '{cfg.dataset}' not cached and fetch='never'.",
            )

        # Fetch and cache
        return self._fetch_and_cache_builtin(cfg.dataset, cache_path)

    def _fetch_and_cache_builtin(self, dataset_key: str, cache_path: Path) -> Path:
        """Fetch and cache a built-in dataset.

        Args:
            dataset_key: Dataset identifier in registry
            cache_path: Path to cache the dataset

        Returns:
            Path to the cached dataset file

        Raises:
            RuntimeError: If fetch fails

        """
        import shutil
        import time

        from ..utils.cache_dirs import ensure_dir_writable
        from ..utils.locks import file_lock, get_lock_path

        def _retry_io(fn, attempts=5, base_sleep=0.05):
            """Retry I/O operations with exponential backoff."""
            last_error = None
            sleep_time = base_sleep
            for attempt in range(attempts):
                try:
                    return fn()
                except OSError as e:
                    last_error = e
                    if attempt == attempts - 1:
                        raise
                    time.sleep(sleep_time)
                    sleep_time *= 2
            raise last_error

        import importlib

        # Get loader function for built-in dataset
        if dataset_key not in BUILT_IN_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_key}")

        module_name, func_name = BUILT_IN_DATASETS[dataset_key]
        module = importlib.import_module(module_name)
        loader = getattr(module, func_name)

        cache_root = ensure_dir_writable(cache_path.parent)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        lock_path = get_lock_path(cache_path)
        with file_lock(lock_path):
            # Double-check inside lock (another process may have fetched)
            if cache_path.exists():
                return cache_path

            # Fetch
            logger.info(f"Fetching dataset {dataset_key} into cache")
            produced = Path(loader(encoded=True)).resolve()
            tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")

            if produced != cache_path:
                # Move atomically into cache location via temp + replace
                _retry_io(lambda: shutil.move(str(produced), str(tmp)))
                _retry_io(lambda: os.replace(str(tmp), str(cache_path)))
            else:
                # Fetcher wrote directly to final location
                cache_path.touch(exist_ok=True)

        return cache_path

    def _ensure_dataset_availability(self, requested_path: Path) -> Path:
        """Ensure dataset is available, fetching if necessary.

        Args:
            requested_path: The path where the dataset should be located

        Returns:
            Path to the available dataset file

        Raises:
            FileNotFoundError: If dataset cannot be fetched or is not available

        """
        import shutil
        import time

        from ..utils.cache_dirs import ensure_dir_writable, resolve_data_root
        from ..utils.locks import file_lock, get_lock_path

        def _retry_io(fn, attempts=5, base_sleep=0.05):
            """Retry I/O operations with exponential backoff."""
            last_error = None
            sleep_time = base_sleep
            for attempt in range(attempts):
                try:
                    return fn()
                except OSError as e:
                    last_error = e
                    if attempt == attempts - 1:
                        raise
                    time.sleep(sleep_time)
                    sleep_time *= 2
            raise last_error  # Should never reach here, but for type checker

        cfg = self.config.data
        ds_key = cfg.dataset

        # Custom dataset: user path, just ensure parent exists if needed
        if ds_key == "custom":
            if not requested_path.exists():
                if cfg.offline:
                    raise FileNotFoundError(
                        f"Data file not found and offline is true.\n"
                        f"Path: {requested_path}\n"
                        f"Provide the file or set offline: false",
                    )
                # For custom paths, we don't fetch - user must provide
                raise FileNotFoundError(
                    f"Custom data file not found: {requested_path}\nPlease provide the file at this location.",
                )
            return requested_path

        # Built-in dataset: ensure cache exists, then mirror to requested if different
        cache_root = ensure_dir_writable(resolve_data_root())
        final_cache_path = (cache_root / f"{ds_key}.csv").resolve()
        final_cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file already exists and fetch policy allows it
        if final_cache_path.exists() and cfg.fetch != "always":
            return requested_path if requested_path == final_cache_path else final_cache_path

        # Check offline mode
        if cfg.offline:
            if not final_cache_path.exists():
                raise FileNotFoundError(
                    f"Data file not found and offline is true.\n"
                    f"Cache: {final_cache_path}\n"
                    f"Run with offline: false to download, or provide the file.",
                )
            return final_cache_path

        lock_path = get_lock_path(final_cache_path)
        with file_lock(lock_path):
            # Respect fetch policy inside lock so only one thread refreshes
            if cfg.fetch == "always":
                try:
                    final_cache_path.unlink()
                except FileNotFoundError:
                    pass

            # Fetch if needed
            if not final_cache_path.exists():
                logger.info(f"Fetching dataset {spec.key} into cache")
                produced = Path(spec.fetch_fn()).resolve()
                tmp = final_cache_path.with_suffix(final_cache_path.suffix + ".tmp")

                if produced != final_cache_path:
                    # Move atomically into cache location via temp + replace
                    _retry_io(lambda: shutil.move(str(produced), str(tmp)))
                    _retry_io(lambda: os.replace(str(tmp), str(final_cache_path)))
                else:
                    # Fetcher wrote directly to final location
                    final_cache_path.touch(exist_ok=True)

            # Mirror to requested path if different (inside lock to avoid races)
            if requested_path != final_cache_path:
                requested_path.parent.mkdir(parents=True, exist_ok=True)

                def _mirror():
                    if requested_path.exists():
                        return  # Already mirrored
                    try:
                        # Try hard link first (fastest, zero copy)
                        os.link(str(final_cache_path), str(requested_path))
                    except OSError:
                        # Fall back to copy (cross-device or unsupported filesystem)
                        shutil.copy2(str(final_cache_path), str(requested_path))

                _retry_io(_mirror)

        return requested_path


def run_audit_pipeline(
    config: AuditConfig,
    progress_callback: Callable | None = None,
    selected_explainer: str | None = None,
    requested_model: str | None = None,
) -> AuditResults:
    """Convenience function to run audit pipeline.

    Args:
        config: Validated audit configuration
        progress_callback: Optional progress callback function
        selected_explainer: Pre-selected explainer name to use (avoids re-selection)
        requested_model: Originally requested model type (for fallback tracking)

    Returns:
        Audit results

    """
    kwargs = {}
    if selected_explainer is not None:
        kwargs["selected_explainer"] = selected_explainer
    if requested_model is not None:
        kwargs["requested_model"] = requested_model

    pipeline = AuditPipeline(config, **kwargs)
    return pipeline.run(progress_callback)
