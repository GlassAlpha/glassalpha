"""Template rendering engine for audit reports.

This module provides the core template rendering functionality for generating
professional audit reports from audit results. It handles data preparation,
template processing, and asset management.
"""

import base64
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import jinja2

# Lazy import matplotlib to avoid dependency for HTML-only reports
try:
    from matplotlib.figure import Figure

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    Figure = None  # type: ignore
    _MATPLOTLIB_AVAILABLE = False

# Friend's spec: Use importlib.resources for template loading from installed package
try:
    from importlib.resources import files

    IMPORTLIB_RESOURCES_AVAILABLE = True
except ImportError:
    # Fallback for older Python versions
    IMPORTLIB_RESOURCES_AVAILABLE = False
    files = None

from glassalpha.pipeline.audit import AuditResults

from .plots import AuditPlotter, create_fairness_plots, create_performance_plots, create_shap_plots

logger = logging.getLogger(__name__)


def validate_output_path(output_path: Path) -> None:
    """Validate output path before attempting to save.

    Args:
        output_path: Path to validate

    Raises:
        ValueError: If path is invalid or too long

    """
    path_str = str(output_path.resolve())
    filename = output_path.name

    # Check filename length (most filesystems have 255 char limit)
    if len(filename) > 255:
        raise ValueError(
            f"❌ Filename too long: {len(filename)} characters (limit: 255)\n\n"
            f"Current filename: {filename[:50]}...\n\n"
            f"💡 Use a shorter filename:\n"
            f"   --output audit_report.html\n"
            f"   --output report_20251004.html",
        )

    # Check full path length (be conservative for Windows compatibility)
    # Windows typically has 260 char limit, Linux/Mac ~4096
    if len(path_str) > 4000:  # Leave some margin
        raise ValueError(
            f"❌ Path too long: {len(path_str)} characters\n\n"
            f"System limit: ~4096 characters (varies by OS)\n"
            f"Windows limit: 260 characters (unless long paths enabled)\n\n"
            f"💡 Use a shorter path:\n"
            f"   --output ~/audit.html\n"
            f"   --output /tmp/report.html",
        )


class AuditReportRenderer:
    """Professional template renderer for audit reports."""

    def __init__(self, template_dir: Path | None = None) -> None:
        """Initialize the audit report renderer.

        Args:
            template_dir: Directory containing Jinja2 templates

        """
        # Friend's spec: Use package resources for template loading from installed package
        if template_dir is None:
            # Try to use PackageLoader which is designed for package resources
            try:
                loader = jinja2.PackageLoader("glassalpha.report", "templates")
                self.template_dir = Path("glassalpha.report.templates")  # Virtual path for logging
                logger.debug("Using PackageLoader for template resources")
            except (ImportError, AttributeError, ValueError):
                # Fall back to filesystem loader
                self.template_dir = Path(__file__).parent / "templates"
                loader = jinja2.FileSystemLoader(str(self.template_dir))
                logger.debug(f"Falling back to FileSystemLoader: {self.template_dir}")
        else:
            # Use provided template_dir with filesystem loader
            self.template_dir = template_dir
            loader = jinja2.FileSystemLoader(str(self.template_dir))
            logger.debug(f"Using provided template directory: {template_dir}")

        # Configure Jinja2 environment
        self.env = jinja2.Environment(
            loader=loader,
            autoescape=jinja2.select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add custom filters
        self.env.filters["format_number"] = self._format_number
        self.env.filters["format_percentage"] = self._format_percentage
        self.env.filters["format_datetime"] = self._format_datetime
        self.env.filters["unique"] = lambda x: list(set(x)) if x else []
        self.env.filters["safe_get"] = lambda d, k, default=None: d.get(k, default) if isinstance(d, dict) else default

        # Initialize plotter
        self.plotter = AuditPlotter(style="professional")

        logger.info(f"Initialized AuditReportRenderer with template directory: {self.template_dir}")

    def render_audit_report(
        self,
        audit_results: AuditResults,
        template_name: str = "standard_audit.html",
        output_path: Path | None = None,
        *,
        embed_plots: bool = True,
        compact: bool = False,
        **template_vars: Any,
    ) -> str:
        """Render complete audit report from audit results.

        Args:
            audit_results: Complete audit results from pipeline
            template_name: Name of the template file to use
            output_path: Optional path to save rendered HTML
            embed_plots: Whether to embed plots as base64 images
            compact: If True, exclude individual fairness matched pairs from HTML (saves ~50-80MB)
            **template_vars: Additional variables to pass to template: Any

        Returns:
            Rendered HTML content as string

        """
        logger.info(f"Rendering audit report using template: {template_name}")

        # Prepare template context
        context = self._prepare_template_context(audit_results, embed_plots=embed_plots, compact=compact)
        context.update(template_vars)

        # Add guard to prevent template rendering errors
        self._validate_template_context(context)

        # Load and render template
        template = self.env.get_template(template_name)
        rendered_html = template.render(**context)

        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            validate_output_path(output_path)  # Validate before attempting to save
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as f:
                f.write(rendered_html)

            logger.info(f"Saved rendered report to: {output_path}")

        return rendered_html

    def _prepare_template_context(
        self,
        audit_results: AuditResults,
        *,
        embed_plots: bool = True,
        compact: bool = False,
    ) -> dict[str, Any]:
        """Prepare comprehensive template context from audit results.

        Args:
            audit_results: Audit results to process
            embed_plots: Whether to create and embed plots
            compact: If True, exclude individual fairness matched pairs from HTML

        Returns:
            Dictionary with template context variables

        """
        logger.debug("Preparing template context from audit results")

        # Use centralized context building for consistency and safety
        context = self._build_report_context(audit_results, embed_plots=embed_plots, compact=compact)

        # Add metadata and descriptions
        context.update(
            {
                "metric_descriptions": self._get_metric_descriptions(),
                "plot_descriptions": self._get_plot_descriptions(),
                "shap_descriptions": self._get_shap_descriptions(),
            },
        )

        logger.debug(f"Template context prepared with {len(context)} variables")
        return context

    def _validate_template_context(self, context: dict[str, Any]) -> None:
        """Validate template context to prevent rendering errors.

        Args:
            context: Template context dictionary

        Raises:
            TypeError: If context contains invalid types for template rendering

        """
        # Check that metrics is a dict if present
        if "metrics" in context and not isinstance(context["metrics"], dict):
            raise TypeError("Template context error: 'metrics' must be a dict")

        # Check that model_performance is a dict if present
        if "model_performance" in context and not isinstance(context["model_performance"], dict):
            raise TypeError("Template context error: 'model_performance' must be a dict")

        # Check that fairness_analysis is a dict if present
        if "fairness_analysis" in context and not isinstance(context["fairness_analysis"], dict):
            raise TypeError("Template context error: 'fairness_analysis' must be a dict")

        # Check that drift_analysis is a dict if present
        if "drift_analysis" in context and not isinstance(context["drift_analysis"], dict):
            raise TypeError("Template context error: 'drift_analysis' must be a dict")

        # Check that feature_importances is a dict with float values if present
        if "feature_importances" in context:
            if not isinstance(context["feature_importances"], dict):
                raise TypeError("Template context error: 'feature_importances' must be a dict")

            # Check that all values are numeric (int or float)
            for feature, importance in context["feature_importances"].items():
                if not isinstance(importance, (int, float)):
                    raise TypeError(
                        f"Template context error: 'feature_importances' values must be numeric, got {type(importance)} for feature '{feature}'",
                    )

    def _build_report_context(
        self,
        audit_results: AuditResults,
        *,
        embed_plots: bool = True,
        compact: bool = False,
    ) -> dict[str, Any]:
        """Build normalized report context from audit results.

        Args:
            audit_results: Audit results to process
            embed_plots: Whether to create and embed plots
            compact: If True, exclude large data structures (matched pairs)

        Returns:
            Normalized context dictionary safe for template rendering

        """
        from .context import normalize_audit_context

        # Get base context with proper normalization
        context = normalize_audit_context(audit_results, compact=compact)

        # Determine generation date (fixed when SOURCE_DATE_EPOCH is set)
        from glassalpha.utils.determinism import get_deterministic_timestamp

        # Get seed from audit results for deterministic timestamp
        seed = audit_results.execution_info.get("random_seed") if audit_results.execution_info else None

        generation_timestamp = get_deterministic_timestamp(seed=seed)
        generation_date = generation_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")

        # Add basic audit information
        context.update(
            {
                "report_title": "Machine Learning Model Audit Report",
                "generation_date": generation_date,
                "version": "1.0.0",
            },
        )

        # Extract audit metadata
        if audit_results.execution_info:
            context.update(
                {
                    "audit_id": audit_results.execution_info.get("audit_id"),
                    "audit_profile": audit_results.execution_info.get("audit_profile"),
                    "strict_mode": audit_results.execution_info.get("strict_mode", False),
                    "preprocessing_info": audit_results.execution_info.get("preprocessing"),
                    "preprocessing_unknown_rates": audit_results.execution_info.get("preprocessing_unknown_rates"),
                },
            )

        # Generate and embed plots if requested
        if embed_plots:
            plots = self._generate_plots(audit_results)
            context.update(
                {
                    "shap_plots": plots.get("shap", {}),
                    "performance_plots": plots.get("performance", {}),
                    "fairness_plots": plots.get("fairness", {}),
                },
            )

        return context

    def _generate_plots(self, audit_results: AuditResults) -> dict[str, dict[str, str]]:
        """Generate plots and return as base64-encoded data URLs.

        Args:
            audit_results: Audit results containing plot data

        Returns:
            Dictionary with plot categories and base64 data URLs

        """
        logger.debug("Generating plots for template embedding")
        plots = {"shap": {}, "performance": {}, "fairness": {}}

        try:
            # Generate SHAP plots
            if audit_results.explanations:
                feature_names = list(audit_results.schema_info.get("features", [])) if audit_results.schema_info else []
                shap_figures = create_shap_plots(audit_results.explanations, feature_names)

                for plot_name, figure in shap_figures.items():
                    plots["shap"][plot_name] = self._figure_to_base64(figure)

                logger.debug(f"Generated {len(shap_figures)} SHAP plots")

            # Generate performance plots
            if audit_results.model_performance:
                perf_figures = create_performance_plots(audit_results.model_performance)

                for plot_name, figure in perf_figures.items():
                    plots["performance"][plot_name] = self._figure_to_base64(figure)

                logger.debug(f"Generated {len(perf_figures)} performance plots")

            # Generate fairness plots
            if audit_results.fairness_analysis:
                fairness_figures = create_fairness_plots(audit_results.fairness_analysis)

                for plot_name, figure in fairness_figures.items():
                    plots["fairness"][plot_name] = self._figure_to_base64(figure)

                logger.debug(f"Generated {len(fairness_figures)} fairness plots")

        except Exception as e:
            logger.warning(f"Failed to generate some plots: {e}")

        return plots

    def _figure_to_base64(self, figure: Any) -> str:
        """Convert matplotlib figure to base64 data URL.

        Args:
            figure: Matplotlib figure object

        Returns:
            Base64 data URL string for embedding in HTML

        Raises:
            ImportError: If matplotlib not available

        """
        if not _MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for plot generation. "
                "Install with: pip install 'glassalpha[all]' or pip install matplotlib",
            )

        buffer = BytesIO()
        # Reduce DPI from 150 to 96 for smaller file size (still looks good on screen)
        # Use 'tight' bbox_inches to minimize whitespace
        # Note: 'optimize' parameter requires matplotlib >= 3.10, so we omit it
        figure.savefig(
            buffer,
            format="png",
            dpi=96,  # Reduced from 150 to 96 (standard screen DPI)
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        buffer.seek(0)

        # Try to compress PNG with PIL if available (reduces file size 30-50%)
        try:
            from PIL import Image

            # Load PNG from buffer
            img = Image.open(buffer)
            compressed_buffer = BytesIO()
            # Save with optimize=True (PIL supports this)
            img.save(compressed_buffer, format="PNG", optimize=True)
            compressed_buffer.seek(0)
            image_data = compressed_buffer.getvalue()
            compressed_buffer.close()
        except ImportError:
            # PIL not available, use uncompressed PNG
            image_data = buffer.getvalue()

        buffer.close()

        # Encode as base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Create data URL
        return f"data:image/png;base64,{image_base64}"

    def _get_metric_descriptions(self) -> dict[str, str]:
        """Get descriptions for performance metrics."""
        return {
            "accuracy": "Overall classification accuracy across all classes",
            "precision": "Proportion of positive predictions that were correct",
            "recall": "Proportion of actual positives that were correctly identified",
            "f1": "Harmonic mean of precision and recall",
            "auc_roc": "Area under the ROC curve - discrimination ability",
            "classification_report": "Comprehensive performance summary",
            "demographic_parity": "Equal positive prediction rates across groups",
            "equal_opportunity": "Equal true positive rates across groups",
            "equalized_odds": "Equal TPR and FPR across groups",
            "predictive_parity": "Equal precision across groups",
        }

    def _get_plot_descriptions(self) -> dict[str, str]:
        """Get descriptions for different plot types."""
        return {
            "summary": "Overall performance metrics dashboard",
            "confusion_matrix": "True vs predicted labels breakdown",
            "roc_curve": "True positive rate vs false positive rate",
            "feature_importance": "Most influential features for predictions",
        }

    def _get_shap_descriptions(self) -> dict[str, str]:
        """Get descriptions for SHAP plot types."""
        return {
            "global_importance": "Average impact of each feature across all predictions",
            "summary": "Distribution of feature impacts across all samples",
            "waterfall": "Step-by-step explanation of individual prediction",
        }

    def _format_number(self, value: Any, decimals: int = 3) -> str:
        """Format numeric values for display."""
        if value is None:
            return "N/A"

        if isinstance(value, (int, float)):
            if decimals == 0:
                return f"{value:,.0f}"
            return f"{value:.{decimals}f}"

        return str(value)

    def _format_percentage(self, value: Any, decimals: int = 1) -> str:
        """Format values as percentages."""
        if value is None:
            return "N/A"

        if isinstance(value, (int, float)):
            return f"{value * 100:.{decimals}f}%"

        return str(value)

    def _format_datetime(self, value: Any, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format datetime values."""
        if value is None:
            return "N/A"

        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                return dt.strftime(format_string)
            except ValueError:
                return value

        if hasattr(value, "strftime"):
            return value.strftime(format_string)

        return str(value)

    def _normalize_metrics(self, metrics: Any) -> dict[str, Any]:
        """Normalize metrics for template compatibility.

        Contract compliance: Handle metrics that may be numbers or dicts.
        Template expects metrics[key] to have attributes, but sometimes
        metrics[key] is just a float (e.g. accuracy: 0.868).

        Args:
            metrics: Raw metrics from audit results

        Returns:
            Normalized metrics where values are always dicts

        """
        if not metrics:
            return {}

        mp = dict(metrics) if isinstance(metrics, dict) else {}

        # Normalize each metric - if it's a number, wrap it for template access
        for k, v in list(mp.items()):
            if isinstance(v, (int, float)):
                mp[k] = {"value": float(v)}

        return mp

    def _normalize_explanations(self, explanations: dict[str, Any]) -> dict[str, Any]:
        """Normalize explanations to ensure all importances are scalar floats for template rendering.

        Args:
            explanations: Raw explanations dictionary

        Returns:
            Normalized explanations with scalar importance values

        """
        if not explanations:
            return explanations

        # Create a copy to avoid modifying the original
        normalized = explanations.copy()

        # Normalize global_importance if it exists
        if "global_importance" in normalized:
            normalized["global_importance"] = self._normalize_importance_dict(
                normalized["global_importance"],
            )

        return normalized

    def _normalize_importance_dict(self, importance_dict: dict[str, Any]) -> dict[str, float]:
        """Normalize importance dictionary to ensure all values are scalar floats.

        Args:
            importance_dict: Dictionary with feature names and importance values

        Returns:
            Dictionary with scalar float importance values

        """
        if not importance_dict:
            return {}

        normalized = {}
        for feature, importance in importance_dict.items():
            try:
                # Convert to numpy array and take mean if it's multi-dimensional
                import numpy as np

                arr = np.asarray(importance)
                if arr.ndim == 0:
                    # Already a scalar
                    normalized[feature] = float(arr)
                else:
                    # Multi-dimensional - take mean to get scalar
                    normalized[feature] = float(np.mean(arr))
            except (TypeError, ValueError, AttributeError):
                # Fallback - try to convert directly to float
                try:
                    normalized[feature] = float(importance)
                except (TypeError, ValueError):
                    # If all else fails, use 0.0
                    normalized[feature] = 0.0
                    logger.warning(f"Could not normalize importance for feature {feature}: {importance}")

        return normalized


def render_audit_report(
    audit_results: AuditResults,
    output_path: Path | None = None,
    template_name: str = "standard_audit.html",
    compact: bool = False,
    **template_vars: Any,
) -> str:
    """Convenience function to render audit report.

    Args:
        audit_results: Complete audit results from pipeline
        output_path: Optional path to save HTML file
        template_name: Template file to use
        compact: If True, exclude matched pairs from HTML (saves ~50-80MB)
        **template_vars: Additional template variables: Any

    Returns:
        Rendered HTML content

    """
    renderer = AuditReportRenderer()
    return renderer.render_audit_report(
        audit_results=audit_results,
        output_path=output_path,
        template_name=template_name,
        compact=compact,
        **template_vars,
    )
