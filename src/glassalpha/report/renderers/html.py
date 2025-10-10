# SPDX-License-Identifier: Apache-2.0
"""HTML renderer using PackageLoader - fixes template packaging issues.

This module ensures templates are loaded correctly from installed wheels
using Jinja2's PackageLoader instead of filesystem paths.
"""

import logging
import os

from jinja2 import Environment, PackageLoader, select_autoescape

from glassalpha.pipeline.audit import AuditResults

logger = logging.getLogger(__name__)


class HTMLRenderer:
    """HTML renderer with proper PackageLoader for wheel compatibility."""

    def __init__(self) -> None:
        """Initialize HTML renderer with PackageLoader."""
        # Use PackageLoader - designed for installed packages/wheels
        self.env = Environment(
            loader=PackageLoader("glassalpha.report", "templates"),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Friend's spec: Optional assert for packaging validation
        if os.getenv("GLASSALPHA_ASSERT_PACKAGING") == "1":
            try:
                from importlib.resources import files  # noqa: PLC0415

                assert files("glassalpha.report.templates").joinpath("standard_audit.html").is_file()  # noqa: S101
            except (ImportError, AssertionError) as e:
                msg = f"Template packaging validation failed: {e}"
                raise RuntimeError(msg) from e

        logger.info("Initialized HTML renderer with PackageLoader")

    def render_template(self, template_name: str, context: dict) -> str:
        """Render template with given context.

        Args:
            template_name: Name of template file
            context: Template context dictionary

        Returns:
            Rendered HTML string

        """
        template = self.env.get_template(template_name)
        return template.render(**context)

    def prepare_metrics_context(self, audit_results: AuditResults) -> dict:
        """Prepare metrics as dicts (not scalars) to prevent 'float has no attribute' errors.

        Args:
            audit_results: Audit results with metrics

        Returns:
            Context dictionary with metrics as mappings

        """
        context = {}

        # Ensure performance metrics are dicts, not scalars
        if audit_results.model_performance:
            if isinstance(audit_results.model_performance, dict):
                # Convert any scalar values to proper dict format
                performance = {}
                for key, value in audit_results.model_performance.items():
                    if isinstance(value, (int, float)):
                        performance[key] = {"value": float(value), "formatted": f"{value:.3f}"}
                    else:
                        performance[key] = value
                context["performance"] = performance
            else:
                # Handle case where performance is a single scalar (shouldn't happen but be safe)
                context["performance"] = {"accuracy": {"value": 0.0, "formatted": "N/A"}}
                logger.warning("model_performance was not a dict, using default")

        # Same approach for other metric categories
        if audit_results.fairness_analysis:
            context["fairness"] = audit_results.fairness_analysis

        if audit_results.drift_analysis:
            context["drift"] = audit_results.drift_analysis

        return context
