"""Tabular compliance audit profile for Phase 1.

This profile defines the component set for regulatory compliance audits
of tabular machine learning models (XGBoost, LightGBM, LogisticRegression).
"""

import logging
from typing import Any

from .registry import ProfileRegistry

logger = logging.getLogger(__name__)


@ProfileRegistry.register("tabular_compliance", priority=10)
class TabularComplianceProfile:
    """Profile for tabular model compliance audits.

    This is the primary profile for Phase 1, focusing on tree-based models
    and their regulatory compliance requirements.
    """

    name = "tabular_compliance"
    version = "1.0.0"

    # Compatible models for Phase 1
    compatible_models = [
        "xgboost",
        "lightgbm",
        "logistic_regression",
        "random_forest",
        "passthrough",  # For testing
    ]

    # Required metrics for compliance
    required_metrics = [
        # Performance metrics
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        # Fairness metrics
        "demographic_parity",
        "equal_opportunity",
        "statistical_parity",
        # Drift metrics
        "psi",  # Population Stability Index
        "kl_divergence",
    ]

    # Optional metrics that can be included
    optional_metrics = [
        "confusion_matrix",
        "calibration",
        "feature_importance",
        "partial_dependence",
        "noop",  # For testing
    ]

    # Report template for tabular audits
    report_template = "standard_audit"

    # Explainer priority for tabular models
    explainer_priority = [
        "treeshap",  # Best for tree models
        "kernelshap",  # Fallback for any model
        "noop",  # Last resort for testing
    ]

    @staticmethod
    def validate_config(config: dict[str, Any]) -> bool:
        """Validate configuration for tabular compliance.

        Args:
            config: Configuration to validate

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid

        """
        # Run validation (no parent class now)

        # Additional tabular-specific validation
        data_config = config.get("data", {})

        # Check for protected attributes (needed for fairness) - warn if missing but don't fail
        if "protected_attributes" not in data_config:
            logger.warning(
                "Profile 'tabular_compliance' recommends 'protected_attributes' in data configuration for fairness analysis",
            )

        # Check for schema (needed for determinism) - warn if missing but don't fail
        if "schema_path" not in data_config and "data_schema" not in data_config:
            logger.warning("Profile 'tabular_compliance' recommends data schema for deterministic validation")

        # Check explainer configuration
        explainer_config = config.get("explainers", {})
        if not explainer_config.get("priority"):
            # Provide default if not specified
            # Order: model-specific (coefficients for linear) → SHAP (if available) → permutation (fallback) → noop (last resort)
            config["explainers"] = config.get("explainers", {})
            config["explainers"]["priority"] = ["coefficients", "treeshap", "kernelshap", "permutation", "noop"]

        # Check metrics configuration
        metrics_config = config.get("metrics", {})
        if not metrics_config:
            # Set default metrics if not specified
            config["metrics"] = {
                "performance": ["accuracy", "precision", "recall", "f1", "auc_roc"],
                "fairness": ["demographic_parity", "equal_opportunity"],
                "drift": ["psi"],
            }

        # Validate recourse configuration if present
        if "recourse" in config:
            recourse_config = config["recourse"]
            if recourse_config.get("enabled", False) and "immutable_features" not in recourse_config:
                raise ValueError("Recourse requires 'immutable_features' to be specified")

        return True

    @staticmethod
    def apply_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
        """Apply tabular compliance defaults to configuration.

        Args:
            cfg: User configuration dictionary

        Returns:
            Configuration with defaults applied

        """
        out = dict(cfg)

        # Explainers - prefer model-specific (coefficients for linear), then SHAP, then zero-dep fallbacks
        # Order: coefficients (linear models) → treeshap (tree models) → kernelshap (general) → permutation (fallback) → noop (last resort)
        if "explainers" not in out or not out["explainers"].get("priority"):
            out["explainers"] = {
                "strategy": "first_compatible",
                "priority": ["coefficients", "treeshap", "kernelshap", "permutation", "noop"],
            }
        elif "explainers" in out:
            # Update existing explainer config with defaults
            existing = out["explainers"]
            existing.setdefault("strategy", "first_compatible")
            existing.setdefault("priority", ["coefficients", "treeshap", "kernelshap", "permutation", "noop"])

        # Metrics - performance and fairness
        if "metrics" not in out:
            out["metrics"] = {
                "performance": ["accuracy", "precision", "recall", "f1", "auc_roc"],
                "fairness": ["demographic_parity", "equal_opportunity"],
                "drift": ["psi", "kl_divergence"],
            }
        else:
            # Update existing metrics config with defaults
            existing = out["metrics"]
            existing.setdefault("performance", ["accuracy", "precision", "recall", "f1", "auc_roc"])
            existing.setdefault("fairness", ["demographic_parity", "equal_opportunity"])
            existing.setdefault("drift", ["psi", "kl_divergence"])

        # Report template
        if "report" not in out:
            out["report"] = {
                "template": "standard_audit.html",
                # Don't set output_format - let CLI detect from file extension
            }
        else:
            # Update existing report config with defaults
            existing = out["report"]
            existing.setdefault("template", "standard_audit.html")
            # Don't override output_format if user explicitly set it
            # existing.setdefault("output_format", "pdf")  # Removed - let CLI detect from extension

        # Reproducibility settings
        if "reproducibility" not in out:
            out["reproducibility"] = {
                "random_seed": 42,
                "deterministic": True,
                "capture_environment": True,
            }
        else:
            # Update existing reproducibility config with defaults
            existing = out["reproducibility"]
            existing.setdefault("random_seed", 42)
            existing.setdefault("deterministic", True)
            existing.setdefault("capture_environment", True)

        return out

    @staticmethod
    def required_extras(cfg: dict[str, Any]) -> list[str]:
        """Get required extras for this profile.

        Args:
            cfg: Configuration dictionary

        Returns:
            List of required extra dependency groups

        """
        # Tabular compliance typically needs the tabular stack
        return ["tabular"]

    @staticmethod
    def get_default_config() -> dict[str, Any]:
        """Get default configuration for this profile.

        Returns:
            Default configuration dictionary

        """
        return {
            "audit_profile": "tabular_compliance",
            "model": {
                "type": "xgboost",  # Most common
                "path": None,  # Must be provided
            },
            "data": {
                "path": None,  # Must be provided
                "schema_path": None,  # Must be provided
                "protected_attributes": [],  # Must be provided
            },
            "explainers": {
                "strategy": "first_compatible",
                "priority": ["treeshap", "kernelshap", "noop"],
                "config": {
                    "treeshap": {
                        "max_samples": 1000,
                        "check_additivity": True,
                    },
                    "kernelshap": {
                        "n_samples": 100,
                    },
                },
            },
            "metrics": {
                "performance": ["accuracy", "precision", "recall", "f1", "auc_roc"],
                "fairness": ["demographic_parity", "equal_opportunity"],
                "drift": ["psi"],
            },
            "recourse": {
                "enabled": True,
                "immutable_features": [],
                "monotonic_constraints": {},
                "cost_function": "weighted_l1",
            },
            "report": {
                "template": "standard_audit.html",
                "output_format": "pdf",
                "include_sections": [
                    "lineage",
                    "data_schema",
                    "global_explanations",
                    "local_explanations",
                    "fairness",
                    "drift",
                    "recourse",
                    "assumptions",
                ],
            },
            "reproducibility": {
                "random_seed": 42,
                "deterministic": True,
                "capture_environment": True,
            },
            "manifest": {
                "enabled": True,
                "include_git_sha": True,
                "include_config_hash": True,
                "include_data_hash": True,
            },
            "strict_mode": False,
        }
