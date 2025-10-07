"""Context normalization utilities for audit report rendering.

This module provides utilities to normalize audit results for template
rendering, ensuring consistent data structures that prevent Jinja2 errors.

Design Note: Dual Metric Paths
-------------------------------
The system intentionally supports two metric structures:

1. **Pipeline Path (Production)**: Rich semantic structure
   - Example: {"accuracy": {"accuracy": 0.87, "n_samples": 100}}
   - Provides full audit context (value + metadata)
   - Used by AuditPipeline for complete audit trails

2. **Normalization Path (Tests/Mocks)**: Minimal safe structure
   - Example: {"accuracy": {"value": 0.87}}
   - Sanitizes arbitrary input for template safety
   - Used by tests, external integrations, edge cases

Templates handle BOTH via defensive extraction (checks .value, then .accuracy, then direct).
This separation is intentional: don't consolidate unless you want to lose audit richness OR
force verbose test structures.
"""

from typing import Any


def normalize_metrics(metrics: Any) -> dict[str, Any]:  # noqa: ANN401
    """Normalize metrics for template compatibility.

    Contract compliance: Handle metrics that may be numbers or dicts.
    Template expects metrics[key] to have accessible attributes/values,
    but sometimes metrics[key] is just a float (e.g. accuracy: 0.868).

    This prevents "TypeError: 'float' object has no attribute 'accuracy'"
    errors in Jinja2 templates.

    Args:
        metrics: Raw metrics from audit results (dict or None)

    Returns:
        Normalized metrics where values are always accessible as objects

    Example:
        Input: {"accuracy": 0.868, "f1": {"value": 0.75, "details": {...}}}
        Output: {"accuracy": {"value": 0.868}, "f1": {"value": 0.75, "details": {...}}}

    """
    if not metrics:
        return {}

    normalized = dict(metrics) if isinstance(metrics, dict) else {}

    # Normalize each metric - if it's a number, wrap it for template access
    for key, value in list(normalized.items()):
        if isinstance(value, (int, float)):
            normalized[key] = {"value": float(value)}

    return normalized


def normalize_audit_context(audit_results: Any) -> dict[str, Any]:  # noqa: ANN401
    """Normalize complete audit results for template rendering.

    Args:
        audit_results: Audit results object with various attributes

    Returns:
        Normalized context dict safe for template rendering

    """
    context = {}

    # Safely extract and normalize metrics with explicit naming
    if hasattr(audit_results, "model_performance"):
        context["model_performance"] = normalize_metrics(audit_results.model_performance)
        context["performance_metrics"] = context["model_performance"]  # Compatibility alias

        # E10+: Extract calibration CIs to top level for template convenience
        if isinstance(audit_results.model_performance, dict):
            calibration_ci = audit_results.model_performance.get("calibration_ci")
            if calibration_ci and isinstance(calibration_ci, dict) and "error" not in calibration_ci:
                context["calibration_ci"] = calibration_ci

    if hasattr(audit_results, "fairness_analysis"):
        context["fairness_analysis"] = audit_results.fairness_analysis
        context["fairness_metrics"] = context["fairness_analysis"]  # Compatibility alias

        # Extract nested features for template convenience
        if isinstance(audit_results.fairness_analysis, dict):
            # E10: Group fairness confidence intervals
            # Extract CIs for each metric (demographic_parity_ci, equal_opportunity_ci, etc.)
            fairness_cis = {}
            for key, value in audit_results.fairness_analysis.items():
                if key.endswith("_ci") and isinstance(value, dict):
                    fairness_cis[key] = value
            if fairness_cis:
                context["fairness_confidence_intervals"] = fairness_cis

            # E5.1: Intersectional fairness
            intersectional = audit_results.fairness_analysis.get("intersectional")
            if intersectional:
                context["intersectional_fairness"] = intersectional

            # E11: Individual fairness
            individual = audit_results.fairness_analysis.get("individual_fairness")
            if individual and isinstance(individual, dict) and "error" not in individual:
                # Flatten nested structure for template
                flattened = {}

                # Extract consistency score fields
                if "consistency_score" in individual and isinstance(individual["consistency_score"], dict):
                    consistency = individual["consistency_score"]
                    flattened["consistency_score"] = consistency.get("consistency_score", 0.0)

                # Extract matched pairs fields
                if "matched_pairs" in individual and isinstance(individual["matched_pairs"], dict):
                    matched = individual["matched_pairs"]
                    matched_pairs_list = matched.get("matched_pairs", [])
                    flattened["matched_pairs_count"] = len(matched_pairs_list)

                    # Calculate avg and max prediction diff from matched pairs
                    if matched_pairs_list:
                        diffs = [p.get("prediction_diff", 0.0) for p in matched_pairs_list]
                        flattened["avg_prediction_diff"] = sum(diffs) / len(diffs) if diffs else 0.0
                        flattened["max_prediction_diff"] = max(diffs) if diffs else 0.0
                    else:
                        flattened["avg_prediction_diff"] = 0.0
                        flattened["max_prediction_diff"] = 0.0

                # Extract flip test fields
                if "flip_test" in individual and isinstance(individual["flip_test"], dict):
                    flip = individual["flip_test"]
                    flip_changes = flip.get("flip_changes_prediction", [])
                    flattened["flip_test_violations"] = len(flip_changes)

                context["individual_fairness"] = flattened

            # E12: Dataset bias
            dataset_bias = audit_results.fairness_analysis.get("dataset_bias")
            if dataset_bias and isinstance(dataset_bias, dict) and "error" not in dataset_bias:
                context["dataset_bias"] = dataset_bias

    if hasattr(audit_results, "drift_analysis"):
        context["drift_analysis"] = audit_results.drift_analysis
        context["drift_metrics"] = context["drift_analysis"]  # Compatibility alias

    # E6+: Extract stability analysis (perturbation sweeps)
    if hasattr(audit_results, "stability_analysis"):
        stability = audit_results.stability_analysis
        if stability and isinstance(stability, dict):
            context["stability_analysis"] = stability
            # Extract perturbation results for template convenience
            if "robustness_score" in stability:
                context["perturbation_results"] = stability

    # Safely extract other results with defaults
    context.update(
        {
            "explanations": getattr(audit_results, "explanations", {}),
            "data_summary": getattr(audit_results, "data_summary", {}),
            "schema_info": getattr(audit_results, "schema_info", {}),
            "model_info": getattr(audit_results, "model_info", {}),
            "selected_components": getattr(audit_results, "selected_components", {}),
            "execution_info": getattr(audit_results, "execution_info", {}),
            "manifest": getattr(audit_results, "manifest", {}),
            "success": getattr(audit_results, "success", False),
            "error_message": getattr(audit_results, "error_message", None),
        },
    )

    # Add feature_importances mapping for template compatibility
    explanations = context.get("explanations", {})
    if "global_importance" in explanations:
        context["feature_importances"] = explanations["global_importance"]

    return context


def safe_get_nested(data: Any, *keys: str, default: Any = None) -> Any:  # noqa: ANN401
    """Safely get nested dictionary/object values for templates.

    Args:
        data: Source data (dict, object, or None)
        *keys: Sequence of keys/attributes to traverse
        default: Default value if path doesn't exist

    Returns:
        Value at nested path or default

    Example:
        safe_get_nested(results, "execution_info", "audit_id", default="unknown")

    """
    current = data

    for key in keys:
        if current is None:
            return default

        current = current.get(key) if isinstance(current, dict) else getattr(current, key, None)

        if current is None:
            return default

    return current if current is not None else default
