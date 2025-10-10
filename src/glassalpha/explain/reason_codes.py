"""Reason code extraction for ECOA-compliant adverse action notices.

This module extracts top-N negative feature contributions from SHAP values
to generate regulatory-compliant reason codes for adverse decisions.

ECOA Requirements:
- Must provide specific reasons (not just "credit score")
- Must rank by importance (top-N negative contributions)
- Typically 2-4 reasons (default 4)
- Must be understandable to applicant
- Must not mention protected attributes

Architecture:
- Reuses existing SHAP explainers (no new explainer needed)
- Pure functions for deterministic output
- Protected attribute filtering
- Seeded tie-breaking for reproducibility
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default protected attributes per ECOA/Fair Lending laws
DEFAULT_PROTECTED_ATTRIBUTES = [
    "age",
    "gender",
    "sex",
    "race",
    "ethnicity",
    "national_origin",
    "nationality",
    "religion",
    "marital_status",
    "disability",
    "foreign_worker",
    "color",
]


@dataclass(frozen=True)
class ReasonCode:
    """Single reason code with feature contribution.

    Attributes:
        feature: Feature name (human-readable)
        contribution: SHAP value (negative = pushed toward denial)
        feature_value: Original feature value for context
        rank: Importance rank (1 = most negative)

    """

    feature: str
    contribution: float
    feature_value: float | str
    rank: int


@dataclass(frozen=True)
class ReasonCodeResult:
    """Complete reason code extraction result with audit trail.

    Attributes:
        instance_id: Unique instance identifier
        prediction: Model prediction (probability)
        decision: Human-readable decision ("approved" or "denied")
        reason_codes: Top-N negative contributions
        excluded_features: Protected attributes excluded from codes
        timestamp: ISO 8601 timestamp (UTC)
        model_hash: Hash of model for provenance
        seed: Random seed used for tie-breaking

    """

    instance_id: str | int
    prediction: float
    decision: str
    reason_codes: list[ReasonCode]
    excluded_features: list[str]
    timestamp: str
    model_hash: str
    seed: int


def extract_reason_codes(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_values: pd.Series | np.ndarray,
    instance_id: str | int,
    prediction: float,
    threshold: float = 0.5,
    top_n: int = 4,
    protected_attributes: list[str] | None = None,
    model_hash: str | None = None,
    seed: int = 42,
) -> ReasonCodeResult:
    """Extract top-N negative feature contributions for adverse action notice.

    This function identifies the features that most negatively impacted a prediction,
    suitable for ECOA-compliant adverse action notices.

    Args:
        shap_values: SHAP values for single instance (1D array of shape [n_features])
        feature_names: List of feature names matching SHAP values
        feature_values: Feature values for instance (Series or 1D array)
        instance_id: Unique instance identifier
        prediction: Model prediction (probability for binary classification)
        threshold: Decision threshold (default 0.5)
        top_n: Number of reason codes to extract (ECOA typical: 4)
        protected_attributes: Features to exclude (default: DEFAULT_PROTECTED_ATTRIBUTES)
        model_hash: Optional model hash for provenance
        seed: Random seed for deterministic tie-breaking

    Returns:
        ReasonCodeResult with top-N negative contributions and audit trail

    Raises:
        ValueError: If SHAP values don't match feature names
        ValueError: If no negative contributions exist after exclusions

    Examples:
        >>> shap_values = np.array([-0.5, 0.3, -0.2, 0.1])
        >>> feature_names = ["debt", "income", "duration", "savings"]
        >>> feature_values = pd.Series([5000, 30000, 24, 1000])
        >>> result = extract_reason_codes(
        ...     shap_values=shap_values,
        ...     feature_names=feature_names,
        ...     feature_values=feature_values,
        ...     instance_id=42,
        ...     prediction=0.35,
        ...     threshold=0.5,
        ...     top_n=2,
        ...     seed=42,
        ... )
        >>> result.decision
        'denied'
        >>> len(result.reason_codes)
        2
        >>> result.reason_codes[0].feature
        'debt'

    """
    # Validate inputs
    if len(shap_values.shape) != 1:
        msg = f"Expected 1D SHAP values, got shape {shap_values.shape}"
        raise ValueError(msg)

    if len(shap_values) != len(feature_names):
        msg = f"SHAP values ({len(shap_values)}) don't match feature names ({len(feature_names)})"
        raise ValueError(msg)

    # Convert feature values to array if needed
    if isinstance(feature_values, pd.Series):
        feature_values_array = feature_values.values
    else:
        feature_values_array = np.asarray(feature_values)

    if len(feature_values_array) != len(feature_names):
        msg = f"Feature values ({len(feature_values_array)}) don't match feature names ({len(feature_names)})"
        raise ValueError(msg)

    # Use default protected attributes if none provided
    if protected_attributes is None:
        protected_attributes = DEFAULT_PROTECTED_ATTRIBUTES

    # Normalize protected attribute names (lowercase for matching)
    protected_set = {attr.lower() for attr in protected_attributes}

    # Filter out protected attributes
    excluded = []
    valid_indices = []
    for i, name in enumerate(feature_names):
        if name.lower() in protected_set:
            excluded.append(name)
        else:
            valid_indices.append(i)

    if not valid_indices:
        msg = "All features are protected attributes - cannot generate reason codes"
        raise ValueError(msg)

    # Extract valid SHAP values and feature info
    valid_shap = shap_values[valid_indices]
    valid_names = [feature_names[i] for i in valid_indices]
    valid_values = feature_values_array[valid_indices]

    # Find negative contributions (features pushing toward denial)
    negative_mask = valid_shap < 0
    negative_indices = np.where(negative_mask)[0]

    if len(negative_indices) == 0:
        msg = "No negative contributions found - cannot generate reason codes for approved decision"
        raise ValueError(msg)

    # Extract negative contributions
    negative_shap = valid_shap[negative_indices]
    negative_names = [valid_names[i] for i in negative_indices]
    negative_values = valid_values[negative_indices]

    # Sort by SHAP magnitude (most negative first)
    # Use seeded random for deterministic tie-breaking
    rng = np.random.default_rng(seed)

    # Create sorting key: (shap_value, random_tiebreaker)
    # This ensures deterministic ordering even with ties
    tiebreakers = rng.random(len(negative_shap))
    sort_keys = list(zip(negative_shap, tiebreakers, strict=False))
    sorted_indices = sorted(
        range(len(negative_shap)),
        key=lambda i: (sort_keys[i][0], sort_keys[i][1]),  # Sort by SHAP, then tiebreaker
    )

    # Take top-N
    top_indices = sorted_indices[: min(top_n, len(sorted_indices))]

    # Build reason codes
    reason_codes = []
    for rank, idx in enumerate(top_indices, start=1):
        reason_codes.append(
            ReasonCode(
                feature=negative_names[idx],
                contribution=float(negative_shap[idx]),
                feature_value=_format_feature_value(negative_values[idx]),
                rank=rank,
            ),
        )

    # Determine decision
    decision = "denied" if prediction < threshold else "approved"

    # Generate model hash if not provided
    if model_hash is None:
        # Create deterministic hash from SHAP values as proxy
        model_hash = hashlib.sha256(shap_values.tobytes()).hexdigest()[:16]

    # Create timestamp
    timestamp = datetime.now(UTC).isoformat()

    return ReasonCodeResult(
        instance_id=instance_id,
        prediction=prediction,
        decision=decision,
        reason_codes=reason_codes,
        excluded_features=excluded,
        timestamp=timestamp,
        model_hash=model_hash,
        seed=seed,
    )


def format_adverse_action_notice(
    result: ReasonCodeResult,
    template_path: Path | None = None,
    organization: str = "[Organization Name]",
    contact_info: str = "[Contact Information]",
) -> str:
    """Format reason codes as ECOA-compliant adverse action notice.

    Args:
        result: ReasonCodeResult from extract_reason_codes
        template_path: Path to custom AAN template (optional)
        organization: Organization name for notice
        contact_info: Contact information for applicant inquiries

    Returns:
        Formatted adverse action notice (plain text)

    Raises:
        FileNotFoundError: If template_path specified but doesn't exist

    Examples:
        >>> result = extract_reason_codes(...)
        >>> notice = format_adverse_action_notice(
        ...     result=result,
        ...     organization="Example Bank",
        ...     contact_info="1-800-555-0199",
        ... )

    """
    # Load template
    if template_path is not None:
        if not template_path.exists():
            msg = f"Template not found: {template_path}"
            raise FileNotFoundError(msg)
        template_text = template_path.read_text()
    else:
        # Use default template
        template_text = _get_default_template()

    # Format reason codes
    reason_lines = []
    for code in result.reason_codes:
        # Format contribution as user-friendly text
        reason_lines.append(
            f"{code.rank}. {_humanize_feature_name(code.feature)}: "
            f"{_explain_contribution(code.feature, code.contribution, code.feature_value)}",
        )
    reason_text = "\n".join(reason_lines)

    # Replace template variables
    notice = template_text.format(
        organization=organization,
        instance_id=result.instance_id,
        prediction=f"{result.prediction:.1%}",
        decision=result.decision.upper(),
        reason_codes=reason_text,
        timestamp=result.timestamp,
        contact_info=contact_info,
        model_hash=result.model_hash,
    )

    return notice


def _format_feature_value(value: Any) -> float | str:  # noqa: ANN401
    """Format feature value for display."""
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


def _humanize_feature_name(name: str) -> str:
    """Convert feature name to human-readable form.

    Examples:
        "debt_ratio" -> "Debt Ratio"
        "employment_duration" -> "Employment Duration"

    """
    return name.replace("_", " ").title()


def _explain_contribution(feature: str, contribution: float, value: Any) -> str:  # noqa: ANN401, ARG001
    """Generate human-readable explanation for contribution.

    Args:
        feature: Feature name (unused in basic implementation)
        contribution: SHAP value (negative, unused in basic implementation)
        value: Feature value

    Returns:
        Human-readable explanation

    Note:
        This is a basic implementation. Future versions may use feature and contribution
        to generate more specific explanations (e.g., "High debt of $5000 was a concern").

    """
    # Generic explanation for negative contribution
    return f"Value of {value} negatively impacted the decision"


def _get_default_template() -> str:
    """Get default adverse action notice template."""
    return """ADVERSE ACTION NOTICE
Equal Credit Opportunity Act (ECOA) Disclosure

{organization}
Date: {timestamp}
Application ID: {instance_id}

DECISION: {decision}
Predicted Score: {prediction}

PRINCIPAL REASONS FOR ADVERSE ACTION:

The following factors most negatively affected your application:

{reason_codes}

IMPORTANT RIGHTS UNDER FEDERAL LAW:

You have the right to a statement of specific reasons why your application
was denied. To obtain the statement, please contact us at:

{contact_info}

We will send you a written statement of reasons for the denial within
30 days of your request. The statement will contain the specific reasons
for the denial.

NOTICE OF RIGHT TO REQUEST CREDIT REPORT:

As set forth by the Fair Credit Reporting Act, you have the right to obtain
a free copy of your credit report from the consumer reporting agency(ies)
used in connection with your application.

This notice was generated by GlassAlpha ML Compliance System.
Model Hash: {model_hash}
Generation Seed: Deterministic (reproducible)

---
This is an automated notice generated for regulatory compliance.
For questions about this decision, please contact us using the information above.
"""
