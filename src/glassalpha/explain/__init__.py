"""Explainer modules for GlassAlpha."""

from .base import ExplainerBase
from .coefficients import CoefficientsExplainer
from .noop import NoOpExplainer
from .permutation import PermutationExplainer
from .reason_codes import (
    DEFAULT_PROTECTED_ATTRIBUTES,
    ReasonCode,
    ReasonCodeResult,
    extract_reason_codes,
    format_adverse_action_notice,
)
from .recourse import (
    RecourseRecommendation,
    RecourseResult,
    generate_recourse,
)
from .registry import ExplainerRegistry

# Discover explainers from entry points (aliases are registered in registry.py)
ExplainerRegistry.discover()

__all__ = [
    "ExplainerBase",
    "ExplainerRegistry",
    "NoOpExplainer",
    "PermutationExplainer",
    # Reason codes (E2) - exported for E2.5 (Recourse) integration
    "ReasonCode",
    "ReasonCodeResult",
    "extract_reason_codes",
    "format_adverse_action_notice",
    "DEFAULT_PROTECTED_ATTRIBUTES",
    # Recourse (E2.5) - counterfactual recommendations
    "RecourseRecommendation",
    "RecourseResult",
    "generate_recourse",
]
