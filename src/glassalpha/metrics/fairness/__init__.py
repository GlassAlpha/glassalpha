"""Fairness metrics for bias detection and evaluation.

This module contains metrics for evaluating model fairness across different
demographic groups, including demographic parity, equalized odds, and
equal opportunity metrics.

Includes E11: Individual fairness metrics for detecting disparate treatment.
Includes E5.1: Basic intersectional fairness for two-way bias analysis.
"""

from .individual import (
    IndividualFairnessMetrics,
    compute_consistency_score,
    counterfactual_flip_test,
    find_matched_pairs,
)
from .intersectional import (
    compute_intersectional_fairness,
    create_intersectional_groups,
    parse_intersection_spec,
)

__all__ = [
    "IndividualFairnessMetrics",
    "compute_consistency_score",
    "compute_intersectional_fairness",
    "counterfactual_flip_test",
    "create_intersectional_groups",
    "find_matched_pairs",
    "parse_intersection_spec",
]
