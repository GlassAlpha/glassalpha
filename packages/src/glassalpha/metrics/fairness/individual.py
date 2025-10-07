"""Individual fairness metrics (E11).

This module implements individual-level fairness analysis to detect disparate treatment.
Complements group fairness metrics (E5) by examining treatment consistency for similar
individuals.

Key Metrics:
1. Consistency Score: Lipschitz-like metric measuring prediction stability for similar individuals
2. Matched Pairs: Identifies individuals with similar features but different predictions
3. Counterfactual Flip Test: Tests if protected attribute changes affect predictions

Legal Context:
- Catches disparate treatment (illegal under anti-discrimination laws)
- Provides evidence for individual-level fairness audits
- Complements group fairness analysis

Dependencies:
- Uses distance metrics (Euclidean, Mahalanobis) to find similar individuals
- Reuses model interface from recourse infrastructure
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import iqr

logger = logging.getLogger(__name__)


# ============================================================================
# Protocol for Model Interface
# ============================================================================


class ModelProtocol(Protocol):
    """Protocol for model with predict_proba method."""

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict probability for instances.

        Args:
            X: Feature DataFrame or array

        Returns:
            Array of shape (n_samples, n_classes) with predicted probabilities

        """


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ConsistencyScoreResult:
    """Results from consistency score computation."""

    consistency_score: float  # Lipschitz constant (max pred_diff / distance)
    similarity_threshold: float  # Distance threshold for "similar"
    n_similar_pairs: int  # Number of pairs below threshold
    n_samples: int
    distance_metric: str
    similarity_percentile: float
    features_used: list[str]
    protected_attributes_excluded: list[str]
    max_prediction_diff: float  # Max prediction difference among similar pairs
    mean_prediction_diff: float  # Mean prediction difference

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "consistency_score": float(self.consistency_score),
            "similarity_threshold": float(self.similarity_threshold),
            "n_similar_pairs": int(self.n_similar_pairs),
            "n_samples": int(self.n_samples),
            "distance_metric": self.distance_metric,
            "similarity_percentile": float(self.similarity_percentile),
            "features_used": self.features_used,
            "protected_attributes_excluded": self.protected_attributes_excluded,
            "max_prediction_diff": float(self.max_prediction_diff),
            "mean_prediction_diff": float(self.mean_prediction_diff),
        }


@dataclass
class MatchedPair:
    """A pair of individuals with similar features but different predictions."""

    instance_1: int
    instance_2: int
    feature_distance: float
    prediction_diff: float
    protected_attributes_differ: bool
    instance_1_prediction: float
    instance_2_prediction: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instance_1": int(self.instance_1),
            "instance_2": int(self.instance_2),
            "feature_distance": float(self.feature_distance),
            "prediction_diff": float(self.prediction_diff),
            "protected_attributes_differ": bool(self.protected_attributes_differ),
            "instance_1_prediction": float(self.instance_1_prediction),
            "instance_2_prediction": float(self.instance_2_prediction),
        }


@dataclass
class MatchedPairsResult:
    """Results from matched pairs analysis."""

    matched_pairs: list[MatchedPair]
    n_comparisons_made: int
    distance_threshold: float
    prediction_diff_threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "matched_pairs": [p.to_dict() for p in self.matched_pairs],
            "n_comparisons_made": int(self.n_comparisons_made),
            "distance_threshold": float(self.distance_threshold),
            "prediction_diff_threshold": float(self.prediction_diff_threshold),
        }


@dataclass
class FlipTestResult:
    """Results from counterfactual flip test."""

    disparate_treatment_rate: float  # % of instances where flip changes prediction
    flip_changes_prediction: list[int]  # Instance IDs where flip changes decision
    protected_attributes_tested: list[str]
    threshold: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "disparate_treatment_rate": float(self.disparate_treatment_rate),
            "flip_changes_prediction": [int(x) for x in self.flip_changes_prediction],
            "protected_attributes_tested": self.protected_attributes_tested,
            "threshold": float(self.threshold),
            "n_samples": int(self.n_samples),
        }


# ============================================================================
# Consistency Score
# ============================================================================


def compute_consistency_score(
    features: pd.DataFrame,
    predictions: np.ndarray,
    protected_attributes: list[str],
    distance_metric: Literal["euclidean", "mahalanobis"] = "euclidean",
    similarity_percentile: float = 90,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute consistency score (Lipschitz-like metric for individual fairness).

    The consistency score measures prediction stability for similar individuals:
        consistency_score = max(|pred_i - pred_j| / distance(features_i, features_j))

    Lower scores indicate better individual fairness (similar people → similar predictions).

    Args:
        features: Feature DataFrame (may include protected attributes)
        predictions: Model predictions (0-1 scale)
        protected_attributes: List of protected attribute names to EXCLUDE from distance
        distance_metric: Distance metric ("euclidean" or "mahalanobis")
        similarity_percentile: Percentile threshold for "similar" (default 90th)
        seed: Random seed for deterministic tie-breaking

    Returns:
        Dictionary with consistency score and metadata

    """
    # Set seed for deterministic behavior
    if seed is not None:
        np.random.seed(seed)

    # Exclude protected attributes from distance computation
    feature_cols = [c for c in features.columns if c not in protected_attributes]
    if not feature_cols:
        raise ValueError("No non-protected features available for distance computation")

    X = features[feature_cols].values
    n_samples = len(X)

    # Validate inputs
    if len(predictions) != n_samples:
        raise ValueError(f"Predictions length {len(predictions)} != features length {n_samples}")

    # Check for NaN values
    if np.isnan(X).any():
        raise ValueError("Features contain NaN values. Please handle missing data before computing individual fairness.")

    # Normalize features for fair distance comparison (use robust scaling)
    X_normalized = _robust_scale(X)

    # Compute pairwise distances
    if distance_metric == "euclidean":
        distances = cdist(X_normalized, X_normalized, metric="euclidean")
    elif distance_metric == "mahalanobis":
        # Compute covariance matrix (add regularization for stability)
        try:
            cov = np.cov(X_normalized.T)
            cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
            distances = cdist(X_normalized, X_normalized, metric="mahalanobis", VI=cov_inv)
        except np.linalg.LinAlgError:
            logger.warning("Mahalanobis distance failed (singular covariance), falling back to Euclidean")
            distances = cdist(X_normalized, X_normalized, metric="euclidean")
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    # Set diagonal to infinity (don't compare instance to itself)
    np.fill_diagonal(distances, np.inf)

    # Compute similarity threshold (percentile of pairwise distances)
    finite_distances = distances[np.isfinite(distances)]
    if len(finite_distances) == 0:
        logger.warning("No finite pairwise distances found")
        return ConsistencyScoreResult(
            consistency_score=np.nan,
            similarity_threshold=np.nan,
            n_similar_pairs=0,
            n_samples=n_samples,
            distance_metric=distance_metric,
            similarity_percentile=similarity_percentile,
            features_used=feature_cols,
            protected_attributes_excluded=protected_attributes,
            max_prediction_diff=0.0,
            mean_prediction_diff=0.0,
        ).to_dict()

    similarity_threshold = np.percentile(finite_distances, similarity_percentile)

    # Find similar pairs (below threshold)
    similar_mask = distances <= similarity_threshold

    # Compute prediction differences
    pred_diffs = np.abs(predictions[:, None] - predictions[None, :])

    # Filter to similar pairs
    similar_pred_diffs = pred_diffs[similar_mask]
    similar_distances = distances[similar_mask]

    if len(similar_pred_diffs) == 0:
        # No similar pairs found
        return ConsistencyScoreResult(
            consistency_score=0.0,
            similarity_threshold=float(similarity_threshold),
            n_similar_pairs=0,
            n_samples=n_samples,
            distance_metric=distance_metric,
            similarity_percentile=similarity_percentile,
            features_used=feature_cols,
            protected_attributes_excluded=protected_attributes,
            max_prediction_diff=0.0,
            mean_prediction_diff=0.0,
        ).to_dict()

    # Compute Lipschitz constant: max(pred_diff / distance)
    # Avoid division by zero
    nonzero_mask = similar_distances > 1e-10
    if not np.any(nonzero_mask):
        # All similar pairs have distance ~0 (identical features)
        # Consistency score is 0 if predictions are same, else very high
        max_pred_diff = np.max(similar_pred_diffs)
        consistency_score = 0.0 if max_pred_diff < 1e-6 else float('inf')
    else:
        lipschitz_values = similar_pred_diffs[nonzero_mask] / similar_distances[nonzero_mask]
        consistency_score = float(np.max(lipschitz_values))

    # Compute statistics
    max_pred_diff = float(np.max(similar_pred_diffs))
    mean_pred_diff = float(np.mean(similar_pred_diffs))

    return ConsistencyScoreResult(
        consistency_score=consistency_score,
        similarity_threshold=float(similarity_threshold),
        n_similar_pairs=int(np.sum(similar_mask)),
        n_samples=n_samples,
        distance_metric=distance_metric,
        similarity_percentile=similarity_percentile,
        features_used=feature_cols,
        protected_attributes_excluded=protected_attributes,
        max_prediction_diff=max_pred_diff,
        mean_prediction_diff=mean_pred_diff,
    ).to_dict()


def _robust_scale(X: np.ndarray) -> np.ndarray:
    """Robust feature scaling using median and IQR (resistant to outliers)."""
    median = np.median(X, axis=0)
    iqr_val = iqr(X, axis=0)
    # Avoid division by zero
    iqr_val = np.where(iqr_val == 0, 1.0, iqr_val)
    return (X - median) / iqr_val


# ============================================================================
# Matched Pairs
# ============================================================================


def find_matched_pairs(
    features: pd.DataFrame,
    predictions: np.ndarray,
    protected_features: pd.DataFrame,
    distance_threshold: float,
    prediction_diff_threshold: float,
    seed: int | None = None,
) -> dict[str, Any]:
    """Find pairs of individuals with similar features but different predictions.

    Identifies potential disparate treatment cases: individuals who are similar on
    non-protected features but receive different predictions.

    Args:
        features: Non-protected features for distance computation
        predictions: Model predictions
        protected_features: Protected attributes (to check if they differ in pairs)
        distance_threshold: Max distance for "similar" features
        prediction_diff_threshold: Min prediction difference to flag
        seed: Random seed for deterministic behavior

    Returns:
        MatchedPairsResult with flagged pairs and metadata

    """
    if seed is not None:
        np.random.seed(seed)

    X = features.values
    n_samples = len(X)

    # Normalize features
    X_normalized = _robust_scale(X)

    # Compute pairwise distances (Euclidean)
    distances = cdist(X_normalized, X_normalized, metric="euclidean")
    np.fill_diagonal(distances, np.inf)

    # Find pairs meeting both conditions:
    # 1. Feature distance < threshold (similar)
    # 2. Prediction difference >= threshold (different outcomes)
    similar_mask = distances <= distance_threshold
    pred_diffs = np.abs(predictions[:, None] - predictions[None, :])
    different_preds_mask = pred_diffs >= prediction_diff_threshold

    # Combined mask
    flagged_mask = similar_mask & different_preds_mask

    # Extract matched pairs
    matched_pairs = []
    n_comparisons = 0

    # Iterate over upper triangle to avoid duplicates
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            n_comparisons += 1

            if flagged_mask[i, j]:
                # Check if protected attributes differ
                protected_differ = _protected_attributes_differ(
                    protected_features.iloc[i],
                    protected_features.iloc[j],
                )

                pair = MatchedPair(
                    instance_1=i,
                    instance_2=j,
                    feature_distance=float(distances[i, j]),
                    prediction_diff=float(pred_diffs[i, j]),
                    protected_attributes_differ=protected_differ,
                    instance_1_prediction=float(predictions[i]),
                    instance_2_prediction=float(predictions[j]),
                )
                matched_pairs.append(pair)

    return MatchedPairsResult(
        matched_pairs=matched_pairs,
        n_comparisons_made=n_comparisons,
        distance_threshold=distance_threshold,
        prediction_diff_threshold=prediction_diff_threshold,
    ).to_dict()


def _protected_attributes_differ(
    instance_1: pd.Series,
    instance_2: pd.Series,
) -> bool:
    """Check if any protected attributes differ between two instances."""
    return not instance_1.equals(instance_2)


# ============================================================================
# Counterfactual Flip Test
# ============================================================================


def counterfactual_flip_test(
    model: ModelProtocol,
    features: pd.DataFrame,
    protected_attributes: list[str],
    threshold: float = 0.5,
    seed: int | None = None,
) -> dict[str, Any]:
    """Test if flipping protected attributes changes predictions (disparate treatment test).

    For each instance, creates counterfactuals by changing each protected attribute to
    all other possible values. If prediction changes, flags as potential disparate treatment.

    Args:
        model: Model with predict_proba method
        features: Feature DataFrame including protected attributes
        protected_attributes: List of protected attribute names to flip
        threshold: Decision threshold (default 0.5)
        seed: Random seed for deterministic behavior

    Returns:
        FlipTestResult with disparate treatment statistics

    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(features)
    flip_changes = []

    for idx in range(n_samples):
        instance = features.iloc[idx:idx+1].copy()

        # Get original prediction
        orig_proba = model.predict_proba(instance)
        if len(orig_proba.shape) == 2:
            orig_pred = float(orig_proba[0, 1])  # Binary classification
        else:
            orig_pred = float(orig_proba[1])

        orig_decision = orig_pred >= threshold

        # Try flipping each protected attribute
        for attr in protected_attributes:
            # Get all possible values for this attribute
            possible_values = features[attr].unique()

            for value in possible_values:
                if value == instance[attr].values[0]:
                    continue  # Skip current value

                # Create counterfactual
                counterfactual = instance.copy()
                counterfactual[attr] = value

                # Predict
                try:
                    cf_proba = model.predict_proba(counterfactual)
                    if len(cf_proba.shape) == 2:
                        cf_pred = float(cf_proba[0, 1])
                    else:
                        cf_pred = float(cf_proba[1])

                    cf_decision = cf_pred >= threshold

                    # Check if decision changed
                    if cf_decision != orig_decision:
                        flip_changes.append(idx)
                        break  # Found disparate treatment for this instance

                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Flip test prediction failed for instance {idx}: {e}")
                    continue

            if idx in flip_changes:
                break  # Already flagged this instance

    # Compute statistics
    disparate_treatment_rate = len(flip_changes) / n_samples if n_samples > 0 else 0.0

    return FlipTestResult(
        disparate_treatment_rate=disparate_treatment_rate,
        flip_changes_prediction=flip_changes,
        protected_attributes_tested=protected_attributes,
        threshold=threshold,
        n_samples=n_samples,
    ).to_dict()


# ============================================================================
# Main Metrics Class
# ============================================================================


class IndividualFairnessMetrics:
    """Compute all individual fairness metrics.

    Combines consistency score, matched pairs, and flip test into unified analysis.
    """

    def __init__(
        self,
        model: ModelProtocol,
        features: pd.DataFrame,
        predictions: np.ndarray,
        protected_attributes: list[str],
        distance_metric: Literal["euclidean", "mahalanobis"] = "euclidean",
        similarity_percentile: float = 90,
        prediction_diff_threshold: float = 0.1,
        threshold: float = 0.5,
        seed: int | None = 42,
    ):
        """Initialize individual fairness metrics.

        Args:
            model: Model with predict_proba method
            features: Feature DataFrame (includes protected attributes)
            predictions: Model predictions
            protected_attributes: List of protected attribute names
            distance_metric: Distance metric for similarity
            similarity_percentile: Percentile threshold for "similar"
            prediction_diff_threshold: Min difference to flag in matched pairs
            threshold: Decision threshold
            seed: Random seed

        """
        self.model = model
        self.features = features
        self.predictions = predictions
        self.protected_attributes = protected_attributes
        self.distance_metric = distance_metric
        self.similarity_percentile = similarity_percentile
        self.prediction_diff_threshold = prediction_diff_threshold
        self.threshold = threshold
        self.seed = seed

    def compute(self) -> dict[str, Any]:
        """Compute all individual fairness metrics.

        Returns:
            Dictionary with consistency_score, matched_pairs, and flip_test results

        """
        # 1. Consistency score
        consistency = compute_consistency_score(
            features=self.features,
            predictions=self.predictions,
            protected_attributes=self.protected_attributes,
            distance_metric=self.distance_metric,
            similarity_percentile=self.similarity_percentile,
            seed=self.seed,
        )

        # 2. Matched pairs
        # Extract non-protected features
        non_protected_features = self.features[[
            c for c in self.features.columns
            if c not in self.protected_attributes
        ]]
        protected_features_df = self.features[self.protected_attributes]

        # Use similarity threshold from consistency score
        matched = find_matched_pairs(
            features=non_protected_features,
            predictions=self.predictions,
            protected_features=protected_features_df,
            distance_threshold=consistency["similarity_threshold"],
            prediction_diff_threshold=self.prediction_diff_threshold,
            seed=self.seed,
        )

        # 3. Counterfactual flip test
        flip = counterfactual_flip_test(
            model=self.model,
            features=self.features,
            protected_attributes=self.protected_attributes,
            threshold=self.threshold,
            seed=self.seed,
        )

        return {
            "consistency_score": consistency,
            "matched_pairs": matched,
            "flip_test": flip,
        }
