"""Adversarial perturbation sweeps for model robustness testing (E6+).

This module implements epsilon-perturbation stability testing to validate
model robustness under small input changes. Critical for regulatory compliance
and production readiness.

Key features:
1. Deterministic perturbations (seeded for reproducibility)
2. Protected feature exclusion (never perturb gender, race, etc.)
3. Configurable epsilon values (default: 1%, 5%, 10% noise)
4. Gate logic (PASS/FAIL/WARNING based on threshold)
5. JSON export for programmatic access

Algorithm:
- For each epsilon ∈ {0.01, 0.05, 0.1}:
  - Add Gaussian noise: X_perturbed = X + ε * σ * N(0,1)
  - Compute predictions on perturbed data
  - Measure max absolute delta: max(|y_pred_perturbed - y_pred_original|)
- Robustness score = max delta across all epsilon values
- Gate: PASS if max_delta < threshold, FAIL if >= 1.5*threshold, else WARNING
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerturbationResult:
    """Container for perturbation sweep results."""

    robustness_score: float
    max_delta: float
    epsilon_values: list[float]
    per_epsilon_deltas: dict[float, float]
    gate_status: str
    threshold: float
    n_samples: int
    n_features_perturbed: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "robustness_score": self.robustness_score,
            "max_delta": self.max_delta,
            "epsilon_values": self.epsilon_values,
            "per_epsilon_deltas": self.per_epsilon_deltas,
            "gate_status": self.gate_status,
            "threshold": self.threshold,
            "n_samples": self.n_samples,
            "n_features_perturbed": self.n_features_perturbed,
        }


def run_perturbation_sweep(
    model: Any,  # noqa: ANN401
    X_test: pd.DataFrame,
    protected_features: list[str],
    epsilon_values: list[float] | None = None,
    threshold: float = 0.15,
    seed: int = 42,
) -> PerturbationResult:
    """Run epsilon-perturbation sweep for model robustness testing.

    Args:
        model: Trained model with predict() or predict_proba() method
        X_test: Test data (DataFrame with feature names)
        protected_features: List of feature names to exclude from perturbation
        epsilon_values: List of epsilon values for perturbation (default: [0.01, 0.05, 0.1])
        threshold: Threshold for gate logic (default: 0.15)
        seed: Random seed for determinism (default: 42)

    Returns:
        PerturbationResult with robustness score, deltas, and gate status

    Raises:
        ValueError: If all features are protected, epsilon values invalid, or data empty

    Algorithm:
        For each epsilon:
            1. Add Gaussian noise to non-protected features: X' = X + ε * σ * N(0,1)
            2. Compute predictions on perturbed data
            3. Measure max delta: max(|y_pred' - y_pred|)
        Robustness score = max delta across all epsilon values
        Gate: PASS if max_delta < threshold, FAIL if >= 1.5*threshold, else WARNING

    """
    # Set seed for deterministic perturbations
    rng = np.random.RandomState(seed)

    # Default epsilon values
    if epsilon_values is None:
        epsilon_values = [0.01, 0.05, 0.1]

    # Sort epsilon values for consistent ordering
    epsilon_values = sorted(epsilon_values)

    # Validate epsilon values
    if any(eps <= 0 for eps in epsilon_values):
        raise ValueError("All epsilon values must be positive")

    # Validate data
    if X_test.empty:
        raise ValueError("X_test cannot be empty")

    # Identify non-protected features
    non_protected_features = [col for col in X_test.columns if col not in protected_features]

    if len(non_protected_features) == 0:
        raise ValueError(
            f"No non-protected features available for perturbation. All {len(X_test.columns)} features are protected.",
        )

    logger.info(
        f"Running perturbation sweep: {len(non_protected_features)} features, "
        f"{len(epsilon_values)} epsilon values, seed={seed}",
    )

    # Get original predictions
    if hasattr(model, "predict_proba"):
        y_pred_original = model.predict_proba(X_test)
        # For binary classification, use positive class probability
        if y_pred_original.ndim == 2 and y_pred_original.shape[1] == 2:
            y_pred_original = y_pred_original[:, 1]
    else:
        y_pred_original = model.predict(X_test)

    # Convert to numpy for consistency
    y_pred_original = np.asarray(y_pred_original).flatten()

    # Compute per-epsilon deltas
    per_epsilon_deltas = {}

    for epsilon in epsilon_values:
        # Create perturbed copy
        X_perturbed = X_test.copy()

        # Add Gaussian noise to non-protected features
        for feature in non_protected_features:
            # Compute feature std dev for scaling
            sigma = X_test[feature].std()

            # Handle zero variance features (no perturbation needed)
            if sigma == 0:
                continue

            # Add scaled Gaussian noise: X' = X + ε * σ * N(0,1)
            noise = rng.randn(len(X_test)) * sigma * epsilon
            X_perturbed[feature] = X_test[feature] + noise

        # Get predictions on perturbed data
        if hasattr(model, "predict_proba"):
            y_pred_perturbed = model.predict_proba(X_perturbed)
            if y_pred_perturbed.ndim == 2 and y_pred_perturbed.shape[1] == 2:
                y_pred_perturbed = y_pred_perturbed[:, 1]
        else:
            y_pred_perturbed = model.predict(X_perturbed)

        y_pred_perturbed = np.asarray(y_pred_perturbed).flatten()

        # Compute max absolute delta (L∞ norm)
        delta = np.max(np.abs(y_pred_perturbed - y_pred_original))
        per_epsilon_deltas[epsilon] = float(delta)

        logger.debug(f"Epsilon {epsilon:.3f}: max delta = {delta:.6f}")

    # Compute robustness score (max delta across all epsilon values)
    max_delta = max(per_epsilon_deltas.values())
    robustness_score = max_delta

    # Determine gate status
    if max_delta < threshold:
        gate_status = "PASS"
    elif max_delta >= 1.5 * threshold:
        gate_status = "FAIL"
    else:
        gate_status = "WARNING"

    logger.info(
        f"Perturbation sweep complete: robustness_score={robustness_score:.6f}, gate={gate_status}",
    )

    return PerturbationResult(
        robustness_score=robustness_score,
        max_delta=max_delta,
        epsilon_values=epsilon_values,
        per_epsilon_deltas=per_epsilon_deltas,
        gate_status=gate_status,
        threshold=threshold,
        n_samples=len(X_test),
        n_features_perturbed=len(non_protected_features),
    )
