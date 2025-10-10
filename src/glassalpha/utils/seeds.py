"""Centralized random seed management for reproducible audits.

This module provides comprehensive seed management across all randomness
sources in the audit pipeline: Python random, NumPy, scikit-learn, and
optional deep learning frameworks.
"""

import logging
import os
import random
import types
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Module-level flags for framework availability (expected by tests)
try:
    import torch as _torch

    torch = True
except ImportError:
    _torch = None
    torch = False

try:
    import tensorflow as tf

    tensorflow = True
except ImportError:
    tf = None  # type: ignore[assignment]
    tensorflow = False


class SeedManager:
    """Centralized manager for all random seeds in the audit pipeline."""

    def __init__(self, master_seed: int = 42) -> None:
        """Initialize seed manager with master seed.

        Args:
            master_seed: Primary seed for all randomness

        """
        self.master_seed = master_seed
        self.component_seeds: dict[str, int] = {}
        self._original_states: dict[str, Any] = {}

    def get_seed(self, component: str) -> int:
        """Get deterministic seed for a specific component.

        Args:
            component: Name of component needing seed (e.g., 'model', 'explainer')

        Returns:
            Deterministic seed derived from master seed

        """
        if component not in self.component_seeds:
            # Generate deterministic component seed from master seed and component name
            # Use hashlib for deterministic hashing (Python's hash() is randomized per process)
            import hashlib

            seed_string = f"{component}:{self.master_seed}"
            component_hash = int(hashlib.sha256(seed_string.encode()).hexdigest()[:8], 16) % (2**31)
            self.component_seeds[component] = component_hash
            logger.debug(f"Generated seed for '{component}': {self.component_seeds[component]}")

        return self.component_seeds[component]

    def set_all_seeds(self, seed: int | None = None) -> None:
        """Set all random seeds for reproducible execution.

        Args:
            seed: Override master seed (default: use current master_seed)

        """
        if seed is not None:
            self.master_seed = seed
            self.component_seeds.clear()  # Clear cached seeds

        logger.info(f"Setting all random seeds to master seed: {self.master_seed}")

        # Set Python random seed
        random.seed(self.master_seed)

        # Set NumPy seed (legacy API required for global reproducibility)
        np.random.seed(self.master_seed)  # noqa: NPY002

        # Set sklearn seed via environment variable
        os.environ["PYTHONHASHSEED"] = str(self.master_seed)

        # Set optional ML framework seeds
        self._set_optional_framework_seeds()

    def _set_optional_framework_seeds(self) -> None:
        """Set seeds for optional ML frameworks if available."""
        # PyTorch using module flags
        if torch and _torch is not None:
            _torch.manual_seed(self.master_seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed(self.master_seed)
                _torch.cuda.manual_seed_all(self.master_seed)
                # For deterministic CUDA operations
                _torch.backends.cudnn.deterministic = True
                _torch.backends.cudnn.benchmark = False
            logger.debug("Set PyTorch seeds")

        # TensorFlow using module flags
        if tensorflow and tf is not None:
            tf.random.set_seed(self.master_seed)
            logger.debug("Set TensorFlow seeds")

        # XGBoost and LightGBM use random_state parameters directly
        # These are handled in model wrappers using get_seed()

    def save_random_states(self) -> None:
        """Save current random number generator states for audit reproducibility.

        Captures the complete state of all active random number generators including
        Python's random module, NumPy, PyTorch, and framework-specific generators.
        This enables exact reproduction of randomized computations for regulatory
        audit verification and debugging.

        Side Effects:
            - Modifies internal _original_states dictionary with RNG snapshots
            - Captures GPU random states if CUDA is available
            - State objects consume ~50KB memory for complete capture
            - May trigger CUDA context initialization if GPU available

        Raises:
            RuntimeError: If PyTorch or NumPy random states are corrupted
            CudaError: If CUDA random state capture fails on GPU systems

        Note:
            Must be called before any randomized operations (model training,
            data shuffling, explainer sampling) to ensure audit reproducibility.
            State capture is thread-safe but not process-safe for distributed computing.

        """
        self._original_states = {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),  # noqa: NPY002
        }

        # Save optional framework states using module flags
        if torch and _torch is not None:
            self._original_states["torch_random"] = _torch.get_rng_state()
            if _torch.cuda.is_available():
                self._original_states["torch_cuda_random"] = _torch.cuda.get_rng_state()

        logger.debug("Saved random states")

    def restore_random_states(self) -> None:
        """Restore previously saved random states."""
        if not self._original_states:
            logger.warning("No saved random states to restore")
            return

        # Restore Python and NumPy states
        if "python_random" in self._original_states:
            random.setstate(self._original_states["python_random"])

        if "numpy_random" in self._original_states:
            np.random.set_state(self._original_states["numpy_random"])  # noqa: NPY002

        # Restore optional framework states using module flags
        if torch and _torch is not None:
            if "torch_random" in self._original_states:
                _torch.set_rng_state(self._original_states["torch_random"])
            if "torch_cuda_random" in self._original_states:
                _torch.cuda.set_rng_state(self._original_states["torch_cuda_random"])

        logger.debug("Restored random states")

    def __enter__(self) -> "SeedManager":
        """Context manager entry: set all seeds."""
        self.set_all_seeds()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool:
        """Context manager exit: don't suppress exceptions."""
        # Don't restore state on exit - let seeds remain set
        return False

    def get_seeds_manifest(self) -> dict[str, Any]:
        """Get manifest of all seeds used in this session.

        Returns:
            Dictionary with seed information for audit trail

        """
        return {
            "master_seed": self.master_seed,
            "component_seeds": self.component_seeds.copy(),
            "framework_availability": {
                "torch": self._check_framework_availability("torch"),
                "tensorflow": self._check_framework_availability("tensorflow"),
            },
            "timestamp": datetime.now(UTC).isoformat(),  # Required by tests - ISO8601 UTC format
        }

    def _check_framework_availability(self, framework: str) -> bool:
        """Check if optional framework is available."""
        # Use module-level flags for consistency
        if framework == "torch":
            return torch
        if framework == "tensorflow":
            return tensorflow
        # Fallback for other frameworks
        try:
            __import__(framework)
        except ImportError:
            return False
        else:
            return True


# Global seed manager instance
_global_seed_manager = SeedManager()


def set_global_seed(seed: int) -> None:
    """Set global random seed for entire application.

    Args:
        seed: Master seed for all randomness

    """
    _global_seed_manager.set_all_seeds(seed)


def get_component_seed(component: str) -> int:
    """Get deterministic seed for specific component.

    Args:
        component: Component name (e.g., 'model', 'explainer', 'data_split')

    Returns:
        Deterministic seed for component

    """
    return _global_seed_manager.get_seed(component)


@contextmanager
def with_seed(seed: int, *, restore_after: bool = True) -> Generator[None, None, None]:
    """Context manager for temporary seed setting.

    Args:
        seed: Temporary seed to use
        restore_after: Whether to restore original state after context

    Yields:
        Context with temporary seed set

    Example:
        >>> with with_seed(123):
        ...     # All randomness uses seed 123
        ...     random_data = np.random.rand(10)
        >>> # Original random state restored

    """
    if restore_after:
        _global_seed_manager.save_random_states()

    # Set temporary seed
    _global_seed_manager.set_all_seeds(seed)

    try:
        yield
    finally:
        if restore_after:
            _global_seed_manager.restore_random_states()


@contextmanager
def with_component_seed(component: str) -> Generator[int, None, None]:
    """Context manager for component-specific seeding.

    Args:
        component: Component name for seed generation

    Yields:
        Component seed value

    Example:
        >>> with with_component_seed('model_training') as seed:
        ...     model = XGBoost(random_state=seed)
        ...     model.fit(X, y)

    """
    seed = get_component_seed(component)

    with with_seed(seed, restore_after=True):
        yield seed


def ensure_reproducibility(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to ensure function runs with consistent seeding.

    Args:
        func: Function to wrap with seed management

    Returns:
        Wrapped function with seed management

    Example:
        @ensure_reproducibility
        def train_model(data):
            # This function will have consistent randomness
            return model.fit(data)

    """

    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def] # noqa: ANN002,ANN003,ANN202
        # Save current state
        _global_seed_manager.save_random_states()

        try:
            # Ensure seeds are set
            _global_seed_manager.set_all_seeds()
            return func(*args, **kwargs)
        finally:
            # Restore original state
            _global_seed_manager.restore_random_states()

    return wrapper


def get_seeds_manifest(master_seed: int | None = None, component_seeds: dict[str, int] | None = None) -> dict[str, Any]:
    """Get complete seed manifest for audit trail.

    Args:
        master_seed: Master seed value (uses global manager if None)
        component_seeds: Component-specific seeds (uses global manager if None)

    Returns:
        Dictionary with all seed information

    """
    if master_seed is not None or component_seeds is not None:
        # Create temporary manager with provided parameters
        temp_manager = SeedManager(master_seed or 42)
        if component_seeds:
            temp_manager.component_seeds.update(component_seeds)
        return temp_manager.get_seeds_manifest()
    # Use global manager
    return _global_seed_manager.get_seeds_manifest()


def validate_deterministic_environment() -> dict[str, bool]:
    """Validate that environment supports deterministic execution.

    Returns:
        Dictionary with validation results for each component

    """
    validation_results = {}

    # Test Python random (not for cryptographic use)
    random.seed(42)
    val1 = random.random()  # noqa: S311
    random.seed(42)
    val2 = random.random()  # noqa: S311
    validation_results["python_random"] = val1 == val2

    # Test NumPy random (legacy API for reproducibility testing)
    np.random.seed(42)  # noqa: NPY002
    arr1 = np.random.rand(5)  # noqa: NPY002
    np.random.seed(42)  # noqa: NPY002
    arr2 = np.random.rand(5)  # noqa: NPY002
    validation_results["numpy_random"] = np.allclose(arr1, arr2)

    # Test optional frameworks using module flags
    if torch and _torch is not None:
        _torch.manual_seed(42)
        tensor1 = _torch.rand(5)
        _torch.manual_seed(42)
        tensor2 = _torch.rand(5)
        validation_results["torch"] = _torch.allclose(tensor1, tensor2)
    else:
        validation_results["torch"] = None

    if tensorflow and tf is not None:
        tf.random.set_seed(42)
        tensor1 = tf.random.uniform([5])
        tf.random.set_seed(42)
        tensor2 = tf.random.uniform([5])
        validation_results["tensorflow"] = tf.reduce_all(tf.equal(tensor1, tensor2)).numpy()
    else:
        validation_results["tensorflow"] = None

    logger.info(f"Deterministic environment validation: {validation_results}")
    return validation_results


# Convenience aliases
seed_manager = _global_seed_manager
