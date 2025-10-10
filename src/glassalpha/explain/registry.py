"""Registry for explainer classes and compatibility detection."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Iterable

from glassalpha.constants import NO_EXPLAINER_MSG
from glassalpha.core.decor_registry import DecoratorFriendlyRegistry

logger = logging.getLogger(__name__)


def _has(mod: str) -> bool:
    """Check if a module is available."""
    return importlib.util.find_spec(mod) is not None


# Single source of truth for the exact error copy the contract test expects.
# If the test already asserts a different string, set this constant to match it.
ERR_UNSUPPORTED_MODEL = "No compatible explainer found for model type '{model_type}'."

SUPPORTED_FAMILIES: dict[str, list[str]] = {
    # sensible defaults per family; order = preference
    "logistic_regression": ["coefficients", "permutation", "kernelshap"],
    "linear_regression": ["coefficients", "permutation", "kernelshap"],
    "linear_svc": ["coefficients", "permutation", "kernelshap"],
    "random_forest": ["treeshap", "permutation", "kernelshap"],
    "extra_trees": ["treeshap", "permutation", "kernelshap"],
    "xgboost": ["treeshap", "permutation", "kernelshap"],
    "lightgbm": ["treeshap", "permutation", "kernelshap"],
    # add others you officially support here
}

REQUIRES: dict[str, list[str]] = {
    "kernelshap": ["shap"],
    "treeshap": ["shap"],
    "coefficients": [],
    "permutation": [],
    "noop": [],
}


def _available(name: str) -> bool:
    # If it's a built-in explainer with known dependencies, check those
    if name in REQUIRES:
        return all(_has(m) for m in REQUIRES[name])

    # Otherwise, check if it's registered in the registry (for dynamic registrations)
    # We need to access the global ExplainerRegistry which is defined later in this module
    import sys

    if "glassalpha.explain.registry" in sys.modules:
        registry = sys.modules["glassalpha.explain.registry"]
        if hasattr(registry, "ExplainerRegistry"):
            return registry.ExplainerRegistry.has(name)

    return False


def _first_available(candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if _available(c):
            return c
    return None


def _first_compatible(candidates: Iterable[str], model_type: str) -> str | None:
    """Find first explainer that is both available and compatible with model type.

    Also respects enterprise licensing - enterprise components are skipped unless
    a valid license is present.

    Args:
        candidates: List of explainer names to check
        model_type: Model type to check compatibility against

    Returns:
        First compatible explainer name, or None if none found

    """
    import sys

    # Get registry if available
    registry = None
    if "glassalpha.explain.registry" in sys.modules:
        registry_module = sys.modules["glassalpha.explain.registry"]
        if hasattr(registry_module, "ExplainerRegistry"):
            registry = registry_module.ExplainerRegistry

    # Check if enterprise features are enabled
    from glassalpha.core.features import is_enterprise as has_license  # noqa: PLC0415

    for c in candidates:
        # First check if available (registered and dependencies met)
        if not _available(c):
            continue

        # Skip enterprise components without license
        if registry is not None and registry.is_enterprise(c) and not has_license():
            continue

        # If registry is available, check compatibility
        if registry is not None:
            if registry.is_compatible(c, model_type):
                return c
        else:
            # Fallback: if we can't check compatibility, assume available means compatible
            return c

    return None


def select_explainer(model_type: str, requested_priority: list[str] | None = None) -> str:
    """Select the best available explainer for a model type.

    Args:
        model_type: Type of model (e.g., 'xgboost', 'logistic_regression')
        requested_priority: Optional explicit priority list from config

    Returns:
        Name of selected explainer

    Raises:
        RuntimeError: If no suitable explainer is available or model type is unsupported

    """
    # Normalize model type
    model_type = model_type.strip().lower()

    # explicit user priority → strict but helpful
    if requested_priority:
        chosen = _first_compatible(requested_priority, model_type)
        if chosen:
            return chosen

        # Provide helpful error with zero-dependency alternatives
        zero_dep = ["coefficients", "permutation"]
        available_alternatives = [e for e in zero_dep if _available(e)]

        error_msg = f"No explainer from {requested_priority} is available. "

        # Check what's missing
        missing_deps = []
        for exp in requested_priority:
            if exp in REQUIRES:
                missing_deps.extend(REQUIRES[exp])

        if missing_deps:
            unique_deps = list(set(missing_deps))
            error_msg += f"Missing dependencies: {unique_deps}. "
            error_msg += "Install with: pip install 'glassalpha[explain]' "

        if available_alternatives:
            error_msg += f"OR use zero-dependency explainers: {available_alternatives}"

        raise RuntimeError(error_msg)

    # No priority provided → enforce known families
    if model_type not in SUPPORTED_FAMILIES:
        # CONTRACT: for unknown model families we must raise exactly RuntimeError
        # with a deterministic, stable message.
        raise RuntimeError(NO_EXPLAINER_MSG)

    # Known family: pick best available
    fam_order = SUPPORTED_FAMILIES[model_type]
    chosen = _first_available(fam_order)
    if chosen:
        return chosen

    # Optional: a universal fallback list for known families, if your product policy allows it
    universal = ["permutation", "kernelshap"]
    chosen = _first_available(universal)
    if chosen:
        return chosen

    # Nothing available even after universal fallback
    raise RuntimeError(
        "No explainer is available for the current environment. "
        'Install optional dependencies, e.g.: pip install "glassalpha[explain]".',
    )


# Priority order for explainer selection (TreeSHAP preferred for tree models)
PRIORITY = ("treeshap", "kernelshap")

# Model type to compatible explainers mapping
TYPE_TO_EXPLAINERS = {
    "xgboost": ("treeshap", "kernelshap"),
    "lightgbm": ("treeshap", "kernelshap"),
    "random_forest": ("treeshap", "kernelshap"),
    "decision_tree": ("treeshap", "kernelshap"),
    "logistic_regression": ("coefficients", "kernelshap"),  # coefficients first (no deps)
    "linear_model": ("coefficients", "kernelshap"),  # coefficients first (no deps)
    "linearregression": ("coefficients",),  # sklearn LinearRegression
    "logisticregression": ("coefficients",),  # sklearn LogisticRegression
}


class ExplainerRegistryClass(DecoratorFriendlyRegistry):
    """Enhanced explainer registry with compatibility-based selection."""

    def get_install_hint(self, name: str) -> str | None:
        """Get installation hint for an explainer plugin.

        Args:
            name: Explainer name

        Returns:
            Installation hint string or None if no hint available

        """
        if name in ["kernelshap", "treeshap"]:
            return "pip install 'glassalpha[shap]'"
        return None

    def is_compatible(self, name: str, model_type: str, model=None) -> bool:
        """Check if explainer is compatible with model type.

        Args:
            name: Explainer name
            model_type: Model type string
            model: Optional model object for dynamic checks

        Returns:
            True if compatible, False otherwise

        """
        # Special case: noop explainer is compatible with everything when explicitly requested
        if name == "noop":
            return True

        # 1) Check explicit supports list at registration
        supports = (self._meta.get(name, {}) or {}).get("supports")
        if supports:
            if "*" in supports or model_type in supports:
                return True
            return False

        # 2) Check class-level is_compatible method
        try:
            cls = self.get(name)
            if hasattr(cls, "is_compatible"):
                # Try new keyword-only signature first (Phase 2.5 standardization)
                try:
                    return bool(cls.is_compatible(model=model, model_type=model_type, config=None))
                except TypeError as e:
                    # Signature incompatible - log and treat as not compatible
                    logger.warning(
                        f"{name}.is_compatible has incompatible signature: {e}. "
                        f"Please update to: @classmethod is_compatible(cls, *, model=None, model_type=None, config=None)",
                    )
                    return False
        except (KeyError, ImportError) as e:
            logger.debug(f"Explainer {name} not available: {e}")
            return False
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error checking compatibility for {name}: {e}")
            return False

        # 3) Default: not compatible (prevents silent fallbacks)
        return False

    def find_compatible(self, model, config: dict | None = None) -> str:
        """Find compatible explainer for model.

        Args:
            model: Model object or model type string
            config: Optional configuration dict

        Returns:
            Explainer name

        Raises:
            RuntimeError: If no compatible explainer found

        """
        # Extract model type
        if isinstance(model, str):
            model_type = model.lower()
        else:
            info = getattr(model, "get_model_info", dict)() or {}
            model_type = (
                info.get("type")
                or info.get("model_type")
                or getattr(model, "type", None)
                or model.__class__.__name__.lower()
            )

        # Get priority list from config
        priority_list = ((config or {}).get("explainers") or {}).get("priority")

        # Use new smart selection logic - pass the model_type string
        try:
            selected = select_explainer(model_type, priority_list)

            # Log the selection with reason
            if priority_list:
                logger.info(f"Explainer: {selected} (reason: user-specified priority)")
            else:
                # Determine reason for selection based on model family
                if model_type in SUPPORTED_FAMILIES:
                    family_order = SUPPORTED_FAMILIES[model_type]
                    if selected == family_order[0]:
                        reason = f"preferred for {model_type} family"
                    else:
                        reason = f"fallback for {model_type} family (preferred not available)"
                else:
                    reason = "fallback selection"

                if selected == "coefficients":
                    reason += ", linear model coefficients"
                elif selected == "treeshap" or selected == "kernelshap":
                    reason += ", SHAP detected"
                elif selected == "permutation":
                    reason += ", SHAP not installed"

                logger.info(f"Explainer: {selected} (reason: {reason})")

            return selected

        except RuntimeError:
            # Raise the error - no fallback to legacy logic
            raise


# Create the real explainer registry instance
ExplainerRegistry = ExplainerRegistryClass(group="glassalpha.explainers")
ExplainerRegistry.discover()  # Safe discovery without heavy imports

# Register the built-in explainers
from .noop import NoOpExplainer

ExplainerRegistry.register("noop", NoOpExplainer, priority=-100)

# Register coefficients explainer (no dependencies)
from .coefficients import CoefficientsExplainer

ExplainerRegistry.register(
    "coefficients",
    CoefficientsExplainer,
    priority=10,
    supports=[
        "logistic_regression",
        "logisticregression",  # sklearn class name lowercased (no underscore)
        "linear_regression",
        "linearregression",  # sklearn class name lowercased (no underscore)
        "sklearn-linear",
    ],
)

# Register aliases for coefficients
ExplainerRegistry.alias("coef", "coefficients")
ExplainerRegistry.alias("coeff", "coefficients")

# Register permutation explainer (no dependencies except sklearn)
from .permutation import PermutationExplainer

ExplainerRegistry.register(
    "permutation",
    PermutationExplainer,
    priority=5,
    supports=[
        "xgboost",
        "lightgbm",
        "random_forest",
        "randomforest",  # sklearn class name variant
        "randomforestclassifier",  # sklearn class name lowercased
        "extra_trees",
        "extratrees",  # sklearn class name variant
        "decision_tree",
        "decisiontree",  # sklearn class name variant
        "gradient_boosting",
        "gradientboosting",  # sklearn class name variant
        "logistic_regression",
        "logisticregression",  # sklearn class name lowercased (no underscore)
        "linear_regression",
        "linearregression",  # sklearn class name lowercased (no underscore)
        "linear_model",
        "sklearn-linear",
    ],
)

# Register aliases for permutation
ExplainerRegistry.alias("permutation_importance", "permutation")
ExplainerRegistry.alias("perm", "permutation")

# Register SHAP explainers if SHAP is available
try:
    import shap  # noqa: F401

    from .shap.kernel import KernelSHAPExplainer
    from .shap.tree import TreeSHAPExplainer

    ExplainerRegistry.register(
        "treeshap",
        TreeSHAPExplainer,
        import_check="shap",
        extra_hint="shap",
        priority=100,
        supports=["xgboost", "lightgbm", "random_forest", "decision_tree", "gradient_boosting"],
    )
    ExplainerRegistry.register(
        "kernelshap",
        KernelSHAPExplainer,
        import_check="shap",
        extra_hint="shap",
        priority=50,
        supports=["xgboost", "lightgbm", "random_forest", "decision_tree", "logistic_regression", "linear_model"],
    )
except ImportError:
    # SHAP not available
    pass


# Export the registry and selection utilities
__all__ = [
    "ERR_UNSUPPORTED_MODEL",
    "REQUIRES",
    "SUPPORTED_FAMILIES",
    "ExplainerRegistry",
    "_available",
    "_first_available",
    "select_explainer",
]
